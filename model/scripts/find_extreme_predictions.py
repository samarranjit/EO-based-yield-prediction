#!/usr/bin/env python
"""Locate physically implausible predictions (e.g. negative yield) and map them to place.

Chip-level metrics cannot find these: a handful of wild pixels inside a
~3000-pixel chip barely moves that chip's mean RMSE, so they hide in the
aggregate. This re-runs inference over the test year's qualifying chips, finds
every pixel outside a plausible range, and reports exactly which chip, which
pixel, which lat/lon and which county each one falls in.

Fast (minutes, GPU): loads the saved norm_stats.json rather than recomputing
fold statistics, so it skips the ~1h pass that `evaluate` does.

Example:
  uv run python scripts/find_extreme_predictions.py \
      --config configs/experiments/maryland_soybeans.yaml \
      checkpoint=outputs/runs/maryland_soybeans/test2024/checkpoints/farm-periodic-119.ckpt \
      state=MD year=2024 min_valid=0
"""
import sys
from pathlib import Path

import numpy as np

from farm_us.cli import _expand_overrides, _kv
from farm_us.config import load_config


def main(argv):
    config = None
    rest = []
    it = iter(argv)
    for a in it:
        if a == "--config":
            config = next(it)
        else:
            rest.append(a)
    cfg = load_config(config, _expand_overrides(rest))
    state = _kv(rest, "state")
    year = _kv(rest, "year")
    ckpt = _kv(rest, "checkpoint")
    min_valid = float(_kv(rest, "min_valid") or 0.0)
    if not (state and year and ckpt):
        print("Need state=, year=, checkpoint=")
        return
    year = int(year)

    import geopandas as gpd
    import pandas as pd
    import rasterio
    import torch
    from rasterio.features import rasterize

    from farm_us.data.normalization import NormStats, find_norm_stats
    from farm_us.data.raster_readers import build_reader
    from farm_us.evaluation.inference import predict_and_compare_test_year
    from farm_us.training.lightning_module import FarmLightningModule
    from farm_us.utils.geospatial import pixel_to_lonlat

    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_path = find_norm_stats(ckpt)
    if norm_path is None:
        print(f"Could not find norm_stats.json in any parent directory of {ckpt}")
        return
    norm = NormStats.load(norm_path)
    print(f"device={device}  norm_stats={norm_path}")

    reader = build_reader(cfg.data)
    lm = FarmLightningModule.load_from_checkpoint(ckpt, cfg=cfg, use_dummy=False)
    res = predict_and_compare_test_year(lm.model, reader, cfg, norm, state, year, device=device)
    pred, actual = res["prediction"], res["actual"]

    finite = np.isfinite(pred)
    bad = finite & (pred < min_valid)
    n_valid = int(finite.sum())
    n_bad = int(bad.sum())
    print(f"\nvalid predicted pixels: {n_valid:,}")
    print(f"pixels below {min_valid}: {n_bad:,}  ({100.0 * n_bad / max(1, n_valid):.6f}%)")
    if n_bad == 0:
        print("No implausible predictions found.")
        return
    print(f"predicted range over flagged pixels: {pred[bad].min():.3f} .. {pred[bad].max():.3f}")

    with rasterio.open(reader._label_path(state, year)) as ds:
        transform, crs = ds.transform, ds.crs
        h, w = ds.height, ds.width

    counties = gpd.read_file(cfg.data.counties_path)
    if counties.crs != crs:
        counties = counties.to_crs(crs)
    county_idx = rasterize(
        [(g, i + 1) for i, g in enumerate(counties.geometry)],
        out_shape=(h, w), transform=transform, fill=0, dtype="int32",
    )

    chip = cfg.data.chip_size
    rows, cols = np.where(bad)
    records = []
    for r, c in zip(rows, cols, strict=False):
        lon, lat = pixel_to_lonlat(transform, float(r), float(c), str(crs))
        ci = int(county_idx[r, c])
        records.append({
            "row": int(r), "col": int(c),
            "chip_row_off": (int(r) // chip) * chip, "chip_col_off": (int(c) // chip) * chip,
            "sample_id": f"{cfg.data.crop.lower()}_{state}_{year}_r{(int(r)//chip)*chip}_c{(int(c)//chip)*chip}",
            "predicted": float(pred[r, c]),
            "actual": float(actual[r, c]) if np.isfinite(actual[r, c]) else np.nan,
            "lat": lat, "lon": lon,
            "county": counties.iloc[ci - 1]["NAME"] if ci > 0 else "OUTSIDE_COUNTIES",
            "county_geoid": counties.iloc[ci - 1]["GEOID"] if ci > 0 else None,
        })

    df = pd.DataFrame(records)
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"extreme_predictions_{state}_{year}_{Path(ckpt).stem}.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== by county ===")
    print(df.groupby("county").agg(
        n_pixels=("predicted", "size"),
        min_pred=("predicted", "min"),
        mean_pred=("predicted", "mean"),
        mean_actual=("actual", "mean"),
    ).sort_values("n_pixels", ascending=False).to_string())

    print("\n=== by chip ===")
    print(df.groupby("sample_id").agg(
        n_pixels=("predicted", "size"),
        min_pred=("predicted", "min"),
        mean_actual=("actual", "mean"),
    ).sort_values("n_pixels", ascending=False).head(15).to_string())

    print(f"\nwrote {out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main(sys.argv[1:])
