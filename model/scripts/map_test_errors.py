#!/usr/bin/env python
"""Georeferenced map of LOYO test-year prediction error.

Restricted to chips with real imagery (same qualifying-chip filter used in
training/eval) -- unlike scripts/predict_raster.py, which tiles the entire
state grid and would silently predict from zero-filled input outside a
chip-gated imagery download's footprint.

Writes 3 GeoTIFFs (predicted yield, actual yield, residual = pred - actual)
plus a PNG overlay: actual yield as background, error as a semi-transparent
diverging layer on top, so high-error areas are visible in geographic context.

Example:
  uv run python scripts/map_test_errors.py \
      --config configs/experiments/maryland_soybeans.yaml \
      checkpoint=outputs/runs/maryland_soybeans/test2024/checkpoints/farm-032-0.0000.ckpt \
      state=MD year=2024
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
    if not (state and year and ckpt):
        print("Need state=, year=, checkpoint= (and real imagery + a built manifest on disk).")
        return
    year = int(year)

    import rasterio
    import torch

    from farm_us.data.normalization import NormStats, find_norm_stats
    from farm_us.data.raster_readers import build_reader
    from farm_us.evaluation.inference import predict_and_compare_test_year, write_geotiff
    from farm_us.evaluation.plots import error_overlay_map
    from farm_us.training.lightning_module import FarmLightningModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}")

    # train_fold writes norm_stats.json at the run root, but checkpoints are often
    # filed into subdirectories -- search upward rather than assume a fixed depth.
    norm_path = find_norm_stats(ckpt)
    if norm_path is None:
        print(f"Could not find norm_stats.json in any parent directory of {ckpt}")
        return
    print(f"Using norm stats: {norm_path}")
    norm = NormStats.load(norm_path)

    reader = build_reader(cfg.data)
    lm = FarmLightningModule.load_from_checkpoint(ckpt, cfg=cfg, use_dummy=False)
    res = predict_and_compare_test_year(lm.model, reader, cfg, norm, state, year, device=device)

    with rasterio.open(reader._label_path(state, year)) as ds:
        transform, crs = ds.transform, ds.crs

    out_dir = Path("outputs/predictions/decoder_only")
    stem = f"{cfg.data.crop.lower()}_{state}_{year}"

    # _pred / _actual / _residual share ONE footprint (comparison_mask) so the
    # maps are directly comparable. _pred_full is the unrestricted model surface
    # -- it legitimately covers crop pixels with no test label, so it must never
    # be compared against _actual.
    write_geotiff(out_dir / f"{stem}_pred.tif", res["prediction"], transform, crs, nodata=cfg.data.nodata)
    write_geotiff(out_dir / f"{stem}_actual.tif", res["actual"], transform, crs, nodata=cfg.data.nodata)
    write_geotiff(out_dir / f"{stem}_residual.tif", res["residual"], transform, crs, nodata=cfg.data.nodata)
    write_geotiff(out_dir / f"{stem}_pred_full.tif", res["prediction_full"], transform, crs, nodata=cfg.data.nodata)
    write_geotiff(
        out_dir / f"{stem}_comparison_mask.tif",
        res["comparison_mask"].astype("float32"), transform, crs, nodata=cfg.data.nodata,
    )

    n_cmp = int(res["comparison_mask"].sum())
    n_full = int(np.isfinite(res["prediction_full"]).sum())
    print(f"comparable pixels: {n_cmp:,}   full prediction surface: {n_full:,}")
    if n_full:
        print(f"  ({n_full - n_cmp:,} predicted pixels have no valid test label; excluded from comparison)")

    error_overlay_map(
        res["actual"], res["residual"], out_dir / f"{stem}_error_map.png",
        title=f"{state} {year}: actual yield (background) + prediction error (overlay)",
    )
    print(f"wrote {out_dir}/{stem}_{{pred,actual,residual,pred_full,comparison_mask}}.tif and _error_map.png")


if __name__ == "__main__":
    main(sys.argv[1:])
