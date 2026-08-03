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

    from farm_us.data.normalization import NormStats
    from farm_us.data.raster_readers import build_reader
    from farm_us.evaluation.inference import predict_and_compare_test_year, write_geotiff
    from farm_us.evaluation.plots import error_overlay_map
    from farm_us.training.lightning_module import FarmLightningModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}")

    # norm_stats.json lives next to the checkpoint (train_fold saves it there), NOT
    # at the generic cfg.norm.stats_path default -- that path is never actually written.
    norm_path = Path(ckpt).parent.parent / "norm_stats.json"
    if not norm_path.exists():
        print(f"Could not find {norm_path} (expected next to the checkpoint's run directory).")
        return
    norm = NormStats.load(norm_path)

    reader = build_reader(cfg.data)
    lm = FarmLightningModule.load_from_checkpoint(ckpt, cfg=cfg, use_dummy=False)
    res = predict_and_compare_test_year(lm.model, reader, cfg, norm, state, year, device=device)

    with rasterio.open(reader._label_path(state, year)) as ds:
        transform, crs = ds.transform, ds.crs

    out_dir = Path("outputs/predictions")
    stem = f"{cfg.data.crop.lower()}_{state}_{year}"
    write_geotiff(out_dir / f"{stem}_pred.tif", res["prediction"], transform, crs, nodata=cfg.data.nodata)
    write_geotiff(out_dir / f"{stem}_actual.tif", res["actual"], transform, crs, nodata=cfg.data.nodata)
    write_geotiff(out_dir / f"{stem}_residual.tif", res["residual"], transform, crs, nodata=cfg.data.nodata)
    error_overlay_map(
        res["actual"], res["residual"], out_dir / f"{stem}_error_map.png",
        title=f"{state} {year}: actual yield (background) + prediction error (overlay)",
    )
    print(f"wrote {out_dir}/{stem}_{{pred,actual,residual}}.tif and _error_map.png")


if __name__ == "__main__":
    main(sys.argv[1:])
