#!/usr/bin/env python
"""Georeferenced tiled inference over a state-year raster → predicted-yield GeoTIFF.

Requires real HLS imagery on disk and a trained checkpoint. Example:
  python scripts/predict_raster.py --config configs/experiments/us_soybeans.yaml \
      checkpoint=/path/to/farm.ckpt state=IL year=2018
"""
import sys

from farm_us.cli import _expand_overrides, _kv
from farm_us.config import load_config


def main(argv):
    # parse --config and overrides
    config = None
    rest = []
    it = iter(argv)
    for a in it:
        if a == "--config":
            config = next(it)
        else:
            rest.append(a)
    cfg = load_config(config, _expand_overrides(rest))
    state = _kv(rest, "state"); year = _kv(rest, "year"); ckpt = _kv(rest, "checkpoint")
    if not (state and year and ckpt):
        print("Need state=, year=, checkpoint= (and real imagery on disk).")
        return
    import rasterio

    from farm_us.data.normalization import NormStats
    from farm_us.data.raster_readers import build_reader
    from farm_us.evaluation.inference import predict_state_year, write_geotiff
    from farm_us.training.lightning_module import FarmLightningModule

    reader = build_reader(cfg.data)
    norm = NormStats.load(cfg.norm.stats_path)
    lm = FarmLightningModule.load_from_checkpoint(ckpt, cfg=cfg)
    res = predict_state_year(lm.model, reader, cfg, norm, state, int(year))
    with rasterio.open(reader._label_path(state, int(year))) as ds:
        transform, crs = ds.transform, ds.crs
    out = f"outputs/predictions/{cfg.data.crop.lower()}_{state}_{year}_pred.tif"
    write_geotiff(out, res["prediction"], transform, crs, nodata=cfg.data.nodata)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
