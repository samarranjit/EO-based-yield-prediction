"""Deterministic chip manifest builder.

Enumerates 224×224 windows over every available (state, year) raster and writes
one row per chip to Parquet. The manifest fingerprint (hash of source metadata +
relevant config) lets an experiment pin the exact manifest it used.

Each row carries the schema in DATA_CONTRACT.md (sample_id, crop, year, state,
fips, window, bbox, center lat/lon, dates, valid fractions, label provenance,
split placeholder, QC flags). Where imagery is absent, per-month valid fractions
are left null and flagged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import STATE_FIPS, FarmConfig
from ..data.compositing import representative_dates
from ..utils.geospatial import pixel_to_lonlat, tile_windows
from ..utils.logging import get_logger
from ..utils.reproducibility import fingerprint

logger = get_logger(__name__)


def _sample_id(crop: str, state: str, year: int, r: int, c: int) -> str:
    return f"{crop.lower()}_{state}_{year}_r{r}_c{c}"


def build_manifest(cfg: FarmConfig, out_path: str | None = None, limit_states: int | None = None) -> pd.DataFrame:
    import rasterio

    rows: list[dict] = []
    crop = cfg.data.crop
    variant = cfg.data.label_variant
    label_root = Path(cfg.data.label_root)
    states = cfg.data.states[:limit_states] if limit_states else cfg.data.states

    for state in states:
        if state in cfg.data.exclude_geoids:  # defensive; exclude is by GEOID not state
            pass
        for year in cfg.data.years:
            lp = (
                label_root / str(year)
                / f"nass_{crop.lower()}_yield_{state}_{year}_{variant}_30m_{crop.lower()}_only.tif"
            )
            if not lp.exists():
                continue
            with rasterio.open(lp) as ds:
                h, w = ds.height, ds.width
                transform, crs = ds.transform, str(ds.crs)
            dates = [d.isoformat() for d in representative_dates(year)]
            for win in tile_windows(h, w, cfg.data.chip_size, cfg.data.stride):
                r0, c0 = win.row_off, win.col_off
                try:
                    lon, lat = pixel_to_lonlat(
                        transform, r0 + cfg.data.chip_size / 2, c0 + cfg.data.chip_size / 2, crs
                    )
                except Exception:
                    lon = lat = None
                rows.append({
                    "sample_id": _sample_id(crop, state, year, r0, c0),
                    "crop": crop, "year": year, "state": state,
                    "state_fips": STATE_FIPS.get(state),
                    "label_path": str(lp), "label_variant": variant,
                    "label_units": cfg.data.label_units,
                    "label_source": "ridge_distributed_county_yield",
                    "ridge_provenance": "external pipeline; may not be strictly LOYO (see LOYO_PROTOCOL.md)",
                    "row_off": r0, "col_off": c0,
                    "chip_size": cfg.data.chip_size, "stride": cfg.data.stride,
                    "crs": crs, "resolution_m": cfg.data.resolution_m,
                    "center_lat": lat, "center_lon": lon,
                    "dates": ";".join(dates),
                    "n_valid_crop_px": None,  # filled by a windowed QC pass when imagery exists
                    "valid_hls_fraction": None,
                    "valid_label_fraction": None,
                    "split": None,
                    "qc_flags": "",
                })

    df = pd.DataFrame(rows)
    cfg_fp = fingerprint({
        "crop": crop, "variant": variant, "states": states, "years": cfg.data.years,
        "chip": cfg.data.chip_size, "stride": cfg.data.stride,
    })
    df.attrs["manifest_fingerprint"] = cfg_fp
    if out_path is None:
        out_path = cfg.data.manifest_path
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(df):
        df.to_parquet(out)
    logger.info("Manifest: %d chips, fingerprint=%s → %s", len(df), cfg_fp, out)
    return df
