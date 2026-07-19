"""Data inventory: discover on-disk labels / CDL masks / imagery and summarize
CRS, resolution, size, band count, no-data, available state-years, and gaps.

Writes docs/DATA_INVENTORY.md-friendly JSON to outputs/data_inventory.json.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from ..config import FarmConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

_LABEL_RE = re.compile(r"nass_.+_yield_(?P<state>[A-Z]{2})_(?P<year>\d{4})_")
_CDL_RE = re.compile(r"cdl_.+_(?P<state>[A-Z]{2})_(?P<year>\d{4})\.tif")


def _raster_meta(path: Path) -> dict:
    try:
        import rasterio

        with rasterio.open(path) as ds:
            return {
                "width": ds.width, "height": ds.height, "bands": ds.count,
                "dtype": ds.dtypes[0], "crs": str(ds.crs),
                "res": list(ds.res), "nodata": ds.nodata,
            }
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def scan(cfg: FarmConfig, sample_meta: bool = True) -> dict:
    inv: dict = {
        "crop": cfg.data.crop,
        "roots": {
            "label_root": cfg.data.label_root,
            "cdl_root": cfg.data.cdl_root,
            "imagery_root": cfg.data.imagery_root,
        },
        "labels": defaultdict(list),
        "cdl": defaultdict(list),
        "imagery_present": Path(cfg.data.imagery_root).exists(),
        "sample_meta": {},
        "missing_state_years": [],
    }

    label_root = Path(cfg.data.label_root)
    if label_root.exists():
        for p in label_root.rglob("*.tif"):
            m = _LABEL_RE.search(p.name)
            if m:
                inv["labels"][m["state"]].append(int(m["year"]))
                if sample_meta and "label" not in inv["sample_meta"]:
                    inv["sample_meta"]["label"] = {"path": str(p), **_raster_meta(p)}

    cdl_root = Path(cfg.data.cdl_root)
    if cdl_root.exists():
        for p in cdl_root.glob("*.tif"):
            m = _CDL_RE.search(p.name)
            if m:
                inv["cdl"][m["state"]].append(int(m["year"]))
                if sample_meta and "cdl" not in inv["sample_meta"]:
                    inv["sample_meta"]["cdl"] = {"path": str(p), **_raster_meta(p)}

    for st in cfg.data.states:
        for yr in cfg.data.years:
            has_label = yr in inv["labels"].get(st, [])
            has_cdl = yr in inv["cdl"].get(st, [])
            if not (has_label and has_cdl):
                inv["missing_state_years"].append(
                    {"state": st, "year": yr, "label": has_label, "cdl": has_cdl}
                )

    inv["labels"] = {k: sorted(v) for k, v in inv["labels"].items()}
    inv["cdl"] = {k: sorted(v) for k, v in inv["cdl"].items()}
    return inv


def write_inventory(cfg: FarmConfig, out_json: str = "outputs/data_inventory.json") -> dict:
    inv = scan(cfg)
    p = Path(out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(inv, indent=2, default=str))
    logger.info("Wrote inventory → %s", p)
    return inv
