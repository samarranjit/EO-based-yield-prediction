"""
Create 30 m yield-label rasters from NASS county yield data + CDL crop masks.

For each (state, year) pair two GeoTIFFs are written:

  nearest      — all crop pixels in a county share the same county yield value
  bilinear10km — county yields rasterised at ~10 km then bilinear-resampled to 30 m
                 (FARM-style smooth surface; softens sharp county boundaries)

Non-crop pixels are set to NODATA.
Counties in EXCLUDE_GEOIDS are masked out (leakage protection).

    python scripts/04_make_yield_label_rasters.py
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.ops import unary_union
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    NASS_CSV, COUNTIES_GPKG, CDL_MASK_DIR, YIELD_LABEL_DIR,
    CROP_NAME, VALUE_COL, NODATA,
    EXCLUDE_GEOIDS, STATE_FIPS, TARGET_CRS,
    TARGET_SCALE_M, COARSE_RES_M,
    START_YEAR, END_YEAR, STATE_ALPHA,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_yield_df() -> pd.DataFrame:
    df = pd.read_csv(NASS_CSV, dtype={"GEOID": str})
    df = df[~df["GEOID"].isin(EXCLUDE_GEOIDS)]
    return df


def load_counties() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(COUNTIES_GPKG)
    if "GEOID" not in gdf.columns:
        gdf["GEOID"] = gdf["STATEFP"].str.zfill(2) + gdf["COUNTYFP"].str.zfill(3)
    return gdf


# ---------------------------------------------------------------------------
# Rasterisation helpers
# ---------------------------------------------------------------------------

def _shapes(gdf: gpd.GeoDataFrame, value_col: str):
    """Yield (geometry, value) pairs, skipping null values."""
    for geom, val in zip(gdf.geometry, gdf[value_col]):
        if geom is not None and not np.isnan(val):
            yield geom, float(val)


def make_nearest_label(
    gdf: gpd.GeoDataFrame,
    profile: dict,
    value_col: str,
    nodata: float,
) -> np.ndarray:
    """Rasterise county polygons with their yield values at the CDL 30 m grid."""
    return rasterize(
        shapes=list(_shapes(gdf, value_col)),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=nodata,
        dtype="float32",
        all_touched=False,
    )


def make_bilinear_label(
    gdf: gpd.GeoDataFrame,
    profile: dict,
    value_col: str,
    nodata: float,
    coarse_res_m: int,
) -> np.ndarray:
    """
    Build a smooth yield surface:
      1. Rasterise county yields onto a coarse (~10 km) grid.
      2. Bilinear-resample back to 30 m.

    This mimics the FARM-style label where county boundaries are blended
    rather than hard-cut, producing a spatially gradual yield surface.
    """
    t = profile["transform"]
    h, w = profile["height"], profile["width"]
    crs = profile["crs"]

    left   = t.c
    top    = t.f
    right  = left + w * t.a
    bottom = top  + h * t.e   # t.e is negative (north-up raster)

    factor   = max(1, coarse_res_m // int(abs(t.a)))
    coarse_h = max(1, h // factor)
    coarse_w = max(1, w // factor)
    coarse_t = from_bounds(left, bottom, right, top, coarse_w, coarse_h)

    coarse_arr = rasterize(
        shapes=list(_shapes(gdf, value_col)),
        out_shape=(coarse_h, coarse_w),
        transform=coarse_t,
        fill=nodata,
        dtype="float32",
        all_touched=False,
    )

    fine_arr = np.full((h, w), nodata, dtype="float32")
    reproject(
        source=coarse_arr,
        destination=fine_arr,
        src_transform=coarse_t,
        src_crs=crs,
        dst_transform=t,
        dst_crs=crs,
        resampling=Resampling.bilinear,
        src_nodata=nodata,
        dst_nodata=nodata,
    )
    return fine_arr


def make_state_boundary_mask(
    state_counties: gpd.GeoDataFrame,
    profile: dict,
) -> np.ndarray:
    """
    Rasterise the dissolved state boundary as a binary mask (1 = inside, 0 = outside).
    Used to prevent bilinear bleed across state lines.
    """
    state_union = unary_union(state_counties.geometry)
    return rasterize(
        shapes=[(state_union, 1)],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=False,
    )


def apply_crop_mask(arr: np.ndarray, mask: np.ndarray, nodata: float) -> np.ndarray:
    """Zero out pixels where mask is False/0; mask may be boolean or uint8."""
    out = arr.copy()
    out[~mask.astype(bool)] = nodata
    return out


def save_raster(arr: np.ndarray, profile: dict, path: Path, nodata: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = {**profile, "dtype": "float32", "count": 1, "nodata": nodata, "compress": "lzw"}
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype("float32"), 1)


# ---------------------------------------------------------------------------
# Per state-year processing
# ---------------------------------------------------------------------------

def process(
    state: str,
    year: int,
    counties: gpd.GeoDataFrame,
    yield_df: pd.DataFrame,
) -> str:
    crop = CROP_NAME.lower()
    cdl_path = CDL_MASK_DIR / f"cdl_{crop}_{state}_{year}.tif"
    if not cdl_path.exists():
        return f"SKIP (CDL not found): {cdl_path.name}"

    with rasterio.open(cdl_path) as src:
        crop_mask = src.read(1)
        profile   = src.profile.copy()

    # Subset counties to this state
    state_fips   = STATE_FIPS[state]
    state_counties = counties[counties["STATEFP"] == state_fips].copy()

    year_yield = yield_df[
        (yield_df["year"] == year) & (yield_df["state"] == state)
    ][["GEOID", VALUE_COL]]

    if year_yield.empty:
        return f"SKIP (no yield data): {state} {year}"

    merged = state_counties.merge(year_yield, on="GEOID", how="left")
    merged = merged[merged[VALUE_COL].notna()].copy()
    if merged.empty:
        return f"SKIP (no county-yield join): {state} {year}"

    if merged.crs != profile["crs"]:
        merged = merged.to_crs(profile["crs"])
    if state_counties.crs != profile["crs"]:
        state_counties = state_counties.to_crs(profile["crs"])

    # State boundary mask — built from ALL counties (not just those with yield data)
    # to get the full legal state outline. Prevents bilinear bleed across state lines.
    state_mask = make_state_boundary_mask(state_counties, profile)
    combined_mask = (crop_mask == 1) & (state_mask == 1)

    prefix          = f"nass_{crop}_yield_{state}_{year}"
    county_val_dir  = YIELD_LABEL_DIR / "county_values" / str(year)
    bilinear_dir    = YIELD_LABEL_DIR / "bilinear"      / str(year)

    # --- nearest (one value per county) ---
    nearest = make_nearest_label(merged, profile, VALUE_COL, NODATA)
    nearest = apply_crop_mask(nearest, combined_mask, NODATA)
    save_raster(nearest, profile, county_val_dir / f"{prefix}_nearest_30m_{crop}_only.tif", NODATA)

    # --- bilinear (smooth FARM-style surface) ---
    bilinear = make_bilinear_label(merged, profile, VALUE_COL, NODATA, COARSE_RES_M)
    bilinear = apply_crop_mask(bilinear, combined_mask, NODATA)
    save_raster(bilinear, profile, bilinear_dir / f"{prefix}_bilinear10km_30m_{crop}_only.tif", NODATA)

    valid_px = int((nearest != NODATA).sum())
    return f"OK  {valid_px:>10,} crop px"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading NASS yield data…")
    yield_df = load_yield_df()
    print(f"  {len(yield_df)} county-year records")

    print("Loading county boundaries…")
    counties = load_counties()
    print(f"  {len(counties)} counties")

    combos = [(s, y) for s in STATE_ALPHA for y in range(START_YEAR, END_YEAR + 1)]
    print(f"\nProcessing {len(combos)} state-year combinations…\n")

    for state, year in tqdm(combos, desc="Yield labels"):
        msg = process(state, year, counties, yield_df)
        tqdm.write(f"  {state} {year}  {msg}")

    print("\nDone.")


if __name__ == "__main__":
    main()
