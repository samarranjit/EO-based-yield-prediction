#!/usr/bin/env python3
# ============================================================
# NASA CMR-STAC / LP DAAC HLS v2.0 COUNTY VI SUMMARY
# WITH TIMING DIAGNOSTICS + PARALLEL SCENE PROCESSING + PIXEL GEOTIFF EXPORT
# ============================================================

import argparse
import calendar
import json
import math
import random
import sys
import tempfile
import os
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping
from pystac_client import Client
from tqdm import tqdm


# ------------------------------------------------------------
# FIPS map
# ------------------------------------------------------------
STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}


# ------------------------------------------------------------
# HLS band mapping
# ------------------------------------------------------------
S30_BANDS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "swir1": "B11",
    "fmask": "Fmask",
}

L30_BANDS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B05",
    "swir1": "B06",
    "fmask": "Fmask",
}

VI_NAMES = ["NDVI", "EVI", "GCVI", "NDWI"]

REFLECTANCE_SCALE = 0.0001
EPS = 1e-6


# ============================================================
# Timing helpers
# ============================================================

def format_runtime(seconds):
    seconds = float(seconds)
    minutes = seconds / 60
    hours = seconds / 3600

    if seconds < 60:
        return f"{seconds:.1f} sec"
    elif minutes < 60:
        return f"{minutes:.2f} min"
    else:
        return f"{hours:.2f} hr"


def rounded_timing_dict(timing):
    out = {}
    for k, v in timing.items():
        if isinstance(v, str):
            out[k] = v
        elif "count" in k or k.endswith("_reads"):
            out[k] = int(v)
        else:
            out[k] = round(float(v), 3)
    return out


def print_county_timing_summary(timing):
    print("\n---------------- TIMING SUMMARY ----------------")
    print(f"Total county runtime:       {format_runtime(timing.get('county_total_sec', 0))}")
    print(f"Load CDL/crop grid:         {format_runtime(timing.get('load_cdl_crop_grid_sec', 0))}")
    print(f"Search HLS STAC:            {format_runtime(timing.get('search_hls_items_sec', 0))}")
    print(f"Remote read + warp total:   {format_runtime(timing.get('remote_read_warp_total_sec', 0))}")
    print(f"Process HLS items total:    {format_runtime(timing.get('process_hls_items_loop_sec', 0))}")
    print(f"Accumulate VI values:       {format_runtime(timing.get('accumulate_vi_sec', 0))}")
    print(f"Final reduce/summary:       {format_runtime(timing.get('final_reduce_summary_sec', 0))}")
    print(f"Pixel GeoTIFF write:        {format_runtime(timing.get('pixel_tif_write_sec', 0))}")
    print(f"CSV write:                  {format_runtime(timing.get('csv_write_sec', 0))}")
    print(f"JSON log write:             {format_runtime(timing.get('json_log_write_sec', 0))}")
    print(f"Intentional sleep time:     {format_runtime(timing.get('sleep_after_item_sec', 0))}")

    reads = timing.get("remote_asset_reads", 0)
    if reads > 0:
        avg = timing.get("remote_read_warp_total_sec", 0) / reads
        print(f"Remote asset reads:         {int(reads)}")
        print(f"Avg sec / remote asset:     {avg:.3f}")

    used = timing.get("hls_items_used", 0)
    if used > 0:
        avg_item = timing.get("process_hls_items_loop_sec", 0) / used
        print(f"Avg sec / used HLS item:    {avg_item:.3f}")

    print("------------------------------------------------\n")


# ============================================================
# Utility helpers
# ============================================================

def sleep_with_jitter(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds + random.uniform(0, min(1.0, seconds * 0.25)))


def retry_call(label, func, max_retries=4, base_sleep=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            wait = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            print(
                f"  WARNING: {label} failed on attempt "
                f"{attempt}/{max_retries}: {e}",
                flush=True,
            )   
            print(
                f"  Sleeping {wait:.1f} seconds before retry...",
                flush=True,
            )
            time.sleep(wait)
    raise last_err


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def append_row_to_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def safe_float(x):
    if x is None:
        return np.nan
    try:
        if np.isnan(x):
            return np.nan
    except TypeError:
        pass
    return float(x)


def cloud_cover(item):
    val = item.properties.get("eo:cloud_cover")
    if val is None:
        val = item.properties.get("cloud_cover")
    if val is None:
        val = item.properties.get("CLOUD_COVERAGE")
    return None if val is None else float(val)


def item_datetime(item):
    return item.properties.get("datetime", "")


def window_covering_bounds(bounds, transform):
    w = from_bounds(*bounds, transform=transform)

    col0 = math.floor(w.col_off)
    row0 = math.floor(w.row_off)
    col1 = math.ceil(w.col_off + w.width)
    row1 = math.ceil(w.row_off + w.height)

    return Window(
        col_off=col0,
        row_off=row0,
        width=max(1, col1 - col0),
        height=max(1, row1 - row0),
    )


# ============================================================
# Boundary loading
# ============================================================

def download_counties_2018():
    url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip"

    def _download():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return r.content

    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "cb_2018_us_county_500k.zip"
        zip_path.write_bytes(
            retry_call("download Census county boundaries", _download, max_retries=4)
        )
        gdf = gpd.read_file(zip_path)

    return gdf.to_crs("EPSG:4326")


def select_counties(counties, state_abbr, county_fips=None, county_name=None):
    state_abbr = state_abbr.upper()
    state_fips = STATE_ABBR_TO_FIPS[state_abbr]

    out = counties[counties["STATEFP"] == state_fips].copy()

    if county_fips:
        county_fips = county_fips.zfill(3)
        out = out[out["COUNTYFP"] == county_fips].copy()

    if county_name:
        out = out[out["NAME"].str.lower().str.contains(county_name.lower())].copy()

    if out.empty:
        raise ValueError("No county matched your state/county filters.")

    out = out.sort_values(["GEOID"]).reset_index(drop=True)
    return out


# ============================================================
# CDL / crop mask
# ============================================================

def load_county_crop_grid(cdl_path, county_row_4326, cdl_mode, include_double_crop):
    with rasterio.open(cdl_path) as src:
        if src.crs is None:
            raise ValueError("CDL raster has no CRS.")

        county_gdf = gpd.GeoDataFrame(
            [county_row_4326],
            geometry=[county_row_4326.geometry],
            crs="EPSG:4326",
        ).to_crs(src.crs)

        geom = county_gdf.geometry.iloc[0]
        window = window_covering_bounds(geom.bounds, src.transform)

        cdl = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=0,
        )

        target_transform = src.window_transform(window)
        target_crs = src.crs
        target_shape = cdl.shape

    county_mask = rasterize(
        [(geom, 1)],
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    if cdl_mode == "binary":
        crop_mask = cdl > 0
    elif cdl_mode == "raw":
        crop_mask = cdl == 5
        if include_double_crop:
            crop_mask = crop_mask | (cdl == 26)
    else:
        raise ValueError("cdl_mode must be 'binary' or 'raw'.")

    crop_mask = crop_mask & county_mask

    return {
        "crop_mask": crop_mask,
        "target_transform": target_transform,
        "target_crs": target_crs,
        "height": target_shape[0],
        "width": target_shape[1],
    }


# ============================================================
# NASA CMR-STAC / LP DAAC HLS v2.0
# ============================================================

CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
CMR_COLLECTION_L30 = "HLSL30_2.0"
CMR_COLLECTION_S30 = "HLSS30_2.0"
S30_START_DATE = "2015-11-28"


def prepare_earthdata_auth(args):
    """
    Prepare local Earthdata authentication for GDAL/rasterio HTTPS COG reads.

    Recommended setup:
      pip install earthaccess
      python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"

    This function also tries to run earthaccess.login(...) for convenience.
    GDAL/rasterio then uses ~/.netrc and a cookie jar for authenticated HTTPS reads.
    """
    if args.cookie_file is None:
        args.cookie_file = args.out_dir / "earthdata_gdal_cookies.txt"
    args.cookie_file.parent.mkdir(parents=True, exist_ok=True)
    args.cookie_file.touch(exist_ok=True)

    os.environ["EARTHDATA_GDAL_COOKIE_FILE"] = str(args.cookie_file)

    if args.netrc_file is not None:
        os.environ["EARTHDATA_NETRC_FILE"] = str(args.netrc_file)

    if not args.earthdata_login:
        print("Earthdata login step skipped because --no-earthdata-login was used.")
        print("GDAL will still try to use your existing ~/.netrc / cookie file.")
        return

    try:
        import earthaccess

        print("Checking Earthdata Login with earthaccess...")
        # strategy='interactive' opens a browser/input flow if needed.
        # persist=True saves credentials for future runs.
        earthaccess.login(strategy=args.earthdata_strategy, persist=True)
        print("Earthdata Login ready.")
    except ImportError:
        print("\nWARNING: earthaccess is not installed.")
        print("Install it with: pip install earthaccess")
        print("Or create a ~/.netrc with your Earthdata username/password.")
        print("Continuing anyway; raster reads may fail if GDAL cannot authenticate.\n")
    except Exception as e:
        print(f"\nWARNING: earthaccess login check failed: {e}")
        print("Continuing anyway; raster reads may fail if GDAL cannot authenticate.\n")


def gdal_http_env_options():
    """
    GDAL options for authenticated Earthdata HTTPS COG reads.

    The key pieces are:
      - GDAL_HTTP_NETRC=YES so GDAL can use ~/.netrc Earthdata credentials.
      - Cookie file/jar so redirects/auth cookies can be reused.
    """
    opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",

        # Prevent remote reads from hanging too long
        "GDAL_HTTP_CONNECTTIMEOUT": "20",
        "GDAL_HTTP_TIMEOUT": "180",

        # Retry transient LP DAAC / Earthdata responses
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "5",

        "GDAL_HTTP_NETRC": "YES",
    }

    cookie_file = os.environ.get("EARTHDATA_GDAL_COOKIE_FILE")
    if cookie_file:
        opts["GDAL_HTTP_COOKIEFILE"] = cookie_file
        opts["GDAL_HTTP_COOKIEJAR"] = cookie_file

    netrc_file = os.environ.get("EARTHDATA_NETRC_FILE")
    if netrc_file:
        opts["GDAL_HTTP_NETRC_FILE"] = netrc_file

    return opts


def open_pc_catalog():
    """
    Kept the old function name so the rest of the script changes minimally.
    This now opens NASA CMR-STAC / LPCLOUD instead of Microsoft Planetary Computer.
    """
    return Client.open(CMR_STAC_URL)


def cmr_collections_for_date_range(start_date, end_date):
    """
    HLSL30 starts in 2013. HLSS30 starts in late 2015.
    For crop seasons ending before Sentinel HLS exists, avoid querying S30.
    """
    collections = [CMR_COLLECTION_L30]
    if end_date >= S30_START_DATE:
        collections.append(CMR_COLLECTION_S30)
    return collections


def search_hls_items(
    catalog,
    county_geom_4326,
    start_date,
    end_date,
    max_scene_cloud,
    max_retries,
    cmr_page_limit=100,
    cmr_max_items=1000,
    cmr_spatial_mode="bbox",
):
    """
    Search NASA CMR-STAC / LPCLOUD HLS v2.0 items.

    Important CMR-STAC stability choices:
      1) Search one collection at a time instead of combined L30+S30.
      2) Use a small page limit; CMR-STAC can fail with large limits.
      3) Default to bbox search because it is less fragile than complex county polygons.

    bbox search may return a few extra scenes, but the raster read is still cropped/warped
    to the county CDL grid later, so this is safe for the county mean calculation.
    """
    geom = mapping(county_geom_4326)
    bbox = list(county_geom_4326.bounds)  # [minx, miny, maxx, maxy] in EPSG:4326
    dt = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    collections = cmr_collections_for_date_range(start_date, end_date)

    cmr_page_limit = max(1, min(int(cmr_page_limit), 250))
    cmr_max_items = max(1, int(cmr_max_items))
    cmr_spatial_mode = str(cmr_spatial_mode).lower().strip()

    print(f"CMR-STAC collections: {collections}")
    print(f"CMR-STAC spatial mode: {cmr_spatial_mode}")
    print(f"CMR-STAC page limit: {cmr_page_limit}")
    print(f"CMR-STAC max items per collection: {cmr_max_items}")

    items = []

    for collection in collections:
        def one_search(collection=collection):
            kwargs = {
                "collections": [collection],
                "datetime": dt,
                "limit": cmr_page_limit,
                "max_items": cmr_max_items,
            }

            if cmr_spatial_mode == "intersects":
                kwargs["intersects"] = geom
            elif cmr_spatial_mode == "bbox":
                kwargs["bbox"] = bbox
            else:
                raise ValueError("cmr_spatial_mode must be 'bbox' or 'intersects'.")

            search = catalog.search(**kwargs)
            return list(search.items())

        try:
            collection_items = retry_call(
                f"{collection} CMR-STAC search",
                one_search,
                max_retries=max_retries,
            )
            print(f"  {collection}: {len(collection_items)} items before cloud filter")
            items.extend(collection_items)
        except Exception as ee:
            print(f"  WARNING: {collection} search failed and will be skipped: {ee}")

    dedup = {}
    for item in items:
        dedup[item.id] = item
    items = list(dedup.values())

    filtered = []
    for item in items:
        cc = cloud_cover(item)
        if cc is None or cc <= max_scene_cloud:
            filtered.append(item)

    filtered = sorted(filtered, key=lambda x: (item_datetime(x), x.id))
    return filtered


def is_s30_item(item):
    cid = getattr(item, "collection_id", "") or ""
    iid = getattr(item, "id", "") or ""
    return cid in [CMR_COLLECTION_S30, "hls2-s30"] or "HLS.S30" in iid or ".S30." in iid or "S30" in iid


def is_l30_item(item):
    cid = getattr(item, "collection_id", "") or ""
    iid = getattr(item, "id", "") or ""
    return cid in [CMR_COLLECTION_L30, "hls2-l30"] or "HLS.L30" in iid or ".L30." in iid or "L30" in iid


def band_map_for_item(item):
    if is_s30_item(item):
        return S30_BANDS
    if is_l30_item(item):
        return L30_BANDS

    raise ValueError(f"Cannot identify HLS collection for item {item.id}")

def warm_up_earthdata_gdal(items, args):
    if len(items) == 0:
        return

    item = items[0]
    band_map = band_map_for_item(item)
    href = item.assets[band_map["fmask"]].href

    print("\nWarming up Earthdata/GDAL remote access...")
    print(f"Warm-up item: {item.id}")
    print(f"Warm-up href: {href}", flush=True)

    def _warmup():
        with rasterio.Env(**gdal_http_env_options()):
            with rasterio.open(href) as src:
                _ = src.crs
                _ = src.transform
                _ = src.width
                _ = src.height
        return True

    retry_call(
        "Earthdata/GDAL warm-up",
        _warmup,
        max_retries=args.max_retries,
        base_sleep=5.0,
    )

    print("Earthdata/GDAL warm-up complete.\n", flush=True)

def read_asset_to_target_grid(
    href,
    target_crs,
    target_transform,
    height,
    width,
    resampling,
    out_dtype="float32",
    dst_nodata=None,
    max_retries=4,
    timing=None,
    timing_key="remote_read_warp_total_sec",
):
    def _read():
        t0 = time.perf_counter()

        with rasterio.Env(**gdal_http_env_options()):
            with rasterio.open(href) as src:
                src_nodata = src.nodata
                with WarpedVRT(
                    src,
                    crs=target_crs,
                    transform=target_transform,
                    width=width,
                    height=height,
                    resampling=resampling,
                    src_nodata=src_nodata,
                    nodata=dst_nodata if dst_nodata is not None else src_nodata,
                ) as vrt:
                    arr = vrt.read(1, out_dtype=out_dtype)

        elapsed = time.perf_counter() - t0

        if timing is not None:
            timing["remote_read_warp_total_sec"] += elapsed
            timing[timing_key] += elapsed
            timing["remote_asset_reads"] += 1

        return arr

    return retry_call("read NASA Earthdata COG window", _read, max_retries=max_retries)


def hls_clear_mask(fmask, mask_adjacent_cloud=True, mask_high_aerosol=True):
    fmask = fmask.astype("uint8")

    cloud = ((fmask >> 1) & 1).astype(bool)
    adjacent = ((fmask >> 2) & 1).astype(bool)
    shadow = ((fmask >> 3) & 1).astype(bool)
    snow = ((fmask >> 4) & 1).astype(bool)

    bad = cloud | shadow | snow

    if mask_adjacent_cloud:
        bad = bad | adjacent

    if mask_high_aerosol:
        aerosol = (fmask >> 6) & 3
        bad = bad | (aerosol == 3)

    return ~bad


def scale_hls_reflectance(arr):
    arr = arr.astype("float32")
    arr[arr <= -9990] = np.nan
    arr = arr * REFLECTANCE_SCALE
    return arr


def compute_vi_arrays(blue, green, red, nir, swir1, clear_mask):
    refl_ok = (
        clear_mask
        & np.isfinite(nir) & (nir >= 0) & (nir <= 1.2)
        & np.isfinite(red) & (red >= 0) & (red <= 1.2)
        & np.isfinite(green) & (green >= 0) & (green <= 1.2)
        & np.isfinite(blue) & (blue >= 0) & (blue <= 1.2)
        & np.isfinite(swir1) & (swir1 >= 0) & (swir1 <= 1.2)
    )

    vis = {}

    den = nir + red
    ndvi = np.full(nir.shape, np.nan, dtype="float32")
    m = refl_ok & (np.abs(den) > EPS)
    ndvi[m] = (nir[m] - red[m]) / den[m]
    vis["NDVI"] = ndvi

    evi_den = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.full(nir.shape, np.nan, dtype="float32")
    m = refl_ok & (np.abs(evi_den) > 0.05)
    evi[m] = 2.5 * (nir[m] - red[m]) / evi_den[m]
    vis["EVI"] = evi

    gcvi = np.full(nir.shape, np.nan, dtype="float32")
    green_ok = refl_ok & (green > 0.02) & (green < 1.0)
    gcvi[green_ok] = (nir[green_ok] / green[green_ok]) - 1.0
    vis["GCVI"] = gcvi

    den = nir + swir1
    ndwi = np.full(nir.shape, np.nan, dtype="float32")
    m = refl_ok & (np.abs(den) > EPS)
    ndwi[m] = (nir[m] - swir1[m]) / den[m]
    vis["NDWI"] = ndwi

    return vis


def process_hls_item(item, grid, args, timing=None):
    item_t0 = time.perf_counter()

    band_map = band_map_for_item(item)

    required_assets = list(band_map.values())
    missing = [a for a in required_assets if a not in item.assets]
    if missing:
        available = sorted(list(item.assets.keys()))
        raise KeyError(f"missing assets {missing}; available assets include {available[:30]}")

    target_crs = grid["target_crs"]
    target_transform = grid["target_transform"]
    height = grid["height"]
    width = grid["width"]

    fmask = read_asset_to_target_grid(
        item.assets[band_map["fmask"]].href,
        target_crs,
        target_transform,
        height,
        width,
        resampling=Resampling.nearest,
        out_dtype="uint8",
        dst_nodata=255,
        max_retries=args.max_retries,
        timing=timing,
        timing_key="remote_read_fmask_sec",
    )

    t0 = time.perf_counter()
    clear = hls_clear_mask(
        fmask,
        mask_adjacent_cloud=args.mask_adjacent_cloud,
        mask_high_aerosol=args.mask_high_aerosol,
    )
    if timing is not None:
        timing["fmask_decode_sec"] += time.perf_counter() - t0

    blue = scale_hls_reflectance(
        read_asset_to_target_grid(
            item.assets[band_map["blue"]].href,
            target_crs,
            target_transform,
            height,
            width,
            resampling=Resampling.bilinear,
            out_dtype="float32",
            max_retries=args.max_retries,
            timing=timing,
            timing_key="remote_read_blue_sec",
        )
    )

    green = scale_hls_reflectance(
        read_asset_to_target_grid(
            item.assets[band_map["green"]].href,
            target_crs,
            target_transform,
            height,
            width,
            resampling=Resampling.bilinear,
            out_dtype="float32",
            max_retries=args.max_retries,
            timing=timing,
            timing_key="remote_read_green_sec",
        )
    )

    red = scale_hls_reflectance(
        read_asset_to_target_grid(
            item.assets[band_map["red"]].href,
            target_crs,
            target_transform,
            height,
            width,
            resampling=Resampling.bilinear,
            out_dtype="float32",
            max_retries=args.max_retries,
            timing=timing,
            timing_key="remote_read_red_sec",
        )
    )

    nir = scale_hls_reflectance(
        read_asset_to_target_grid(
            item.assets[band_map["nir"]].href,
            target_crs,
            target_transform,
            height,
            width,
            resampling=Resampling.bilinear,
            out_dtype="float32",
            max_retries=args.max_retries,
            timing=timing,
            timing_key="remote_read_nir_sec",
        )
    )

    swir1 = scale_hls_reflectance(
        read_asset_to_target_grid(
            item.assets[band_map["swir1"]].href,
            target_crs,
            target_transform,
            height,
            width,
            resampling=Resampling.bilinear,
            out_dtype="float32",
            max_retries=args.max_retries,
            timing=timing,
            timing_key="remote_read_swir1_sec",
        )
    )

    t0 = time.perf_counter()
    vis = compute_vi_arrays(blue, green, red, nir, swir1, clear)
    if timing is not None:
        timing["vi_compute_sec"] += time.perf_counter() - t0
        timing["process_hls_item_total_sec"] += time.perf_counter() - item_t0

    return vis



def process_one_item_parallel(item, grid, args):
    """
    Worker-thread wrapper for processing one HLS scene.

    It returns VI arrays and a local timing dictionary.
    The main thread merges timing and updates sums/counts to avoid race conditions.
    """
    local_timing = defaultdict(float)

    try:
        vis = process_hls_item(
            item=item,
            grid=grid,
            args=args,
            timing=local_timing,
        )

        return {
            "ok": True,
            "item_id": item.id,
            "collection_id": item.collection_id,
            "vis": vis,
            "timing": dict(local_timing),
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "item_id": getattr(item, "id", None),
            "collection_id": getattr(item, "collection_id", None),
            "vis": None,
            "timing": dict(local_timing),
            "error": str(e),
        }


def merge_timing(main_timing, worker_timing):
    """
    Merge timing values collected inside a worker thread into the county timing dict.
    """
    for k, v in worker_timing.items():
        main_timing[k] += v


def accumulate_item_vis(vis, crop_mask, sums, counts, timing):
    """
    Accumulate one processed HLS item's VI arrays into county-level sums/counts.

    This must run in the main thread because sums/counts are shared arrays.
    """
    item_has_crop_values = False

    t0 = time.perf_counter()

    for vi in VI_NAMES:
        arr = vis[vi]
        valid = crop_mask & np.isfinite(arr)

        if valid.any():
            sums[vi][valid] += arr[valid]
            counts[vi][valid] += 1
            item_has_crop_values = True

    timing["accumulate_vi_sec"] += time.perf_counter() - t0

    return item_has_crop_values


def process_hls_items_serial(items, grid, args, crop_mask, sums, counts, row_base, log_jsonl, timing):
    """
    Original serial HLS scene loop.
    Kept for --parallel-workers 1 and for debugging comparisons.
    """
    used_items = 0
    skipped_items = 0

    for item in tqdm(items, desc=f"{row_base['county_name']} HLS scenes"):
        try:
            vis = process_hls_item(item, grid, args, timing=timing)

            item_has_crop_values = accumulate_item_vis(
                vis=vis,
                crop_mask=crop_mask,
                sums=sums,
                counts=counts,
                timing=timing,
            )

            if item_has_crop_values:
                used_items += 1
            else:
                skipped_items += 1

        except Exception as e:
            skipped_items += 1
            append_jsonl(
                log_jsonl,
                {
                    **row_base,
                    "status": "item_skipped",
                    "item_id": item.id,
                    "collection_id": item.collection_id,
                    "error": str(e),
                },
            )

        t0 = time.perf_counter()
        sleep_with_jitter(args.sleep_after_item)
        timing["sleep_after_item_sec"] += time.perf_counter() - t0

    return used_items, skipped_items


def process_hls_items_parallel(items, grid, args, crop_mask, sums, counts, row_base, log_jsonl, timing):
    """
    Parallel HLS scene loop.

    Important design choice:
      - Worker threads read/warp/compute VI for separate HLS scenes.
      - The main thread alone updates sums/counts.
      - The number of in-flight futures is bounded to avoid memory blow-up.

    Start with --parallel-workers 2 or 3.
    """
    parallel_workers = max(1, int(args.parallel_workers))
    max_in_flight = max(1, int(args.max_in_flight))

    used_items = 0
    skipped_items = 0

    print(f"Processing HLS scenes in parallel with {parallel_workers} workers")
    print(f"Max in-flight HLS scenes: {max_in_flight}")

    item_iter = iter(items)
    pending = set()
    submitted = 0

    def submit_next(executor):
        nonlocal submitted
        try:
            item = next(item_iter)
        except StopIteration:
            return False

        fut = executor.submit(process_one_item_parallel, item, grid, args)
        pending.add(fut)
        submitted += 1
        return True

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        # Keep only a small number of large-array results in flight.
        initial = min(max_in_flight, len(items))
        for _ in range(initial):
            submit_next(executor)

        with tqdm(total=len(items), desc=f"{row_base['county_name']} HLS scenes parallel") as pbar:
            while pending:
                done_futures, pending = wait(
                                                pending,
                                                timeout=60,
                                                return_when=FIRST_COMPLETED,
                                            )

                if not done_futures:
                    print(
                        f"[HEARTBEAT] No scene finished in 60 seconds. "
                        f"Pending scenes: {len(pending)}",
                        flush=True,
                    )
                    continue

                for fut in done_futures:
                    result = fut.result()

                    merge_timing(timing, result.get("timing", {}))

                    if not result["ok"]:
                        skipped_items += 1
                        append_jsonl(
                            log_jsonl,
                            {
                                **row_base,
                                "status": "item_skipped",
                                "item_id": result["item_id"],
                                "collection_id": result["collection_id"],
                                "error": result["error"],
                            },
                        )
                    else:
                        item_has_crop_values = accumulate_item_vis(
                            vis=result["vis"],
                            crop_mask=crop_mask,
                            sums=sums,
                            counts=counts,
                            timing=timing,
                        )

                        if item_has_crop_values:
                            used_items += 1
                        else:
                            skipped_items += 1

                    pbar.update(1)

                    # Optional small sleep before sending another remote-read job.
                    # For parallel mode, usually run with --sleep-after-item 0.
                    t0 = time.perf_counter()
                    sleep_with_jitter(args.sleep_after_item)
                    timing["sleep_after_item_sec"] += time.perf_counter() - t0

                    submit_next(executor)

    return used_items, skipped_items



# ============================================================
# Pixel GeoTIFF export helpers
# ============================================================

PIXEL_TIF_NODATA = -9999.0


def sanitize_filename_part(value):
    """Make a safe filename component from county names, etc."""
    text = str(value)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        elif ch.isspace():
            safe.append("_")
    out = "".join(safe).strip("_")
    return out if out else "unknown"


def safe_gtiff_block_size(n, target=512):
    """
    Return a block size compatible with tiled GeoTIFFs.
    GTiff block sizes should be multiples of 16.
    """
    if n >= target:
        return target
    block = (int(n) // 16) * 16
    return max(16, block)


def build_county_pixel_tif_path(args, row_base):
    tif_dir = args.pixel_tif_dir
    if tif_dir is None:
        tif_dir = args.out_dir / "pixel_vi_tifs"

    tif_dir.mkdir(parents=True, exist_ok=True)

    county_safe = sanitize_filename_part(row_base["county_name"])
    filename = (
        f"soybeans_{row_base['state']}_{row_base['year']}_"
        f"{row_base['county_fips']}_{county_safe}_"
        f"vi_pixels_nasa_cmr_hls.tif"
    )

    return tif_dir / filename


def write_pixel_vi_geotiff(
    tif_path,
    grid,
    crop_mask,
    valid_pixel_mask,
    vi_mean_pixels,
    counts,
    min_obs,
    row_base,
    args,
):
    """
    Save county-window pixel-level seasonal VI values as a multi-band GeoTIFF.

    The VI bands are written only for soybean pixels passing the seasonal
    validity threshold: crop_mask & min_index_obsCount >= min_obs_season.
    Other pixels are written as nodata. Count/mask bands are included so the
    raster can be filtered later without rerunning HLS extraction.
    """
    height = int(grid["height"])
    width = int(grid["width"])
    shape = (height, width)

    band_arrays = []
    band_names = []

    # Seasonal mean VI bands for valid soybean pixels only.
    for vi in VI_NAMES:
        out = np.full(shape, PIXEL_TIF_NODATA, dtype="float32")
        src = vi_mean_pixels[vi]
        m = valid_pixel_mask & np.isfinite(src)
        out[m] = src[m].astype("float32")
        band_arrays.append(out)
        band_names.append(f"{vi}_mean")

    # Per-VI observation count bands.
    # These are useful later for QA/filtering before pseudo-yield generation.
    for vi in VI_NAMES:
        out = np.zeros(shape, dtype="float32")
        out[crop_mask] = counts[vi][crop_mask].astype("float32")
        band_arrays.append(out)
        band_names.append(f"{vi}_obsCount")

    # Minimum observation count across all VI bands.
    out = np.zeros(shape, dtype="float32")
    out[crop_mask] = min_obs[crop_mask].astype("float32")
    band_arrays.append(out)
    band_names.append("min_index_obsCount")

    # Crop and valid masks for later use.
    out = np.zeros(shape, dtype="float32")
    out[crop_mask] = 1.0
    band_arrays.append(out)
    band_names.append("crop_mask")

    out = np.zeros(shape, dtype="float32")
    out[valid_pixel_mask] = 1.0
    band_arrays.append(out)
    band_names.append("valid_vi_mask")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(band_arrays),
        "dtype": "float32",
        "crs": grid["target_crs"],
        "transform": grid["target_transform"],
        "nodata": PIXEL_TIF_NODATA,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": safe_gtiff_block_size(width),
        "blockysize": safe_gtiff_block_size(height),
        "BIGTIFF": "IF_SAFER",
    }

    tif_path.parent.mkdir(parents=True, exist_ok=True)

    if tif_path.exists() and not args.overwrite:
        print(f"Pixel GeoTIFF already exists; skipping: {tif_path}")
        return str(tif_path)

    with rasterio.open(tif_path, "w", **profile) as dst:
        for idx, (name, arr) in enumerate(zip(band_names, band_arrays), start=1):
            dst.write(arr, idx)
            dst.set_band_description(idx, name)

        dst.update_tags(
            source="NASA CMR-STAC / LP DAAC HLS v2.0",
            crop="soybeans",
            state=str(row_base["state"]),
            state_fips=str(row_base["state_fips"]),
            county_fips=str(row_base["county_fips"]),
            county_name=str(row_base["county_name"]),
            year=str(row_base["year"]),
            season_start=str(row_base["season_start"]),
            season_end=str(row_base["season_end"]),
            min_obs_season=str(args.min_obs_season),
            vi_bands=",".join(VI_NAMES),
            valid_pixel_definition="crop_mask AND min_index_obsCount >= min_obs_season",
            nodata=str(PIXEL_TIF_NODATA),
        )

    print(f"Saved pixel GeoTIFF: {tif_path}")
    return str(tif_path)

# ============================================================
# County processing
# ============================================================

def summarize_county(county_row, catalog, args, out_csv, log_jsonl, fail_jsonl):
    county_t0 = time.perf_counter()
    timing = defaultdict(float)

    state = args.state.upper()
    state_fips = STATE_ABBR_TO_FIPS[state]
    geoid = county_row["GEOID"]
    county_name = county_row["NAME"]

    print("\n============================================================")
    print(f"Processing {state} {county_name} County ({geoid}), {args.year}")
    print("============================================================")

    row_base = {
        "state": state,
        "state_fips": state_fips,
        "county_fips": geoid,
        "county_name": county_name,
        "year": args.year,
        "crop": "soybeans",
        "season_start": args.start_date,
        "season_end": args.end_date,
    }

    def add_timing_to_row(row):
        timing["county_total_sec"] = time.perf_counter() - county_t0
        row["county_runtime_sec"] = round(timing["county_total_sec"], 2)
        row["county_runtime_min"] = round(timing["county_total_sec"] / 60, 3)
        row["remote_asset_reads"] = int(timing.get("remote_asset_reads", 0))
        row["remote_read_warp_sec"] = round(timing.get("remote_read_warp_total_sec", 0), 3)
        row["remote_read_warp_min"] = round(timing.get("remote_read_warp_total_sec", 0) / 60, 3)
        row["search_hls_items_sec"] = round(timing.get("search_hls_items_sec", 0), 3)
        row["load_cdl_crop_grid_sec"] = round(timing.get("load_cdl_crop_grid_sec", 0), 3)
        row["process_hls_items_loop_sec"] = round(timing.get("process_hls_items_loop_sec", 0), 3)
        row["final_reduce_summary_sec"] = round(timing.get("final_reduce_summary_sec", 0), 3)
        row["pixel_tif_write_sec"] = round(timing.get("pixel_tif_write_sec", 0), 3)
        row["pixel_tif_path"] = timing.get("pixel_tif_path", "")
        row["sleep_after_item_sec"] = round(timing.get("sleep_after_item_sec", 0), 3)
        row["parallel_workers"] = int(timing.get("parallel_workers", args.parallel_workers))
        row["max_in_flight"] = int(timing.get("max_in_flight", args.max_in_flight))
        return row

    try:
        # ----------------------------
        # Load CDL/crop grid
        # ----------------------------
        t0 = time.perf_counter()
        grid = load_county_crop_grid(
            args.cdl,
            county_row,
            args.cdl_mode,
            args.include_double_crop,
        )
        timing["load_cdl_crop_grid_sec"] += time.perf_counter() - t0

        crop_mask = grid["crop_mask"]
        crop_pixel_count = int(crop_mask.sum())

        print(f"Crop pixels: {crop_pixel_count:,}")

        if crop_pixel_count == 0:
            row = {
                **row_base,
                "NDVI_mean": np.nan,
                "EVI_mean": np.nan,
                "GCVI_mean": np.nan,
                "NDWI_mean": np.nan,
                "min_index_obsCount": np.nan,
                "crop_pixel_count": 0,
                "valid_pixel_count": 0,
                "valid_pixel_fraction": 0.0,
            }

            row = add_timing_to_row(row)

            t0 = time.perf_counter()
            append_row_to_csv(out_csv, row)
            timing["csv_write_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            append_jsonl(
                log_jsonl,
                {
                    **row_base,
                    "status": "no_crop_pixels",
                    "timing": rounded_timing_dict(timing),
                },
            )
            timing["json_log_write_sec"] += time.perf_counter() - t0

            print_county_timing_summary(timing)
            return row

        # ----------------------------
        # Search HLS items
        # ----------------------------
        t0 = time.perf_counter()
        items = search_hls_items(
            catalog,
            county_row.geometry,
            args.start_date,
            args.end_date,
            args.max_scene_cloud,
            args.max_retries,
            cmr_page_limit=args.cmr_page_limit,
            cmr_max_items=args.cmr_max_items,
            cmr_spatial_mode=args.cmr_spatial_mode,
        )
        timing["search_hls_items_sec"] += time.perf_counter() - t0

        l30_count = sum(1 for i in items if is_l30_item(i))
        s30_count = sum(1 for i in items if is_s30_item(i))

        print(f"HLS items after cloud filter: {len(items)} total ({l30_count} L30, {s30_count} S30)")

        warm_up_earthdata_gdal(items, args)

        shape = crop_mask.shape

        sums = {vi: np.zeros(shape, dtype="float64") for vi in VI_NAMES}
        counts = {vi: np.zeros(shape, dtype="uint16") for vi in VI_NAMES}

        used_items = 0
        skipped_items = 0

        # ----------------------------
        # Process HLS scenes
        # ----------------------------
        loop_t0 = time.perf_counter()

        timing["parallel_workers"] = int(args.parallel_workers)
        timing["max_in_flight"] = int(args.max_in_flight)

        if int(args.parallel_workers) <= 1:
            used_items, skipped_items = process_hls_items_serial(
                items=items,
                grid=grid,
                args=args,
                crop_mask=crop_mask,
                sums=sums,
                counts=counts,
                row_base=row_base,
                log_jsonl=log_jsonl,
                timing=timing,
            )
        else:
            used_items, skipped_items = process_hls_items_parallel(
                items=items,
                grid=grid,
                args=args,
                crop_mask=crop_mask,
                sums=sums,
                counts=counts,
                row_base=row_base,
                log_jsonl=log_jsonl,
                timing=timing,
            )

        timing["process_hls_items_loop_sec"] += time.perf_counter() - loop_t0
        timing["hls_items_used"] = used_items
        timing["hls_items_skipped"] = skipped_items

        # ----------------------------
        # Final reduction / county summary
        # ----------------------------
        t0 = time.perf_counter()

        with np.errstate(invalid="ignore", divide="ignore"):
            vi_mean_pixels = {}
            for vi in VI_NAMES:
                vi_mean = np.full(shape, np.nan, dtype="float32")
                m = counts[vi] > 0
                vi_mean[m] = (sums[vi][m] / counts[vi][m]).astype("float32")
                vi_mean_pixels[vi] = vi_mean

        min_obs = np.minimum.reduce([counts[vi] for vi in VI_NAMES])
        valid_pixel_mask = crop_mask & (min_obs >= args.min_obs_season)

        valid_pixel_count = int(valid_pixel_mask.sum())
        valid_pixel_fraction = (
            float(valid_pixel_count / crop_pixel_count)
            if crop_pixel_count > 0
            else np.nan
        )

        row = {
            **row_base,
            "NDVI_mean": np.nan,
            "EVI_mean": np.nan,
            "GCVI_mean": np.nan,
            "NDWI_mean": np.nan,
            "min_index_obsCount": np.nan,
            "crop_pixel_count": crop_pixel_count,
            "valid_pixel_count": valid_pixel_count,
            "valid_pixel_fraction": valid_pixel_fraction,
        }

        if valid_pixel_count > 0:
            for vi in VI_NAMES:
                vals = vi_mean_pixels[vi][valid_pixel_mask]
                vals = vals[np.isfinite(vals)]
                row[f"{vi}_mean"] = float(np.nanmean(vals)) if vals.size else np.nan

            row["min_index_obsCount"] = float(np.nanmean(min_obs[valid_pixel_mask]))

        timing["final_reduce_summary_sec"] += time.perf_counter() - t0

        # ----------------------------
        # Optional pixel-level GeoTIFF export
        # ----------------------------
        if args.save_pixel_tif:
            t0 = time.perf_counter()
            tif_path = build_county_pixel_tif_path(args, row_base)
            pixel_tif_path = write_pixel_vi_geotiff(
                tif_path=tif_path,
                grid=grid,
                crop_mask=crop_mask,
                valid_pixel_mask=valid_pixel_mask,
                vi_mean_pixels=vi_mean_pixels,
                counts=counts,
                min_obs=min_obs,
                row_base=row_base,
                args=args,
            )
            timing["pixel_tif_path"] = pixel_tif_path
            timing["pixel_tif_write_sec"] += time.perf_counter() - t0

        row = add_timing_to_row(row)

        # ----------------------------
        # Write output files
        # ----------------------------
        t0 = time.perf_counter()
        append_row_to_csv(out_csv, row)
        timing["csv_write_sec"] += time.perf_counter() - t0

        log_record = {
            **row_base,
            "status": "done",
            "hls_items_after_filter": len(items),
            "hls_l30_items": l30_count,
            "hls_s30_items": s30_count,
            "hls_items_used": used_items,
            "hls_items_skipped": skipped_items,
            "crop_pixel_count": crop_pixel_count,
            "valid_pixel_count": valid_pixel_count,
            "valid_pixel_fraction": valid_pixel_fraction,
            "pixel_tif_path": timing.get("pixel_tif_path", ""),
            "timing": rounded_timing_dict(timing),
        }

        t0 = time.perf_counter()
        append_jsonl(log_jsonl, log_record)
        timing["json_log_write_sec"] += time.perf_counter() - t0

        print("Saved row:")
        print(pd.DataFrame([row]).to_string(index=False))

        print_county_timing_summary(timing)

        return row

    except Exception as e:
        timing["county_total_sec"] = time.perf_counter() - county_t0

        fail_record = {
            **row_base,
            "status": "failed_county",
            "error": str(e),
            "timing": rounded_timing_dict(timing),
        }

        append_jsonl(fail_jsonl, fail_record)

        print(f"FAILED county {county_name} ({geoid}): {e}")
        print_county_timing_summary(timing)

        return None


def completed_geoids(out_csv: Path, state: str, year: int):
    if not out_csv.exists():
        return set()

    try:
        df = pd.read_csv(out_csv, dtype={"county_fips": str, "state": str})
        df = df[(df["state"] == state) & (df["year"] == year)]
        return set(df["county_fips"].astype(str))
    except Exception:
        return set()


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="County-scale soybean VI summary from NASA CMR-STAC / LP DAAC HLS v2.0 L30+S30."
    )

    parser.add_argument("--state", required=True, help="State abbreviation, e.g., DE")
    parser.add_argument("--year", required=True, type=int, help="Crop year, e.g., 2022")
    parser.add_argument("--cdl", required=True, type=Path, help="Path to local CDL or soybean mask GeoTIFF")
    parser.add_argument("--cdl-mode", choices=["binary", "raw"], default="binary",
                        help="binary = soybean pixels >0; raw = CDL class 5 and optionally 26")
    parser.add_argument("--exclude-double-crop", action="store_true",
                        help="For raw CDL mode, exclude CDL class 26.")
    parser.add_argument("--county-fips", default=None,
                        help="Optional 3-digit county FIPS, e.g., 001")
    parser.add_argument("--county-name", default=None,
                        help="Optional county name filter, e.g., Kent")
    parser.add_argument("--out-dir", type=Path, default=Path("./nasa_cmr_hls_vi_outputs"))
    parser.add_argument("--save-pixel-tif", action="store_true",
                        help="Save county-window pixel-level seasonal VI GeoTIFFs for valid soybean pixels.")
    parser.add_argument("--pixel-tif-dir", type=Path, default=None,
                        help="Directory for pixel GeoTIFFs. Default: OUT_DIR/pixel_vi_tifs")

    parser.add_argument("--start-date", default=None,
                        help="Season start date. Default: YEAR-04-01")
    parser.add_argument("--end-date", default=None,
                        help="Season end date. Default: YEAR-11-15")

    parser.add_argument("--max-scene-cloud", type=float, default=70.0)
    parser.add_argument("--min-obs-season", type=int, default=3)

    parser.add_argument("--mask-adjacent-cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-high-aerosol", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sleep-after-item", type=float, default=0.4)
    parser.add_argument("--sleep-after-county", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=4)

    parser.add_argument("--earthdata-login", action=argparse.BooleanOptionalAction, default=True,
                        help="Use earthaccess.login(...) to prepare NASA Earthdata authentication.")
    parser.add_argument("--earthdata-strategy", default="interactive",
                        choices=["interactive", "environment", "netrc", "all"],
                        help="earthaccess login strategy. Use interactive first; environment/netrc for automated runs.")
    parser.add_argument("--netrc-file", type=Path, default=None,
                        help="Optional path to a .netrc file for Earthdata credentials. Default uses ~/.netrc.")
    parser.add_argument("--cookie-file", type=Path, default=None,
                        help="Optional GDAL cookie jar path. Default: OUT_DIR/earthdata_gdal_cookies.txt")

    parser.add_argument("--cmr-spatial-mode", choices=["bbox", "intersects"], default="bbox",
                        help="Spatial filter for CMR-STAC search. bbox is more stable; intersects is stricter.")
    parser.add_argument("--cmr-page-limit", type=int, default=100,
                        help="CMR-STAC page size. Keep <=250 to avoid known CMR-STAC response-size errors.")
    parser.add_argument("--cmr-max-items", type=int, default=1000,
                        help="Maximum STAC items to request per collection before cloud filtering.")

    parser.add_argument("--parallel-workers", type=int, default=1,
                        help="Number of HLS scenes to process in parallel. Start with 2 or 3.")

    parser.add_argument("--max-in-flight", type=int, default=None,
                        help="Maximum HLS scene jobs allowed in memory at once. Default: parallel-workers.")

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    state_t0 = time.perf_counter()
    state_timing = defaultdict(float)

    args = parse_args()
    args.state = args.state.upper()

    if args.state not in STATE_ABBR_TO_FIPS:
        raise ValueError(f"Unknown state abbreviation: {args.state}")

    if not args.cdl.exists():
        raise FileNotFoundError(f"CDL file does not exist: {args.cdl}")

    args.include_double_crop = not args.exclude_double_crop

    if args.max_in_flight is None:
        args.max_in_flight = max(1, int(args.parallel_workers))
    else:
        args.max_in_flight = max(1, int(args.max_in_flight))

    args.parallel_workers = max(1, int(args.parallel_workers))

    if args.start_date is None:
        args.start_date = f"{args.year}-04-01"
    if args.end_date is None:
        args.end_date = f"{args.year}-11-15"

    if args.year < 2013:
        print(
            "\nWARNING: HLSL30 v2.0 starts in 2013. This year may return zero HLS items.\n"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prepare_earthdata_auth(args)

    out_csv = args.out_dir / f"soybeans_{args.state}_{args.year}_county_vi_summary_nasa_cmr_hls.csv"
    log_jsonl = args.out_dir / f"soybeans_{args.state}_{args.year}_processing_log.jsonl"
    fail_jsonl = args.out_dir / f"soybeans_{args.state}_{args.year}_failures.jsonl"

    print("\n============================================================")
    print("NASA CMR-STAC / LP DAAC HLS v2.0 County VI Summary")
    print("============================================================")
    print(f"State:       {args.state}")
    print(f"Year:        {args.year}")
    print(f"CDL:         {args.cdl}")
    print(f"CDL mode:    {args.cdl_mode}")
    print(f"Season:      {args.start_date} to {args.end_date}")
    print(f"Workers:     {args.parallel_workers}")
    print(f"Max in-flight scenes: {args.max_in_flight}")
    print(f"CMR-STAC:    {CMR_STAC_URL}")
    print(f"CMR spatial: {args.cmr_spatial_mode}")
    print(f"CMR page limit: {args.cmr_page_limit}")
    print(f"CMR max items:  {args.cmr_max_items}")
    print(f"Cookie file: {args.cookie_file}")
    print(f"Output CSV:  {out_csv}")
    print(f"Save pixel GeoTIFFs: {args.save_pixel_tif}")
    print(f"Pixel GeoTIFF dir:   {args.pixel_tif_dir if args.pixel_tif_dir is not None else args.out_dir / 'pixel_vi_tifs'}")
    print("============================================================\n")

    t0 = time.perf_counter()
    counties_all = download_counties_2018()
    state_timing["download_counties_sec"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    counties = select_counties(
        counties_all,
        args.state,
        county_fips=args.county_fips,
        county_name=args.county_name,
    )
    state_timing["select_counties_sec"] += time.perf_counter() - t0

    print("Selected counties:")
    print(counties[["GEOID", "NAME"]].to_string(index=False))

    t0 = time.perf_counter()
    catalog = open_pc_catalog()
    state_timing["open_pc_catalog_sec"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    done = completed_geoids(out_csv, args.state, args.year)
    state_timing["read_completed_csv_sec"] += time.perf_counter() - t0

    print(f"\nAlready completed counties in output CSV: {len(done)}")

    processed_counties = 0
    skipped_counties = 0

    for _, county_row in counties.iterrows():
        geoid = str(county_row["GEOID"])

        if (not args.overwrite) and (geoid in done):
            print(f"\nSkipping already completed county {county_row['NAME']} ({geoid})")
            skipped_counties += 1
            continue

        summarize_county(
            county_row=county_row,
            catalog=catalog,
            args=args,
            out_csv=out_csv,
            log_jsonl=log_jsonl,
            fail_jsonl=fail_jsonl,
        )

        processed_counties += 1

        t0 = time.perf_counter()
        sleep_with_jitter(args.sleep_after_county)
        state_timing["sleep_after_county_sec"] += time.perf_counter() - t0

    state_timing["state_year_total_sec"] = time.perf_counter() - state_t0

    state_summary = {
        "state": args.state,
        "year": args.year,
        "status": "state_year_done",
        "processed_counties": processed_counties,
        "skipped_counties": skipped_counties,
        "state_timing": rounded_timing_dict(state_timing),
    }

    append_jsonl(log_jsonl, state_summary)

    print("\n============================================================")
    print("Done.")
    print(f"State-year runtime: {format_runtime(state_timing['state_year_total_sec'])}")
    print(f"Processed counties: {processed_counties}")
    print(f"Skipped counties:   {skipped_counties}")
    print(f"Summary CSV: {out_csv}")
    print(f"Log JSONL:   {log_jsonl}")
    print(f"Failures:    {fail_jsonl}")
    print("============================================================")


if __name__ == "__main__":
    main()
