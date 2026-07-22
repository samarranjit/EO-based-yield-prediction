#!/usr/bin/env python3
"""
State-year HLS VI downloader and county summarizer.

Workflow
--------
1. Search HLS once for the complete state and crop season.
2. Download each required HLS asset once with threads.
3. Process counties from the local cache with multiple processes.
4. Preserve the original CSV and county-pixel GeoTIFF formats.
5. Resume from the existing CSV, downloaded files, and county TIFFs.
6. Delete the temporary scene cache only after every county is complete.

This replaces repeated county-by-county remote COG reads while keeping the
same output names and data columns as the previous script.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from shapely import wkb
from shapely.geometry import box, mapping
from tqdm import tqdm


# ============================================================
# Constants
# ============================================================

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

# Preserve the VI pipeline's original broad-NIR choice for S30.
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

REQUIRED_ASSETS = ["blue", "green", "red", "nir", "swir1", "fmask"]
VI_NAMES = ["NDVI", "EVI", "GCVI", "NDWI"]

REFLECTANCE_SCALE = 0.0001
EPS = 1e-6
PIXEL_TIF_NODATA = -9999.0

CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
CMR_COLLECTION_L30 = "HLSL30_2.0"
CMR_COLLECTION_S30 = "HLSS30_2.0"
S30_START_DATE = "2015-11-28"

COUNTY_BOUNDARY_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2018/shp/"
    "cb_2018_us_county_500k.zip"
)

# Worker globals are loaded once per process.
_WORKER_SCENES: list[dict[str, Any]] = []
_WORKER_CONFIG: dict[str, Any] = {}
_THREAD_LOCAL = threading.local()
_USE_EARTHACCESS_SESSION = False


# ============================================================
# Small helpers
# ============================================================

def format_runtime(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} sec"
    if seconds < 3600:
        return f"{seconds / 60:.2f} min"
    return f"{seconds / 3600:.2f} hr"


def sleep_with_jitter(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds + random.uniform(0, min(1.0, seconds * 0.25)))


def retry_call(label, func, max_retries=4, base_sleep=2.0):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            delay = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            print(
                f"WARNING: {label} failed ({attempt}/{max_retries}): {exc}; "
                f"retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    raise last_error


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, default=str), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def append_row_to_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", index=False, header=write_header)


def clean_existing_csv(path: Path) -> None:
    """Remove accidental duplicate county rows while preserving the format."""
    if not path.exists():
        return
    try:
        frame = pd.read_csv(path, dtype={"county_fips": str, "state": str})
    except pd.errors.EmptyDataError:
        path.unlink(missing_ok=True)
        return
    needed = {"state", "year", "county_fips"}
    if needed.issubset(frame.columns):
        frame = frame.drop_duplicates(
            subset=["state", "year", "county_fips"], keep="last"
        )
        frame.to_csv(path, index=False)


def completed_geoids(
    path: Path,
    state: str,
    year: int,
    require_pixel_tif: bool = False,
    pixel_tif_dir: Path | None = None,
) -> set[str]:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, dtype={"county_fips": str, "state": str})
        frame = frame[(frame["state"] == state) & (frame["year"] == year)]
        completed = set()

        for _, row in frame.iterrows():
            geoid = str(row["county_fips"])
            crop_pixels = float(row.get("crop_pixel_count", 1) or 0)

            if require_pixel_tif and crop_pixels > 0:
                if pixel_tif_dir is None:
                    continue
                county_safe = sanitize_filename_part(row.get("county_name", "unknown"))
                tif_path = pixel_tif_dir / (
                    f"soybeans_{state}_{year}_{geoid}_{county_safe}_"
                    "vi_pixels_nasa_cmr_hls.tif"
                )
                if not validate_raster(tif_path):
                    continue

            completed.add(geoid)

        return completed
    except Exception:  # noqa: BLE001
        return set()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def sanitize_filename_part(value) -> str:
    text = str(value)
    safe = [
        character if character.isalnum() or character in "-_" else "_"
        for character in text
    ]
    return "".join(safe).strip("_") or "unknown"


def cloud_cover(item) -> float | None:
    for key in ("eo:cloud_cover", "cloud_cover", "CLOUD_COVERAGE"):
        value = item.properties.get(key)
        if value is not None:
            return float(value)
    return None


def item_datetime(item) -> str:
    return item.properties.get("datetime", "")


def is_s30_item(item) -> bool:
    collection = getattr(item, "collection_id", "") or ""
    item_id = getattr(item, "id", "") or ""
    return collection == CMR_COLLECTION_S30 or "S30" in item_id


def is_l30_item(item) -> bool:
    collection = getattr(item, "collection_id", "") or ""
    item_id = getattr(item, "id", "") or ""
    return collection == CMR_COLLECTION_L30 or "L30" in item_id


def band_map_for_item(item) -> dict[str, str]:
    if is_s30_item(item):
        return S30_BANDS
    if is_l30_item(item):
        return L30_BANDS
    raise ValueError(f"Cannot identify HLS collection for {item.id}")


def collections_for_end_date(end_date: str) -> list[str]:
    collections = [CMR_COLLECTION_L30]
    if end_date >= S30_START_DATE:
        collections.append(CMR_COLLECTION_S30)
    return collections


def bbox_intersects(a: list[float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


# ============================================================
# Counties and CDL grid
# ============================================================

def download_counties_2018(max_retries: int) -> gpd.GeoDataFrame:
    def download_bytes():
        response = requests.get(COUNTY_BOUNDARY_URL, timeout=120)
        response.raise_for_status()
        return response.content

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "counties.zip"
        path.write_bytes(
            retry_call(
                "download Census county boundaries",
                download_bytes,
                max_retries=max_retries,
            )
        )
        counties = gpd.read_file(path)

    return counties.to_crs("EPSG:4326")


def select_counties(
    counties: gpd.GeoDataFrame,
    state: str,
    county_fips: str | None = None,
    county_name: str | None = None,
) -> gpd.GeoDataFrame:
    state_fips = STATE_ABBR_TO_FIPS[state]
    selected = counties[counties["STATEFP"] == state_fips].copy()

    if county_fips:
        selected = selected[selected["COUNTYFP"] == county_fips.zfill(3)].copy()
    if county_name:
        selected = selected[
            selected["NAME"].str.lower().str.contains(county_name.lower())
        ].copy()
    if selected.empty:
        raise ValueError("No county matched the requested filters.")

    return selected.sort_values("GEOID").reset_index(drop=True)


def window_covering_bounds(bounds, transform) -> Window:
    window = from_bounds(*bounds, transform=transform)
    col0 = math.floor(window.col_off)
    row0 = math.floor(window.row_off)
    col1 = math.ceil(window.col_off + window.width)
    row1 = math.ceil(window.row_off + window.height)
    return Window(col0, row0, max(1, col1 - col0), max(1, row1 - row0))


def load_county_crop_grid(
    cdl_path: str | Path,
    county_geometry_4326,
    cdl_mode: str,
    include_double_crop: bool,
) -> dict[str, Any]:
    with rasterio.open(cdl_path) as source:
        if source.crs is None:
            raise ValueError("CDL raster has no CRS.")

        county = gpd.GeoSeries([county_geometry_4326], crs="EPSG:4326").to_crs(
            source.crs
        ).iloc[0]
        window = window_covering_bounds(county.bounds, source.transform)
        cdl = source.read(1, window=window, boundless=True, fill_value=0)
        target_transform = source.window_transform(window)
        target_crs = source.crs

    county_mask = rasterize(
        [(county, 1)],
        out_shape=cdl.shape,
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
            crop_mask |= cdl == 26
    else:
        raise ValueError("cdl_mode must be 'binary' or 'raw'.")

    crop_mask &= county_mask

    return {
        "crop_mask": crop_mask,
        "target_transform": target_transform,
        "target_crs": target_crs,
        "height": cdl.shape[0],
        "width": cdl.shape[1],
    }


# ============================================================
# Earthdata authentication and threaded downloads
# ============================================================

def prepare_earthdata_auth(args) -> None:
    global _USE_EARTHACCESS_SESSION

    if args.netrc_file:
        os.environ["NETRC"] = str(args.netrc_file)

    if not args.earthdata_login:
        print("Earthdata login skipped; using existing netrc credentials.")
        return

    try:
        import earthaccess

        print("Checking Earthdata Login with earthaccess...")
        earthaccess.login(strategy=args.earthdata_strategy, persist=True)
        _USE_EARTHACCESS_SESSION = hasattr(
            earthaccess, "get_requests_https_session"
        )
        print("Earthdata Login ready.")
    except ImportError:
        print("WARNING: earthaccess is not installed; falling back to requests/netrc.")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: earthaccess login failed: {exc}; using requests/netrc.")


def get_download_session() -> requests.Session:
    if hasattr(_THREAD_LOCAL, "session"):
        return _THREAD_LOCAL.session

    session = None
    if _USE_EARTHACCESS_SESSION:
        try:
            import earthaccess

            session = earthaccess.get_requests_https_session()
        except Exception:  # noqa: BLE001
            session = None

    if session is None:
        session = requests.Session()
        session.trust_env = True

    session.headers.update({"User-Agent": "hls-county-vi-cache/1.0"})
    _THREAD_LOCAL.session = session
    return session


def validate_raster(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as source:
            if source.width <= 0 or source.height <= 0 or source.count < 1:
                return False
            source.read(1, window=Window(0, 0, 1, 1))
        return True
    except Exception:  # noqa: BLE001
        return False


def download_one_asset(task: dict[str, Any], args) -> dict[str, Any]:
    destination = Path(task["path"])
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if validate_raster(destination):
        return {**task, "status": "cached"}
    destination.unlink(missing_ok=True)

    session = get_download_session()
    last_error = None

    for attempt in range(1, args.max_retries + 1):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}

            with session.get(
                task["url"],
                stream=True,
                allow_redirects=True,
                headers=headers,
                timeout=(30, args.download_read_timeout),
            ) as response:
                if response.status_code == 416:
                    if validate_raster(partial):
                        partial.replace(destination)
                        return {**task, "status": "resumed"}
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("server rejected an invalid partial download")

                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                mode = "ab" if append else "wb"

                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)

            if not validate_raster(partial):
                raise RuntimeError("downloaded file is not a readable raster")

            partial.replace(destination)
            return {**task, "status": "downloaded"}

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.max_retries:
                delay = 2 ** (attempt - 1) + random.uniform(0, 2)
                print(
                    f"WARNING: download failed {task['item_id']} {task['asset']} "
                    f"({attempt}/{args.max_retries}): {exc}; retry in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)

    return {**task, "status": "failed", "error": str(last_error)}


def download_manifest_assets(manifest: dict[str, Any], args) -> None:
    tasks = []
    for scene in manifest["items"]:
        for asset, metadata in scene["assets"].items():
            tasks.append(
                {
                    "item_id": scene["id"],
                    "asset": asset,
                    "url": metadata["url"],
                    "path": metadata["path"],
                }
            )

    print(
        f"Scene cache: {len(manifest['items'])} scenes, "
        f"{len(tasks)} required assets",
        flush=True,
    )

    failed = []
    completed = 0
    executor = ThreadPoolExecutor(max_workers=max(1, args.download_workers))
    pending = {executor.submit(download_one_asset, task, args): task for task in tasks}

    try:
        while pending:
            done, _ = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            if not done:
                print(
                    f"[HEARTBEAT] No download finished in 60 seconds; "
                    f"{len(pending)} pending",
                    flush=True,
                )
                continue

            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {**task, "status": "failed", "error": str(exc)}

                completed += 1
                if result["status"] == "failed":
                    failed.append(result)
                    print(
                        f"[DOWNLOAD FAILED] {result['item_id']} {result['asset']}: "
                        f"{result.get('error')}",
                        flush=True,
                    )
                else:
                    print(
                        f"[DOWNLOAD DONE] {completed}/{len(tasks)} "
                        f"{result['item_id']} {result['asset']} ({result['status']})",
                        flush=True,
                    )
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    if failed:
        raise RuntimeError(
            f"{len(failed)} scene assets failed to download; rerun to resume"
        )


# ============================================================
# State-level CMR-STAC search and manifest
# ============================================================

def search_state_items(
    state_geometry_4326,
    start_date: str,
    end_date: str,
    args,
):
    catalog = Client.open(CMR_STAC_URL)
    state_bbox = list(state_geometry_4326.bounds)
    datetime_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    items = []

    for collection in collections_for_end_date(end_date):
        def run_search(collection=collection):
            search = catalog.search(
                collections=[collection],
                datetime=datetime_range,
                bbox=state_bbox,
                limit=max(1, min(args.cmr_page_limit, 250)),
                max_items=max(1, args.cmr_max_items),
            )
            return list(search.items())

        found = retry_call(
            f"{collection} state search",
            run_search,
            max_retries=args.max_retries,
        )
        print(f"{collection}: {len(found)} items before filters")
        items.extend(found)

    deduplicated = {item.id: item for item in items}
    filtered = []

    for item in deduplicated.values():
        cover = cloud_cover(item)
        if cover is not None and cover > args.max_scene_cloud:
            continue
        if item.bbox and not box(*item.bbox).intersects(state_geometry_4326):
            continue
        filtered.append(item)

    return sorted(filtered, key=lambda item: (item_datetime(item), item.id))


def build_or_load_manifest(
    cache_dir: Path,
    state: str,
    year: int,
    start_date: str,
    end_date: str,
    state_geometry_4326,
    args,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = cache_dir / state / str(year) / "scene_manifest.json"

    if manifest_path.exists() and not args.refresh_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = (state, year, start_date, end_date)
        actual = (
            manifest.get("state"),
            manifest.get("year"),
            manifest.get("start_date"),
            manifest.get("end_date"),
        )
        if actual == expected:
            print(f"Using existing scene manifest: {manifest_path}")
            return manifest_path, manifest
        raise ValueError(
            "Existing manifest was built for different state/year/dates. "
            "Use --refresh-manifest."
        )

    items = search_state_items(
        state_geometry_4326,
        start_date,
        end_date,
        args,
    )
    if not items:
        raise RuntimeError("No HLS scenes remained after state/cloud filtering.")

    scene_root = manifest_path.parent / "scenes"
    manifest_items = []

    for item in items:
        band_map = band_map_for_item(item)
        missing = [band_map[name] for name in REQUIRED_ASSETS if band_map[name] not in item.assets]
        if missing:
            print(f"WARNING: skipping {item.id}; missing assets {missing}")
            continue

        scene_directory = scene_root / safe_name(item.id)
        assets = {}
        for canonical in REQUIRED_ASSETS:
            asset_id = band_map[canonical]
            assets[canonical] = {
                "asset_id": asset_id,
                "url": item.assets[asset_id].href,
                "path": str(scene_directory / f"{canonical}.tif"),
            }

        manifest_items.append(
            {
                "id": item.id,
                "collection": item.collection_id,
                "datetime": item_datetime(item),
                "bbox": list(item.bbox) if item.bbox else None,
                "cloud_cover": cloud_cover(item),
                "assets": assets,
            }
        )

    if not manifest_items:
        raise RuntimeError("No HLS scenes had all required VI assets.")

    manifest = {
        "state": state,
        "year": year,
        "start_date": start_date,
        "end_date": end_date,
        "max_scene_cloud": args.max_scene_cloud,
        "items": manifest_items,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Saved scene manifest: {manifest_path}")
    return manifest_path, manifest


# ============================================================
# Local scene processing
# ============================================================

def hls_clear_mask(
    fmask: np.ndarray,
    mask_adjacent_cloud: bool = True,
    mask_high_aerosol: bool = True,
) -> np.ndarray:
    fmask = fmask.astype("uint8")
    cloud = ((fmask >> 1) & 1).astype(bool)
    adjacent = ((fmask >> 2) & 1).astype(bool)
    shadow = ((fmask >> 3) & 1).astype(bool)
    snow = ((fmask >> 4) & 1).astype(bool)
    bad = cloud | shadow | snow
    if mask_adjacent_cloud:
        bad |= adjacent
    if mask_high_aerosol:
        aerosol = (fmask >> 6) & 3
        bad |= aerosol == 3
    return ~bad


def read_local_asset(
    path: str,
    target_crs,
    target_transform,
    height: int,
    width: int,
    resampling: Resampling,
    out_dtype: str,
    destination_nodata=None,
) -> np.ndarray:
    with rasterio.open(path) as source:
        source_nodata = source.nodata
        with WarpedVRT(
            source,
            crs=target_crs,
            transform=target_transform,
            width=width,
            height=height,
            resampling=resampling,
            src_nodata=source_nodata,
            nodata=(
                destination_nodata
                if destination_nodata is not None
                else source_nodata
            ),
        ) as vrt:
            return vrt.read(1, out_dtype=out_dtype)


def scale_reflectance(array: np.ndarray) -> np.ndarray:
    array = array.astype("float32")
    array[array <= -9990] = np.nan
    return array * REFLECTANCE_SCALE


def compute_vi_arrays(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    clear: np.ndarray,
) -> dict[str, np.ndarray]:
    reflectance_ok = (
        clear
        & np.isfinite(nir) & (nir >= 0) & (nir <= 1.2)
        & np.isfinite(red) & (red >= 0) & (red <= 1.2)
        & np.isfinite(green) & (green >= 0) & (green <= 1.2)
        & np.isfinite(blue) & (blue >= 0) & (blue <= 1.2)
        & np.isfinite(swir1) & (swir1 >= 0) & (swir1 <= 1.2)
    )

    outputs = {}

    denominator = nir + red
    ndvi = np.full(nir.shape, np.nan, dtype="float32")
    valid = reflectance_ok & (np.abs(denominator) > EPS)
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    outputs["NDVI"] = ndvi

    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.full(nir.shape, np.nan, dtype="float32")
    valid = reflectance_ok & (np.abs(denominator) > 0.05)
    evi[valid] = 2.5 * (nir[valid] - red[valid]) / denominator[valid]
    outputs["EVI"] = evi

    gcvi = np.full(nir.shape, np.nan, dtype="float32")
    valid = reflectance_ok & (green > 0.02) & (green < 1.0)
    gcvi[valid] = nir[valid] / green[valid] - 1.0
    outputs["GCVI"] = gcvi

    denominator = nir + swir1
    ndwi = np.full(nir.shape, np.nan, dtype="float32")
    valid = reflectance_ok & (np.abs(denominator) > EPS)
    ndwi[valid] = (nir[valid] - swir1[valid]) / denominator[valid]
    outputs["NDWI"] = ndwi

    return outputs


def process_local_scene(scene: dict[str, Any], grid: dict[str, Any]) -> dict[str, np.ndarray]:
    assets = scene["assets"]
    target = {
        "target_crs": grid["target_crs"],
        "target_transform": grid["target_transform"],
        "height": grid["height"],
        "width": grid["width"],
    }

    fmask = read_local_asset(
        assets["fmask"]["path"],
        target["target_crs"],
        target["target_transform"],
        target["height"],
        target["width"],
        Resampling.nearest,
        "uint8",
        destination_nodata=255,
    )
    clear = hls_clear_mask(
        fmask,
        _WORKER_CONFIG["mask_adjacent_cloud"],
        _WORKER_CONFIG["mask_high_aerosol"],
    )

    arrays = {}
    for band in ("blue", "green", "red", "nir", "swir1"):
        arrays[band] = scale_reflectance(
            read_local_asset(
                assets[band]["path"],
                target["target_crs"],
                target["target_transform"],
                target["height"],
                target["width"],
                Resampling.bilinear,
                "float32",
            )
        )

    return compute_vi_arrays(
        arrays["blue"],
        arrays["green"],
        arrays["red"],
        arrays["nir"],
        arrays["swir1"],
        clear,
    )


# ============================================================
# Pixel GeoTIFF output (same format as the original script)
# ============================================================

def safe_gtiff_block_size(size: int, target: int = 512) -> int:
    if size >= target:
        return target
    return max(16, (int(size) // 16) * 16)


def build_pixel_tif_path(row_base: dict[str, Any]) -> Path:
    directory = Path(_WORKER_CONFIG["pixel_tif_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    county_name = sanitize_filename_part(row_base["county_name"])
    return directory / (
        f"soybeans_{row_base['state']}_{row_base['year']}_"
        f"{row_base['county_fips']}_{county_name}_"
        "vi_pixels_nasa_cmr_hls.tif"
    )


def write_pixel_vi_geotiff(
    path: Path,
    grid: dict[str, Any],
    crop_mask: np.ndarray,
    valid_pixel_mask: np.ndarray,
    vi_mean_pixels: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    minimum_observations: np.ndarray,
    row_base: dict[str, Any],
) -> str:
    shape = (int(grid["height"]), int(grid["width"]))
    arrays = []
    names = []

    for vi in VI_NAMES:
        output = np.full(shape, PIXEL_TIF_NODATA, dtype="float32")
        mask = valid_pixel_mask & np.isfinite(vi_mean_pixels[vi])
        output[mask] = vi_mean_pixels[vi][mask]
        arrays.append(output)
        names.append(f"{vi}_mean")

    for vi in VI_NAMES:
        output = np.zeros(shape, dtype="float32")
        output[crop_mask] = counts[vi][crop_mask]
        arrays.append(output)
        names.append(f"{vi}_obsCount")

    output = np.zeros(shape, dtype="float32")
    output[crop_mask] = minimum_observations[crop_mask]
    arrays.append(output)
    names.append("min_index_obsCount")

    output = np.zeros(shape, dtype="float32")
    output[crop_mask] = 1.0
    arrays.append(output)
    names.append("crop_mask")

    output = np.zeros(shape, dtype="float32")
    output[valid_pixel_mask] = 1.0
    arrays.append(output)
    names.append("valid_vi_mask")

    if path.exists() and not _WORKER_CONFIG["overwrite"]:
        if validate_raster(path):
            return str(path)
        path.unlink(missing_ok=True)

    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": len(arrays),
        "dtype": "float32",
        "crs": grid["target_crs"],
        "transform": grid["target_transform"],
        "nodata": PIXEL_TIF_NODATA,
        "compress": "deflate",
        "predictor": 2,
        "BIGTIFF": "IF_SAFER",
    }
    if shape[0] >= 16 and shape[1] >= 16:
        profile.update(
            tiled=True,
            blockxsize=safe_gtiff_block_size(shape[1]),
            blockysize=safe_gtiff_block_size(shape[0]),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        for index, (name, array) in enumerate(zip(names, arrays), start=1):
            destination.write(array, index)
            destination.set_band_description(index, name)

        destination.update_tags(
            source="NASA CMR-STAC / LP DAAC HLS v2.0",
            crop="soybeans",
            state=str(row_base["state"]),
            state_fips=str(row_base["state_fips"]),
            county_fips=str(row_base["county_fips"]),
            county_name=str(row_base["county_name"]),
            year=str(row_base["year"]),
            season_start=str(row_base["season_start"]),
            season_end=str(row_base["season_end"]),
            min_obs_season=str(_WORKER_CONFIG["min_obs_season"]),
            vi_bands=",".join(VI_NAMES),
            valid_pixel_definition=(
                "crop_mask AND min_index_obsCount >= min_obs_season"
            ),
            nodata=str(PIXEL_TIF_NODATA),
        )

    return str(path)


# ============================================================
# Multiprocessing county worker
# ============================================================

def initialize_county_worker(manifest_path: str, config: dict[str, Any]) -> None:
    global _WORKER_SCENES, _WORKER_CONFIG
    os.environ["GDAL_NUM_THREADS"] = "1"
    _WORKER_CONFIG = config
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _WORKER_SCENES = manifest["items"]


def build_output_row(
    row_base: dict[str, Any],
    metrics: dict[str, Any],
    timing: dict[str, float],
) -> dict[str, Any]:
    county_seconds = timing.get("county_total_sec", 0.0)
    return {
        **row_base,
        **metrics,
        "county_runtime_sec": round(county_seconds, 2),
        "county_runtime_min": round(county_seconds / 60, 3),
        "remote_asset_reads": 0,
        "remote_read_warp_sec": 0.0,
        "remote_read_warp_min": 0.0,
        "search_hls_items_sec": 0.0,
        "load_cdl_crop_grid_sec": round(timing.get("load_grid_sec", 0.0), 3),
        "process_hls_items_loop_sec": round(timing.get("process_scenes_sec", 0.0), 3),
        "final_reduce_summary_sec": round(timing.get("reduce_sec", 0.0), 3),
        "pixel_tif_write_sec": round(timing.get("pixel_tif_sec", 0.0), 3),
        "pixel_tif_path": timing.get("pixel_tif_path", ""),
        "sleep_after_item_sec": 0.0,
        "parallel_workers": int(_WORKER_CONFIG["process_workers"]),
        "max_in_flight": int(_WORKER_CONFIG["process_workers"]),
    }


def process_county_worker(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    timing: dict[str, Any] = defaultdict(float)
    geometry = wkb.loads(bytes.fromhex(payload["geometry_wkb_hex"]))

    row_base = {
        "state": _WORKER_CONFIG["state"],
        "state_fips": _WORKER_CONFIG["state_fips"],
        "county_fips": payload["geoid"],
        "county_name": payload["name"],
        "year": _WORKER_CONFIG["year"],
        "crop": "soybeans",
        "season_start": _WORKER_CONFIG["start_date"],
        "season_end": _WORKER_CONFIG["end_date"],
    }

    try:
        grid_start = time.perf_counter()
        grid = load_county_crop_grid(
            _WORKER_CONFIG["cdl_path"],
            geometry,
            _WORKER_CONFIG["cdl_mode"],
            _WORKER_CONFIG["include_double_crop"],
        )
        timing["load_grid_sec"] = time.perf_counter() - grid_start

        crop_mask = grid["crop_mask"]
        crop_pixel_count = int(crop_mask.sum())

        if crop_pixel_count == 0:
            timing["county_total_sec"] = time.perf_counter() - started
            metrics = {
                "NDVI_mean": np.nan,
                "EVI_mean": np.nan,
                "GCVI_mean": np.nan,
                "NDWI_mean": np.nan,
                "min_index_obsCount": np.nan,
                "crop_pixel_count": 0,
                "valid_pixel_count": 0,
                "valid_pixel_fraction": 0.0,
            }
            return {
                "ok": True,
                "row": build_output_row(row_base, metrics, timing),
                "used_items": 0,
                "skipped_items": 0,
                "relevant_items": 0,
                "relevant_l30": 0,
                "relevant_s30": 0,
                "scene_failures": [],
            }

        county_bounds = geometry.bounds
        relevant = [
            scene
            for scene in _WORKER_SCENES
            if scene.get("bbox") is None
            or bbox_intersects(scene["bbox"], county_bounds)
        ]
        if not relevant:
            raise RuntimeError("No cached HLS scenes intersect this county.")

        shape = crop_mask.shape
        sums = {vi: np.zeros(shape, dtype="float64") for vi in VI_NAMES}
        counts = {vi: np.zeros(shape, dtype="uint16") for vi in VI_NAMES}
        used_items = 0
        skipped_items = 0
        scene_failures = []

        scene_start = time.perf_counter()
        with rasterio.Env(GDAL_CACHEMAX=int(_WORKER_CONFIG["gdal_cache_mb"])):
            for scene in relevant:
                try:
                    vi_arrays = process_local_scene(scene, grid)
                    item_has_values = False
                    for vi in VI_NAMES:
                        valid = crop_mask & np.isfinite(vi_arrays[vi])
                        if valid.any():
                            sums[vi][valid] += vi_arrays[vi][valid]
                            counts[vi][valid] += 1
                            item_has_values = True
                    if item_has_values:
                        used_items += 1
                    else:
                        skipped_items += 1
                except Exception as exc:  # noqa: BLE001
                    skipped_items += 1
                    scene_failures.append(
                        {"item_id": scene["id"], "error": str(exc)}
                    )

        timing["process_scenes_sec"] = time.perf_counter() - scene_start

        if used_items == 0:
            raise RuntimeError(
                f"All {len(relevant)} relevant cached scenes failed or had no valid crop pixels."
            )

        reduce_start = time.perf_counter()
        vi_mean_pixels = {}
        for vi in VI_NAMES:
            mean = np.full(shape, np.nan, dtype="float32")
            valid = counts[vi] > 0
            mean[valid] = (sums[vi][valid] / counts[vi][valid]).astype("float32")
            vi_mean_pixels[vi] = mean

        minimum_observations = np.minimum.reduce([counts[vi] for vi in VI_NAMES])
        valid_pixel_mask = crop_mask & (
            minimum_observations >= _WORKER_CONFIG["min_obs_season"]
        )
        valid_pixel_count = int(valid_pixel_mask.sum())
        valid_fraction = valid_pixel_count / crop_pixel_count

        metrics = {
            "NDVI_mean": np.nan,
            "EVI_mean": np.nan,
            "GCVI_mean": np.nan,
            "NDWI_mean": np.nan,
            "min_index_obsCount": np.nan,
            "crop_pixel_count": crop_pixel_count,
            "valid_pixel_count": valid_pixel_count,
            "valid_pixel_fraction": float(valid_fraction),
        }

        if valid_pixel_count > 0:
            for vi in VI_NAMES:
                values = vi_mean_pixels[vi][valid_pixel_mask]
                values = values[np.isfinite(values)]
                metrics[f"{vi}_mean"] = (
                    float(np.nanmean(values)) if values.size else np.nan
                )
            metrics["min_index_obsCount"] = float(
                np.nanmean(minimum_observations[valid_pixel_mask])
            )

        timing["reduce_sec"] = time.perf_counter() - reduce_start

        if _WORKER_CONFIG["save_pixel_tif"]:
            tif_start = time.perf_counter()
            tif_path = build_pixel_tif_path(row_base)
            timing["pixel_tif_path"] = write_pixel_vi_geotiff(
                tif_path,
                grid,
                crop_mask,
                valid_pixel_mask,
                vi_mean_pixels,
                counts,
                minimum_observations,
                row_base,
            )
            timing["pixel_tif_sec"] = time.perf_counter() - tif_start

        timing["county_total_sec"] = time.perf_counter() - started
        return {
            "ok": True,
            "row": build_output_row(row_base, metrics, timing),
            "used_items": used_items,
            "skipped_items": skipped_items,
            "relevant_items": len(relevant),
            "relevant_l30": sum("L30" in scene["id"] for scene in relevant),
            "relevant_s30": sum("S30" in scene["id"] for scene in relevant),
            "scene_failures": scene_failures,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "state": row_base["state"],
            "year": row_base["year"],
            "county_fips": row_base["county_fips"],
            "county_name": row_base["county_name"],
            "error": str(exc),
            "county_runtime_sec": round(time.perf_counter() - started, 2),
        }


# ============================================================
# CLI and main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download state-year HLS scenes once, then build county soybean VI "
            "summaries and pixel GeoTIFFs locally."
        )
    )

    parser.add_argument("--state", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--cdl", required=True, type=Path)
    parser.add_argument("--cdl-mode", choices=["binary", "raw"], default="binary")
    parser.add_argument("--exclude-double-crop", action="store_true")
    parser.add_argument("--county-fips", default=None)
    parser.add_argument("--county-name", default=None)

    parser.add_argument("--out-dir", type=Path, default=Path("./nasa_cmr_hls_vi_outputs"))
    parser.add_argument("--pixel-tif-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--save-pixel-tif", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--refresh-manifest", action="store_true")

    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-scene-cloud", type=float, default=70.0)
    parser.add_argument("--min-obs-season", type=int, default=3)

    parser.add_argument(
        "--mask-adjacent-cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mask-high-aerosol",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--process-workers", type=int, default=4)
    parser.add_argument("--gdal-cache-mb", type=int, default=256)
    parser.add_argument("--download-read-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=6)

    parser.add_argument("--earthdata-login", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--earthdata-strategy",
        choices=["interactive", "environment", "netrc", "all"],
        default="netrc",
    )
    parser.add_argument("--netrc-file", type=Path, default=None)
    parser.add_argument("--cookie-file", type=Path, default=None)  # compatibility

    parser.add_argument("--cmr-page-limit", type=int, default=100)
    parser.add_argument("--cmr-max-items", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")

    # Retained only so old command lines do not fail.
    parser.add_argument("--parallel-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-in-flight", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sleep-after-item", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--sleep-after-county", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--cmr-spatial-mode", default="bbox", help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> int:
    state_started = time.perf_counter()
    args = parse_args()
    args.state = args.state.upper()

    if args.state not in STATE_ABBR_TO_FIPS:
        raise ValueError(f"Unknown state abbreviation: {args.state}")
    if not args.cdl.exists():
        raise FileNotFoundError(f"CDL file does not exist: {args.cdl}")

    args.download_workers = max(1, int(args.download_workers))
    args.process_workers = max(1, int(args.process_workers))
    args.include_double_crop = not args.exclude_double_crop
    args.start_date = args.start_date or f"{args.year}-04-01"
    args.end_date = args.end_date or f"{args.year}-11-15"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.pixel_tif_dir is None:
        args.pixel_tif_dir = args.out_dir / "pixel_vi_tifs"

    output_csv = args.out_dir / (
        f"soybeans_{args.state}_{args.year}_county_vi_summary_nasa_cmr_hls.csv"
    )
    log_jsonl = args.out_dir / (
        f"soybeans_{args.state}_{args.year}_processing_log.jsonl"
    )
    failure_jsonl = args.out_dir / (
        f"soybeans_{args.state}_{args.year}_failures.jsonl"
    )

    clean_existing_csv(output_csv)
    prepare_earthdata_auth(args)

    print("\n============================================================")
    print("Scene-cached HLS County VI Summary")
    print("============================================================")
    print(f"State/year:       {args.state} {args.year}")
    print(f"Season:           {args.start_date} to {args.end_date}")
    print(f"CDL:              {args.cdl}")
    print(f"Scene cache:      {args.cache_dir}")
    print(f"Download workers: {args.download_workers}")
    print(f"County processes: {args.process_workers}")
    print(f"Output CSV:       {output_csv}")
    print(f"Pixel TIFF dir:   {args.pixel_tif_dir}")
    print("============================================================\n")

    counties_all = download_counties_2018(args.max_retries)
    all_state_counties = select_counties(counties_all, args.state)
    selected_counties = select_counties(
        counties_all,
        args.state,
        county_fips=args.county_fips,
        county_name=args.county_name,
    )

    all_state_geoids = set(all_state_counties["GEOID"].astype(str))
    done = set() if args.overwrite else completed_geoids(
        output_csv,
        args.state,
        args.year,
        require_pixel_tif=args.save_pixel_tif,
        pixel_tif_dir=args.pixel_tif_dir,
    )
    remaining = selected_counties[
        ~selected_counties["GEOID"].astype(str).isin(done)
    ].copy()

    print(f"Counties in state:     {len(all_state_counties)}")
    print(f"Already completed:     {len(done & all_state_geoids)}")
    print(f"Selected still needed: {len(remaining)}")

    cache_state_year = args.cache_dir / args.state / str(args.year)

    if remaining.empty:
        print("All selected counties are already complete.")
        if all_state_geoids.issubset(done) and cache_state_year.exists() and not args.keep_cache:
            shutil.rmtree(cache_state_year)
            print(f"Removed completed scene cache: {cache_state_year}")
        return 0

    if hasattr(all_state_counties.geometry, "union_all"):
        state_geometry = all_state_counties.geometry.union_all()
    else:
        state_geometry = all_state_counties.geometry.unary_union
    manifest_path, manifest = build_or_load_manifest(
        args.cache_dir,
        args.state,
        args.year,
        args.start_date,
        args.end_date,
        state_geometry,
        args,
    )

    print(
        f"HLS scenes after filters: {len(manifest['items'])} "
        f"({sum('L30' in item['id'] for item in manifest['items'])} L30, "
        f"{sum('S30' in item['id'] for item in manifest['items'])} S30)"
    )
    download_manifest_assets(manifest, args)

    worker_config = {
        "state": args.state,
        "state_fips": STATE_ABBR_TO_FIPS[args.state],
        "year": args.year,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "cdl_path": str(args.cdl),
        "cdl_mode": args.cdl_mode,
        "include_double_crop": args.include_double_crop,
        "min_obs_season": args.min_obs_season,
        "mask_adjacent_cloud": args.mask_adjacent_cloud,
        "mask_high_aerosol": args.mask_high_aerosol,
        "save_pixel_tif": args.save_pixel_tif,
        "pixel_tif_dir": str(args.pixel_tif_dir),
        "overwrite": args.overwrite,
        "process_workers": args.process_workers,
        "gdal_cache_mb": args.gdal_cache_mb,
    }

    payloads = [
        {
            "geoid": str(row["GEOID"]),
            "name": str(row["NAME"]),
            "geometry_wkb_hex": wkb.dumps(row.geometry).hex(),
        }
        for _, row in remaining.iterrows()
    ]

    failed_counties = []
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=args.process_workers,
        mp_context=context,
        initializer=initialize_county_worker,
        initargs=(str(manifest_path), worker_config),
    )
    pending = {
        executor.submit(process_county_worker, payload): payload
        for payload in payloads
    }

    try:
        with tqdm(total=len(payloads), desc=f"{args.state} {args.year} counties") as progress:
            while pending:
                finished, _ = wait(
                    pending,
                    timeout=120,
                    return_when=FIRST_COMPLETED,
                )
                if not finished:
                    print(
                        f"[HEARTBEAT] No county finished in 120 seconds; "
                        f"{len(pending)} pending",
                        flush=True,
                    )
                    continue

                for future in finished:
                    payload = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "ok": False,
                            "state": args.state,
                            "year": args.year,
                            "county_fips": payload["geoid"],
                            "county_name": payload["name"],
                            "error": str(exc),
                        }

                    if result["ok"]:
                        append_row_to_csv(output_csv, result["row"])
                        append_jsonl(
                            log_jsonl,
                            {
                                "state": args.state,
                                "year": args.year,
                                "county_fips": result["row"]["county_fips"],
                                "county_name": result["row"]["county_name"],
                                "status": "done",
                                "hls_items_after_filter": result["relevant_items"],
                                "hls_l30_items": result["relevant_l30"],
                                "hls_s30_items": result["relevant_s30"],
                                "hls_items_used": result["used_items"],
                                "hls_items_skipped": result["skipped_items"],
                                "crop_pixel_count": result["row"]["crop_pixel_count"],
                                "valid_pixel_count": result["row"]["valid_pixel_count"],
                                "valid_pixel_fraction": result["row"]["valid_pixel_fraction"],
                                "pixel_tif_path": result["row"]["pixel_tif_path"],
                                "scene_failures": result["scene_failures"],
                            },
                        )
                        print(
                            f"[COUNTY DONE] {args.state} {args.year} "
                            f"{result['row']['county_name']} "
                            f"({result['row']['county_fips']})",
                            flush=True,
                        )
                    else:
                        failed_counties.append(result)
                        append_jsonl(
                            failure_jsonl,
                            {**result, "status": "failed_county"},
                        )
                        print(
                            f"[COUNTY FAILED] {payload['name']} ({payload['geoid']}): "
                            f"{result.get('error')}",
                            flush=True,
                        )
                    progress.update(1)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    clean_existing_csv(output_csv)
    done_after = completed_geoids(
        output_csv,
        args.state,
        args.year,
        require_pixel_tif=args.save_pixel_tif,
        pixel_tif_dir=args.pixel_tif_dir,
    )
    state_complete = all_state_geoids.issubset(done_after)

    append_jsonl(
        log_jsonl,
        {
            "state": args.state,
            "year": args.year,
            "status": "state_year_done" if state_complete else "state_year_incomplete",
            "completed_counties": len(done_after & all_state_geoids),
            "total_counties": len(all_state_geoids),
            "failed_this_run": len(failed_counties),
            "state_runtime_sec": round(time.perf_counter() - state_started, 2),
        },
    )

    if failed_counties or not state_complete:
        print(
            f"State-year incomplete: {len(done_after & all_state_geoids)}/"
            f"{len(all_state_geoids)} counties complete. Rerun to resume.",
            flush=True,
        )
        return 2

    if cache_state_year.exists() and not args.keep_cache:
        shutil.rmtree(cache_state_year)
        print(f"Removed temporary scene cache: {cache_state_year}")

    print("\n============================================================")
    print(f"Completed {args.state} {args.year}")
    print(f"Runtime: {format_runtime(time.perf_counter() - state_started)}")
    print(f"CSV: {output_csv}")
    print(f"Pixel TIFFs: {args.pixel_tif_dir}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())