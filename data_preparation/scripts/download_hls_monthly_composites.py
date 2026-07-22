"""
Download each required HLS asset once, process it locally, and keep only
soybean-qualified 224 x 224 chip windows in sparse monthly GeoTIFFs.

Pipeline for each state/year/month:
  1. Find CDL windows whose soybean fraction meets the threshold.
  2. Query CMR-STAC for HLS L30/S30 granules overlapping those windows.
  3. Download the seven required assets per granule once to a resumable cache.
  4. Use a process pool to build chip composites from the local scene files.
  5. Write only qualifying chip windows into a sparse statewide GeoTIFF.
  6. Delete the downloaded scene cache after the month completes.

Output contract:
  {out_dir}/{year}/hls_soybeans_{STATE}_{YEAR}_{MON}.tif

Bands:
  BLUE, GREEN, RED, NIR_NARROW, SWIR1, SWIR2, OBS_COUNT

The reflectance bands remain raw HLS DN. The model applies the 1e-4 scale.
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
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import geopandas as gpd
import numpy as np
import rasterio
import requests
from affine import Affine
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

# Import the existing project configuration.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CDL_MASK_DIR, END_YEAR, START_YEAR, STATE_ALPHA, STATE_FIPS  # noqa: E402

# config.COUNTIES_GPKG points at data/counties/, which is empty -- the real file
# lives under data/county_maps/. Default to the actual location instead.
DEFAULT_COUNTY_GPKG = CDL_MASK_DIR.parent / "county_maps" / "selected_states_counties_2023.gpkg"


# =============================================================================
# Constants
# =============================================================================

CANONICAL_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR1", "SWIR2"]
ALL_BANDS = CANONICAL_BANDS + ["OBS_COUNT"]

S30_ASSET = {
    "BLUE": "B02",
    "GREEN": "B03",
    "RED": "B04",
    "NIR_NARROW": "B8A",
    "SWIR1": "B11",
    "SWIR2": "B12",
    "FMASK": "Fmask",
}
L30_ASSET = {
    "BLUE": "B02",
    "GREEN": "B03",
    "RED": "B04",
    "NIR_NARROW": "B05",
    "SWIR1": "B06",
    "SWIR2": "B07",
    "FMASK": "Fmask",
}

MONTHS = ["APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV"]
MONTH_NUM = {
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
}

CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
CMR_COLLECTION_L30 = "HLSL30_2.0"
CMR_COLLECTION_S30 = "HLSS30_2.0"
S30_START_DATE = "2015-11-28"

HLS_NODATA = -9999.0
HLS_INPUT_FILL_THRESHOLD = -9990.0
FMASK_NODATA = 255
MANIFEST_VERSION = 1

_DOWNLOAD_TLS = threading.local()
_WORKER_CONTEXT: dict[str, Any] | None = None


# =============================================================================
# Generic helpers
# =============================================================================


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def sleep_backoff(attempt: int, base: float = 2.0, maximum: float = 90.0) -> None:
    seconds = min(maximum, base * (2 ** max(0, attempt - 1)))
    seconds += random.uniform(0.0, min(2.0, seconds * 0.2))
    time.sleep(seconds)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def affine_to_list(transform: Affine) -> list[float]:
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]


def affine_from_list(values: list[float]) -> Affine:
    return Affine(*values)


def window_tuple(window: Window) -> tuple[int, int, int, int]:
    return (
        int(window.row_off),
        int(window.col_off),
        int(window.height),
        int(window.width),
    )


def tuple_window(values: tuple[int, int, int, int] | list[int]) -> Window:
    row, col, height, width = map(int, values)
    return Window(col, row, width, height)


def progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.json")


def valid_local_raster(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return src.width > 0 and src.height > 0 and src.count > 0
    except Exception:
        return False


# =============================================================================
# Earthdata authentication and resumable threaded downloads
# =============================================================================


def prepare_earthdata_auth(args: argparse.Namespace) -> None:
    if not args.earthdata_login:
        print("Earthdata login skipped; using existing credentials/netrc.")
        return

    import earthaccess

    print("Checking Earthdata Login with earthaccess...")
    earthaccess.login(strategy=args.earthdata_strategy, persist=True)
    print("Earthdata Login ready.")


def get_download_session() -> requests.Session:
    """One authenticated session per download thread, with a netrc fallback."""
    session = getattr(_DOWNLOAD_TLS, "session", None)
    if session is not None:
        return session

    try:
        import earthaccess

        session = earthaccess.get_requests_https_session()
    except Exception:
        # requests automatically consults ~/.netrc when trust_env is True.
        session = requests.Session()
        session.trust_env = True

    session.headers.update({"User-Agent": "farm-us-hls-scene-cache/1.0"})
    _DOWNLOAD_TLS.session = session
    return session


def reset_download_session() -> None:
    session = getattr(_DOWNLOAD_TLS, "session", None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
    _DOWNLOAD_TLS.session = None


def expected_download_size(response: requests.Response, resumed_bytes: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        length = int(content_length)
        return resumed_bytes + length if response.status_code == 206 else length

    return None


def download_one_asset(
    url: str,
    destination: str,
    retries: int,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    """Download one file with .part resume, authentication refresh, and validation."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if valid_local_raster(dest):
        return str(dest)

    if dest.exists():
        dest.unlink(missing_ok=True)

    partial = Path(str(dest) + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            session = get_download_session()
            resumed = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={resumed}-"} if resumed > 0 else {}

            with session.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(connect_timeout, read_timeout),
            ) as response:
                # A completed .part file can produce 416 on a resume request.
                if response.status_code == 416 and partial.exists():
                    partial.replace(dest)
                    if valid_local_raster(dest):
                        return str(dest)
                    dest.unlink(missing_ok=True)
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("HTTP 416 but the local partial file was invalid")

                response.raise_for_status()

                if resumed > 0 and response.status_code == 206:
                    mode = "ab"
                else:
                    mode = "wb"
                    resumed = 0

                expected_size = expected_download_size(response, resumed)

                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            if expected_size is not None and partial.stat().st_size != expected_size:
                raise IOError(
                    f"size mismatch for {dest.name}: "
                    f"got {partial.stat().st_size}, expected {expected_size}"
                )

            partial.replace(dest)
            if not valid_local_raster(dest):
                dest.unlink(missing_ok=True)
                raise IOError(f"downloaded file is not a readable raster: {dest}")

            return str(dest)

        except Exception as exc:
            last_error = exc
            reset_download_session()
            if attempt < retries:
                print(
                    f"  download retry {attempt}/{retries}: {dest.name}: {exc}",
                    flush=True,
                )
                sleep_backoff(attempt)

    raise RuntimeError(f"failed to download {url}: {last_error}")


def download_scene_assets(
    scenes: list[dict[str, Any]],
    month_cache: Path,
    args: argparse.Namespace,
) -> None:
    tasks: dict[str, tuple[str, str]] = {}

    for scene in scenes:
        item_dir = month_cache / "scenes" / safe_name(scene["id"])
        for logical_name, asset in scene["assets"].items():
            destination = item_dir / asset["filename"]
            asset["local_path"] = str(destination.relative_to(month_cache))
            tasks[str(destination)] = (asset["url"], str(destination))

    pending = []
    for url, destination in tasks.values():
        path = Path(destination)
        if not valid_local_raster(path):
            if path.exists():
                path.unlink(missing_ok=True)
            pending.append((url, destination))

    print(
        f"    scene cache: {len(tasks) - len(pending)}/{len(tasks)} assets already ready; "
        f"{len(pending)} to download"
    )

    if not pending:
        return

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        future_map = {
            executor.submit(
                download_one_asset,
                url,
                destination,
                args.download_retries,
                args.http_connect_timeout,
                args.http_read_timeout,
            ): destination
            for url, destination in pending
        }

        with tqdm(total=len(future_map), desc="    downloading HLS assets") as progress:
            for future in as_completed(future_map):
                destination = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"{destination}: {exc}")
                progress.update(1)

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            f"{len(failures)} HLS assets failed to download. First failures:\n{preview}"
        )

    invalid = [destination for _, destination in tasks.values() if not valid_local_raster(Path(destination))]
    if invalid:
        raise RuntimeError(f"{len(invalid)} cached assets failed final raster validation")


# =============================================================================
# CDL grid and chip gating
# =============================================================================


def load_target_grid(cdl_path: Path) -> dict[str, Any]:
    with rasterio.open(cdl_path) as src:
        if src.crs is None:
            raise ValueError(f"CDL raster has no CRS: {cdl_path}")
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }


def load_state_boundary_mask(
    county_gpkg: Path,
    state_fips: str,
    grid: dict[str, Any],
) -> np.ndarray:
    """Rasterize the requested state's county polygons onto the CDL grid.

    The CDL raster's rectangular extent can include pixels physically located
    in a neighboring state (its bounding box isn't clipped to the state
    boundary), so a plain `cdl > 0.5` threshold alone can offer another
    state's soybean pixels as "qualifying" for this state. This mask fixes
    that by restricting to the actual state polygon.
    """
    counties = gpd.read_file(county_gpkg)
    if "STATEFP" not in counties.columns:
        raise ValueError(f"{county_gpkg} has no STATEFP column")

    state_counties = counties[counties["STATEFP"] == state_fips]
    if state_counties.empty:
        raise ValueError(f"No counties found for STATEFP={state_fips!r} in {county_gpkg}")

    if state_counties.crs != grid["crs"]:
        state_counties = state_counties.to_crs(grid["crs"])

    mask = rasterize(
        [(geom, 1) for geom in state_counties.geometry],
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask.astype(bool)


def compute_qualifying_windows(
    cdl_path: Path,
    chip_size: int,
    min_crop_fraction: float,
    state_mask: np.ndarray | None = None,
) -> list[Window]:
    with rasterio.open(cdl_path) as src:
        crop = src.read(1) > 0.5

    if state_mask is not None:
        if state_mask.shape != crop.shape:
            raise ValueError(
                f"state_mask shape {state_mask.shape} does not match CDL raster shape {crop.shape}"
            )
        crop = crop & state_mask

    height, width = crop.shape
    windows: list[Window] = []

    for row in range(0, height, chip_size):
        for col in range(0, width, chip_size):
            if row + chip_size > height or col + chip_size > width:
                continue
            block = crop[row : row + chip_size, col : col + chip_size]
            if float(block.mean()) >= min_crop_fraction:
                windows.append(Window(col, row, chip_size, chip_size))

    return windows


def qualifying_bbox_wgs84(grid: dict[str, Any], windows: list[Window]) -> tuple[float, float, float, float]:
    min_row = min(int(w.row_off) for w in windows)
    min_col = min(int(w.col_off) for w in windows)
    max_row = max(int(w.row_off + w.height) for w in windows)
    max_col = max(int(w.col_off + w.width) for w in windows)

    union = Window(min_col, min_row, max_col - min_col, max_row - min_row)
    bounds = rasterio.windows.bounds(union, grid["transform"])
    return transform_bounds(grid["crs"], "EPSG:4326", *bounds, densify_pts=21)


def scene_pixel_extent(item_bbox: list[float], grid: dict[str, Any]) -> list[int]:
    target_bounds = transform_bounds("EPSG:4326", grid["crs"], *item_bbox, densify_pts=21)
    floating = from_bounds(*target_bounds, transform=grid["transform"])

    row0 = max(0, math.floor(floating.row_off))
    col0 = max(0, math.floor(floating.col_off))
    row1 = min(grid["height"], math.ceil(floating.row_off + floating.height))
    col1 = min(grid["width"], math.ceil(floating.col_off + floating.width))

    return [row0, row1, col0, col1]


def extent_intersects_qualifying_chip(
    extent: list[int],
    qualifying_cells: set[tuple[int, int]],
    chip_size: int,
) -> bool:
    row0, row1, col0, col1 = extent
    if row1 <= row0 or col1 <= col0:
        return False

    chip_row0 = max(0, row0 // chip_size)
    chip_row1 = max(0, (row1 - 1) // chip_size)
    chip_col0 = max(0, col0 // chip_size)
    chip_col1 = max(0, (col1 - 1) // chip_size)

    for chip_row in range(chip_row0, chip_row1 + 1):
        for chip_col in range(chip_col0, chip_col1 + 1):
            if (chip_row, chip_col) in qualifying_cells:
                return True
    return False


# =============================================================================
# CMR-STAC search and persistent scene manifest
# =============================================================================


def month_date_range(year: int, month: str) -> tuple[str, str]:
    month_number = MONTH_NUM[month]
    start = f"{year}-{month_number:02d}-01"
    if month == "NOV":
        end = f"{year}-11-15"
    else:
        last_day = 30 if month_number in (4, 6, 9, 11) else 31
        end = f"{year}-{month_number:02d}-{last_day:02d}"
    return start, end


def collections_for_range(end_date: str) -> list[str]:
    collections = [CMR_COLLECTION_L30]
    if end_date >= S30_START_DATE:
        collections.append(CMR_COLLECTION_S30)
    return collections


def band_map_for_item(item: Any) -> dict[str, str]:
    item_id = getattr(item, "id", "") or ""
    collection_id = getattr(item, "collection_id", "") or ""

    if "S30" in item_id or collection_id == CMR_COLLECTION_S30:
        return S30_ASSET
    if "L30" in item_id or collection_id == CMR_COLLECTION_L30:
        return L30_ASSET
    raise ValueError(f"cannot identify HLS product for {item_id}")


def cloud_cover(item: Any) -> float | None:
    value = item.properties.get("eo:cloud_cover")
    return None if value is None else float(value)


def retry_stac_search(label: str, function: Any, retries: int) -> list[Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return function()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(f"  {label} retry {attempt}/{retries}: {exc}")
                sleep_backoff(attempt)
    raise RuntimeError(f"{label} failed: {last_error}")


def search_month_items(
    catalog: Client,
    bbox_wgs84: tuple[float, float, float, float],
    year: int,
    month: str,
    args: argparse.Namespace,
) -> list[Any]:
    start_date, end_date = month_date_range(year, month)
    datetime_filter = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    items: list[Any] = []

    for collection in collections_for_range(end_date):
        def one_search(collection_id: str = collection) -> list[Any]:
            search = catalog.search(
                collections=[collection_id],
                datetime=datetime_filter,
                bbox=list(bbox_wgs84),
                limit=args.cmr_page_limit,
                max_items=args.cmr_max_items,
            )
            return list(search.items())

        found = retry_stac_search(
            f"{collection} {month} CMR-STAC search",
            one_search,
            args.max_retries,
        )
        items.extend(found)

    deduplicated = {item.id: item for item in items}
    filtered = []
    for item in deduplicated.values():
        coverage = cloud_cover(item)
        if coverage is None or coverage <= args.max_scene_cloud:
            filtered.append(item)

    return sorted(
        filtered,
        key=lambda item: (item.properties.get("datetime", ""), item.id),
    )


def asset_filename(item_id: str, logical_name: str, url: str) -> str:
    remote_name = unquote(Path(urlparse(url).path).name)
    if remote_name:
        return remote_name
    return f"{safe_name(item_id)}_{logical_name}.tif"


def manifest_config(
    state: str,
    year: int,
    month: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "state": state,
        "year": year,
        "month": month,
        "chip_size": args.chip_size,
        "min_crop_fraction": args.min_crop_fraction,
        "max_scene_cloud": args.max_scene_cloud,
    }


def build_or_load_manifest(
    catalog: Client,
    state: str,
    year: int,
    month: str,
    grid: dict[str, Any],
    qualifying_windows: list[Window],
    month_cache: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest_path = month_cache / "scene_manifest.json"
    expected_config = manifest_config(state, year, month, args)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != expected_config:
            raise ValueError(
                f"cache manifest settings differ: {manifest_path}. "
                "Use --overwrite to rebuild this month."
            )
        return manifest

    bbox_wgs84 = qualifying_bbox_wgs84(grid, qualifying_windows)
    items = search_month_items(catalog, bbox_wgs84, year, month, args)

    qualifying_cells = {
        (int(window.row_off) // args.chip_size, int(window.col_off) // args.chip_size)
        for window in qualifying_windows
    }

    scenes: list[dict[str, Any]] = []
    skipped_missing_assets = 0

    for item in items:
        if not item.bbox:
            continue

        extent = scene_pixel_extent(list(item.bbox), grid)
        if not extent_intersects_qualifying_chip(extent, qualifying_cells, args.chip_size):
            continue

        band_map = band_map_for_item(item)
        required_keys = ["FMASK"] + CANONICAL_BANDS
        missing = [band_map[key] for key in required_keys if band_map[key] not in item.assets]
        if missing:
            skipped_missing_assets += 1
            continue

        assets: dict[str, dict[str, str]] = {}
        for logical_name in required_keys:
            asset_key = band_map[logical_name]
            url = item.assets[asset_key].href
            assets[logical_name] = {
                "asset_key": asset_key,
                "url": url,
                "filename": asset_filename(item.id, logical_name, url),
            }

        scenes.append(
            {
                "id": item.id,
                "collection": getattr(item, "collection_id", ""),
                "datetime": item.properties.get("datetime", ""),
                "bbox_wgs84": list(item.bbox),
                "pixel_extent": extent,
                "assets": assets,
            }
        )

    manifest = {
        "config": expected_config,
        "query_bbox_wgs84": list(bbox_wgs84),
        "scenes_after_filter": len(scenes),
        "skipped_missing_assets": skipped_missing_assets,
        "scenes": scenes,
    }
    save_json_atomic(manifest_path, manifest)
    return manifest


# =============================================================================
# Multiprocessing workers: local scene files -> chip composites
# =============================================================================


def initialize_worker(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context

    # Avoid multiplying GDAL's internal threads by the number of Python processes.
    os.environ["GDAL_NUM_THREADS"] = "1"
    os.environ["GDAL_CACHEMAX"] = str(context["gdal_cache_mb"])


def hls_clear_mask(fmask: np.ndarray, mask_adjacent: bool, mask_high_aerosol: bool) -> np.ndarray:
    fmask = fmask.astype("uint8", copy=False)
    fill = fmask == FMASK_NODATA
    cloud = ((fmask >> 1) & 1).astype(bool)
    adjacent = ((fmask >> 2) & 1).astype(bool)
    shadow = ((fmask >> 3) & 1).astype(bool)
    snow = ((fmask >> 4) & 1).astype(bool)

    bad = fill | cloud | shadow | snow
    if mask_adjacent:
        bad |= adjacent
    if mask_high_aerosol:
        aerosol = (fmask >> 6) & 3
        bad |= aerosol == 3
    return ~bad


def extent_overlaps_window(extent: list[int], window_values: tuple[int, int, int, int]) -> bool:
    row, col, height, width = window_values
    row0, row1, col0, col1 = extent
    return not (
        row + height <= row0
        or row >= row1
        or col + width <= col0
        or col >= col1
    )


def open_scene_vrts(
    scene: dict[str, Any],
    context: dict[str, Any],
    stack: ExitStack,
) -> dict[str, WarpedVRT]:
    grid = context["grid"]
    transform = affine_from_list(grid["transform"])
    crs = grid["crs"]

    vrts: dict[str, WarpedVRT] = {}
    for logical_name in ["FMASK"] + CANONICAL_BANDS:
        local_path = scene["assets"][logical_name]["absolute_path"]
        source = stack.enter_context(rasterio.open(local_path))

        if logical_name == "FMASK":
            vrt = WarpedVRT(
                source,
                crs=crs,
                transform=transform,
                width=grid["width"],
                height=grid["height"],
                resampling=Resampling.nearest,
                src_nodata=source.nodata,
                nodata=FMASK_NODATA,
            )
        else:
            vrt = WarpedVRT(
                source,
                crs=crs,
                transform=transform,
                width=grid["width"],
                height=grid["height"],
                resampling=Resampling.bilinear,
                src_nodata=source.nodata,
                nodata=HLS_NODATA,
            )

        vrts[logical_name] = stack.enter_context(vrt)

    return vrts


def process_chip_batch(
    batch: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context was not initialized")

    context = _WORKER_CONTEXT
    scenes = context["scenes"]
    chip_count = len(batch)

    sums = [
        np.zeros((len(CANONICAL_BANDS), height, width), dtype="float32")
        for row, col, height, width in batch
    ]
    counts = [
        np.zeros((height, width), dtype="uint16")
        for row, col, height, width in batch
    ]
    relevant_scenes = np.zeros(chip_count, dtype="uint16")
    successful_scenes = np.zeros(chip_count, dtype="uint16")
    failed_scenes = np.zeros(chip_count, dtype="uint16")
    errors: list[str] = []
    bad_scene_ids: set[str] = set()

    for scene in scenes:
        relevant_indexes = [
            index
            for index, window_values in enumerate(batch)
            if extent_overlaps_window(scene["pixel_extent"], window_values)
        ]
        if not relevant_indexes:
            continue

        for index in relevant_indexes:
            relevant_scenes[index] += 1

        try:
            with ExitStack() as stack:
                vrts = open_scene_vrts(scene, context, stack)

                for index in relevant_indexes:
                    window = tuple_window(batch[index])
                    fmask = vrts["FMASK"].read(1, window=window, out_dtype="uint8")
                    valid = hls_clear_mask(
                        fmask,
                        context["mask_adjacent_cloud"],
                        context["mask_high_aerosol"],
                    )

                    arrays: list[np.ndarray] = []
                    for band in CANONICAL_BANDS:
                        array = vrts[band].read(1, window=window, out_dtype="float32")
                        array[array <= HLS_INPUT_FILL_THRESHOLD] = np.nan
                        arrays.append(array)
                        valid &= np.isfinite(array)

                    if np.any(valid):
                        for band_index, array in enumerate(arrays):
                            sums[index][band_index][valid] += array[valid]
                        counts[index][valid] += 1

                    successful_scenes[index] += 1

        except Exception as exc:
            bad_scene_ids.add(scene["id"])
            for index in relevant_indexes:
                failed_scenes[index] += 1
            if len(errors) < 20:
                errors.append(f"{scene['id']}: {exc}")

    results: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    failed_chips: list[tuple[int, int, int, int]] = []

    for index, window_values in enumerate(batch):
        row, col, height, width = window_values

        # No returned HLS granule covers this chip: write an explicit empty chip.
        if relevant_scenes[index] == 0:
            output = np.full((len(ALL_BANDS), height, width), HLS_NODATA, dtype="float32")
            output[-1] = 0.0
            results.append((window_values, output))
            continue

        # Do not silently build a mean that omits a downloaded scene. A failed
        # local scene causes this chip to remain incomplete and the scene cache
        # is invalidated for a clean re-download on the next wrapper restart.
        if failed_scenes[index] > 0 or successful_scenes[index] == 0:
            failed_chips.append(window_values)
            continue

        output = np.full((len(ALL_BANDS), height, width), HLS_NODATA, dtype="float32")
        has_data = counts[index] > 0

        for band_index in range(len(CANONICAL_BANDS)):
            mean = sums[index][band_index] / np.maximum(counts[index], 1)
            output[band_index][has_data] = mean[has_data]

        output[-1] = counts[index].astype("float32")
        results.append((window_values, output))

    return {
        "results": results,
        "failed_chips": failed_chips,
        "scene_errors": errors,
        "bad_scene_ids": sorted(bad_scene_ids),
    }


# =============================================================================
# Sparse output and chip-level recovery
# =============================================================================


def safe_block_size(size: int, target: int) -> int:
    block = min(size, target)
    block = max(16, (block // 16) * 16)
    return block


def create_or_load_output(
    output_path: Path,
    grid: dict[str, Any],
    qualifying_windows: list[Window],
    state: str,
    year: int,
    month: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    prog_path = progress_path(output_path)
    total_chips = len(qualifying_windows)

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        prog_path.unlink(missing_ok=True)

    if output_path.exists() and not prog_path.exists():
        raise RuntimeError(
            f"{output_path} exists without {prog_path}; use --overwrite after inspection"
        )

    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": grid["height"],
            "width": grid["width"],
            "count": len(ALL_BANDS),
            "dtype": "float32",
            "crs": grid["crs"],
            "transform": grid["transform"],
            "nodata": HLS_NODATA,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": safe_block_size(grid["width"], args.chip_size),
            "blockysize": safe_block_size(grid["height"], args.chip_size),
            "SPARSE_OK": "TRUE",
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(output_path, "w", **profile) as dst:
            for index, name in enumerate(ALL_BANDS, start=1):
                dst.set_band_description(index, name)
            dst.update_tags(
                source="NASA HLS v2.0 local scene cache",
                state=state,
                year=str(year),
                month=month,
                chip_size=str(args.chip_size),
                min_crop_fraction=str(args.min_crop_fraction),
                values="raw HLS DN; apply scale 0.0001 in model reader",
            )

        progress = {
            "state": state,
            "year": year,
            "month": month,
            "chip_size": args.chip_size,
            "min_crop_fraction": args.min_crop_fraction,
            "total_chips": total_chips,
            "completed_chips": [],
            "complete": False,
        }
        save_json_atomic(prog_path, progress)
    else:
        progress = json.loads(prog_path.read_text(encoding="utf-8"))
        if (
            progress.get("chip_size") != args.chip_size
            or progress.get("min_crop_fraction") != args.min_crop_fraction
            or progress.get("total_chips") != total_chips
        ):
            raise ValueError(
                f"progress settings differ for {output_path}; use --overwrite to rebuild"
            )

    completed = {
        (int(values[0]), int(values[1]))
        for values in progress.get("completed_chips", [])
    }
    return progress, completed


def update_progress(
    prog_path: Path,
    progress: dict[str, Any],
    completed: set[tuple[int, int]],
    total_chips: int,
) -> None:
    progress["completed_chips"] = [list(values) for values in sorted(completed)]
    progress["complete"] = len(completed) == total_chips
    save_json_atomic(prog_path, progress)


def chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def build_worker_context(
    grid: dict[str, Any],
    scenes: list[dict[str, Any]],
    month_cache: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    local_scenes = json.loads(json.dumps(scenes))
    for scene in local_scenes:
        for asset in scene["assets"].values():
            asset["absolute_path"] = str(month_cache / asset["local_path"])

    return {
        "grid": {
            "crs": grid["crs"].to_wkt(),
            "transform": affine_to_list(grid["transform"]),
            "width": grid["width"],
            "height": grid["height"],
        },
        "scenes": local_scenes,
        "mask_adjacent_cloud": args.mask_adjacent_cloud,
        "mask_high_aerosol": args.mask_high_aerosol,
        "gdal_cache_mb": args.gdal_cache_mb,
    }


def write_worker_result(
    dst: rasterio.io.DatasetWriter,
    worker_result: dict[str, Any],
    completed: set[tuple[int, int]],
    failure_log: Path,
    state: str,
    year: int,
    month: str,
) -> tuple[int, list[tuple[int, int, int, int]], list[str]]:
    written = 0
    for window_values, array in worker_result["results"]:
        row, col, height, width = window_values
        dst.write(array, window=Window(col, row, width, height))
        completed.add((row, col))
        written += 1

    for failed in worker_result.get("failed_chips", []):
        append_jsonl(
            failure_log,
            {
                "state": state,
                "year": year,
                "month": month,
                "status": "chip_failed_all_local_scenes",
                "chip": list(failed),
            },
        )

    if worker_result.get("scene_errors"):
        append_jsonl(
            failure_log,
            {
                "state": state,
                "year": year,
                "month": month,
                "status": "local_scene_read_errors",
                "errors": worker_result["scene_errors"],
            },
        )

    return written, worker_result.get("failed_chips", []), worker_result.get("bad_scene_ids", [])


def process_remaining_chips(
    output_path: Path,
    grid: dict[str, Any],
    qualifying_windows: list[Window],
    scenes: list[dict[str, Any]],
    month_cache: Path,
    state: str,
    year: int,
    month: str,
    args: argparse.Namespace,
    failure_log: Path,
) -> bool:
    progress, completed = create_or_load_output(
        output_path,
        grid,
        qualifying_windows,
        state,
        year,
        month,
        args,
    )
    prog_path = progress_path(output_path)
    total_chips = len(qualifying_windows)

    if progress.get("complete") is True and len(completed) == total_chips:
        print("    monthly output already complete")
        return True

    remaining = [
        window_tuple(window)
        for window in qualifying_windows
        if (int(window.row_off), int(window.col_off)) not in completed
    ]
    print(f"    local processing: {len(remaining)}/{total_chips} chips remaining")

    if not remaining:
        update_progress(prog_path, progress, completed, total_chips)
        return True

    batches = chunked(remaining, args.chips_per_task)
    context = build_worker_context(grid, scenes, month_cache, args)
    failed_batches: list[list[tuple[int, int, int, int]]] = []
    bad_scene_ids: set[str] = set()
    chips_since_save = 0

    with rasterio.open(output_path, "r+") as dst:
        try:
            mp_context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.process_workers,
                mp_context=mp_context,
                initializer=initialize_worker,
                initargs=(context,),
            ) as executor:
                future_map = {
                    executor.submit(process_chip_batch, batch): batch
                    for batch in batches
                }

                with tqdm(total=len(remaining), desc=f"    {state} {year} {month} local chips") as bar:
                    for future in as_completed(future_map):
                        batch = future_map[future]
                        try:
                            result = future.result()
                            written, failed_chips, bad_scenes = write_worker_result(
                                dst,
                                result,
                                completed,
                                failure_log,
                                state,
                                year,
                                month,
                            )
                            if failed_chips:
                                failed_batches.append(failed_chips)
                            bad_scene_ids.update(bad_scenes)
                            chips_since_save += written
                            bar.update(len(batch))

                            if chips_since_save >= args.progress_save_every:
                                update_progress(prog_path, progress, completed, total_chips)
                                chips_since_save = 0

                        except Exception as exc:
                            failed_batches.append(batch)
                            append_jsonl(
                                failure_log,
                                {
                                    "state": state,
                                    "year": year,
                                    "month": month,
                                    "status": "process_batch_failed",
                                    "chips": [list(values) for values in batch],
                                    "error": str(exc),
                                },
                            )
                            bar.update(len(batch))

        except Exception as exc:
            # A broken process pool should not throw away successful chip writes.
            append_jsonl(
                failure_log,
                {
                    "state": state,
                    "year": year,
                    "month": month,
                    "status": "process_pool_failed",
                    "error": str(exc),
                },
            )
            failed_batches = chunked(
                [
                    values
                    for values in remaining
                    if (values[0], values[1]) not in completed
                ],
                args.chips_per_task,
            )

        update_progress(prog_path, progress, completed, total_chips)

        # A local scene read error usually indicates a damaged cached asset.
        # Remove that scene so the next wrapper restart downloads it cleanly.
        if bad_scene_ids:
            print(f"    invalidating {len(bad_scene_ids)} damaged scene cache directories")
            for scene_id in bad_scene_ids:
                shutil.rmtree(month_cache / "scenes" / safe_name(scene_id), ignore_errors=True)

        # Fallback: retry process-level failures once in the main process. Scene
        # data failures wait for the next restart because their cache was removed.
        if failed_batches and not bad_scene_ids:
            print(f"    fallback: retrying {len(failed_batches)} failed batches serially")
            initialize_worker(context)
            for batch in tqdm(failed_batches, desc="    serial fallback batches"):
                batch = [values for values in batch if (values[0], values[1]) not in completed]
                if not batch:
                    continue
                try:
                    result = process_chip_batch(batch)
                    _, still_failed, _ = write_worker_result(
                        dst,
                        result,
                        completed,
                        failure_log,
                        state,
                        year,
                        month,
                    )
                    if still_failed:
                        append_jsonl(
                            failure_log,
                            {
                                "state": state,
                                "year": year,
                                "month": month,
                                "status": "serial_fallback_left_incomplete_chips",
                                "chips": [list(values) for values in still_failed],
                            },
                        )
                    update_progress(prog_path, progress, completed, total_chips)
                except Exception as exc:
                    append_jsonl(
                        failure_log,
                        {
                            "state": state,
                            "year": year,
                            "month": month,
                            "status": "serial_fallback_failed",
                            "chips": [list(values) for values in batch],
                            "error": str(exc),
                        },
                    )

    update_progress(prog_path, progress, completed, total_chips)
    return len(completed) == total_chips


# =============================================================================
# Month, state-year, and main orchestration
# =============================================================================


def process_month(
    catalog: Client,
    state: str,
    year: int,
    month: str,
    grid: dict[str, Any],
    qualifying_windows: list[Window],
    args: argparse.Namespace,
    log_path: Path,
    failure_log: Path,
) -> bool:
    output_path = args.out_dir / str(year) / f"hls_soybeans_{state}_{year}_{month}.tif"
    prog_path = progress_path(output_path)
    month_cache = args.scene_cache_dir / state / str(year) / month

    if output_path.exists() and prog_path.exists() and not args.overwrite:
        progress = json.loads(prog_path.read_text(encoding="utf-8"))
        if progress.get("complete") is True:
            print(f"  {state} {year} {month}: already complete")
            if args.delete_scene_cache and month_cache.exists():
                shutil.rmtree(month_cache, ignore_errors=True)
            return True

    if args.overwrite and month_cache.exists():
        shutil.rmtree(month_cache)

    month_cache.mkdir(parents=True, exist_ok=True)
    manifest = build_or_load_manifest(
        catalog,
        state,
        year,
        month,
        grid,
        qualifying_windows,
        month_cache,
        args,
    )
    scenes = manifest["scenes"]

    l30 = sum("L30" in scene["id"] for scene in scenes)
    s30 = sum("S30" in scene["id"] for scene in scenes)
    print(
        f"  {state} {year} {month}: {len(scenes)} required granules "
        f"({l30} L30, {s30} S30), {len(qualifying_windows)} soybean chips"
    )

    if args.dry_run:
        return True
    if not scenes:
        raise RuntimeError(f"no HLS granules found for {state} {year} {month}")

    download_scene_assets(scenes, month_cache, args)
    save_json_atomic(month_cache / "scene_manifest.json", manifest)

    complete = process_remaining_chips(
        output_path,
        grid,
        qualifying_windows,
        scenes,
        month_cache,
        state,
        year,
        month,
        args,
        failure_log,
    )

    append_jsonl(
        log_path,
        {
            "state": state,
            "year": year,
            "month": month,
            "status": "done" if complete else "incomplete",
            "scenes": len(scenes),
            "qualifying_chips": len(qualifying_windows),
            "output": str(output_path),
        },
    )

    if complete and args.delete_scene_cache:
        shutil.rmtree(month_cache, ignore_errors=True)
        print("    deleted temporary scene cache")

    return complete


def process_state_year(
    catalog: Client,
    state: str,
    year: int,
    args: argparse.Namespace,
    log_path: Path,
    failure_log: Path,
) -> bool:
    cdl_path = args.cdl_dir / f"cdl_soybeans_{state}_{year}.tif"
    if not cdl_path.exists():
        append_jsonl(
            failure_log,
            {"state": state, "year": year, "status": "cdl_missing", "path": str(cdl_path)},
        )
        print(f"WARNING: missing CDL: {cdl_path}")
        return False

    grid = load_target_grid(cdl_path)
    qualifying_windows = compute_qualifying_windows(
        cdl_path,
        args.chip_size,
        args.min_crop_fraction,
    )
    if args.max_chips is not None:
        qualifying_windows = qualifying_windows[: args.max_chips]

    print(
        f"\n=== {state} {year}: {len(qualifying_windows)} qualifying "
        f"{args.chip_size}x{args.chip_size} chips ==="
    )

    if not qualifying_windows:
        append_jsonl(log_path, {"state": state, "year": year, "status": "no_qualifying_chips"})
        return True

    all_complete = True
    for month in args.months:
        try:
            complete = process_month(
                catalog,
                state,
                year,
                month,
                grid,
                qualifying_windows,
                args,
                log_path,
                failure_log,
            )
            all_complete &= complete
        except Exception as exc:
            all_complete = False
            print(f"WARNING: {state} {year} {month} failed: {exc}")
            append_jsonl(
                failure_log,
                {
                    "state": state,
                    "year": year,
                    "month": month,
                    "status": "month_failed",
                    "error": str(exc),
                },
            )

    return all_complete


def parse_args() -> argparse.Namespace:
    default_processes = max(1, min(8, (os.cpu_count() or 2) - 1))

    parser = argparse.ArgumentParser(
        description="Download HLS scenes once, then build sparse soybean-chip monthly composites locally."
    )
    parser.add_argument("--states", nargs="+", default=STATE_ALPHA)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(START_YEAR, END_YEAR + 1)))
    parser.add_argument("--months", nargs="+", default=MONTHS, choices=MONTHS)
    parser.add_argument("--cdl-dir", type=Path, default=CDL_MASK_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scene-cache-dir", type=Path, default=None)

    parser.add_argument("--chip-size", type=int, default=224)
    parser.add_argument("--min-crop-fraction", type=float, default=0.05)
    parser.add_argument("--max-scene-cloud", type=float, default=70.0)
    parser.add_argument("--mask-adjacent-cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-high-aerosol", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--process-workers", type=int, default=default_processes)
    parser.add_argument("--chips-per-task", type=int, default=8)
    parser.add_argument("--gdal-cache-mb", type=int, default=256)
    parser.add_argument("--progress-save-every", type=int, default=8)

    parser.add_argument("--download-retries", type=int, default=8)
    parser.add_argument("--http-connect-timeout", type=int, default=30)
    parser.add_argument("--http-read-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--cmr-page-limit", type=int, default=100)
    parser.add_argument("--cmr-max-items", type=int, default=5000)

    parser.add_argument("--earthdata-login", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--earthdata-strategy",
        choices=["interactive", "environment", "netrc", "all"],
        default="netrc",
    )
    parser.add_argument("--delete-scene-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-chips", type=int, default=None, help="Debug-only chip cap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.states = [state.upper() for state in args.states]
    args.download_workers = max(1, args.download_workers)
    args.process_workers = max(1, args.process_workers)
    args.chips_per_task = max(1, args.chips_per_task)
    args.progress_save_every = max(1, args.progress_save_every)

    if args.scene_cache_dir is None:
        args.scene_cache_dir = args.out_dir / "_scene_cache"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.scene_cache_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.out_dir / "_logs" / "hls_scene_cached_log.jsonl"
    failure_log = args.out_dir / "_logs" / "hls_scene_cached_failures.jsonl"

    print("============================================================")
    print("HLS scene-cache monthly composite builder")
    print(f"States:            {args.states}")
    print(f"Years:             {args.years}")
    print(f"Months:            {args.months}")
    print(f"Download threads:  {args.download_workers}")
    print(f"Local processes:   {args.process_workers}")
    print(f"Chips per task:    {args.chips_per_task}")
    print(f"Scene cache:       {args.scene_cache_dir}")
    print(f"Delete cache:      {args.delete_scene_cache}")
    print("============================================================")

    prepare_earthdata_auth(args)
    catalog = Client.open(CMR_STAC_URL)

    all_complete = True
    try:
        for year in args.years:
            for state in args.states:
                all_complete &= process_state_year(
                    catalog,
                    state,
                    year,
                    args,
                    log_path,
                    failure_log,
                )
    except KeyboardInterrupt:
        print("\nInterrupted. Cached scenes and completed chip progress were preserved.")
        raise SystemExit(130)

    print("\n============================================================")
    print("Complete." if all_complete else "Incomplete work remains; rerun the same command to resume.")
    print(f"Log:      {log_path}")
    print(f"Failures: {failure_log}")
    print("============================================================")

    raise SystemExit(0 if all_complete else 2)


if __name__ == "__main__":
    main()
