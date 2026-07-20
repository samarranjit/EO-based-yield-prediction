#!/usr/bin/env python3
# ============================================================
# NASA CMR-STAC / LP DAAC HLS v2.0 -> SOYBEAN-CHIP-GATED MONTHLY 6-BAND COMPOSITES
#
# Produces the raw-reflectance imagery farm_us.GeotiffMonthlyReader expects:
#   {imagery_root}/{year}/hls_{crop}_{STATE}_{YEAR}_{MON}.tif   (6 bands + 1 QA band)
#
# This is a different product from download_county_wise_summary_table_from_azure_hls.py,
# which computes seasonal per-county VI means (NDVI/EVI/GCVI/NDWI) for the ridge
# label-distribution step. This script instead writes cloud-masked MEAN composites
# of the 6 raw reflectance bands Prithvi was pretrained on, one GeoTIFF per
# (state, year, month), on the same statewide grid as the existing CDL rasters.
#
# Storage is chip-gated, not statewide: the raster's logical extent still covers
# the whole state (so it's a drop-in match for GeotiffMonthlyReader's windowed
# reads), but pixel data is only ever fetched/written for 224x224 chip windows
# whose CDL soybean fraction clears --min-crop-fraction. Everything else is left
# untouched (SPARSE_OK) rather than fetched, composited, and thrown away.
#   Statewide CDL -> 224x224 windows -> soybean-pixel fraction per window
#     -> skip windows below --min-crop-fraction -> fetch+write only the rest.
# See the chat: chip-gating saves ~2x versus raw crop-pixel fraction (~9x),
# because a permissive 5% threshold means most chips in dense soy states (IL/IA/IN)
# qualify anyway. --chip-size/--min-crop-fraction MUST match the model's actual
# data config (currently chip_size=224, stride=224, min_crop_fraction=0.05 in
# configs/data/us_soybeans_hls.yaml) -- if that config changes later, already-
# downloaded state-years need re-running with matching values or they will have
# silent gaps for newly-qualifying chips.
#
# Robustness, ported/adapted from download_county_wise_summary_table_from_azure_hls.py:
#   - rate limiting with jitter between remote reads and between months
#   - a warm-up read before each month's main loop
#   - persistent JSONL logs (per-item failures, per-month/state-year outcomes)
#   - resumable at CHIP granularity via a .progress.json sidecar per output file,
#     so an interrupted multi-day run can restart without redoing finished work
#   - one invocation can sweep --states x --years x --months, skipping/logging
#     missing CDL rasters and continuing past per-month or per-state-year failures
#
# Values written are raw DN (matching the existing HLS convention used elsewhere
# in this pipeline) -- the model applies hls_scale (1e-4) itself when reading.
# ============================================================

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from pystac_client import Client
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import STATE_ALPHA, START_YEAR, END_YEAR, CDL_MASK_DIR  # noqa: E402

# ------------------------------------------------------------
# Band mapping: canonical farm_us band order -> HLS asset id.
# NOTE: includes SWIR2 (B12/B07), which the VI-summary script omits because
# none of NDVI/EVI/GCVI/NDWI need it. farm_us requires all 6 bands.
# ------------------------------------------------------------
CANONICAL_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR1", "SWIR2"]
ALL_BANDS = CANONICAL_BANDS + ["OBS_COUNT"]  # QA band: clear observations behind each pixel's mean

S30_ASSET = {
    "BLUE": "B02", "GREEN": "B03", "RED": "B04",
    "NIR_NARROW": "B8A", "SWIR1": "B11", "SWIR2": "B12",
    "FMASK": "Fmask",
}
L30_ASSET = {
    "BLUE": "B02", "GREEN": "B03", "RED": "B04",
    "NIR_NARROW": "B05", "SWIR1": "B06", "SWIR2": "B07",
    "FMASK": "Fmask",
}

HLS_NODATA = -9999.0
HLS_INPUT_FILL_THRESHOLD = -9990  # raw asset fill values are <= this

MONTHS = ["APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV"]
MONTH_NUM = {"APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11}

CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
CMR_COLLECTION_L30 = "HLSL30_2.0"
CMR_COLLECTION_S30 = "HLSS30_2.0"
S30_START_DATE = "2015-11-28"


# ============================================================
# Small helpers
# ============================================================

def sleep_with_jitter(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds + random.uniform(0, min(1.0, seconds * 0.25)))


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def retry_call(label, func, max_retries=4, base_sleep=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            sleep = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            print(f"  WARNING: {label} failed ({attempt}/{max_retries}): {e}; retrying in {sleep:.1f}s")
            time.sleep(sleep)
    raise last_err


# ============================================================
# Auth + GDAL env (same approach as download_county_wise_summary_table_from_azure_hls.py)
# ============================================================

def prepare_earthdata_auth(out_dir: Path, do_login: bool, strategy: str):
    cookie_file = out_dir / "earthdata_gdal_cookies.txt"
    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    cookie_file.touch(exist_ok=True)
    os.environ["EARTHDATA_GDAL_COOKIE_FILE"] = str(cookie_file)

    if not do_login:
        print("Earthdata login step skipped (--no-earthdata-login). Relying on ~/.netrc.")
        return
    try:
        import earthaccess
        print("Checking Earthdata Login with earthaccess...")
        earthaccess.login(strategy=strategy, persist=True)
        print("Earthdata Login ready.")
    except ImportError:
        print("WARNING: earthaccess not installed. `pip install earthaccess` or set up ~/.netrc.")
    except Exception as e:
        print(f"WARNING: earthaccess login check failed: {e}")


def gdal_http_env_options():
    opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "GDAL_HTTP_CONNECTTIMEOUT": "20",
        "GDAL_HTTP_TIMEOUT": "180",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "5",
        "GDAL_HTTP_NETRC": "YES",
    }
    cookie_file = os.environ.get("EARTHDATA_GDAL_COOKIE_FILE")
    if cookie_file:
        opts["GDAL_HTTP_COOKIEFILE"] = cookie_file
        opts["GDAL_HTTP_COOKIEJAR"] = cookie_file
    return opts


# ============================================================
# Target grid: reuse the existing statewide CDL raster's grid so imagery
# lines up pixel-for-pixel with labels/CDL (DATA_CONTRACT requires this).
# ============================================================

def load_target_grid(cdl_path: Path):
    with rasterio.open(cdl_path) as src:
        return {"crs": src.crs, "transform": src.transform, "width": src.width, "height": src.height}


def state_bbox_wgs84(grid):
    return transform_bounds(
        grid["crs"], "EPSG:4326",
        *rasterio.transform.array_bounds(grid["height"], grid["width"], grid["transform"]),
    )


# ============================================================
# Chip gating: which 224x224 windows are worth fetching imagery for at all.
# Computed purely from the local CDL raster -- no network needed. farm_us's
# own manifest builder doesn't apply this filter yet (crop-fraction QC is
# deferred until imagery exists, per manifest.py), so this recomputes the
# same chip_size/stride/min_crop_fraction logic independently; keep the
# CLI defaults in sync with configs/data/us_soybeans_hls.yaml.
# ============================================================

def compute_qualifying_windows(cdl_path: Path, chip_size: int, min_crop_fraction: float) -> list[Window]:
    """Only full chip_size x chip_size windows qualify -- Prithvi expects complete
    224x224 chips, so partial edge windows are skipped rather than padded."""
    with rasterio.open(cdl_path) as ds:
        crop = ds.read(1) > 0.5
    h, w = crop.shape
    windows = []
    for row0 in range(0, h, chip_size):
        for col0 in range(0, w, chip_size):
            if row0 + chip_size > h or col0 + chip_size > w:
                continue
            block = crop[row0:row0 + chip_size, col0:col0 + chip_size]
            if block.mean() >= min_crop_fraction:
                windows.append(Window(col0, row0, chip_size, chip_size))
    return windows


# ============================================================
# CMR-STAC search (per month date window)
# ============================================================

def month_date_range(year: int, mon: str) -> tuple[str, str]:
    m = MONTH_NUM[mon]
    start = f"{year}-{m:02d}-01"
    if mon == "NOV":
        end = f"{year}-11-15"
    else:
        last_day = 30 if m in (4, 6, 9, 11) else 31
        end = f"{year}-{m:02d}-{last_day:02d}"
    return start, end


def collections_for_range(end_date: str):
    cols = [CMR_COLLECTION_L30]
    if end_date >= S30_START_DATE:
        cols.append(CMR_COLLECTION_S30)
    return cols


def search_month_items(catalog, bbox_4326, year, mon, max_scene_cloud, max_retries, page_limit=100, max_items=1000):
    start_date, end_date = month_date_range(year, mon)
    dt = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    items = []
    for collection in collections_for_range(end_date):
        def one_search(collection=collection):
            search = catalog.search(
                collections=[collection], datetime=dt, bbox=list(bbox_4326),
                limit=page_limit, max_items=max_items,
            )
            return list(search.items())

        try:
            found = retry_call(f"{collection} {mon} search", one_search, max_retries=max_retries)
            items.extend(found)
        except Exception as e:
            print(f"  WARNING: {collection} {mon} search failed: {e}")

    dedup = {i.id: i for i in items}
    items = list(dedup.values())

    filtered = []
    for item in items:
        cc = item.properties.get("eo:cloud_cover")
        if cc is None or cc <= max_scene_cloud:
            filtered.append(item)
    return filtered


def band_map_for_item(item):
    iid = getattr(item, "id", "") or ""
    if "S30" in iid:
        return S30_ASSET
    if "L30" in iid:
        return L30_ASSET
    raise ValueError(f"Cannot identify HLS collection for item {iid}")


def item_intersects_chip(item, chip_bounds_4326):
    ib = item.bbox  # [minx, miny, maxx, maxy]
    cb = chip_bounds_4326
    return not (ib[2] < cb[0] or ib[0] > cb[2] or ib[3] < cb[1] or ib[1] > cb[3])


def warm_up_earthdata(items, args):
    if not items:
        return
    item = items[0]
    band_map = band_map_for_item(item)
    href = item.assets[band_map["FMASK"]].href

    def _warmup():
        with rasterio.Env(**gdal_http_env_options()):
            with rasterio.open(href) as src:
                _ = src.crs, src.transform, src.width, src.height
        return True

    try:
        retry_call("Earthdata/GDAL warm-up", _warmup, max_retries=args.max_retries, base_sleep=5.0)
    except Exception as e:
        print(f"  WARNING: warm-up failed (continuing anyway): {e}")


# ============================================================
# Windowed remote reads
# ============================================================

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


def read_asset_to_grid(href, crs, transform, height, width, resampling, out_dtype, max_retries):
    def _read():
        with rasterio.Env(**gdal_http_env_options()):
            with rasterio.open(href) as src:
                with WarpedVRT(
                    src, crs=crs, transform=transform, width=width, height=height,
                    resampling=resampling, src_nodata=src.nodata, nodata=src.nodata,
                ) as vrt:
                    return vrt.read(1, out_dtype=out_dtype)
    return retry_call("read Earthdata COG window", _read, max_retries=max_retries)


def process_item_for_chip(item, chip_crs, chip_transform, chip_h, chip_w, args):
    band_map = band_map_for_item(item)
    fmask = read_asset_to_grid(
        item.assets[band_map["FMASK"]].href, chip_crs, chip_transform, chip_h, chip_w,
        Resampling.nearest, "uint8", args.max_retries,
    )
    clear = hls_clear_mask(fmask, args.mask_adjacent_cloud, args.mask_high_aerosol)

    bands = {}
    for band in CANONICAL_BANDS:
        arr = read_asset_to_grid(
            item.assets[band_map[band]].href, chip_crs, chip_transform, chip_h, chip_w,
            Resampling.bilinear, "float32", args.max_retries,
        )
        arr[arr <= HLS_INPUT_FILL_THRESHOLD] = np.nan
        bands[band] = arr
    return bands, clear


# ============================================================
# Chip compositing (bounded-concurrency, rate-limited, one item at a time
# submitted as prior ones complete -- mirrors the parallel item loop in
# download_county_wise_summary_table_from_azure_hls.py for real pacing,
# not just a bounded thread pool).
# ============================================================

def empty_chip(height: int, width: int) -> np.ndarray:
    """A chip with no intersecting HLS scenes at all: reflectance bands nodata,
    OBS_COUNT explicitly 0 (not nodata) -- distinguishes 'no scene covers this
    chip' from 'a storage block was never written'."""
    out = np.full((len(ALL_BANDS), height, width), HLS_NODATA, dtype="float32")
    out[-1] = 0.0
    return out


def composite_chip(items, chip_window, grid, args, log_jsonl, state, year, mon):
    chip_transform = rasterio.windows.transform(chip_window, grid["transform"])
    chip_h, chip_w = chip_window.height, chip_window.width
    chip_bounds_4326 = transform_bounds(
        grid["crs"], "EPSG:4326", *rasterio.windows.bounds(chip_window, grid["transform"]),
    )

    relevant = [it for it in items if item_intersects_chip(it, chip_bounds_4326)]
    if not relevant:
        return empty_chip(chip_h, chip_w)

    sums = {b: np.zeros((chip_h, chip_w), dtype="float32") for b in CANONICAL_BANDS}
    counts = np.zeros((chip_h, chip_w), dtype="uint16")

    max_in_flight = max(1, args.parallel_workers)
    item_iter = iter(relevant)
    successful_items = 0
    failed_items = 0

    with ThreadPoolExecutor(max_workers=max_in_flight) as executor:
        fut_to_item = {}

        def submit_next():
            try:
                item = next(item_iter)
            except StopIteration:
                return False
            fut = executor.submit(process_item_for_chip, item, grid["crs"], chip_transform, chip_h, chip_w, args)
            fut_to_item[fut] = item
            return True

        for _ in range(min(max_in_flight, len(relevant))):
            submit_next()

        while fut_to_item:
            done, _ = wait(list(fut_to_item.keys()), timeout=60, return_when=FIRST_COMPLETED)
            if not done:
                print(f"  [HEARTBEAT] no read finished in 60s; {len(fut_to_item)} pending")
                continue
            for fut in done:
                item = fut_to_item.pop(fut)
                try:
                    bands, clear = fut.result()
                    successful_items += 1
                    valid = clear.copy()
                    for b in CANONICAL_BANDS:
                        valid &= np.isfinite(bands[b])
                    for b in CANONICAL_BANDS:
                        sums[b][valid] += bands[b][valid]
                    counts[valid] += 1
                except Exception as e:
                    failed_items += 1
                    append_jsonl(log_jsonl, {
                        "state": state, "year": year, "month": mon,
                        "chip": [chip_window.row_off, chip_window.col_off],
                        "status": "item_failed", "item_id": item.id, "error": str(e),
                    })
                sleep_with_jitter(args.sleep_after_item)
                submit_next()

    if successful_items == 0:
        # Every intersecting scene failed (likely transient Earthdata/network trouble)
        # -- raise so this chip stays incomplete and is retried on the next run,
        # instead of being written as an empty chip and marked done forever.
        raise RuntimeError(
            f"All {len(relevant)} relevant HLS scenes failed for "
            f"chip row={chip_window.row_off}, col={chip_window.col_off} ({failed_items} failures)"
        )

    out = np.full((len(ALL_BANDS), chip_h, chip_w), HLS_NODATA, dtype="float32")
    has_data = counts > 0
    for i, b in enumerate(CANONICAL_BANDS):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_dn = sums[b] / np.maximum(counts, 1)
        out[i][has_data] = mean_dn[has_data].astype("float32")  # raw DN; model applies hls_scale itself
    out[len(CANONICAL_BANDS)] = counts.astype("float32")  # 0 where no observation, not nodata
    return out


# ============================================================
# Per-chip resumability: a .progress.json sidecar records which qualifying
# chip windows have already been written into the output GeoTIFF, so an
# interrupted run (Ctrl-C, crash, network outage) resumes without redoing
# finished chips or corrupting the partially-written raster.
# ============================================================

def progress_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".progress.json")


def save_progress(prog_path: Path, data: dict) -> None:
    tmp = prog_path.with_suffix(prog_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(prog_path)  # atomic on POSIX; avoids a torn progress file on crash


def load_or_init_progress(prog_path: Path, chip_size: int, min_crop_fraction: float, total_chips: int, state, year, mon):
    if prog_path.exists():
        data = json.loads(prog_path.read_text())
        if (data.get("chip_size") != chip_size
                or data.get("min_crop_fraction") != min_crop_fraction
                or data.get("total_chips") != total_chips):
            raise ValueError(
                f"{prog_path} was built with chip_size={data.get('chip_size')}/"
                f"min_crop_fraction={data.get('min_crop_fraction')}/total_chips={data.get('total_chips')}, "
                f"but this run uses chip_size={chip_size}/min_crop_fraction={min_crop_fraction}/"
                f"total_chips={total_chips}. Pass --overwrite to restart this month from scratch, "
                "or match the original --chip-size/--min-crop-fraction."
            )
        return data
    return {
        "state": state, "year": year, "month": mon,
        "chip_size": chip_size, "min_crop_fraction": min_crop_fraction,
        "total_chips": total_chips, "completed_chips": [], "complete": False,
    }


# ============================================================
# One (state, year, month) composite
# ============================================================

def composite_month(catalog, grid, qualifying_windows, state, year, mon, out_path, args, log_jsonl, fail_jsonl):
    t0 = time.perf_counter()
    bbox_4326 = state_bbox_wgs84(grid)
    items = search_month_items(
        catalog, bbox_4326, year, mon, args.max_scene_cloud, args.max_retries,
        page_limit=args.cmr_page_limit, max_items=args.cmr_max_items,
    )
    l30 = sum(1 for i in items if "L30" in i.id)
    s30 = sum(1 for i in items if "S30" in i.id)
    print(f"  {state} {year} {mon}: {len(items)} scenes after cloud filter ({l30} L30, {s30} S30), "
          f"{len(qualifying_windows)} qualifying chips")

    if args.dry_run:
        return {"state": state, "year": year, "month": mon, "items": len(items), "l30": l30, "s30": s30}

    if not items:
        print(f"    no scenes found; month left absent (reader's missing_month_policy handles this)")
        append_jsonl(log_jsonl, {"state": state, "year": year, "month": mon, "status": "no_scenes"})
        return {"state": state, "year": year, "month": mon, "items": 0, "l30": 0, "s30": 0}

    if not qualifying_windows:
        print(f"    no chips clear --min-crop-fraction; nothing to fetch for this state-year")
        append_jsonl(log_jsonl, {"state": state, "year": year, "month": mon, "status": "no_qualifying_chips"})
        return {"state": state, "year": year, "month": mon, "items": len(items), "l30": l30, "s30": s30}

    prog_path = progress_path(out_path)
    total_chips = len(qualifying_windows)

    if out_path.exists() and not args.overwrite:
        if prog_path.exists():
            progress = load_or_init_progress(prog_path, args.chip_size, args.min_crop_fraction, total_chips, state, year, mon)
            if progress["complete"]:
                print(f"    already complete, skipping ({out_path})")
                return {"state": state, "year": year, "month": mon, "items": len(items), "l30": l30, "s30": s30, "skipped": True}
        else:
            raise RuntimeError(
                f"{out_path} exists but {prog_path} is missing. A crash may have created the "
                "file before progress was ever recorded, so completion cannot be verified. "
                "Inspect it or rerun with --overwrite."
            )
    else:
        if args.overwrite:
            out_path.unlink(missing_ok=True)
            prog_path.unlink(missing_ok=True)
        progress = None

    fresh = not out_path.exists()
    if fresh:
        block = min(args.chip_size, grid["width"], grid["height"])
        profile = {
            "driver": "GTiff", "height": grid["height"], "width": grid["width"],
            "count": len(ALL_BANDS), "dtype": "float32", "crs": grid["crs"],
            "transform": grid["transform"], "nodata": HLS_NODATA,
            "compress": "deflate", "predictor": 2, "tiled": True,
            "blockxsize": block, "blockysize": block,
            "SPARSE_OK": "TRUE",  # blocks we never dst.write() into cost ~0 disk
            "BIGTIFF": "IF_SAFER",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, name in enumerate(ALL_BANDS, start=1):
                dst.set_band_description(i, name)
        progress = {
            "state": state, "year": year, "month": mon,
            "chip_size": args.chip_size, "min_crop_fraction": args.min_crop_fraction,
            "total_chips": total_chips, "completed_chips": [], "complete": False,
        }
        save_progress(prog_path, progress)
    elif progress is None:
        progress = load_or_init_progress(prog_path, args.chip_size, args.min_crop_fraction, total_chips, state, year, mon)

    completed_set = {tuple(c) for c in progress["completed_chips"]}
    remaining = [c for c in qualifying_windows if (c.row_off, c.col_off) not in completed_set]
    print(f"    {len(remaining)}/{total_chips} qualifying chips remaining")

    warm_up_earthdata(items, args)

    # Two levels of concurrency: up to chip_workers chips in flight at once, each
    # internally reading its own intersecting scenes with up to parallel_workers
    # threads (see composite_chip). Effective concurrent Earthdata connections is
    # roughly chip_workers x parallel_workers -- keep that product reasonable.
    print(f"    concurrency: {args.chip_workers} chip workers x {args.parallel_workers} "
          f"item workers each (~{args.chip_workers * args.parallel_workers} concurrent reads)")

    since_last_save = 0
    write_lock = threading.Lock()

    def process_one(chip_window, dst):
        nonlocal since_last_save
        try:
            chip_arr = composite_chip(items, chip_window, grid, args, log_jsonl, state, year, mon)
        except Exception as e:
            append_jsonl(fail_jsonl, {
                "state": state, "year": year, "month": mon,
                "chip": [chip_window.row_off, chip_window.col_off],
                "status": "chip_failed", "error": str(e),
            })
            print(f"    WARNING: chip ({chip_window.row_off},{chip_window.col_off}) failed, "
                  f"left for next resume: {e}")
            return
        # composite_chip always returns a real array now (empty_chip for no-coverage,
        # or raises on total read failure) -- never None, so always write.
        # rasterio dataset writes aren't thread-safe concurrently, so serialize them --
        # the network-bound compositing work above this line still runs in parallel.
        with write_lock:
            dst.write(chip_arr, window=chip_window)
            progress["completed_chips"].append([chip_window.row_off, chip_window.col_off])
            since_last_save += 1
            if since_last_save >= args.progress_save_every:
                save_progress(prog_path, progress)
                since_last_save = 0

    with rasterio.open(out_path, "r+") as dst:
        with ThreadPoolExecutor(max_workers=args.chip_workers) as chip_executor:
            futures = [chip_executor.submit(process_one, cw, dst) for cw in remaining]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"    {state} {year} {mon} chips"):
                pass

    # Final flush: covers any tail shorter than progress_save_every. A crash between
    # periodic saves redoes at most progress_save_every chips (already-written pixel
    # data is simply recomposited and rewritten, which is harmless).
    progress["complete"] = len(progress["completed_chips"]) == total_chips
    save_progress(prog_path, progress)

    elapsed = time.perf_counter() - t0
    status = "done" if progress["complete"] else "incomplete_chips_remain"
    append_jsonl(log_jsonl, {
        "state": state, "year": year, "month": mon, "status": status,
        "items": len(items), "l30": l30, "s30": s30, "qualifying_chips": total_chips,
        "elapsed_sec": round(elapsed, 1),
    })
    sleep_with_jitter(args.sleep_after_month)
    return {"state": state, "year": year, "month": mon, "items": len(items), "l30": l30, "s30": s30}


def process_state_year(catalog, state, year, args, log_jsonl, fail_jsonl):
    cdl_path = args.cdl_dir / f"cdl_soybeans_{state}_{year}.tif"
    if not cdl_path.exists():
        print(f"WARNING: CDL missing for {state} {year}, skipping ({cdl_path})")
        append_jsonl(fail_jsonl, {"state": state, "year": year, "status": "cdl_missing"})
        return []

    grid = load_target_grid(cdl_path)
    qualifying_windows = compute_qualifying_windows(cdl_path, args.chip_size, args.min_crop_fraction)
    cap_note = ""
    if args.max_chips is not None:
        qualifying_windows = qualifying_windows[:args.max_chips]
        cap_note = f" (capped to --max-chips={args.max_chips} for testing)"
    print(f"  {state} {year}: {len(qualifying_windows)} chips clear --min-crop-fraction={args.min_crop_fraction}{cap_note}")

    results = []
    for mon in args.months:
        out_path = args.out_dir / str(year) / f"hls_soybeans_{state}_{year}_{mon}.tif"
        try:
            results.append(composite_month(
                catalog, grid, qualifying_windows, state, year, mon, out_path, args, log_jsonl, fail_jsonl,
            ))
        except Exception as e:
            print(f"WARNING: {state} {year} {mon} failed entirely, continuing: {e}")
            append_jsonl(fail_jsonl, {"state": state, "year": year, "month": mon, "status": "month_failed", "error": str(e)})
    return results


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Build soybean-chip-gated monthly 6-band HLS composites for farm_us.")
    p.add_argument("--states", nargs="+", default=STATE_ALPHA)
    p.add_argument("--years", nargs="+", type=int, default=list(range(START_YEAR, END_YEAR + 1)))
    p.add_argument("--months", nargs="+", default=MONTHS, choices=MONTHS)
    p.add_argument("--cdl-dir", type=Path, default=CDL_MASK_DIR, help="Directory of cdl_soybeans_{STATE}_{YEAR}.tif rasters (defines the target grid + gating).")
    p.add_argument("--out-dir", required=True, type=Path, help="Imagery root; writes to {out-dir}/{year}/hls_soybeans_{STATE}_{YEAR}_{MON}.tif")
    p.add_argument("--chip-size", type=int, default=224, help="MUST match configs/data/us_soybeans_hls.yaml chip_size.")
    p.add_argument("--min-crop-fraction", type=float, default=0.05, help="MUST match configs/data/us_soybeans_hls.yaml min_crop_fraction.")
    p.add_argument("--max-scene-cloud", type=float, default=70.0)
    p.add_argument("--mask-adjacent-cloud", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mask-high-aerosol", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--chip-workers", type=int, default=2, help="Chips processed concurrently. Effective concurrent Earthdata reads ~= chip-workers x parallel-workers.")
    p.add_argument("--parallel-workers", type=int, default=3, help="Scene reads processed concurrently, per chip.")
    p.add_argument("--sleep-after-item", type=float, default=0.4, help="Jittered pause after each remote scene read.")
    p.add_argument("--sleep-after-month", type=float, default=5.0, help="Jittered pause after each (state,year,month) completes.")
    p.add_argument("--progress-save-every", type=int, default=25, help="Rewrite .progress.json after this many completed chips, not every single one.")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--cmr-page-limit", type=int, default=100)
    p.add_argument("--cmr-max-items", type=int, default=1000)
    p.add_argument("--earthdata-login", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--earthdata-strategy", default="interactive", choices=["interactive", "environment", "netrc", "all"])
    p.add_argument("--max-chips", type=int, default=None, help="Debug: cap qualifying chips processed per state-year, for a quick smoke test.")
    p.add_argument("--dry-run", action="store_true", help="Search + report scene/chip counts and estimated output size; no reads/writes.")
    p.add_argument("--overwrite", action="store_true", help="Restart matching months from scratch, ignoring any existing progress.")
    return p.parse_args()


def main():
    args = parse_args()
    args.states = [s.upper() for s in args.states]

    combos = [(s, y) for y in args.years for s in args.states]
    total_qualifying_px = 0
    missing_cdl = []
    print("Scanning CDL rasters for chip-gating (local only, no network)...")
    for s, y in combos:
        cdl = args.cdl_dir / f"cdl_soybeans_{s}_{y}.tif"
        if not cdl.exists():
            missing_cdl.append((s, y))
            continue
        windows = compute_qualifying_windows(cdl, args.chip_size, args.min_crop_fraction)
        total_qualifying_px += sum(w.width * w.height for w in windows)

    est_gb = total_qualifying_px * len(ALL_BANDS) * 4 * len(args.months) / 1e9
    print("============================================================")
    print(f"State-years requested: {len(combos)}  (missing CDL, will be skipped: {len(missing_cdl)})")
    if missing_cdl:
        preview = ", ".join(f"{s}-{y}" for s, y in missing_cdl[:10])
        print(f"  Missing: {preview}{' ...' if len(missing_cdl) > 10 else ''}")
    print(f"Months per state-year: {args.months}")
    print(f"Chip gating: chip_size={args.chip_size}, min_crop_fraction={args.min_crop_fraction} "
          "(MUST match configs/data/us_soybeans_hls.yaml)")
    print(f"Estimated uncompressed size (chip-gated): {est_gb:.1f} GB (deflate roughly halves this)")
    print(f"Dry run: {args.dry_run}")
    print("============================================================\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = args.out_dir / "_logs" / "hls_download_log.jsonl"
    fail_jsonl = args.out_dir / "_logs" / "hls_download_failures.jsonl"

    prepare_earthdata_auth(args.out_dir, args.earthdata_login, args.earthdata_strategy)
    catalog = Client.open(CMR_STAC_URL)

    try:
        for year in args.years:
            for state in args.states:
                print(f"\n=== {state} {year} ===")
                process_state_year(catalog, state, year, args, log_jsonl, fail_jsonl)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed chips/months are saved (.progress.json + written GeoTIFFs).")
        print("Rerun the same command to resume from where this left off.")
        raise SystemExit(130)

    print("\n============================================================")
    print("Done.")
    print(f"Log:      {log_jsonl}")
    print(f"Failures: {fail_jsonl}")
    print("============================================================")


if __name__ == "__main__":
    main()
