#!/usr/bin/env python
# ============================================================
# Download PRISM 800 m weather for 2014-2024 (monthly by default, daily opt-in)
#
# Variables: ppt, tmean, vpdmin, vpdmax  (vpdmean is DERIVED in script 09)
#
# Run from data_preparation/:
#   python scripts/06_download_prism.py --estimate        # cost, no download
#   python scripts/06_download_prism.py --pilot           # 1 month, all vars
#   python scripts/06_download_prism.py --coverage        # what is on disk
#   python scripts/06_download_prism.py --workers 12      # full run
#
# PRISM HAS NO SPATIAL SUBSETTING
# --------------------------------
# Confirmed from the official web-service doc (PRISM_downloads_web_service.pdf,
# 26 Mar 2025), whose complete request syntax is
#     /get/<region>/<res>/<element>/<date>?format=[nc|asc|bil]
# <region> is only us | ak | hi | pr -- whole regions, no bounding box. Every
# request transfers a CONUS grid regardless of the footprint we want; that part
# cannot be reduced. What we CAN control is how many grids we ask for and how
# much we keep on disk afterward:
#
#   --temporal monthly (default)  528 files, ~28 GB transferred. PRISM serves
#       monthly ppt/tmean/vpdmin/vpdmax directly -- exact totals/means, not an
#       approximation, since ppt's monthly total and tmean's monthly mean are
#       linear aggregates of the daily values.
#   --temporal daily              16,072 files, ~262 GB transferred. Only worth
#       it for a NON-LINEAR index (growing degree days, stress-day counts, a
#       percentile) that cannot be recovered from monthly grids later.
#
# By default, every downloaded grid is immediately clipped to the study
# footprint and the CONUS original is deleted -- storing 262 GB of ground the
# model never reads would be pure waste. The clip is a plain rasterio window
# read on PRISM's own grid: no resampling, no reprojection.
# --keep-conus retains the full CONUS rasters; --no-clip stores only the raw
# zips and defers all raster work.
#
# PRISM'S PUBLISHED DOWNLOAD LIMIT
# ---------------------------------
# "if a file is downloaded twice in a 24-hour period, no more downloads of that
#  file will be allowed during that period. Repeated excessive download
#  activity may result in IP address blocking, at our discretion."
# One request plus one retry already spends that budget; a third attempt would
# be refused by the server and risks the whole IP being blocked, which would
# take out every other script too. So PRISM_MAX_RETRIES = 2, not the usual 5,
# and a file that fails twice is only recoverable the next day.
# ============================================================

from __future__ import annotations

import argparse
import calendar
import csv
import random
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QC_DIR = DATA_DIR / "qc"
COUNTIES_GPKG = DATA_DIR / "county_maps" / "selected_states_counties_2023.gpkg"

# --------------------------------------------------------------------------- #
# Scope -- the five states the current FARM-US model trains on (cornbelt5).
# Widen this when the model's data.states widens, and re-run --estimate first.
# --------------------------------------------------------------------------- #
EXOG_STATE_FIPS = {"IA": "19", "IL": "17", "IN": "18", "MD": "24", "MN": "27"}
START_YEAR = 2014
END_YEAR = 2024

# --------------------------------------------------------------------------- #
# PRISM endpoint (verified live 2026-09-26/27; see header for the doc citation)
# --------------------------------------------------------------------------- #
PRISM_BASE = "https://services.nacse.org/prism/data/get"
PRISM_REGION = "us"
PRISM_RES = "800m"
PRISM_VARS = ["ppt", "tmean", "vpdmin", "vpdmax"]
PRISM_DERIVED = {"vpdmean": (("vpdmin", "vpdmax"), "mean")}  # built by script 09
PRISM_TEMPORAL = "monthly"       # "monthly" or "daily" -- see header

# Measured throughput, used for the --estimate figures.
PRISM_BYTES_PER_FILE = 17_516_313          # daily 800m zip, e.g. ppt 20200715
PRISM_MONTHLY_BYTES_PER_FILE = 56_832_756  # monthly 800m zip, e.g. ppt 202007
PRISM_SECONDS_PER_FILE = 1.84              # single-stream daily download
PRISM_MEASURED_WORKERS = 8
PRISM_MEASURED_FILES_PER_SEC = 1.38        # measured on the July-2020 ppt pilot
PRISM_CLIPPED_BYTES_PER_FILE = 3_110_000   # measured after clip to 5-state bbox

PRISM_MAX_RETRIES = 2          # PRISM's own limit -- see header
BACKOFF_BASE_SECONDS = 2.0
REQUEST_TIMEOUT = (30, 300)    # (connect, read)
USER_AGENT = "FARM-US-research/1.0 (yield modelling; contact via repo)"
DEFAULT_WORKERS = 8

PRISM_DIR = DATA_DIR / "prism"
PRISM_RAW_DIR = PRISM_DIR / "raw"
PRISM_CLIP_DIR = PRISM_DIR / "daily_clipped"
PRISM_MONTHLY_DIR = PRISM_DIR / "monthly"


def prism_url(var: str, date_str: str) -> str:
    """``date_str`` is YYYYMMDD (daily) or YYYYMM (monthly); PRISM infers which."""
    return f"{PRISM_BASE}/{PRISM_REGION}/{PRISM_RES}/{var}/{date_str}"


def prism_raw_path(var: str, date_str: str) -> Path:
    return PRISM_RAW_DIR / var / date_str[:4] / f"prism_{var}_{PRISM_RES}_{date_str}.zip"


def prism_clip_path(var: str, date_str: str) -> Path:
    """A monthly download is already the final product -> straight to MONTHLY_DIR,
    using the same naming script 09 and the model side expect."""
    if len(date_str) == 6:
        return PRISM_MONTHLY_DIR / var / f"prism_{var}_{date_str}.tif"
    return PRISM_CLIP_DIR / var / date_str[:4] / f"prism_{var}_{date_str}.tif"


def date_range(start_year: int, end_year: int) -> list[str]:
    out, d, last = [], date(start_year, 1, 1), date(end_year, 12, 31)
    while d <= last:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def month_range(start_year: int, end_year: int, months: list[int] | None = None) -> list[str]:
    keep = set(months) if months else set(range(1, 13))
    return [f"{y}{m:02d}" for y in range(start_year, end_year + 1) for m in range(1, 13) if m in keep]


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _study_geometry():
    """Union of the 5 study-state polygons, in EPSG:4326.

    NOTE ON SHAPE: IA/IL/IN/MN cluster in the Corn Belt, but MD sits ~500 miles
    east on the coast. The BOUNDING BOX of this union therefore also covers
    DE/MI/MO/OH/PA/VA/WI, none of which are in the study -- that rectangle is
    unavoidable geometry (a bbox is a rectangle), not a bug. clip_to_region masks
    every pixel outside the actual polygons, so what lands on disk matches the 5
    states even though the intermediate window is a much larger rectangle.
    """
    import geopandas as gpd

    if not COUNTIES_GPKG.exists():
        raise FileNotFoundError(f"{COUNTIES_GPKG} not found -- run scripts/02_download_counties.py first")
    gdf = gpd.read_file(COUNTIES_GPKG)
    gdf = gdf[gdf["STATEFP"].isin(EXOG_STATE_FIPS.values())]
    if gdf.empty:
        raise ValueError(f"No counties for STATEFP in {sorted(EXOG_STATE_FIPS.values())} in {COUNTIES_GPKG}")
    return gdf.to_crs("EPSG:4326").geometry.union_all()


def region_bounds_4326() -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of the 5-state footprint, in EPSG:4326."""
    return tuple(_study_geometry().bounds)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Download machinery (atomic writes, integrity checks, backoff on 429/503)
# --------------------------------------------------------------------------- #

_local = threading.local()


def _session() -> requests.Session:
    """One Session per thread -- requests.Session is not documented thread-safe."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _local.session = s
    return s


@dataclass
class Result:
    path: Path
    status: str          # "downloaded" | "skipped" | "missing" | "failed"
    n_bytes: int = 0
    message: str = ""


def _valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and len(z.namelist()) > 0
    except Exception:
        return False


def verify_zip(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and _valid_zip(path)


def verify_tif(path: Path) -> bool:
    """Existence + a TIFF magic-number check, to catch an HTML error page saved
    with a .tif name -- the realistic failure mode against a retired endpoint."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as fh:
        return fh.read(4) in (b"II*\x00", b"MM\x00*")


def fetch_zip(url: str, dest: Path, overwrite: bool = False,
              max_retries: int = PRISM_MAX_RETRIES) -> Result:
    """Download a PRISM zip with retries, atomically, and verify it.

    Never raises: one failed file out of thousands must not abort a multi-hour
    job. The caller aggregates failures into the summary/coverage report.
    """
    if not overwrite and verify_zip(dest):
        return Result(dest, "skipped", dest.stat().st_size)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last = ""

    for attempt in range(1, max_retries + 1):
        try:
            with _session().get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code == 404:
                    return Result(dest, "missing", 0, "HTTP 404")
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()

                # The retired endpoint returns HTTP 200 with a 430-byte HTML
                # notice instead of a zip -- Content-Type catches that trap
                # regardless of what the status code claims.
                ctype = r.headers.get("Content-Type", "")
                if "text/html" in ctype:
                    body = r.raw.read(400, decode_content=True)[:200].decode("utf-8", "replace")
                    return Result(dest, "failed", 0, f"server returned HTML not data: {body.strip()[:120]}")

                n = 0
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                            n += len(chunk)

            if not _valid_zip(part):
                raise OSError(f"zip integrity check failed after {n} bytes")
            part.replace(dest)
            return Result(dest, "downloaded", n)

        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            last = f"{type(exc).__name__}: {exc}"
            part.unlink(missing_ok=True)
            if attempt < max_retries:
                sleep = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(sleep * (0.5 + random.random()))  # full jitter

    return Result(dest, "failed", 0, f"{max_retries} attempts (PRISM's own limit); last: {last}")


def run_pool(jobs, worker, n_workers: int, desc: str) -> list[Result]:
    """I/O-bound -> threads, not processes: the GIL releases during socket waits."""
    from tqdm import tqdm

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(worker, j): j for j in jobs}
        bar = tqdm(as_completed(futures), total=len(futures), desc=desc, unit="file")
        for fut in bar:
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append(Result(Path("?"), "failed", 0, f"{type(exc).__name__}: {exc}"))
            ok = sum(r.status in ("downloaded", "skipped") for r in results)
            bar.set_postfix(ok=ok, fail=sum(r.status == "failed" for r in results))
    return results


def summarize(results: list[Result], label: str) -> None:
    from collections import Counter

    counts = Counter(r.status for r in results)
    print(f"\n{label}")
    for k in ("downloaded", "skipped", "missing", "failed"):
        if counts.get(k):
            print(f"  {k:<11} {counts[k]:>7}")
    print(f"  bytes on disk this run: {human_bytes(sum(r.n_bytes for r in results))}")

    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"\n  {len(failed)} FAILED (re-run to retry -- they are skipped once valid):")
        for r in failed[:15]:
            print(f"    {r.path.name}: {r.message}")
        if len(failed) > 15:
            print(f"    ... and {len(failed) - 15} more")


# --------------------------------------------------------------------------- #
# Clip a downloaded CONUS grid to the study footprint
# --------------------------------------------------------------------------- #

def _apply_state_mask(data, transform, geometry, nodata_val: float):
    """Set every pixel outside ``geometry`` to ``nodata_val``, in place semantics.

    The bounding-box window is a rectangle spanning IA/MN in the west to MD in
    the east and therefore also covers states not in the study (see
    _study_geometry). This rasterizes the actual state polygons onto the
    window's own grid and masks everything outside them, so the stored raster
    reflects the 5 states, not the rectangle that contains them. Deflate
    compresses the resulting nodata blocks to nearly nothing, so this also
    shrinks storage rather than costing anything.
    """
    from rasterio.features import geometry_mask

    # geometry_mask returns True OUTSIDE the shapes by default -- exactly the
    # "mask these out" boolean we want, with no inversion to get backwards.
    outside = geometry_mask([geometry], out_shape=data.shape[-2:], transform=transform, invert=False)
    data = data.copy()
    data[..., outside] = nodata_val
    return data


def clip_to_region(zip_path: Path, out_path: Path, bounds, geometry) -> tuple[bool, str]:
    """Extract the .tif inside ``zip_path``, window to ``bounds``, then mask to
    ``geometry`` (the actual state polygons, not just the bounding rectangle).

    No resampling, no reprojection for the windowing step: the window comes from
    geographic bounds on PRISM's own grid, so pixel VALUES are a byte-faithful
    subset of the source. Only pixels outside the true state shapes are then
    overwritten with nodata.
    """
    import rasterio
    from rasterio.windows import from_bounds

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".tif")]
        if not names:
            return False, "no .tif inside archive"
        member = names[0]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        part = out_path.with_suffix(".part.tif")
        with z.open(member) as fh, rasterio.MemoryFile(fh.read()) as mem, mem.open() as src:
            win = from_bounds(*bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            data = src.read(window=win)
            win_transform = src.window_transform(win)
            nod = src.nodata if src.nodata is not None else -9999.0
            data = _apply_state_mask(data, win_transform, geometry, nod)

            profile = src.profile
            profile.update(
                height=int(win.height), width=int(win.width), transform=win_transform,
                nodata=nod, compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256, driver="GTiff",
            )
            with rasterio.open(part, "w", **profile) as dst:
                dst.write(data)
                dst.update_tags(
                    prism_source=member, prism_resolution=PRISM_RES,
                    clipped_to="5-state polygons (EPSG:4326), windowed to their union bbox",
                    clip_bounds=",".join(f"{b:.6f}" for b in bounds),
                    masked_outside_states="true",
                )
        part.replace(out_path)
    return True, ""


def remask_existing(path: Path, geometry) -> Result:
    """Apply the state-polygon mask to an ALREADY-clipped raster, in place.

    For files clipped before this mask existed (e.g. from an early pilot run),
    the source CONUS zip is usually gone (deleted after clipping by default), but
    it is not needed: the already-clipped raster is a strict window superset of
    the true footprint, so masking can be applied directly to it with no
    re-download. Idempotent -- tags record whether a file was already masked, so
    re-running --remask on a fully-masked archive is a fast no-op pass.
    """
    import rasterio

    with rasterio.open(path) as src:
        if src.tags().get("masked_outside_states") == "true":
            return Result(path, "skipped", path.stat().st_size)
        data = src.read()
        transform = src.transform
        nod = src.nodata if src.nodata is not None else -9999.0
        profile = src.profile

    data = _apply_state_mask(data, transform, geometry, nod)
    part = path.with_suffix(".part.tif")
    profile.update(nodata=nod, compress="deflate", predictor=3, tiled=True,
                   blockxsize=256, blockysize=256, driver="GTiff")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(masked_outside_states="true",
                        clipped_to="5-state polygons (EPSG:4326), windowed to their union bbox")
    part.replace(path)
    return Result(path, "downloaded", path.stat().st_size)


# --------------------------------------------------------------------------- #
# Reporting modes
# --------------------------------------------------------------------------- #

def estimate(dates: list[str], n_workers: int, clip: bool, temporal: str, variables: list[str]) -> None:
    monthly = temporal == "monthly"
    n_files = len(dates) * len(variables)
    per_file = PRISM_MONTHLY_BYTES_PER_FILE if monthly else PRISM_BYTES_PER_FILE
    raw = n_files * per_file
    seq = n_files * PRISM_SECONDS_PER_FILE * (per_file / PRISM_BYTES_PER_FILE)

    minx, miny, maxx, maxy = region_bounds_4326()
    conus_area = (125.0 - 66.9) * (49.4 - 24.5)
    frac = ((maxx - minx) * (maxy - miny)) / conus_area
    clipped = (n_files * PRISM_CLIPPED_BYTES_PER_FILE if not monthly else raw * frac)

    print("=" * 66)
    print(f"PRISM DOWNLOAD ESTIMATE  ({temporal})")
    print("=" * 66)
    print(f"  endpoint        {PRISM_BASE}/{PRISM_REGION}/{PRISM_RES}/<var>/<date>")
    print(f"  variables       {', '.join(variables)}")
    print(f"  periods         {dates[0]}..{dates[-1]}  ({len(dates)} {'months' if monthly else 'days'})")
    print(f"  files           {n_files:,}")
    print(f"  study footprint lon {minx:.2f}..{maxx:.2f}  lat {miny:.2f}..{maxy:.2f}")
    print()
    print("  PRISM HAS NO BBOX PARAMETER -- every request transfers a CONUS grid.")
    print("  Transfer volume can only be reduced by requesting FEWER grids, which")
    print("  is what --temporal monthly does. Nothing here keeps CONUS on disk.")
    print()
    print(f"  transfer volume        {human_bytes(raw)}  (always downloaded)")
    print(f"  stored, --keep-conus   {human_bytes(raw)}")
    print(f"  stored, clipped        {human_bytes(clipped)}  ({frac * 100:.0f}% of CONUS bbox)")
    print(f"  stored, --no-clip      {human_bytes(raw)}  (zips only)")
    print()
    print(f"  serial time     ~{human_time(seq)}")
    rate = (PRISM_MEASURED_FILES_PER_SEC * (n_workers / PRISM_MEASURED_WORKERS)
            / (per_file / PRISM_BYTES_PER_FILE))
    print(f"  at {n_workers} workers    ~{human_time(n_files / rate)}"
          f"  (measured: {PRISM_MEASURED_FILES_PER_SEC:.2f} daily files/s at {PRISM_MEASURED_WORKERS} workers)")
    print(f"  mode            {'download + clip + delete CONUS' if clip else 'download only'}")
    print()
    if not monthly:
        ratio = raw / (len(dates) / 12 * len(variables) * PRISM_MONTHLY_BYTES_PER_FILE)
        print(f"  *** --temporal daily transfers {ratio:.0f}x more than monthly. ***")
        print("  Only worth it for non-linear indices (growing degree days, stress-day")
        print("  counts, dry spells). Monthly ppt/tmean are already exact totals/means.")
        print()
    print(f"  PRISM limit     max 2 fetches per file per 24 h; retries capped at {PRISM_MAX_RETRIES}.")
    print("                  A file failing twice is recoverable only tomorrow.")
    print("  The >8-worker figure extrapolates linearly from a measurement at 8, which")
    print("  the server will not honour indefinitely. 429s/503s in the failure list ->")
    print("  LOWER --workers; the published terms allow IP blocking.")
    print("=" * 66)

    import shutil
    free = shutil.disk_usage(DATA_DIR).free
    need = clipped if clip else raw
    print(f"  disk free       {human_bytes(free)}")
    print(f"  projected need  {human_bytes(need)}")
    print("  *** INSUFFICIENT DISK ***" if need > free * 0.9 else f"  headroom        {human_bytes(free - need)}")


def coverage(dates: list[str], clip: bool, variables: list[str]) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    out = QC_DIR / "prism_coverage.csv"
    years = sorted({d[:4] for d in dates})

    rows = []
    print(f"{'var':<9}{'year':<7}{'expected':>9}{'present':>9}{'missing':>9}")
    print("-" * 43)
    for var in variables:
        for year in years:
            want = [d for d in dates if d[:4] == year]
            present = 0
            for d in want:
                p = prism_clip_path(var, d) if clip else prism_raw_path(var, d)
                if (verify_tif(p) if clip else verify_zip(p)):
                    present += 1
            missing = len(want) - present
            rows.append({"variable": var, "year": year, "expected": len(want),
                         "present": present, "missing": missing})
            flag = "" if missing == 0 else "  <-- incomplete"
            print(f"{var:<9}{year:<7}{len(want):>9}{present:>9}{missing:>9}{flag}")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "year", "expected", "present", "missing"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    print(f"total missing day/month-variables: {sum(r['missing'] for r in rows):,}")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Download PRISM 800 m weather (monthly by default).")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"parallel downloads (default {DEFAULT_WORKERS}; be polite -- shared server)")
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--vars", nargs="+", default=PRISM_VARS, choices=PRISM_VARS)
    ap.add_argument("--temporal", choices=["monthly", "daily"], default=PRISM_TEMPORAL,
                    help="monthly (default, 528 files ~28 GB) or daily (16,072 files ~262 GB)")
    ap.add_argument("--months", nargs="+", type=int, default=None,
                    help="restrict to these months (e.g. 4 5 6 7 8 9 10 11 for Apr-Nov)")
    ap.add_argument("--pilot", action="store_true", help="July 2020 only -- verifies the full path end to end")
    ap.add_argument("--estimate", action="store_true", help="print cost and exit")
    ap.add_argument("--coverage", action="store_true", help="report what is on disk and exit")
    ap.add_argument("--keep-conus", action="store_true", help="keep full CONUS rasters (needs ~262/28 GB)")
    ap.add_argument("--no-clip", action="store_true", help="store raw zips only; defer raster work")
    ap.add_argument("--remask", action="store_true",
                    help="one-time fix for files clipped before state-polygon masking existed: "
                         "sets pixels outside the true state shapes to nodata, IN PLACE, with no "
                         "re-download. Safe to run any time; already-masked files are skipped.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    clip = not args.no_clip
    monthly = args.temporal == "monthly"

    if args.remask:
        geometry = _study_geometry()
        existing = sorted(PRISM_CLIP_DIR.rglob("*.tif")) + sorted(PRISM_MONTHLY_DIR.rglob("*.tif"))
        existing = [p for p in existing if not p.name.endswith(".part.tif")]
        if not existing:
            print("No clipped PRISM rasters found -- nothing to remask.")
            return
        print(f"Remasking {len(existing)} existing raster(s) to the 5 state polygons "
              f"(no re-download; source zips are not needed)...")
        results = run_pool(existing, lambda p: remask_existing(p, geometry), args.workers, "remask")
        summarize(results, "PRISM remask")
        return

    if monthly:
        dates = month_range(args.start_year, args.end_year, args.months)
    else:
        dates = date_range(args.start_year, args.end_year)
        if args.months:
            dates = [d for d in dates if int(d[4:6]) in set(args.months)]
    if args.pilot:
        dates = ["202007"] if monthly else [d for d in dates if d.startswith("202007")]
        print(f"PILOT MODE ({args.temporal}): {len(dates)} period(s) x {len(args.vars)} vars "
              f"= {len(dates) * len(args.vars)} files\n")

    if args.estimate:
        estimate(dates, args.workers, clip, args.temporal, args.vars)
        return
    if args.coverage:
        coverage(dates, clip, args.vars)
        return

    if args.overwrite:
        print("WARNING: --overwrite re-fetches files you already hold. PRISM allows only")
        print("         2 fetches per file per 24h, so this spends that budget and may")
        print("         make those files unavailable for the rest of the day.\n")

    bounds = region_bounds_4326() if clip else None
    geometry = _study_geometry() if clip else None
    jobs = [(v, d) for v in args.vars for d in dates]

    def worker(job):
        var, date_str = job
        zip_path = prism_raw_path(var, date_str)
        tif_path = prism_clip_path(var, date_str)

        if clip and not args.overwrite and verify_tif(tif_path):
            return Result(tif_path, "skipped", tif_path.stat().st_size)

        res = fetch_zip(prism_url(var, date_str), zip_path, overwrite=args.overwrite)
        if res.status in ("failed", "missing"):
            return res
        if not clip:
            return res

        ok, msg = clip_to_region(zip_path, tif_path, bounds, geometry)
        if not ok:
            return Result(tif_path, "failed", 0, msg)
        if not args.keep_conus:
            zip_path.unlink(missing_ok=True)
        return Result(tif_path, res.status, tif_path.stat().st_size)

    print(f"PRISM {PRISM_RES} {args.temporal} | {len(jobs):,} files | {args.workers} workers | "
          f"{'clip to region' if clip else 'raw zips only'}")
    results = run_pool(jobs, worker, args.workers, f"PRISM-{args.temporal}")
    summarize(results, "PRISM download summary")

    if monthly:
        print(f"\nMonthly grids are the final product; they are in {PRISM_MONTHLY_DIR}.")
        print("Next: python scripts/09_preprocess_prism_monthly.py --derive-only")
        print("      (derives vpdmean from vpdmin/vpdmax; ppt and tmean need nothing)")
    else:
        print("\nNext: python scripts/09_preprocess_prism_monthly.py --estimate")


if __name__ == "__main__":
    main()
