#!/usr/bin/env python
# ============================================================
# Download POLARIS 30 m soil property tiles for the study footprint
#
# POLARIS is STATIC -- no year dimension, one value per location per depth. It is
# downloaded once and reused across all 11 years.
#
# Run from data_preparation/:
#   python scripts/07_download_polaris_soil.py --estimate
#   python scripts/07_download_polaris_soil.py --pilot
#   python scripts/07_download_polaris_soil.py --coverage
#   python scripts/07_download_polaris_soil.py --workers 12
#
#   # configurable properties / depths, per the experiment plan:
#   python scripts/07_download_polaris_soil.py --props clay sand om bd theta_s \
#       --depths 0_5 5_15 15_30 30_60
#
# TILE SELECTION
# --------------
# POLARIS ships 1x1-degree tiles. Tiles are selected by INTERSECTION WITH THE
# STATE GEOMETRY, not by bounding box: the five states span lon -97..-75 because
# MD sits far east of MN, so the bbox is mostly empty. Measured: 336 tiles by
# bbox vs 113 by geometry -- a 3x saving on ~44 GB.
#
# UNITS ARE NOT TOUCHED HERE
# --------------------------
# Tiles are stored exactly as served, including the log10 encoding of om (and of
# ksat/hb/alpha if you add them). Back-transformation happens in script 08, at
# the point of depth averaging, where getting it wrong would actually corrupt a
# number. Keeping the raw archive faithful means script 08 can be re-run after a
# bug fix without re-downloading.
# ============================================================

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QC_DIR = DATA_DIR / "qc"
COUNTIES_GPKG = DATA_DIR / "county_maps" / "selected_states_counties_2023.gpkg"

# Five states the current FARM-US model trains on (cornbelt5). Widen this when
# the model's data.states widens, and re-run --estimate first.
EXOG_STATE_FIPS = {"IA": "19", "IL": "17", "IN": "18", "MD": "24", "MN": "27"}

# --------------------------------------------------------------------------- #
# POLARIS endpoint. Directory listing + Readme fetched 2026-09-26 from
#     http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/
# Tile naming keeps the sign on BOTH longitudes: lat3637_lon-90-89.tif spans
# lat 36..37, lon -90..-89. (lat3637_lon-9089.tif does not exist and 404s.)
# --------------------------------------------------------------------------- #
POLARIS_BASE = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
POLARIS_PROPS = ["clay", "sand", "om", "bd"]
POLARIS_STAT = "mean"
POLARIS_ALL_DEPTHS = {
    "0_5": 5.0, "5_15": 10.0, "15_30": 15.0, "30_60": 30.0, "60_100": 40.0, "100_200": 100.0,
}
POLARIS_DEFAULT_DEPTHS = ["0_5", "5_15", "15_30", "30_60", "60_100", "100_200"]  # 0-200 cm

# From the Readme, verbatim: "hb, alpha, ksat, om are in log10 space." Of the
# default property set, only `om`. Used by script 08, kept here too so a
# download-time check can warn if a requested prop needs the log-aware step.
POLARIS_LOG10_PROPS = {"hb", "alpha", "ksat", "om"}

POLARIS_BYTES_PER_TILE = 34_000_000  # mean of clay/om/bd/sand at 0_5, measured

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0
REQUEST_TIMEOUT = (30, 300)
USER_AGENT = "FARM-US-research/1.0 (yield modelling; contact via repo)"
DEFAULT_WORKERS = 8

POLARIS_DIR = DATA_DIR / "soil" / "polaris"
POLARIS_RAW_DIR = POLARIS_DIR / "raw"


def region_bounds_4326() -> tuple[float, float, float, float]:
    import geopandas as gpd

    if not COUNTIES_GPKG.exists():
        raise FileNotFoundError(f"{COUNTIES_GPKG} not found -- run scripts/02_download_counties.py first")
    gdf = gpd.read_file(COUNTIES_GPKG)
    gdf = gdf[gdf["STATEFP"].isin(EXOG_STATE_FIPS.values())]
    if gdf.empty:
        raise ValueError(f"No counties for STATEFP in {sorted(EXOG_STATE_FIPS.values())} in {COUNTIES_GPKG}")
    return tuple(gdf.to_crs("EPSG:4326").geometry.union_all().bounds)  # type: ignore[return-value]


def polaris_tiles() -> list[tuple[int, int]]:
    """1x1-degree POLARIS tiles that intersect the 5-state geometry (not bbox)."""
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.read_file(COUNTIES_GPKG)
    gdf = gdf[gdf["STATEFP"].isin(EXOG_STATE_FIPS.values())]
    uni = gdf.to_crs("EPSG:4326").geometry.union_all()
    minx, miny, maxx, maxy = uni.bounds
    tiles = []
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            if box(lon, lat, lon + 1, lat + 1).intersects(uni):
                tiles.append((lat, lon))
    return tiles


def polaris_tile_name(lat: int, lon: int) -> str:
    return f"lat{lat}{lat + 1}_lon{lon}{lon + 1}.tif"


def polaris_url(prop: str, depth: str, lat: int, lon: int, stat: str = POLARIS_STAT) -> str:
    return f"{POLARIS_BASE}/{prop}/{stat}/{depth}/{polaris_tile_name(lat, lon)}"


def polaris_raw_path(prop: str, depth: str, lat: int, lon: int, stat: str = POLARIS_STAT) -> Path:
    return POLARIS_RAW_DIR / prop / stat / depth / polaris_tile_name(lat, lon)


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


# --------------------------------------------------------------------------- #
# Download machinery
# --------------------------------------------------------------------------- #

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _local.session = s
    return s


@dataclass
class Result:
    path: Path
    status: str
    n_bytes: int = 0
    message: str = ""


def verify_tif(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as fh:
        return fh.read(4) in (b"II*\x00", b"MM\x00*")


def fetch_tif(url: str, dest: Path, overwrite: bool = False) -> Result:
    if not overwrite and verify_tif(dest):
        return Result(dest, "skipped", dest.stat().st_size)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part.tif")
    last = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _session().get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code == 404:
                    # A genuine upstream hole (open water / outside CONUS), not a failure.
                    return Result(dest, "missing", 0, "HTTP 404")
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()

                n = 0
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                            n += len(chunk)

            if not verify_tif(part):
                raise OSError(f"TIFF magic-number check failed after {n} bytes")
            part.replace(dest)
            return Result(dest, "downloaded", n)

        except Exception as exc:  # noqa: BLE001 - one bad tile must not kill the run
            last = f"{type(exc).__name__}: {exc}"
            part.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                sleep = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(sleep * (0.5 + random.random()))

    return Result(dest, "failed", 0, f"{MAX_RETRIES} attempts; last: {last}")


def content_length(url: str) -> int | None:
    try:
        r = _session().head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    return None


def run_pool(jobs, worker, n_workers: int, desc: str) -> list[Result]:
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
        print(f"\n  {len(failed)} FAILED (re-run to retry):")
        for r in failed[:15]:
            print(f"    {r.path.name}: {r.message}")
        if len(failed) > 15:
            print(f"    ... and {len(failed) - 15} more")


# --------------------------------------------------------------------------- #
# Reporting modes
# --------------------------------------------------------------------------- #

def estimate(props: list[str], depths: list[str], tiles: list[tuple[int, int]],
             n_workers: int, probe: bool) -> None:
    n_files = len(props) * len(depths) * len(tiles)
    per_tile = POLARIS_BYTES_PER_TILE
    if probe:
        print("probing real tile sizes ...")
        sizes = []
        for p in props:
            url = polaris_url(p, depths[0], *tiles[len(tiles) // 2])
            n = content_length(url)
            print(f"  {p:<9} {human_bytes(n) if n else 'unavailable':>12}")
            if n:
                sizes.append(n)
        if sizes:
            per_tile = sum(sizes) / len(sizes)

    total = n_files * per_tile
    minx, miny, maxx, maxy = region_bounds_4326()

    print("=" * 66)
    print("POLARIS DOWNLOAD ESTIMATE")
    print("=" * 66)
    print(f"  endpoint      {POLARIS_BASE}/<prop>/{POLARIS_STAT}/<depth>/<tile>.tif")
    print(f"  properties    {', '.join(props)}")
    print(f"  depths        {', '.join(depths)} cm")
    print(f"  statistic     {POLARIS_STAT}")
    print(f"  footprint     lon {minx:.2f}..{maxx:.2f}  lat {miny:.2f}..{maxy:.2f}")
    print(f"  tiles         {len(tiles)} (geometry-filtered)")
    print(f"  files         {len(props)} props x {len(depths)} depths x {len(tiles)} tiles = {n_files:,}")
    print()
    print(f"  mean tile     {human_bytes(per_tile)}")
    print(f"  raw archive   {human_bytes(total)}")
    print(f"  0-30cm summary (script 08, 1 band per prop): ~{human_bytes(len(props) * len(tiles) * per_tile)}")
    print()
    log_props = [p for p in props if p in POLARIS_LOG10_PROPS]
    print(f"  log10-encoded {', '.join(log_props) if log_props else '(none)'}  -> back-transformed in script 08")
    print()
    for rate in (5, 10):
        print(f"  at {rate:>2} MB/s aggregate: ~{human_time(total / (rate * 1e6))}")
    print(f"  workers       {n_workers}")
    print("=" * 66)

    import shutil
    free = shutil.disk_usage(DATA_DIR).free
    need = total * 1.34
    print(f"  disk free     {human_bytes(free)}")
    print(f"  need (raw + summary) ~{human_bytes(need)}")
    print("  *** INSUFFICIENT DISK ***" if need > free * 0.9 else "")


def coverage(props: list[str], depths: list[str], tiles: list[tuple[int, int]]) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    out = QC_DIR / "polaris_coverage.csv"
    rows = []
    print(f"{'prop':<9}{'depth':<9}{'expected':>9}{'present':>9}{'missing':>9}")
    print("-" * 45)
    for p in props:
        for d in depths:
            present, missing_tiles = 0, []
            for lat, lon in tiles:
                path = polaris_raw_path(p, d, lat, lon)
                if verify_tif(path):
                    present += 1
                else:
                    missing_tiles.append(polaris_tile_name(lat, lon))
            rows.append({"property": p, "depth": d, "expected": len(tiles), "present": present,
                         "missing": len(missing_tiles), "missing_tiles": ";".join(missing_tiles[:40])})
            flag = "" if not missing_tiles else "  <-- incomplete"
            print(f"{p:<9}{d:<9}{len(tiles):>9}{present:>9}{len(missing_tiles):>9}{flag}")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["property", "depth", "expected", "present", "missing", "missing_tiles"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    print(f"total missing: {sum(r['missing'] for r in rows):,}")
    print("\nNote: POLARIS has genuine holes (open water, outside CONUS). A tile that")
    print("404s on every property is absent upstream, not a failed download -- script")
    print("08 treats it as nodata rather than erroring.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download POLARIS 30 m soil tiles.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--props", nargs="+", default=POLARIS_PROPS,
                    help=f"soil properties (default: {' '.join(POLARIS_PROPS)})")
    ap.add_argument("--depths", nargs="+", default=POLARIS_DEFAULT_DEPTHS,
                    choices=list(POLARIS_ALL_DEPTHS), help=f"depth layers cm (default: {' '.join(POLARIS_DEFAULT_DEPTHS)})")
    ap.add_argument("--stat", default=POLARIS_STAT, choices=["mean", "mode", "p5", "p50", "p95"])
    ap.add_argument("--pilot", action="store_true", help="2 tiles x all props/depths")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--probe", action="store_true", help="with --estimate, HEAD real tiles for exact sizes")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tiles = polaris_tiles()
    if args.pilot:
        mid = len(tiles) // 2
        tiles = tiles[mid:mid + 2]
        print(f"PILOT MODE: tiles {[polaris_tile_name(*t) for t in tiles]}\n")

    if args.estimate:
        estimate(args.props, args.depths, tiles, args.workers, args.probe)
        return
    if args.coverage:
        coverage(args.props, args.depths, tiles)
        return

    jobs = [(p, d, lat, lon) for p in args.props for d in args.depths for (lat, lon) in tiles]

    def worker(job):
        prop, depth, lat, lon = job
        return fetch_tif(polaris_url(prop, depth, lat, lon, args.stat),
                         polaris_raw_path(prop, depth, lat, lon, args.stat), overwrite=args.overwrite)

    print(f"POLARIS | {len(jobs):,} tiles | {args.workers} workers | stat={args.stat}")
    results = run_pool(jobs, worker, args.workers, "POLARIS")
    summarize(results, "POLARIS download summary")
    print("\nNext: python scripts/08_preprocess_polaris_depth_summary.py --estimate")


if __name__ == "__main__":
    main()
