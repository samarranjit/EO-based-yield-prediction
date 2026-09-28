#!/usr/bin/env python
# ============================================================
# POLARIS: depth-average raw tiles into a 0-30 cm summary, in LINEAR units
#
# Run from data_preparation/:
#   python scripts/08_preprocess_polaris_depth_summary.py --estimate
#   python scripts/08_preprocess_polaris_depth_summary.py --pilot
#   python scripts/08_preprocess_polaris_depth_summary.py --workers 8
#   python scripts/08_preprocess_polaris_depth_summary.py --validate   # unit QC
#
# This script only touches tiles script 07 already downloaded -- it discovers
# which tiles exist by listing data/soil/polaris/raw/, so it needs no
# geopandas/shapely and cannot drift from what was actually fetched.
#
# THE TWO THINGS THIS SCRIPT EXISTS TO GET RIGHT
# ----------------------------------------------
# 1. LOG10 BACK-TRANSFORM BEFORE AVERAGING.
#    The POLARIS Readme states: "The variables hb, alpha, ksat, om are in log10
#    space." Of the default property set, only `om` is affected.
#
#    Averaging log10 values and exponentiating afterwards computes a GEOMETRIC
#    mean; averaging after back-transforming computes the ARITHMETIC mean that
#    "mean organic matter over 0-30 cm" actually denotes. For a soil profile with
#    om = 3%, 1.5%, 0.8% by layer the two differ by ~8%, and the error is always
#    biased low. So: 10**x FIRST, then weight, then store linear.
#
# 2. THICKNESS WEIGHTING, RENORMALISED PER PIXEL.
#    0-5, 5-15 and 15-30 cm are 5, 10 and 15 cm thick. A plain mean of the three
#    layers over-weights the thin surface layer 3x. Weights are thickness/total,
#    renormalised over VALID layers only where one layer is nodata at a pixel.
#
# Raw tiles are never modified, so this is re-runnable after any fix without
# re-downloading.
# ============================================================

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QC_DIR = DATA_DIR / "qc"

POLARIS_STAT = "mean"
POLARIS_ALL_DEPTHS = {
    "0_5": 5.0, "5_15": 10.0, "15_30": 15.0, "30_60": 30.0, "60_100": 40.0, "100_200": 100.0,
}
POLARIS_DEFAULT_DEPTHS = ["0_5", "5_15", "15_30"]
POLARIS_LOG10_PROPS = {"hb", "alpha", "ksat", "om"}
POLARIS_UNITS = {
    "clay": "%", "sand": "%", "silt": "%", "bd": "g/cm3", "om": "%",
    "theta_s": "m3/m3", "theta_r": "m3/m3", "ksat": "cm/hr", "ph": "pH",
    "hb": "kPa", "alpha": "kPa-1", "lambda": "unitless", "n": "unitless",
}
POLARIS_BYTES_PER_TILE = 34_000_000
DEFAULT_WORKERS = 8
NODATA_OUT = -9999.0

POLARIS_DIR = DATA_DIR / "soil" / "polaris"
POLARIS_RAW_DIR = POLARIS_DIR / "raw"
POLARIS_SUMMARY_DIR = POLARIS_DIR / "weighted_average"

# Physically plausible ranges for the SUMMARY product, in linear units -- a unit
# tripwire, not a scientific filter. `om`'s ceiling is 100 (not ~60) because
# organic soils (peat/histosols) genuinely reach 80-100% and Minnesota has
# extensive peatlands in-footprint (measured 0.007% of pixels on the pilot tiles
# exceed 60%, and they are real). The diagnostic that actually catches a missed
# log10 back-transform is the MEAN: ~3% is correct Corn Belt topsoil, ~0.5 would
# mean the transform did not run.
PLAUSIBLE = {
    "clay": (0.0, 100.0, "%"), "sand": (0.0, 100.0, "%"), "silt": (0.0, 100.0, "%"),
    "bd": (0.3, 2.5, "g/cm3"), "om": (0.01, 100.0, "%"),
    "theta_s": (0.1, 0.8, "m3/m3"), "theta_r": (0.0, 0.4, "m3/m3"),
    "ksat": (0.0, 5000.0, "cm/hr"), "ph": (2.0, 11.0, "pH"),
}


def polaris_tile_name(lat: int, lon: int) -> str:
    return f"lat{lat}{lat + 1}_lon{lon}{lon + 1}.tif"


def polaris_raw_path(prop: str, depth: str, lat: int, lon: int, stat: str = POLARIS_STAT) -> Path:
    return POLARIS_RAW_DIR / prop / stat / depth / polaris_tile_name(lat, lon)


def polaris_summary_path(prop: str, lat: int, lon: int) -> Path:
    return POLARIS_SUMMARY_DIR / prop / polaris_tile_name(lat, lon)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def verify_tif(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as fh:
        return fh.read(4) in (b"II*\x00", b"MM\x00*")


def _parse_tile_name(name: str) -> tuple[int, int] | None:
    """'lat3637_lon-90-89.tif' -> (36, -90). Returns None if unparseable."""
    stem = name[:-4] if name.endswith(".tif") else name
    try:
        lat_part, lon_part = stem.split("_lon")
        lat = int(lat_part[3:5])
        # lon_part is like "-90-89" or "7576" -- split at the second sign boundary
        # by re-deriving from the first integer's expected width.
        if lon_part.startswith("-"):
            # "-90-89": first number is "-90", rest is "-89"
            idx = lon_part.index("-", 1)
            lon = int(lon_part[:idx])
        else:
            lon = int(lon_part[: len(lon_part) // 2])
        return lat, lon
    except (ValueError, IndexError):
        return None


def discovered_tiles(props: list[str], depths: list[str]) -> list[tuple[int, int]]:
    """Which (lat, lon) tiles are actually on disk for the FIRST prop/depth pair.

    Discovering from disk (what script 07 fetched) rather than recomputing state
    geometry means this script has no geopandas/shapely dependency and cannot
    silently drift from what was actually downloaded.
    """
    d = POLARIS_RAW_DIR / props[0] / POLARIS_STAT / depths[0]
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.tif")):
        parsed = _parse_tile_name(f.name)
        if parsed:
            out.append(parsed)
    return out


@dataclass
class Result:
    path: Path
    status: str
    n_bytes: int = 0
    message: str = ""


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
        print(f"\n  {len(failed)} FAILED:")
        for r in failed[:15]:
            print(f"    {r.path.name}: {r.message}")


# --------------------------------------------------------------------------- #

def summarize_tile(prop: str, lat: int, lon: int, depths: dict[str, float],
                   stat: str, overwrite: bool) -> Result:
    """Write the thickness-weighted, linear-unit depth summary for one tile."""
    import rasterio

    out_path = polaris_summary_path(prop, lat, lon)
    if not overwrite and verify_tif(out_path):
        return Result(out_path, "skipped", out_path.stat().st_size)

    is_log = prop in POLARIS_LOG10_PROPS
    acc = wsum = profile = None
    used: list[str] = []

    for depth, thickness in depths.items():
        src_path = polaris_raw_path(prop, depth, lat, lon, stat)
        if not verify_tif(src_path):
            continue
        with rasterio.open(src_path) as src:
            arr = src.read(1).astype("float64")
            nod = src.nodata
            profile = profile or src.profile

        valid = np.isfinite(arr)
        if nod is not None:
            valid &= arr != nod

        if is_log:
            # Back-transform BEFORE weighting -- see header note 1. Clip first: a
            # stray sentinel like -9999 in log10 space would underflow to 0 and be
            # indistinguishable from real data; a large positive would overflow.
            safe = np.where(valid, np.clip(arr, -10.0, 10.0), 0.0)
            vals = np.power(10.0, safe)
        else:
            vals = np.where(valid, arr, 0.0)

        w = np.where(valid, thickness, 0.0)
        acc = (vals * w) if acc is None else acc + vals * w
        wsum = w if wsum is None else wsum + w
        used.append(depth)

    if acc is None or profile is None:
        return Result(out_path, "missing", 0, f"no raw layers for {prop} {lat},{lon}")

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), NODATA_OUT)
    out = out.astype("float32")

    profile.update(dtype="float32", count=1, nodata=NODATA_OUT, compress="deflate",
                   predictor=3, tiled=True, blockxsize=256, blockysize=256, driver="GTiff")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(".part.tif")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(out, 1)
        # Band description is what the model side validates against, exactly as
        # HLS band order is validated from metadata rather than filenames.
        dst.set_band_description(1, f"polaris_{prop}_0_30cm_{POLARIS_UNITS.get(prop, '?')}")
        dst.update_tags(
            polaris_property=prop, polaris_statistic=stat, polaris_depths_used=",".join(used),
            depth_weights_cm=",".join(f"{depths[d]:g}" for d in used),
            units=POLARIS_UNITS.get(prop, "unknown"), log10_source="true" if is_log else "false",
            transform_applied="10**x then thickness-weighted mean" if is_log else "thickness-weighted mean",
            weighting="thickness, renormalised per pixel over valid layers",
        )
    part.replace(out_path)
    return Result(out_path, "downloaded", out_path.stat().st_size)


def validate(props: list[str]) -> None:
    """Sample the summary tiles and check values land in physical ranges."""
    import rasterio

    QC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = QC_DIR / "polaris_summary_validation.csv"
    rows, problems = [], 0

    print(f"{'prop':<9}{'tiles':>6}{'min':>10}{'mean':>10}{'max':>10}{'units':>9}  status")
    print("-" * 66)
    for prop in props:
        tiles = sorted(Path(POLARIS_SUMMARY_DIR / prop).glob("*.tif"))
        if not tiles:
            print(f"{prop:<9}{'0':>6}  no summary tiles found")
            continue
        mins, maxs, means, n = [], [], [], 0
        for t in tiles[:: max(1, len(tiles) // 12)][:12]:
            with rasterio.open(t) as src:
                a = src.read(1)
                v = a[np.isfinite(a) & (a != NODATA_OUT)]
            if v.size:
                mins.append(float(v.min())); maxs.append(float(v.max()))
                means.append(float(v.mean())); n += 1
        if not n:
            print(f"{prop:<9}{len(tiles):>6}  all sampled tiles are empty/nodata")
            continue

        lo, hi, units = PLAUSIBLE.get(prop, (-np.inf, np.inf, "?"))
        vmin, vmax, vmean = min(mins), max(maxs), float(np.mean(means))
        ok = lo <= vmin and vmax <= hi
        status = "OK" if ok else "*** OUT OF RANGE ***"
        problems += 0 if ok else 1
        print(f"{prop:<9}{len(tiles):>6}{vmin:>10.3f}{vmean:>10.3f}{vmax:>10.3f}{units:>9}  {status}")
        rows.append({"property": prop, "n_tiles": len(tiles), "n_sampled": n, "min": vmin,
                     "mean": vmean, "max": vmax, "units": units, "expect_min": lo, "expect_max": hi,
                     "in_range": ok, "log10_source": prop in POLARIS_LOG10_PROPS})

    with open(out_csv, "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {out_csv}")
    if problems:
        print(f"\n{problems} property(ies) out of physical range.")
        print("For a log10 property (om) a value ~10x too small means the")
        print("back-transform did not run. Check POLARIS_LOG10_PROPS.")
    else:
        print("\nAll sampled properties within physical ranges.")


def build_vrt(props: list[str]) -> None:
    """One VRT per property, gluing its summary tiles into a single mosaic view."""
    from osgeo import gdal

    gdal.UseExceptions()
    for prop in props:
        tiles = [str(p) for p in sorted((POLARIS_SUMMARY_DIR / prop).glob("*.tif"))]
        if not tiles:
            continue
        vrt = POLARIS_SUMMARY_DIR / f"{prop}_0_30cm.vrt"
        gdal.BuildVRT(str(vrt), tiles, VRTNodata=NODATA_OUT)
        print(f"  {vrt.name}  ({len(tiles)} tiles)")


def main() -> None:
    ap = argparse.ArgumentParser(description="POLARIS depth summary, log-aware.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--props", nargs="+", default=None,
                    help="default: whatever script 07 downloaded, auto-detected from disk")
    ap.add_argument("--depths", nargs="+", default=POLARIS_DEFAULT_DEPTHS, choices=list(POLARIS_ALL_DEPTHS))
    ap.add_argument("--stat", default=POLARIS_STAT)
    ap.add_argument("--pilot", action="store_true", help="2 tiles only")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--validate", action="store_true", help="physical-range QC and exit")
    ap.add_argument("--vrt", action="store_true", help="also build a per-property VRT mosaic")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    props = args.props
    if props is None:
        if not POLARIS_RAW_DIR.exists():
            print(f"No raw tiles found under {POLARIS_RAW_DIR} -- run script 07 first.")
            return
        props = sorted(p.name for p in POLARIS_RAW_DIR.iterdir() if p.is_dir())
        print(f"--props not given: using what's on disk: {', '.join(props)}")

    depths = {d: POLARIS_ALL_DEPTHS[d] for d in args.depths}
    total_cm = sum(depths.values())
    tiles = discovered_tiles(props, args.depths)
    if not tiles:
        print(f"No downloaded tiles found for {props[0]}/{args.stat}/{args.depths[0]} -- run script 07 first.")
        return
    if args.pilot:
        mid = len(tiles) // 2
        tiles = tiles[mid:mid + 2]

    if args.validate:
        validate(props)
        return

    if args.estimate:
        n_out = len(props) * len(tiles)
        print("=" * 66)
        print("POLARIS DEPTH-SUMMARY ESTIMATE")
        print("=" * 66)
        print(f"  properties   {', '.join(props)}")
        print(f"  depths       {', '.join(args.depths)}  (total {total_cm:g} cm)")
        print("  weights      " + ", ".join(f"{d}={depths[d] / total_cm:.3f}" for d in args.depths))
        print(f"  tiles        {len(tiles)} (discovered from disk)")
        print(f"  reads        {len(props) * len(args.depths) * len(tiles):,} raw tiles")
        print(f"  writes       {n_out:,} summary tiles")
        print(f"  output size  ~{human_bytes(n_out * POLARIS_BYTES_PER_TILE)}")
        log_props = [p for p in props if p in POLARIS_LOG10_PROPS]
        print(f"  log10 props  {', '.join(log_props) if log_props else '(none)'}")
        for p in log_props:
            print(f"                 {p}: 10**x applied BEFORE weighting -> {POLARIS_UNITS.get(p)}")
        print("=" * 66)
        return

    print(f"POLARIS summary | {len(props)} props x {len(tiles)} tiles | "
          f"depths {'+'.join(args.depths)} = {total_cm:g} cm | {args.workers} workers")
    log_props = [p for p in props if p in POLARIS_LOG10_PROPS]
    if log_props:
        print(f"  log10 back-transform will be applied to: {', '.join(log_props)}")

    jobs = [(p, lat, lon) for p in props for (lat, lon) in tiles]

    def worker(job):
        prop, lat, lon = job
        return summarize_tile(prop, lat, lon, depths, args.stat, args.overwrite)

    results = run_pool(jobs, worker, args.workers, "depth-avg")
    summarize(results, "POLARIS depth-summary")

    if args.vrt:
        print("\nbuilding VRT mosaics:")
        build_vrt(props)

    print("\nNext: python scripts/08_preprocess_polaris_depth_summary.py --validate")


if __name__ == "__main__":
    main()
