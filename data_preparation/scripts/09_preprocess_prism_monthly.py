#!/usr/bin/env python
# ============================================================
# PRISM: aggregate clipped daily rasters to monthly, and derive vpdmean
#
# Run from data_preparation/:
#   python scripts/09_preprocess_prism_monthly.py --estimate
#   python scripts/09_preprocess_prism_monthly.py --derive-only   # if script 06 used --temporal monthly
#   python scripts/09_preprocess_prism_monthly.py --pilot
#   python scripts/09_preprocess_prism_monthly.py --workers 8
#   python scripts/09_preprocess_prism_monthly.py --coverage
#
# This script only touches rasters script 06 already downloaded and clipped --
# it never re-derives the study footprint, so it needs no geopandas/shapely.
#
# --derive-only IS THE NORMAL PATH
# ---------------------------------
# script 06's default (--temporal monthly) downloads PRISM's own monthly grids
# directly for ppt/tmean/vpdmin/vpdmax -- those need no aggregation at all. The
# ONLY thing left to compute is vpdmean, which PRISM does not publish. Use
# --derive-only for that case (the common one).
#
# The full daily->monthly AGGREGATION path below (aggregate()) is only needed if
# script 06 ran with --temporal daily, and every variable is per-var:
#   ppt      SUM   mm/day -> mm/month. A mean would report a daily rate labelled
#                  as a monthly total -- ~30x wrong and silent.
#   tmean    MEAN  degC
#   vpdmean  MEAN  hPa, derived per DAY as (vpdmin + vpdmax) / 2
#
# WHY vpdmean IS DERIVED PER-DAY RATHER THAN FROM MONTHLY MEANS
# ---------------------------------------------------------------
# For a plain arithmetic mean the two orders are algebraically identical:
#   mean_t[(min_t + max_t)/2] == (mean_t[min_t] + mean_t[max_t])/2
# so it costs nothing today (confirmed exact to 0.0 on the pilot data, both from
# daily and from monthly inputs). It is still done explicitly rather than
# assumed, because the identity breaks the moment anyone switches to a
# non-linear aggregation (a threshold, a percentile, a stress-day count).
#
# PARTIAL MONTHS ARE REPORTED, NOT SILENTLY AVERAGED (daily path only)
# -----------------------------------------------------------------------
# Each output records n_days_used / n_days_expected in its tags, and
# --require-complete (default on) refuses to write a month missing any day --
# a ppt total from 27 of 31 days is biased low by ~13% and looks perfectly
# normal on a map.
# ============================================================

from __future__ import annotations

import argparse
import calendar
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QC_DIR = DATA_DIR / "qc"

START_YEAR = 2014
END_YEAR = 2024
PRISM_RES = "800m"
PRISM_DERIVED = {"vpdmean": (("vpdmin", "vpdmax"), "mean")}
PRISM_MONTHLY_AGG = {"ppt": "sum", "tmean": "mean", "vpdmean": "mean"}
DEFAULT_WORKERS = 8
NODATA_OUT = -9999.0

PRISM_DIR = DATA_DIR / "prism"
PRISM_CLIP_DIR = PRISM_DIR / "daily_clipped"
PRISM_MONTHLY_DIR = PRISM_DIR / "monthly"


def prism_clip_path(var: str, date_str: str) -> Path:
    if len(date_str) == 6:
        return PRISM_MONTHLY_DIR / var / f"prism_{var}_{date_str}.tif"
    return PRISM_CLIP_DIR / var / date_str[:4] / f"prism_{var}_{date_str}.tif"


def prism_monthly_path(var: str, year: int, month: int) -> Path:
    return PRISM_MONTHLY_DIR / var / f"prism_{var}_{year}{month:02d}.tif"


def verify_tif(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as fh:
        return fh.read(4) in (b"II*\x00", b"MM\x00*")


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


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
    print(f"  bytes written this run: {human_bytes(sum(r.n_bytes for r in results))}")
    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"\n  {len(failed)} FAILED:")
        for r in failed[:15]:
            print(f"    {r.path.name}: {r.message}")


# --------------------------------------------------------------------------- #
# --derive-only: build vpdmean from grids that are ALREADY monthly
# --------------------------------------------------------------------------- #

def derive_from_monthly(var: str, year: int, month: int, overwrite: bool) -> Result:
    import rasterio

    out_path = prism_monthly_path(var, year, month)
    if not overwrite and verify_tif(out_path):
        return Result(out_path, "skipped", out_path.stat().st_size)
    if var not in PRISM_DERIVED:
        return Result(out_path, "failed", 0, f"{var} is not a derived variable")

    inputs, how = PRISM_DERIVED[var]
    if how != "mean":
        return Result(out_path, "failed", 0, f"unsupported derivation {how!r}")

    arrays, profile = [], None
    for src_var in inputs:
        p = prism_monthly_path(src_var, year, month)
        if not verify_tif(p):
            return Result(out_path, "missing", 0, f"no monthly {src_var} for {year}-{month:02d}")
        with rasterio.open(p) as src:
            a = src.read(1).astype("float64")
            nod = src.nodata
            profile = profile or src.profile
        arrays.append(np.where(np.isfinite(a) & ((nod is None) | (a != nod)), a, np.nan))

    if arrays[0].shape != arrays[1].shape:
        return Result(out_path, "failed", 0, "input grids differ in shape")

    stack = np.stack(arrays)
    allnan = np.all(np.isnan(stack), axis=0)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(stack, axis=0)
    out = np.nan_to_num(np.where(allnan, NODATA_OUT, out), nan=NODATA_OUT).astype("float32")

    profile.update(dtype="float32", count=1, nodata=NODATA_OUT, compress="deflate",
                   predictor=3, tiled=True, blockxsize=256, blockysize=256, driver="GTiff")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(".part.tif")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, f"prism_{var}_{year}{month:02d}_hPa")
        dst.update_tags(
            prism_variable=var, prism_resolution=PRISM_RES, aggregation="mean", units="hPa",
            year=str(year), month=f"{month:02d}", derived_from=",".join(inputs),
            derivation="(vpdmin + vpdmax) / 2 on PRISM monthly grids",
            note="exact: equals the per-day derivation averaged over the month",
            source_temporal="monthly",
        )
    part.replace(out_path)
    return Result(out_path, "downloaded", out_path.stat().st_size)


# --------------------------------------------------------------------------- #
# Daily -> monthly aggregation (only if script 06 used --temporal daily)
# --------------------------------------------------------------------------- #

def _daily_paths(var: str, year: int, month: int) -> list[Path]:
    n_days = calendar.monthrange(year, month)[1]
    return [prism_clip_path(var, f"{year}{month:02d}{d:02d}") for d in range(1, n_days + 1)]


def _read_stack(paths: list[Path]):
    import rasterio

    arrays, profile = [], None
    for p in paths:
        if not verify_tif(p):
            continue
        with rasterio.open(p) as src:
            a = src.read(1).astype("float64")
            nod = src.nodata
            profile = profile or src.profile
        arrays.append(np.where(np.isfinite(a) & ((nod is None) | (a != nod)), a, np.nan))
    if not arrays:
        return None, None, 0
    return np.stack(arrays), profile, len(arrays)


def aggregate(var: str, year: int, month: int, require_complete: bool, overwrite: bool) -> Result:
    import rasterio

    out_path = prism_monthly_path(var, year, month)
    if not overwrite and verify_tif(out_path):
        return Result(out_path, "skipped", out_path.stat().st_size)
    n_expected = calendar.monthrange(year, month)[1]

    if var in PRISM_DERIVED:
        inputs, how = PRISM_DERIVED[var]
        stacks, profile, n_used = [], None, None
        for src_var in inputs:
            s, prof, n = _read_stack(_daily_paths(src_var, year, month))
            if s is None:
                return Result(out_path, "missing", 0, f"no daily {src_var} for {year}-{month:02d}")
            stacks.append(s)
            profile = profile or prof
            if n_used is not None and n != n_used:
                return Result(out_path, "failed", 0, f"{inputs[0]}/{src_var} day counts differ ({n_used} vs {n})")
            n_used = n
        if stacks[0].shape != stacks[1].shape:
            return Result(out_path, "failed", 0, "input grids differ in shape")
        daily = np.nanmean(np.stack(stacks), axis=0)
        agg_kind = PRISM_MONTHLY_AGG[var]
    else:
        daily, profile, n_used = _read_stack(_daily_paths(var, year, month))
        if daily is None:
            return Result(out_path, "missing", 0, f"no daily {var} for {year}-{month:02d}")
        agg_kind = PRISM_MONTHLY_AGG[var]

    if require_complete and n_used != n_expected:
        return Result(out_path, "failed", 0,
                       f"incomplete month: {n_used}/{n_expected} days (use --allow-partial)")

    with np.errstate(invalid="ignore"):
        allnan = np.all(np.isnan(daily), axis=0)
        if agg_kind == "sum":
            out = np.where(allnan, NODATA_OUT, np.nansum(daily, axis=0))
        elif agg_kind == "mean":
            out = np.where(allnan, NODATA_OUT, np.nanmean(daily, axis=0))
        else:
            return Result(out_path, "failed", 0, f"unknown aggregation {agg_kind!r}")

    out = np.nan_to_num(out, nan=NODATA_OUT).astype("float32")
    profile.update(dtype="float32", count=1, nodata=NODATA_OUT, compress="deflate",
                   predictor=3, tiled=True, blockxsize=256, blockysize=256, driver="GTiff")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(".part.tif")
    units = {"ppt": "mm/month", "tmean": "degC", "vpdmean": "hPa"}.get(var, "unknown")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, f"prism_{var}_{year}{month:02d}_{units}")
        dst.update_tags(
            prism_variable=var, prism_resolution=PRISM_RES, aggregation=agg_kind, units=units,
            year=str(year), month=f"{month:02d}", n_days_used=str(n_used),
            n_days_expected=str(n_expected), complete="true" if n_used == n_expected else "false",
            derived_from=",".join(PRISM_DERIVED[var][0]) if var in PRISM_DERIVED else "",
        )
    part.replace(out_path)
    return Result(out_path, "downloaded", out_path.stat().st_size)


# --------------------------------------------------------------------------- #

def coverage(variables: list[str], years: list[int]) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = QC_DIR / "prism_monthly_coverage.csv"
    rows = []
    print(f"{'var':<10}{'year':<7}{'months present':>16}{'missing':>9}")
    print("-" * 42)
    for var in variables:
        for year in years:
            present = sum(1 for m in range(1, 13) if verify_tif(prism_monthly_path(var, year, m)))
            rows.append({"variable": var, "year": year, "months_present": present,
                         "months_missing": 12 - present})
            flag = "" if present == 12 else "  <-- incomplete"
            print(f"{var:<10}{year:<7}{present:>16}{12 - present:>9}{flag}")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "year", "months_present", "months_missing"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="PRISM daily -> monthly, with derived vpdmean.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--vars", nargs="+", default=list(PRISM_MONTHLY_AGG))
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)),
                    help="default all 12; the model currently reads APR-NOV (4..11)")
    ap.add_argument("--pilot", action="store_true", help="2020-07 only")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--derive-only", action="store_true",
                    help="inputs are already monthly (script 06 --temporal monthly, the "
                         "default): only compute derived variables, e.g. vpdmean")
    ap.add_argument("--allow-partial", action="store_true",
                    help="daily path only: write months missing days (recorded in tags)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    months = sorted(args.months)
    if args.pilot:
        years, months = [2020], [7]
        print("PILOT MODE: 2020-07\n")

    if args.coverage:
        coverage(args.vars, years)
        return

    if args.derive_only:
        derived = [v for v in args.vars if v in PRISM_DERIVED]
        passthrough = [v for v in args.vars if v not in PRISM_DERIVED]
        if passthrough:
            print(f"already monthly, nothing to do: {', '.join(passthrough)}")
        if not derived:
            print("no derived variables requested -- nothing to do.")
            return
        jobs = [(v, y, m) for v in derived for y in years for m in months]
        if args.estimate:
            print(f"would derive {len(jobs):,} monthly outputs: {', '.join(derived)}")
            return
        print(f"PRISM derive-only | {', '.join(derived)} | {len(jobs):,} outputs | {args.workers} workers")
        results = run_pool(jobs, lambda j: derive_from_monthly(*j, overwrite=args.overwrite),
                           args.workers, "derive")
        summarize(results, "PRISM derived (from monthly)")
        print("\nNext: python scripts/09_preprocess_prism_monthly.py --coverage")
        return

    jobs = [(v, y, m) for v in args.vars for y in years for m in months]

    if args.estimate:
        print("=" * 66)
        print("PRISM MONTHLY AGGREGATION ESTIMATE (daily -> monthly path)")
        print("=" * 66)
        print(f"  variables    {', '.join(args.vars)}")
        for v in args.vars:
            kind = PRISM_MONTHLY_AGG.get(v, "?")
            src = f"derived from {'+'.join(PRISM_DERIVED[v][0])}" if v in PRISM_DERIVED else "direct"
            print(f"                 {v:<9} {kind:<5} {src}")
        print(f"  years        {years[0]}..{years[-1]}")
        print(f"  months       {months}")
        print(f"  outputs      {len(jobs):,} monthly rasters")
        print(f"  reads        ~{len(jobs) * 30:,} daily rasters (vpdmean reads 2 vars/day)")
        print(f"  complete-month policy: {'allow partial' if args.allow_partial else 'require complete'}")
        print("=" * 66)
        print("  If script 06 ran with --temporal monthly (the default), use")
        print("  --derive-only instead -- ppt/tmean/vpdmin/vpdmax need no aggregation.")
        return

    print(f"PRISM monthly aggregation | {len(jobs):,} outputs | {args.workers} workers")

    def worker(job):
        return aggregate(*job, require_complete=not args.allow_partial, overwrite=args.overwrite)

    results = run_pool(jobs, worker, args.workers, "monthly")
    summarize(results, "PRISM monthly aggregation")
    print("\nNext: python scripts/09_preprocess_prism_monthly.py --coverage")


if __name__ == "__main__":
    main()
