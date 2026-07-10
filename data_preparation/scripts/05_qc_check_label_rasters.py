"""
QC check on yield-label rasters produced by script 04.

Reports per-raster statistics (pixel counts, yield distribution, nodata fraction)
and flags rasters with suspiciously few valid pixels or mismatched CRS/resolution.

Writes a summary CSV to data/qc/ and prints a console report.

    python scripts/05_qc_check_label_rasters.py
"""

import sys
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    YIELD_LABEL_DIR, QC_DIR, CROP_NAME,
    STATE_ALPHA, START_YEAR, END_YEAR, NODATA, TARGET_CRS, TARGET_SCALE_M,
)

MIN_VALID_PCT = 0.1   # warn if fewer than 0.1 % of pixels are valid crop pixels


def inspect_raster(path: Path) -> dict:
    with rasterio.open(path) as src:
        arr     = src.read(1).astype("float32")
        profile = src.profile
        crs_str = str(src.crs)
        res_m   = abs(profile["transform"].a)

    valid = arr[arr != NODATA]
    total = arr.size
    n_valid = len(valid)

    return {
        "file":      path.name,
        "total_px":  total,
        "valid_px":  n_valid,
        "nodata_px": total - n_valid,
        "pct_valid": round(100.0 * n_valid / total, 4),
        "min":       round(float(valid.min()),  3) if n_valid else np.nan,
        "p25":       round(float(np.percentile(valid, 25)), 3) if n_valid else np.nan,
        "median":    round(float(np.median(valid)), 3) if n_valid else np.nan,
        "p75":       round(float(np.percentile(valid, 75)), 3) if n_valid else np.nan,
        "max":       round(float(valid.max()),  3) if n_valid else np.nan,
        "mean":      round(float(valid.mean()), 3) if n_valid else np.nan,
        "std":       round(float(valid.std()),  3) if n_valid else np.nan,
        "crs":       crs_str,
        "res_m":     res_m,
        "crs_ok":    crs_str == TARGET_CRS,
        "res_ok":    abs(res_m - TARGET_SCALE_M) < 1,
    }


def main():
    crop = CROP_NAME.lower()
    QC_DIR.mkdir(parents=True, exist_ok=True)

    variant_dirs = {
        "nearest":     YIELD_LABEL_DIR / "county_values",
        "bilinear10km": YIELD_LABEL_DIR / "bilinear",
    }

    raster_paths = []
    for variant, base_dir in variant_dirs.items():
        for year in range(START_YEAR, END_YEAR + 1):
            year_dir = base_dir / str(year)
            if not year_dir.exists():
                continue
            for state in STATE_ALPHA:
                fname = f"nass_{crop}_yield_{state}_{year}_{variant}_30m_{crop}_only.tif"
                fpath = year_dir / fname
                if fpath.exists():
                    raster_paths.append((state, year, variant, fpath))

    if not raster_paths:
        print("No label rasters found. Run script 04 first.")
        return

    print(f"Inspecting {len(raster_paths)} rasters…")
    records = []
    for state, year, variant, fpath in tqdm(raster_paths):
        info = inspect_raster(fpath)
        info.update({"state": state, "year": year, "variant": variant})
        records.append(info)

    df = pd.DataFrame(records)
    out_csv = QC_DIR / f"qc_yield_labels_{crop}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nQC report saved → {out_csv}")

    # --- Summary table ---
    print("\n=== Mean statistics by (year, variant) ===")
    summary = (
        df.groupby(["year", "variant"])[["valid_px", "mean", "std", "pct_valid"]]
        .mean()
        .round(2)
    )
    print(summary.to_string())

    # --- Issue flags ---
    issues = []

    low_valid = df[df["pct_valid"] < MIN_VALID_PCT]
    if not low_valid.empty:
        issues.append(
            f"  WARNING: {len(low_valid)} rasters have < {MIN_VALID_PCT}% valid pixels:\n"
            + low_valid[["file", "pct_valid"]].to_string(index=False)
        )

    empty = df[df["valid_px"] == 0]
    if not empty.empty:
        issues.append(
            f"  ERROR:   {len(empty)} rasters are entirely empty:\n"
            + empty[["file"]].to_string(index=False)
        )

    bad_crs = df[~df["crs_ok"]]
    if not bad_crs.empty:
        issues.append(
            f"  ERROR:   {len(bad_crs)} rasters have unexpected CRS:\n"
            + bad_crs[["file", "crs"]].to_string(index=False)
        )

    bad_res = df[~df["res_ok"]]
    if not bad_res.empty:
        issues.append(
            f"  ERROR:   {len(bad_res)} rasters have unexpected resolution:\n"
            + bad_res[["file", "res_m"]].to_string(index=False)
        )

    if issues:
        print("\n=== Issues ===")
        for msg in issues:
            print(msg)
    else:
        print("\nAll rasters passed QC checks.")

    # --- Missing coverage ---
    expected = {
        (s, y, v)
        for s in STATE_ALPHA
        for y in range(START_YEAR, END_YEAR + 1)
        for v in ("nearest", "bilinear10km")
    }
    found = {(r["state"], r["year"], r["variant"]) for r in records}
    missing = sorted(expected - found)
    if missing:
        print(f"\n=== Missing rasters ({len(missing)}) ===")
        for state, year, variant in missing[:20]:
            print(f"  {state} {year} {variant}")
        if len(missing) > 20:
            print(f"  … and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
