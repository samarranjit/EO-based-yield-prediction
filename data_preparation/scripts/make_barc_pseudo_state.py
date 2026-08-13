#!/usr/bin/env python
"""Build a BARC 'pseudo-state' so the existing pipeline can ingest it unchanged.

Why this exists
---------------
`download_hls_monthly_composites.py` and `model/`'s GeotiffMonthlyReader both key
everything off a `{state}` token and two file-naming patterns. Nothing validates
the token against a real state list, so BARC can ride the *unmodified* scripts as
long as the two files exist with the right names on the right grid. That keeps the
downloader reusable for real states -- no branching, no BARC special case.

Two problems block a plain copy/rename of the BARC yield rasters:

1. GRID. The BARC rasters are 435x251. Both the downloader's
   `compute_qualifying_windows` and the model's chip tiler emit only *whole*
   224x224 chips, so 435x251 yields exactly ONE chip covering the NW corner and
   silently discards the rest of the farm. Worse, BARC's origin sits at
   dx=7526, dy=3074 px from the national grid origin -- 7526/224 = 33.6, not
   integral -- so its chips would not line up with the state chip grid either.
   Both are fixed by snapping outward to the nearest 224-chip boundary, giving a
   672x448 grid = 3x2 = 6 chips that covers all of BARC and coincides with the
   national chip layout.

2. PROVENANCE. Value-wise a rename would work: every consumer thresholds the CDL
   with `> 0.5` (masks.crop_mask_from_cdl default, raster_readers.py), BARC's
   nodata is -9999, and the minimum real yield across all 11 years is 5.73 bu/ac
   -- zero pixels fall in the 0 < v <= 0.5 gap. But a file named
   `cdl_soybeans_BARC_2020.tif` that actually contains yield values is precisely
   what a leakage review flags. The mask never enters the loss (labels come from
   label_root), yet "the CDL file contains the labels" is not a sentence worth
   having to defend. We write a clean {0, 1} mask instead; since the grid forces
   a rewrite anyway, this costs nothing.

Why the mask comes from the yield map, not from CDL
---------------------------------------------------
Deriving the crop mask from Maryland's CDL was measured and rejected. CDL agrees
with BARC's measured-yield footprint on only 11-79% of pixels depending on year
(2016: 209 of 1,944). Since `valid = crop_mask AND label_valid AND hls_valid`, a
CDL-derived mask would discard most of the measured labels -- the entire point of
the BARC experiment. The cause is expected rather than a bug: CDL is a 30 m
classification with real error, and BARC is small research plots, exactly where
it is weakest. BARC's yield monitor is stronger evidence that soybean was grown
in a pixel than CDL is, so the yield footprint IS the crop mask.

Consequence to be aware of: crop_mask then equals the label footprint, so
`valid` reduces to `label_valid AND hls_valid`. That is correct here (CDL adds no
information we trust at BARC) but it is a deliberate deviation from the
training-time mask definition and belongs in model/docs/DECISION_LOG.md.

Outputs (nothing existing is modified)
--------------------------------------
    {out_root}/cdl_masks/cdl_soybeans_BARC_{YEAR}.tif
        float32 {0, 1}, nodata=None -- byte-for-byte the same convention as the
        state CDLs, so no downstream code path behaves differently.

    {out_root}/yield_labels/{YEAR}/nass_soybeans_yield_BARC_{YEAR}_{variant}_30m_soybeans_only.tif
        float32 yield in bu/ac, nodata=-9999. The `nass_` prefix is a cosmetic
        artifact of GeotiffMonthlyReader._label_path -- these are BARC measured
        yields, NOT NASS county statistics. Named this way solely so the reader
        finds them without modification.

Then run the downloader completely unmodified:

    python scripts/download_hls_monthly_composites.py \
        --states BARC --years 2014 ... 2024 \
        --months APR MAY JUN JUL AUG SEP OCT NOV \
        --cdl-dir data/barc_data/cdl_masks \
        --out-dir data/barc_data/HLS_Composites \
        --min-crop-fraction 0.0

`--min-crop-fraction 0.0` is mandatory, not cosmetic: BARC's soybean footprint is
under 3% of any 224x224 chip, so the 0.05 default rejects all six chips and the
downloader reports success having fetched nothing. This script prints the
per-chip fractions so that threshold is a measured choice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

CHIP = 224
RES = 30.0
YIELD_NODATA = -9999.0

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YIELD_DIR = REPO / "data/barc_data/yield_dataset"
DEFAULT_REF_CDL = REPO / "data/cdl_masks/cdl_soybeans_MD_{year}.tif"
DEFAULT_OUT_ROOT = REPO / "data/barc_data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=list(range(2014, 2025)))
    p.add_argument("--yield-dir", type=Path, default=DEFAULT_YIELD_DIR)
    p.add_argument("--ref-cdl", type=str, default=str(DEFAULT_REF_CDL),
                   help="Template for the state CDL defining the national grid; {year} is substituted.")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--state-token", default="BARC", help="Pseudo-state name used in every output filename.")
    p.add_argument("--crop", default="soybeans")
    p.add_argument("--label-variant", default="measured",
                   help="Must match data.label_variant in the model config.")
    p.add_argument("--dry-run", action="store_true", help="Report the plan and write nothing.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def national_grid_origin(ref_cdl: Path) -> tuple[float, float]:
    """Origin of the shared 30 m national grid, read from a real state CDL."""
    with rasterio.open(ref_cdl) as ds:
        t = ds.transform
        if abs(t.a - RES) > 1e-6 or abs(t.e + RES) > 1e-6:
            raise SystemExit(f"{ref_cdl}: expected {RES} m pixels, got {t.a}/{t.e}")
        return t.c, t.f


def target_grid(yield_paths: list[Path], ox: float, oy: float) -> tuple[float, float, int, int]:
    """Union of every year's BARC extent, snapped OUTWARD to 224-chip boundaries.

    Snapping to whole chips (not merely whole pixels) is what makes BARC's chips
    coincide with the state chip grid, so a BARC chip and a Maryland chip covering
    the same ground are the same chip.
    """
    lefts, rights, tops, bottoms = [], [], [], []
    for p in yield_paths:
        with rasterio.open(p) as ds:
            b = ds.bounds
        lefts.append(b.left); rights.append(b.right)
        tops.append(b.top); bottoms.append(b.bottom)
    left, right, top, bottom = min(lefts), max(rights), max(tops), min(bottoms)

    # Floor/ceil in units of whole chips relative to the national origin.
    c0 = int(np.floor((left - ox) / RES / CHIP)) * CHIP
    r0 = int(np.floor((oy - top) / RES / CHIP)) * CHIP
    x0, y0 = ox + c0 * RES, oy - r0 * RES
    ncols = int(np.ceil((right - x0) / RES / CHIP)) * CHIP
    nrows = int(np.ceil((y0 - bottom) / RES / CHIP)) * CHIP
    return x0, y0, ncols, nrows


def read_onto(path: Path, x0: float, y0: float, ncols: int, nrows: int, fill: float) -> np.ndarray:
    """Windowed read of `path` onto the target grid, asserting whole-pixel alignment.

    A fractional offset would mean the source and target grids disagree, and a
    boundless read would silently skew the data by a sub-pixel shift rather than
    fail. We refuse instead -- same contract as raster_readers._aligned_window.
    """
    with rasterio.open(path) as ds:
        x1, y1 = x0 + ncols * RES, y0 - nrows * RES
        w = from_bounds(x0, y1, x1, y0, ds.transform)
        for name, v in (("col_off", w.col_off), ("row_off", w.row_off)):
            if abs(v - round(v)) > 1e-6:
                raise SystemExit(f"{path}: fractional {name}={v} -- grids are not pixel-aligned")
        w = Window(round(w.col_off), round(w.row_off), ncols, nrows)
        return ds.read(1, window=w, boundless=True, fill_value=fill).astype(np.float32)


def write_raster(path: Path, arr: np.ndarray, x0: float, y0: float, crs, nodata: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="float32", crs=crs, transform=rasterio.transform.from_origin(x0, y0, RES, RES),
        nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def main() -> int:
    args = parse_args()
    crop, token = args.crop, args.state_token

    paths = {}
    for yr in args.years:
        p = args.yield_dir / f"barc_{crop}_yield_{yr}_30m.tif"
        if not p.exists():
            print(f"WARNING: missing {p} -- skipping {yr}")
            continue
        paths[yr] = p
    if not paths:
        raise SystemExit(f"No BARC yield rasters found under {args.yield_dir}")

    ref = Path(args.ref_cdl.format(year=sorted(paths)[0]))
    if not ref.exists():
        raise SystemExit(f"Reference CDL not found: {ref}")
    ox, oy = national_grid_origin(ref)
    with rasterio.open(ref) as ds:
        crs = ds.crs
    x0, y0, ncols, nrows = target_grid(list(paths.values()), ox, oy)

    print(f"national grid origin : ({ox:.0f}, {oy:.0f})  from {ref.name}")
    print(f"BARC target grid     : ({x0:.0f}, {y0:.0f})  {ncols} x {nrows}  "
          f"= {ncols // CHIP} x {nrows // CHIP} = {ncols // CHIP * (nrows // CHIP)} chips")
    print(f"offset from national : col={(x0 - ox) / RES:.0f} row={(oy - y0) / RES:.0f} px "
          f"(both must be multiples of {CHIP})")
    print(f"CRS                  : {crs}\n")

    cdl_dir = args.out_root / "cdl_masks"
    lab_root = args.out_root / "yield_labels"

    print(f"{'yr':>5} {'valid px':>9} {'kept':>7} {'chip soybean fraction (row-major)':>44}")
    for yr in sorted(paths):
        y = read_onto(paths[yr], x0, y0, ncols, nrows, YIELD_NODATA)
        valid = np.isfinite(y) & (y != YIELD_NODATA)

        with rasterio.open(paths[yr]) as ds:
            src = ds.read(1).astype(np.float32)
        n_src = int((np.isfinite(src) & (src != YIELD_NODATA)).sum())
        n_kept = int(valid.sum())

        fracs = [valid[r * CHIP:(r + 1) * CHIP, c * CHIP:(c + 1) * CHIP].mean()
                 for r in range(nrows // CHIP) for c in range(ncols // CHIP)]
        flag = "" if n_kept == n_src else "  <-- LOST PIXELS"
        print(f"{yr:>5} {n_src:>9,} {n_kept:>7,} {' '.join(f'{f * 100:5.2f}%' for f in fracs):>44}{flag}")

        if args.dry_run:
            continue

        mask_path = cdl_dir / f"cdl_{crop}_{token}_{yr}.tif"
        label_path = (lab_root / str(yr) /
                      f"nass_{crop}_yield_{token}_{yr}_{args.label_variant}_30m_{crop}_only.tif")
        for p in (mask_path, label_path):
            if p.exists() and not args.overwrite:
                raise SystemExit(f"{p} exists; pass --overwrite to replace")

        # nodata=None matches the state CDLs exactly, so `cdl > 0.5` behaves identically.
        write_raster(mask_path, valid.astype(np.float32), x0, y0, crs, None)
        label = np.where(valid, y, YIELD_NODATA)
        write_raster(label_path, label, x0, y0, crs, YIELD_NODATA)

    print()
    print(f"Every chip fraction above is far below the downloader's default --min-crop-fraction\n"
          f"0.05, which would reject all {ncols // CHIP * (nrows // CHIP)} chips and report success having fetched nothing.\n"
          f"Use --min-crop-fraction 0.0001 (5 px of {CHIP * CHIP:,}): small enough to keep every\n"
          f"chip that holds real BARC data, large enough to skip the all-zero chips -- roughly\n"
          f"half of them, since BARC occupies only part of the padded extent. Do NOT use 0.0;\n"
          f"that downloads the empty chips too.")
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    else:
        print(f"\nwrote {cdl_dir}/cdl_{crop}_{token}_*.tif")
        print(f"wrote {lab_root}/<year>/nass_{crop}_yield_{token}_*_{args.label_variant}_30m_{crop}_only.tif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
