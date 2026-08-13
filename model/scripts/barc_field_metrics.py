#!/usr/bin/env python
"""Aggregate BARC pixel predictions to FIELD level and report metrics.

Why field level is the headline unit for BARC
---------------------------------------------
Prithvi's 14x14 patch embed over 30 m imagery makes one token 420 x 420 m, and
the 30 m output is that token grid bilinearly upsampled. BARC's median field is
~216 m across -- HALF a token -- so per-pixel metrics grade a 420 m predictor
against sub-token structure it cannot express. Measured on BARC 2024: the 2,676
crop pixels fall in 61 tokens, NONE of them 100% BARC (median BARC fill 20.4%).

Field means are the finest unit at which the numbers mean something, so they are
what should be reported. Per-pixel numbers stay useful as a secondary figure but
must be labelled resolution-limited.

Reads the GeoTIFFs written by scripts/map_test_errors.py and the `field_id`
band (band 2) of the ORIGINAL BARC yield raster, windowed onto the same padded
grid. Field IDs are not carried into the padded label written by
data_preparation/scripts/make_barc_pseudo_state.py, which keeps that file a
plain single-band label; they are re-read from source here instead.

Example:
  uv run python scripts/barc_field_metrics.py --year 2024 \
      --pred-dir outputs/predictions/decoder_only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

REPO = Path(__file__).resolve().parents[2]
YIELD_DIR = REPO / "data_preparation/data/barc_data/yield_dataset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--state", default="BARC")
    p.add_argument("--crop", default="soybeans")
    p.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions/decoder_only"))
    p.add_argument("--yield-dir", type=Path, default=YIELD_DIR)
    p.add_argument("--min-px", type=int, default=5,
                   help="Skip fields with fewer than this many comparable pixels.")
    p.add_argument("--top", type=int, default=15, help="Rows to print.")
    p.add_argument("--csv-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    stem = f"{a.crop}_{a.state}_{a.year}"

    with rasterio.open(a.pred_dir / f"{stem}_pred.tif") as d:
        pred, nod, transform = d.read(1), d.nodata, d.transform
        H, W = d.shape
    with rasterio.open(a.pred_dir / f"{stem}_actual.tif") as d:
        act = d.read(1)

    # field_id is band 2 of the source yield raster; window it onto the padded grid.
    src = a.yield_dir / f"barc_{a.crop}_yield_{a.year}_30m.tif"
    with rasterio.open(src) as d:
        if d.count < 2:
            raise SystemExit(f"{src} has {d.count} band(s); expected field_id in band 2")
        b = rasterio.windows.bounds(Window(0, 0, W, H), transform)
        w = from_bounds(b[0], b[1], b[2], b[3], d.transform)
        for nm, v in (("col_off", w.col_off), ("row_off", w.row_off)):
            if abs(v - round(v)) > 1e-6:
                raise SystemExit(f"{src}: fractional {nm}={v} -- grids not pixel-aligned")
        fid = d.read(2, window=Window(round(w.col_off), round(w.row_off), W, H),
                     boundless=True, fill_value=0).astype(int)

    names: dict[int, str] = {}
    fmap = a.yield_dir / "barc_field_id_map.csv"
    if fmap.exists():
        names = {int(r["field_id"]): r["field_name"] for r in csv.DictReader(fmap.open())}

    m = (pred != nod) & (act != nod) & np.isfinite(pred) & np.isfinite(act)
    matched = m & (fid > 0)
    print(f"{a.state} {a.year}: {int(m.sum()):,} comparable px, "
          f"{int(matched.sum()):,} matched to a field")

    rows = []
    for f in sorted(set(fid[matched].tolist())):
        s = matched & (fid == f)
        n = int(s.sum())
        if n < a.min_px:
            continue
        rows.append((f, names.get(f, f"field_{f}"), n, float(pred[s].mean()), float(act[s].mean())))
    if not rows:
        raise SystemExit(f"No field had >= {a.min_px} comparable pixels")

    P = np.array([r[3] for r in rows])
    A = np.array([r[4] for r in rows])

    print(f"\n{'field':<30}{'px':>6}{'pred':>8}{'actual':>8}{'resid':>8}")
    for _f, nm, n, p_, a_ in sorted(rows, key=lambda r: -r[4])[: a.top]:
        print(f"{nm[:29]:<30}{n:>6}{p_:>8.1f}{a_:>8.1f}{p_ - a_:>+8.1f}")
    if len(rows) > a.top:
        print(f"{'... and ' + str(len(rows) - a.top) + ' more':<30}")

    rmse = float(np.sqrt(((P - A) ** 2).mean()))
    print(f"\nFIELD-LEVEL METRICS ({len(rows)} fields, >= {a.min_px} px each)")
    print(f"  pred   mean/std : {P.mean():.2f} / {P.std():.2f}")
    print(f"  actual mean/std : {A.mean():.2f} / {A.std():.2f}")
    print(f"  pearson r       : {np.corrcoef(P, A)[0, 1]:+.3f}")
    print(f"  bias            : {P.mean() - A.mean():+.2f} bu/ac")
    print(f"  RMSE            : {rmse:.2f} bu/ac")
    # A constant predictor scores exactly the target's own std. Any model that
    # cannot beat this has learned nothing about spatial variation, so it is the
    # reference every number above must be read against.
    print(f"  REFERENCE, best constant : {A.std():.2f} bu/ac"
          f"   -> model is {'BETTER' if rmse < A.std() else 'WORSE'} than a constant")
    print(f"  pred_std / actual_std    : {P.std() / A.std():.3f}"
          f"   (<<1 means a collapsed, near-flat prediction surface)")

    if a.csv_out:
        a.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with a.csv_out.open("w", newline="") as fh:
            wtr = csv.writer(fh)
            wtr.writerow(["field_id", "field_name", "n_px", "pred_mean", "actual_mean", "residual"])
            for f, nm, n, p_, a_ in sorted(rows, key=lambda r: r[0]):
                wtr.writerow([f, nm, n, f"{p_:.4f}", f"{a_:.4f}", f"{p_ - a_:.4f}"])
        print(f"\nwrote {a.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
