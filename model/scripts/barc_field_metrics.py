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
must be labelled resolution-limited. This script prints BOTH on the IDENTICAL
pixel mask so the comparison is apples-to-apples: when the field-level Pearson r
is much higher than the pixel-level one, the per-pixel figure was measuring the
resolution limit rather than model skill. (Measured on the zero-shot cornbelt5
checkpoint, BARC 2024: pixel r 0.152 -> field r 0.368.)

Reads the GeoTIFFs written by scripts/map_test_errors.py and the `field_id`
band (band 2) of the ORIGINAL BARC yield raster, windowed onto the same padded
grid. Field IDs are not carried into the padded label written by
data_preparation/scripts/make_barc_pseudo_state.py, which keeps that file a
plain single-band label; they are re-read from source here instead.

Reading the two reference numbers
---------------------------------
`best constant` is the RMSE a predictor that always guesses the field mean would
score -- exactly the target's own std. A model that cannot beat it has learned
nothing about spatial variation, and R2 <= 0 says the same thing in different
units. `pred_std / actual_std` well below 1 means the prediction surface has
collapsed toward a flat value, which is the usual failure mode when a model
trained on smooth county pseudo-labels is asked for within-site variation.

Example:
  uv run python scripts/barc_field_metrics.py --year 2024 \
      --pred-dir outputs/predictions/barc_transfer_5_states_model --plot
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
YIELD_DIR = REPO / "data_preparation/data/barc_data/yield_dataset"

#: Where field-level comparisons land, grouped by the prediction's source
#: directory -- prediction stems are identical across experiments, so a flat
#: layout would have later runs overwrite earlier ones.
AGG_DIR = MODEL_DIR / "outputs/field_aggregation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--state", default="BARC")
    p.add_argument("--crop", default="soybeans")
    p.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions/decoder_only"),
                   help="directory of map_test_errors.py output (<crop>_<state>_<year>_pred.tif)")
    p.add_argument("--yield-dir", type=Path, default=YIELD_DIR)
    p.add_argument("--min-px", type=int, default=5,
                   help="Skip fields with fewer than this many comparable pixels.")
    p.add_argument("--top", type=int, default=15, help="Rows to print.")
    p.add_argument("--csv-out", type=Path, default=None,
                   help="default: outputs/field_aggregation/<pred-dir name>/"
                        "<crop>_<state>_<year>_field_metrics.csv")
    p.add_argument("--plot", action="store_true", help="also write a scatter PNG beside the CSV")
    return p.parse_args()


def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """r2 is against the 1:1 line and is the one that reflects predictive
    accuracy; pearson_r allows a refitted intercept and slope, so a large gap
    between them means correct ranking but bad calibration."""
    if len(obs) < 2:
        return {"n": int(len(obs))}
    resid = pred - obs
    ss_res = float((resid**2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return {
        "n": int(len(obs)),
        "bias": float(resid.mean()),
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid**2).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_r": float(np.corrcoef(obs, pred)[0, 1]),
        "pct_mae": float(100.0 * np.abs(resid).mean() / obs.mean()) if obs.mean() else float("nan"),
    }


def show(label: str, m: dict) -> None:
    if m.get("n", 0) < 2:
        print(f"  [{label}] too few samples (n={m.get('n', 0)})")
        return
    print(f"  [{label}]  n={m['n']}")
    print(f"      bias {m['bias']:+8.3f}   MAE {m['mae']:7.3f}   RMSE {m['rmse']:7.3f}")
    print(f"      R2   {m['r2']:8.4f}   r   {m['pearson_r']:+7.4f}   %MAE {m['pct_mae']:6.2f}")


def scatter(P: np.ndarray, A: np.ndarray, npx: np.ndarray, out_png: Path,
            title: str, m: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    # Measured on y, predicted on x -- matching scripts/county_aggregate.py. A
    # point ABOVE the 1:1 line is a field the model UNDER-predicts; `bias` in the
    # annotation is still pred - measured, so a positive bias sits below the line.
    ax.scatter(P, A, s=np.clip(npx, 10, 160), alpha=0.75, edgecolor="none")
    lo, hi = float(min(P.min(), A.min())), float(max(P.max(), A.max()))
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], lw=1, ls="--", color="0.4", label="1:1")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Predicted field mean (bu/ac)")
    ax.set_ylabel("Measured field mean (bu/ac)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", frameon=False)
    txt = (f"n = {m['n']} fields\nbias = {m['bias']:+.2f}  (pred − meas)\n"
           f"MAE = {m['mae']:.2f}\nRMSE = {m['rmse']:.2f}\n"
           f"R² = {m['r2']:.3f}\nr = {m['pearson_r']:+.3f}")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace")
    ax.text(0.03, 0.03, "marker size ∝ pixels in field", transform=ax.transAxes,
            fontsize=8, color="0.45")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    a = parse_args()
    stem = f"{a.crop}_{a.state}_{a.year}"

    pred_tif = a.pred_dir / f"{stem}_pred.tif"
    act_tif = a.pred_dir / f"{stem}_actual.tif"
    for f in (pred_tif, act_tif):
        if not f.exists():
            raise SystemExit(f"Not found: {f}\nRun scripts/map_test_errors.py for "
                             f"{a.state} {a.year} first.")

    with rasterio.open(pred_tif) as d:
        pred, nod, transform = d.read(1), d.nodata, d.transform
        H, W = d.shape
    with rasterio.open(act_tif) as d:
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
        rows.append((f, names.get(f, f"field_{f}"), n,
                     float(pred[s].mean()), float(act[s].mean()),
                     float(np.median(pred[s])), float(np.median(act[s])),
                     float(pred[s].std()), float(act[s].std())))
    if not rows:
        raise SystemExit(f"No field had >= {a.min_px} comparable pixels")

    P = np.array([r[3] for r in rows])
    A = np.array([r[4] for r in rows])
    NPX = np.array([r[2] for r in rows])

    print(f"\n{'field':<30}{'px':>6}{'pred':>8}{'actual':>8}{'resid':>8}")
    for r in sorted(rows, key=lambda r: -r[4])[: a.top]:
        print(f"{r[1][:29]:<30}{r[2]:>6}{r[3]:>8.1f}{r[4]:>8.1f}{r[3] - r[4]:>+8.1f}")
    if len(rows) > a.top:
        print(f"{'... and ' + str(len(rows) - a.top) + ' more':<30}")

    field_m = regression_metrics(A, P)
    print(f"\nFIELD-LEVEL METRICS ({len(rows)} fields, >= {a.min_px} px each)")
    print(f"  pred   mean/std : {P.mean():.2f} / {P.std():.2f}")
    print(f"  actual mean/std : {A.mean():.2f} / {A.std():.2f}")
    show("field", field_m)
    # A constant predictor scores exactly the target's own std. Any model that
    # cannot beat this has learned nothing about spatial variation, so it is the
    # reference every number above must be read against.
    print(f"  REFERENCE, best constant : {A.std():.2f} bu/ac"
          f"   -> model is {'BETTER' if field_m['rmse'] < A.std() else 'WORSE'} than a constant")
    print(f"  pred_std / actual_std    : {P.std() / A.std():.3f}"
          f"   (<<1 means a collapsed, near-flat prediction surface)")

    # Same mask, pixel level -- the resolution-limited secondary figure.
    print("\nPIXEL-LEVEL METRICS (same mask, resolution-limited -- secondary)")
    show("pixel", regression_metrics(act[matched].astype("float64"),
                                     pred[matched].astype("float64")))

    out_csv = a.csv_out or (AGG_DIR / a.pred_dir.resolve().name / f"{stem}_field_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["field_id", "field_name", "n_px", "pred_mean", "actual_mean",
                      "residual", "pct_residual", "pred_median", "actual_median",
                      "pred_std", "actual_std"])
        for r in sorted(rows, key=lambda r: r[0]):
            f, nm, n, p_, a_, pm, am, ps, as_ = r
            pct = 100.0 * (p_ - a_) / a_ if a_ else float("nan")
            wtr.writerow([f, nm, n, f"{p_:.4f}", f"{a_:.4f}", f"{p_ - a_:.4f}",
                          f"{pct:.2f}", f"{pm:.4f}", f"{am:.4f}", f"{ps:.4f}", f"{as_:.4f}"])
    print(f"\nwrote {out_csv}")

    if a.plot and len(rows) >= 2:
        png = out_csv.with_suffix(".png")
        scatter(P, A, NPX, png, f"{a.state} {a.year} — field mean, predicted vs measured", field_m)
        print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
