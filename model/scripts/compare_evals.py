#!/usr/bin/env python
"""Tabulate several completed evaluations side by side.

Reads each eval directory's saved pixel arrays (eval_pixels.npz) and per-chip
metrics, and prints one row per checkpoint -- useful for comparing checkpoints
on the same held-out test year without re-running inference.

Example:
  uv run python scripts/compare_evals.py \
      outputs/runs/maryland_soybeans/test2024/eval/farm-003-0.0000 \
      outputs/runs/maryland_soybeans/test2024/eval/farm-periodic-099
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from farm_us.evaluation.aggregation import global_pixel_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare completed evaluations.")
    ap.add_argument("eval_dirs", nargs="+", type=Path, help="Directories written by `farm-us evaluate`")
    ap.add_argument("--csv", type=Path, default=None, help="Optional path to save the table as CSV")
    args = ap.parse_args()

    rows = []
    for d in args.eval_dirs:
        npz = d / "eval_pixels.npz"
        chips = d / "per_chip_metrics.csv"
        if not npz.exists():
            print(f"skip {d}: no eval_pixels.npz (evaluation predates pixel persistence)")
            continue

        data = np.load(npz)
        p, t = data["pred"], data["target"]
        m = global_pixel_metrics(p, t, np.ones_like(p))

        row = {
            "checkpoint": d.name,
            "n_pixels": int(p.size),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "bias": m["bias"],
            "r2": m["r2"],
            "pearson_r": m["pearson_r"],
            "pred_mean": float(p.mean()),
            "target_mean": float(t.mean()),
            "pred_std": float(p.std()),
            "target_std": float(t.std()),
        }
        if chips.exists():
            row["n_chips"] = len(pd.read_csv(chips))
        rows.append(row)

    if not rows:
        raise SystemExit("nothing to compare")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    # Prediction spread vs truth is the tell for regression-to-the-mean: a model
    # that hedges toward the mean has a visibly smaller std than the target.
    for r in rows:
        ratio = r["pred_std"] / r["target_std"] if r["target_std"] else float("nan")
        print(f"{r['checkpoint']}: predicted spread is {ratio:.2f}x the true spread "
              f"({'compressed -- hedging toward the mean' if ratio < 0.9 else 'comparable'})")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
