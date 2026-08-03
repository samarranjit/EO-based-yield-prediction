#!/usr/bin/env python
"""Regenerate evaluation plots from a completed run -- no model, no GPU, no re-inference.

`evaluate` saves the raw valid-pixel arrays it plotted (eval_pixels.npz) next to
its figures, so restyling a chart (different alpha, different point size) costs
seconds instead of a full re-evaluation (~1h, almost all of it recomputing fold
normalization statistics).

Examples:
  # restyle in place, overwriting the existing PNGs
  uv run python scripts/replot_eval.py \
      --eval-dir outputs/runs/maryland_soybeans/test2024/eval/farm-periodic-119 \
      --alpha 0.01

  # write variants elsewhere, leaving the originals untouched
  uv run python scripts/replot_eval.py \
      --eval-dir outputs/runs/maryland_soybeans/test2024/eval/farm-periodic-119 \
      --out-dir /tmp/replot --alpha 0.05 --point-size 1
"""
import argparse
from pathlib import Path

import numpy as np

from farm_us.evaluation import plots
from farm_us.evaluation.aggregation import global_pixel_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-render eval plots from saved pixel arrays.")
    ap.add_argument("--eval-dir", type=Path, required=True, help="Directory containing eval_pixels.npz")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write PNGs (default: --eval-dir, overwrites)")
    ap.add_argument("--alpha", type=float, default=0.02, help="Scatter point opacity (lower = more density detail)")
    ap.add_argument("--point-size", type=float, default=2, help="Scatter marker size")
    ap.add_argument("--title", default="Predicted vs Observed")
    args = ap.parse_args()

    npz_path = args.eval_dir / "eval_pixels.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"{npz_path} not found.\n"
            "Only evaluations run after pixel-persistence was added have it. For an older\n"
            "run, re-run `farm-us evaluate` with that checkpoint to regenerate it."
        )

    data = np.load(npz_path)
    p, t = data["pred"], data["target"]
    out_dir = args.out_dir or args.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    m = global_pixel_metrics(p, t, np.ones_like(p))
    print(f"{p.size:,} pixels loaded from {npz_path}")
    print(
        f"  mae={m['mae']:.4f}  rmse={m['rmse']:.4f}  bias={m['bias']:.4f}  "
        f"r2={m['r2']:.4f}  pearson_r={m['pearson_r']:.4f}"
    )

    plots.scatter_obs_pred(
        p, t, out_dir / "scatter_obs_pred.png",
        title=args.title, alpha=args.alpha, point_size=args.point_size,
    )
    plots.residual_hist(p, t, out_dir / "residual_hist.png")
    plots.residual_vs_obs(p, t, out_dir / "residual_vs_obs.png")
    print(f"wrote scatter_obs_pred.png, residual_hist.png, residual_vs_obs.png -> {out_dir}")


if __name__ == "__main__":
    main()
