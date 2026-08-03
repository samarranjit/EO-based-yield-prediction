"""Fold evaluator: run a trained model over the test loader, collect predictions,
compute masked de-standardized metrics at all levels, save CSV/Parquet + plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..training.metrics import TargetScaler, regression_metrics
from . import plots
from .aggregation import global_pixel_metrics, macro_average, per_group_metrics


@torch.no_grad()
def evaluate_loader(model, loader, scaler: TargetScaler, device: str = "cpu") -> pd.DataFrame:
    """Return a per-chip long dataframe with de-standardized pred/target means and
    per-chip metrics, plus accumulates all valid pixels for global metrics."""
    model.eval().to(device)
    rows = []
    px_pred, px_tgt = [], []
    for batch in loader:
        x = batch["image"].to(device)
        tc = batch.get("temporal_coords")
        lc = batch.get("location_coords")
        tc = tc.to(device) if isinstance(tc, torch.Tensor) else None
        lc = lc.to(device) if isinstance(lc, torch.Tensor) else None
        out = model(x, tc, lc, return_aux=False)
        pred = scaler.inverse(out["main"].float().cpu().numpy())
        tgt = scaler.inverse(batch["label"].float().cpu().numpy())
        mask = batch["mask"].cpu().numpy() > 0.5
        for b in range(pred.shape[0]):
            pm, tm, mm = pred[b, 0], tgt[b, 0], mask[b, 0]
            if mm.sum() < 1:
                continue
            m = regression_metrics(pm, tm, mm)
            rows.append({
                "sample_id": batch["sample_id"][b] if "sample_id" in batch else f"b{b}",
                "state": batch["state"][b] if "state" in batch else "NA",
                "year": int(batch["year"][b]) if "year" in batch else -1,
                "pred": float(pm[mm].mean()), "target": float(tm[mm].mean()),
                **{f"chip_{k}": v for k, v in m.items()},
            })
            px_pred.append(pm[mm]); px_tgt.append(tm[mm])
    df = pd.DataFrame(rows)
    if px_pred:
        df.attrs["pixels"] = (np.concatenate(px_pred), np.concatenate(px_tgt))
    return df


def summarize(df: pd.DataFrame, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    if "pixels" in df.attrs:
        p, t = df.attrs["pixels"]
        result["global_pixel"] = global_pixel_metrics(p, t, np.ones_like(p))
        # Persist the raw valid-pixel arrays. They are the ONLY input the three
        # plots below need, and without them a re-plot (different alpha, extra
        # chart type) costs a full re-evaluation -- for the real model that is
        # ~1h, almost all of it recomputing fold normalization stats. float32
        # keeps this ~8 MB per million pixels; see scripts/replot_eval.py.
        np.savez_compressed(
            out_dir / "eval_pixels.npz",
            pred=np.asarray(p, dtype=np.float32),
            target=np.asarray(t, dtype=np.float32),
        )
        plots.scatter_obs_pred(p, t, out_dir / "scatter_obs_pred.png")
        plots.residual_hist(p, t, out_dir / "residual_hist.png")
        plots.residual_vs_obs(p, t, out_dir / "residual_vs_obs.png")

    for level in ("state", "year"):
        if level in df.columns:
            g = per_group_metrics(df.rename(columns={"pred": "pred", "target": "target"}), [level])
            g.to_csv(out_dir / f"metrics_by_{level}.csv", index=False)
            result[f"macro_by_{level}"] = macro_average(g)

    df.to_csv(out_dir / "per_chip_metrics.csv", index=False)
    try:
        df.drop(columns=[], errors="ignore").to_parquet(out_dir / "per_chip_metrics.parquet")
    except Exception:
        pass
    return result
