"""Multi-level metric aggregation and bootstrap confidence intervals.

Metrics are reported at several levels because a single pixel-weighted global
number is dominated by big states/counties. We also aggregate predicted crop
pixels to county level to compare against the *original* NASS county yield
(kept separate from the ridge pseudo-label comparison).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..training.metrics import regression_metrics


def per_group_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """``df`` needs columns: pred, target, plus the group columns (one row per
    valid pixel or per-chip aggregate)."""
    rows = []
    for keys, g in df.groupby(group_cols):
        m = regression_metrics(g["pred"].values, g["target"].values, np.ones(len(g)))
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=False))
        rec.update(m)
        rows.append(rec)
    return pd.DataFrame(rows)


def macro_average(per_group: pd.DataFrame, metrics=("mae", "rmse", "r2", "pearson_r")) -> dict:
    return {f"macro_{k}": float(per_group[k].mean(skipna=True)) for k in metrics}


def global_pixel_metrics(pred, target, mask) -> dict:
    return regression_metrics(pred, target, mask)


def bootstrap_ci(
    df: pd.DataFrame,
    group_col: str,
    metric: str = "rmse",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Resample *groups* (counties/chips), not pixels, for honest CIs."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    stats = []
    per = per_group_metrics(df, [group_col]).set_index(group_col)
    for _ in range(n_boot):
        sample = rng.choice(groups, size=len(groups), replace=True)
        stats.append(np.nanmean(per.loc[sample, metric].values))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.nanmean(stats)), float(lo), float(hi)


def aggregate_to_county(
    prediction: np.ndarray,
    county_id: np.ndarray,
    crop_mask: np.ndarray,
) -> pd.DataFrame:
    """Mean predicted yield over crop pixels within each county."""
    valid = crop_mask & np.isfinite(prediction) & (county_id > 0)
    ids = county_id[valid].astype(int)
    vals = prediction[valid]
    df = pd.DataFrame({"GEOID": ids, "pred": vals})
    return df.groupby("GEOID")["pred"].mean().reset_index()


def compare_county_to_nass(county_pred: pd.DataFrame, nass: pd.DataFrame, year: int) -> pd.DataFrame:
    n = nass[nass["year"] == year][["GEOID", "yield_kg_ha"]].rename(columns={"yield_kg_ha": "nass"})
    county_pred = county_pred.copy()
    county_pred["GEOID"] = county_pred["GEOID"].astype(int)
    n["GEOID"] = n["GEOID"].astype(int)
    merged = county_pred.merge(n, on="GEOID", how="inner")
    return merged
