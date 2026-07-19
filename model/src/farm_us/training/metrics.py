"""Masked regression metrics in physical (de-standardized) units.

Every metric ignores invalid pixels. ``TargetScaler`` handles the standardized
↔ physical inverse transform so metrics are always reported in the original
units (e.g. kg/ha) as well as standardized diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetScaler:
    """Standardize / inverse-transform yield targets (train-fold stats only)."""

    mode: str = "zscore"  # none | zscore | robust
    center: float = 0.0   # mean (zscore) or median (robust)
    scale: float = 1.0    # std (zscore) or IQR (robust)

    def transform(self, y):
        if self.mode == "none":
            return y
        return (y - self.center) / (self.scale if self.scale != 0 else 1.0)

    def inverse(self, y):
        if self.mode == "none":
            return y
        return y * self.scale + self.center

    def to_dict(self):
        return {"mode": self.mode, "center": self.center, "scale": self.scale}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _flatten_valid(pred, target, mask):
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    mask = np.asarray(mask).ravel() > 0.5
    return pred[mask], target[mask]


def regression_metrics(pred, target, mask, denom_for_pct: float | None = None) -> dict[str, float]:
    """Return MAE, RMSE, R², Pearson r, r², bias, and (optional) %MAE.

    Inputs are assumed already in physical units.
    """
    p, t = _flatten_valid(pred, target, mask)
    out: dict[str, float] = {"n": float(p.size)}
    if p.size == 0:
        return {**out, "mae": np.nan, "rmse": np.nan, "r2": np.nan,
                "pearson_r": np.nan, "pearson_r2": np.nan, "bias": np.nan, "pct_mae": np.nan}
    err = p - t
    out["mae"] = float(np.mean(np.abs(err)))
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["bias"] = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if p.size > 1 and p.std() > 0 and t.std() > 0:
        r = float(np.corrcoef(p, t)[0, 1])
    else:
        r = np.nan
    out["pearson_r"] = r
    out["pearson_r2"] = r ** 2 if not np.isnan(r) else np.nan
    denom = denom_for_pct if denom_for_pct is not None else float(np.mean(np.abs(t)))
    out["pct_mae"] = float(out["mae"] / denom * 100.0) if denom and denom > 0 else np.nan
    return out
