"""Validity-mask construction.

We keep explicit, separate masks and only combine them at the point of use. A
pixel is a valid *training/metric* target iff it is crop ∧ has a valid label ∧
has valid HLS in enough months. Crucially we never infer validity from
``yield == 0`` (some true yields are low/zero; zero is also a background value).
"""

from __future__ import annotations

import numpy as np


def crop_mask_from_cdl(cdl: np.ndarray, crop_value: int | None = None) -> np.ndarray:
    """Boolean crop mask from a CDL raster.

    The project's CDL masks are already binarized (0/1). If ``crop_value`` is
    given we compare against the CDL class code instead.
    """
    if crop_value is None:
        return cdl > 0.5
    return cdl == crop_value


def label_valid_mask(label: np.ndarray, nodata: float) -> np.ndarray:
    return np.isfinite(label) & (label != nodata)


def hls_valid_mask_from_months(month_valid: np.ndarray, min_valid_months: int) -> np.ndarray:
    """``month_valid``: [T, H, W] boolean per-month validity → [H, W] boolean."""
    return month_valid.sum(axis=0) >= min_valid_months


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    out = masks[0].astype(bool)
    for m in masks[1:]:
        out = out & m.astype(bool)
    return out


def valid_fraction(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0
