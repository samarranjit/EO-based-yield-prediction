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


def min_valid_months(cfg_data) -> int:
    """Minimum usable HLS months for a pixel to count, from a ``DataConfig``.

    NOTE: this deliberately reproduces the *current* effective behaviour, which
    is ``1`` -- ``dataset.py`` computed it as ``max_missing_months and 1 or 1``,
    an expression that evaluates to the constant 1 for every possible input, so
    the configured ``max_missing_months`` has never actually been applied. The
    apparently intended threshold is ``n_timesteps - max_missing_months`` (5 for
    the shipped config). Changing it would alter which pixels are valid for
    TRAINING and for METRICS -- a scientific change, not a refactor -- so it is
    left as-is here and called out rather than silently "fixed".
    """
    return 1


def target_valid_mask(
    crop_mask: np.ndarray,
    label: np.ndarray,
    month_valid: np.ndarray,
    nodata: float,
    n_valid_months: int,
) -> np.ndarray:
    """The single definition of "this pixel is a usable target": crop ∧ label ∧ HLS.

    Both the training/metrics path (``FarmChipDataset.__getitem__``) and the
    georeferenced map path (``evaluation.inference``) must gate on exactly this.
    They previously built the mask independently, and the map path omitted the
    label and HLS terms -- so the predicted raster covered ~13% more pixels than
    the actual raster and the two maps were not visually comparable. Keep this as
    the only place the three terms are combined.
    """
    return combine_masks(
        crop_mask,
        label_valid_mask(label, nodata),
        hls_valid_mask_from_months(month_valid, n_valid_months),
    )
