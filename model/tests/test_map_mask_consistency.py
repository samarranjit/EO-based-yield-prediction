"""The predicted and actual yield maps must cover exactly the same pixels.

Regression tests for a real defect: the map path gated `prediction` on the crop
mask ALONE while `actual` additionally carried NaN wherever the label raster was
nodata, so the predicted GeoTIFF covered ~13% more pixels than the actual one
(measured on MD 2024: 1,421,110 vs 1,230,484). The two maps were not comparable,
visually or numerically.
"""

import numpy as np

from farm_us.data import masks as M


def _chip(crop, label, month_valid):
    return crop, label, month_valid


def test_target_valid_mask_is_three_way_intersection():
    crop = np.array([[1, 1, 1, 0]], dtype=bool)
    label = np.array([[10.0, np.nan, 12.0, 13.0]], dtype=np.float32)
    month_valid = np.array([[[1, 1, 0, 1]]], dtype=bool)  # [T=1, H=1, W=4]

    m = M.target_valid_mask(crop, label, month_valid, nodata=-9999.0, n_valid_months=1)

    # col0: crop, label ok, hls ok      -> True
    # col1: crop, label NaN             -> False
    # col2: crop, label ok, no hls      -> False
    # col3: not crop                    -> False
    assert m.tolist() == [[True, False, False, False]]


def test_target_valid_mask_rejects_nodata_sentinel():
    """-9999 must be rejected even though it is a finite float."""
    crop = np.ones((1, 2), dtype=bool)
    label = np.array([[-9999.0, 42.0]], dtype=np.float32)
    month_valid = np.ones((1, 1, 2), dtype=bool)
    m = M.target_valid_mask(crop, label, month_valid, nodata=-9999.0, n_valid_months=1)
    assert m.tolist() == [[False, True]]


def test_comparison_surfaces_share_one_footprint():
    """prediction / actual / residual must agree pixel-for-pixel on validity.

    Reproduces the map assembly in
    evaluation.inference.predict_and_compare_test_year without needing a model:
    the model output is dense (finite everywhere), the label is sparse, and the
    comparison surfaces must end up restricted to the intersection.
    """
    h = w = 8
    rng = np.random.default_rng(0)

    crop = rng.random((h, w)) > 0.3
    label = rng.random((h, w)).astype(np.float32) * 50 + 20
    label[rng.random((h, w)) > 0.6] = np.nan          # sparse labels, as in reality
    month_valid = (rng.random((8, h, w)) > 0.2)
    prediction_full = rng.random((h, w)).astype(np.float32) * 50 + 20  # dense

    comparison_mask = M.target_valid_mask(crop, label, month_valid, -9999.0, 1)
    comparison_mask &= np.isfinite(prediction_full)

    prediction = np.where(comparison_mask, prediction_full, np.nan)
    actual = np.where(comparison_mask, label, np.nan)
    residual = prediction - actual

    fp_pred = np.isfinite(prediction)
    fp_actual = np.isfinite(actual)
    fp_resid = np.isfinite(residual)

    assert np.array_equal(fp_pred, fp_actual), "prediction/actual footprints differ"
    assert np.array_equal(fp_pred, fp_resid), "residual footprint differs"
    assert np.array_equal(fp_pred, comparison_mask)

    # And the guard: the FULL surface must be strictly larger, else the test
    # would pass trivially on data that has no label gaps at all.
    assert np.isfinite(prediction_full).sum() > fp_pred.sum()


def test_full_surface_is_not_restricted_by_label_availability():
    """prediction_full intentionally keeps pixels the comparison set drops."""
    h = w = 4
    crop = np.ones((h, w), dtype=bool)
    label = np.full((h, w), np.nan, dtype=np.float32)   # no labels at all
    label[0, 0] = 40.0
    month_valid = np.ones((8, h, w), dtype=bool)
    prediction_full = np.full((h, w), 45.0, dtype=np.float32)

    comparison_mask = M.target_valid_mask(crop, label, month_valid, -9999.0, 1)
    comparison_mask &= np.isfinite(prediction_full)

    assert comparison_mask.sum() == 1, "only the single labelled pixel is comparable"
    assert np.isfinite(prediction_full).sum() == h * w, "full surface must keep every crop pixel"
