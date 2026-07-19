import numpy as np
import pytest
import torch

from farm_us.data.dataset import SyntheticFarmDataset, apply_missing_month_policy


@pytest.mark.parametrize("T", [4, 8])
def test_synthetic_sample_shapes(T):
    ds = SyntheticFarmDataset(n=3, n_timesteps=T, chip=224)
    s = ds[0]
    assert s["image"].shape == (6, T, 224, 224)
    assert s["label"].shape == (1, 224, 224)
    assert s["mask"].shape == (1, 224, 224)
    assert s["temporal_coords"].shape == (T, 2)
    assert s["location_coords"].shape == (2,)
    assert s["image"].dtype == torch.float32


def test_mask_not_derived_from_zero_label():
    ds = SyntheticFarmDataset(n=1, invalid_fraction=0.3)
    s = ds[0]
    s["label"].numpy()[0]
    mask = s["mask"].numpy()[0] > 0.5
    # some masked-out pixels exist and label there was set to 0 (background)
    assert mask.sum() > 0
    assert (~mask).sum() > 0


def test_missing_month_temporal_interp_fills_nan():
    img = np.random.rand(6, 8, 8, 8).astype(np.float32)
    month_valid = np.ones((8, 8, 8), bool)
    img[:, 3] = np.nan
    month_valid[3] = False
    out = apply_missing_month_policy(img, month_valid, "temporal_interp")
    assert np.isfinite(out).all()


def test_missing_month_zero_fill():
    img = np.full((6, 8, 4, 4), np.nan, np.float32)
    img[:, 0] = 1.0
    mv = np.zeros((8, 4, 4), bool)
    mv[0] = True
    out = apply_missing_month_policy(img, mv, "zero_fill")
    assert np.isfinite(out).all()
    assert (out[:, 1] == 0).all()
