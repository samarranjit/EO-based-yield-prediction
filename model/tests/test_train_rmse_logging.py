"""train/main_rmse_phys: training RMSE logged in physical units (bu/ac), and not at validation."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from farm_us.config import FarmConfig
from farm_us.data.dataset import SyntheticFarmDataset
from farm_us.training.lightning_module import FarmLightningModule
from farm_us.training.metrics import TargetScaler

T, H = 4, 28


def _lm_and_batch(scaler):
    cfg = FarmConfig()
    cfg.data.n_timesteps, cfg.data.chip_size = T, H
    lm = FarmLightningModule(cfg, use_dummy=True, dummy_embed_dim=16)
    lm.set_target_scaler(scaler)
    lm.eval()  # deterministic forward: the same pass is reproduced below for the reference value
    batch = next(iter(DataLoader(SyntheticFarmDataset(n=4, n_timesteps=T, chip=H), batch_size=4)))
    logged = {}
    lm.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    return lm, batch, logged


@pytest.mark.parametrize("scaler", [
    TargetScaler(mode="zscore", center=55.0, scale=9.0),
    TargetScaler(mode="robust", center=50.0, scale=12.0),
    TargetScaler(mode="none"),
])
def test_train_rmse_is_physical_units_rmse_over_valid_pixels(scaler):
    lm, batch, logged = _lm_and_batch(scaler)
    with torch.no_grad():
        lm.training_step(batch, 0)
        pred = lm(batch, return_aux=False)["main"].float().numpy()
    mask = batch["mask"].numpy() > 0.5
    assert mask.any()
    p = scaler.inverse(pred)[mask]
    t = scaler.inverse(batch["label"].float().numpy())[mask]
    expected = float(np.sqrt(np.mean((p - t) ** 2)))
    assert logged["train/main_rmse_phys"] == pytest.approx(expected, rel=1e-4)
    if scaler.mode != "none":
        # Guard against forgetting the un-standardization: the standardized-units RMSE
        # (what train/loss is built from) must NOT be what gets logged.
        rmse_z = float(np.sqrt(np.mean((pred[mask] - batch["label"].float().numpy()[mask]) ** 2)))
        assert logged["train/main_rmse_phys"] != pytest.approx(rmse_z, rel=1e-3)


def test_validation_step_does_not_log_the_train_key_or_collide_with_epoch_end_key():
    lm, batch, logged = _lm_and_batch(TargetScaler(mode="zscore", center=55.0, scale=9.0))
    with torch.no_grad():
        lm.validation_step(batch, 0)
    assert "train/main_rmse_phys" not in logged
    # val/main_rmse_phys is logged once per epoch by on_validation_epoch_end (pooled, exact);
    # the per-step hook must not also write that key.
    assert "val/main_rmse_phys" not in logged


def test_epoch_rmse_is_pooled_over_pixels_not_a_mean_of_batch_rmses():
    scaler = TargetScaler(mode="zscore", center=55.0, scale=9.0)
    lm, batch, logged = _lm_and_batch(scaler)
    half = {k: v[:2] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    half["mask"] = half["mask"].clone()
    half["mask"][0] = 0  # batches with unequal valid-pixel counts: pooling now differs from averaging
    errors = []
    lm.on_train_epoch_start()
    with torch.no_grad():
        for b in (batch, half):
            lm.training_step(b, 0)
            mask = b["mask"].numpy() > 0.5
            p = scaler.inverse(lm(b, return_aux=False)["main"].float().numpy())[mask]
            t = scaler.inverse(b["label"].float().numpy())[mask]
            errors.append((p - t) ** 2)
    lm.on_train_epoch_end()
    expected = float(np.sqrt(np.concatenate(errors).mean()))
    assert logged["train/main_rmse_phys_epoch"] == pytest.approx(expected, rel=1e-4)

    lm.on_train_epoch_start()  # the next epoch must not inherit this epoch's pixels
    assert lm._train_n_valid == 0.0


def test_batch_with_no_valid_pixels_logs_no_rmse():
    lm, batch, logged = _lm_and_batch(TargetScaler(mode="zscore", center=55.0, scale=9.0))
    batch["mask"] = torch.zeros_like(batch["mask"])
    with torch.no_grad():
        lm.training_step(batch, 0)
    assert "train/main_rmse_phys" not in logged
    assert logged["train/skipped_zero_valid"] == 1.0
