"""A trailing batch of ONE sample must never reach the model in training mode.

The PPM pools to a 1x1 bin (``ppm_bins`` starts at 1), so a single-sample batch
arrives at BatchNorm as ``[1, C, 1, 1]`` and torch raises
"Expected more than 1 value per channel when training".

This is not hypothetical: it killed a real run at step 1217/1217 of epoch 0
after ~2 hours of compute, because a dataset of 2433 chips with batch_size=2
leaves exactly one sample over. The dataset size had shifted from even to odd as
a side effect of an unrelated raster-alignment fix, so nothing about the change
looked related to batching.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from farm_us.config import FarmConfig
from farm_us.data.dataset import FarmDataModule, SyntheticFarmDataset
from farm_us.models.farm_model import FarmModel


def test_odd_dataset_with_batch_size_two_would_produce_singleton_batch():
    """Guards the premise: without drop_last such a loader really does emit a 1-sample batch."""
    ds = SyntheticFarmDataset(n=5, n_timesteps=4, chip=28)
    sizes = [b["image"].shape[0] for b in DataLoader(ds, batch_size=2, drop_last=False)]
    assert sizes == [2, 2, 1], f"expected a trailing singleton, got {sizes}"


def test_train_dataloader_drops_the_singleton_batch():
    cfg = FarmConfig()
    cfg.train.batch_size = 2
    cfg.train.num_workers = 0

    dm = FarmDataModule(cfg, synthetic=True, n_synth=5)  # odd -> would leave 1 over
    dm.setup()

    train_sizes = [b["image"].shape[0] for b in dm.train_dataloader()]
    assert train_sizes, "train loader yielded nothing"
    assert all(s > 1 for s in train_sizes), f"singleton batch survived: {train_sizes}"


def test_val_and_test_loaders_keep_every_sample():
    """Eval runs under model.eval(); dropping there would silently lose metrics."""
    cfg = FarmConfig()
    cfg.train.batch_size = 2
    cfg.train.num_workers = 0

    dm = FarmDataModule(cfg, synthetic=True, n_synth=5)
    dm.setup()

    for name, loader, ds in [
        ("val", dm.val_dataloader(), dm.val_ds),
        ("test", dm.test_dataloader(), dm.test_ds),
    ]:
        seen = sum(b["image"].shape[0] for b in loader)
        assert seen == len(ds), f"{name} loader dropped samples: {seen} != {len(ds)}"


def test_singleton_batch_really_breaks_the_model_in_train_mode():
    """Pins WHY drop_last is required, so the guard can't be removed as redundant."""
    cfg = FarmConfig()
    model = FarmModel(cfg.model, n_timesteps=4, chip_size=28, use_dummy=True, dummy_embed_dim=16)
    model.train()

    def forward(batch_size):
        x = torch.randn(batch_size, 6, 4, 28, 28)
        tc = torch.zeros(batch_size, 4, 2)
        lc = torch.zeros(batch_size, 2)
        return model(x, tc, lc)

    with pytest.raises(ValueError, match="more than 1 value per channel"):
        forward(1)

    out = forward(2)  # two samples is fine
    assert out["main"].shape[0] == 2
