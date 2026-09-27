"""DetailRefiner: off by default, an exact no-op at init, refiner_only freezes
everything else (including BatchNorm statistics), and init_from may only be
missing the new branch's weights."""

import pytest
import torch

from farm_us.config import FarmConfig, ModelConfig
from farm_us.models.farm_model import FarmModel

T, H = 4, 28


def _inputs(b=2):
    return torch.randn(b, 6, T, H, H), torch.zeros(b, T, 2), torch.zeros(b, 2)


def _model(**overrides):
    cfg = ModelConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return FarmModel(cfg, n_timesteps=T, chip_size=H, use_dummy=True, dummy_embed_dim=16)


def _lm(**model_overrides):
    from farm_us.training.lightning_module import FarmLightningModule

    cfg = FarmConfig()
    for k, v in model_overrides.items():
        setattr(cfg.model, k, v)
    return FarmLightningModule(cfg, use_dummy=True, dummy_embed_dim=16)


def test_refiner_off_by_default():
    m = _model()
    assert m.refiner is None
    assert not any(n.startswith("refiner.") for n, _ in m.named_parameters())


def test_refiner_is_exact_noop_at_init():
    base, refined = _model(), _model(detail_refiner=True)
    missing, unexpected = refined.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and missing and all(k.startswith("refiner.") for k in missing)
    base.eval()
    refined.eval()
    x, tc, lc = _inputs()
    with torch.no_grad():
        assert torch.equal(base(x, tc, lc)["main"], refined(x, tc, lc)["main"])


def test_refiner_only_trains_only_the_refiner():
    m = _model(detail_refiner=True, finetune_mode="refiner_only")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("refiner.") for n in trainable)


def test_refiner_only_keeps_frozen_batchnorm_statistics():
    m = _model(detail_refiner=True, finetune_mode="refiner_only")
    keys = [k for k in m.state_dict() if "running_" in k or "num_batches_tracked" in k]
    before = {k: m.state_dict()[k].clone() for k in keys}
    m.train()
    m(*_inputs())
    assert keys and all(torch.equal(before[k], m.state_dict()[k]) for k in keys)
    assert m.refiner.training and not m.head.training


def test_refiner_only_requires_the_refiner():
    with pytest.raises(ValueError, match="detail_refiner"):
        _model(finetune_mode="refiner_only")


def test_zero_init_refiner_still_learns():
    m = _model(detail_refiner=True, finetune_mode="refiner_only")
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    m.train()
    x, tc, lc = _inputs()
    target = torch.randn(2, 1, H, H)
    for _ in range(3):
        opt.zero_grad()
        ((m(x, tc, lc)["main"] - target) ** 2).mean().backward()
        opt.step()
    assert m.refiner.out.weight.abs().sum() > 0


def test_init_loader_accepts_only_missing_refiner_keys(tmp_path):
    from farm_us.training.run import load_init_weights

    src = tmp_path / "base.ckpt"
    torch.save({"state_dict": _lm().state_dict()}, src)
    load_init_weights(_lm(detail_refiner=True), str(src))


def test_init_loader_rejects_any_other_mismatch(tmp_path):
    from farm_us.training.run import load_init_weights

    state = _lm().state_dict()
    state.pop(next(k for k in state if k.startswith("model.head.")))
    src = tmp_path / "broken.ckpt"
    torch.save({"state_dict": state}, src)
    with pytest.raises(RuntimeError, match="init_from"):
        load_init_weights(_lm(detail_refiner=True), str(src))


def test_refiner_only_frozen_batchnorm_survives_lightning_fit(tmp_path):
    """Regression: Lightning's evaluation loop restores each submodule's `training`
    flag by assignment (_ModuleMode), bypassing FarmModel.train(). Frozen BatchNorm
    layers then ran in train mode and their running statistics drifted. The unit
    test above calls .train() directly, so it cannot see this -- only a real
    Trainer.fit can."""
    import lightning as L
    from torch.utils.data import DataLoader

    from farm_us.data.dataset import SyntheticFarmDataset
    from farm_us.training.lightning_module import FarmLightningModule

    cfg = FarmConfig()
    cfg.data.n_timesteps, cfg.data.chip_size = T, H
    cfg.model.detail_refiner, cfg.model.finetune_mode = True, "refiner_only"
    lm = FarmLightningModule(cfg, use_dummy=True, dummy_embed_dim=16)
    keys = [k for k in lm.state_dict() if "running_" in k or "num_batches_tracked" in k]
    before = {k: lm.state_dict()[k].clone() for k in keys}
    dl = DataLoader(SyntheticFarmDataset(n=4, n_timesteps=T, chip=H), batch_size=2)
    L.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False,
              enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=1,
              limit_train_batches=2, limit_val_batches=1, default_root_dir=tmp_path).fit(lm, dl, dl)
    after = lm.state_dict()
    assert keys and all(torch.equal(before[k], after[k]) for k in keys)
