"""ExogenousRefiner: off by default, an exact no-op at init, exo_refiner_only
freezes everything else (including BatchNorm statistics), init_from may only be
missing the new branch's weights, it is mutually exclusive with the superseded
DetailRefiner, and with no covariates configured it reproduces DetailRefiner
exactly. Mirrors test_detail_refiner.py."""

import pytest
import torch

from farm_us.config import FarmConfig, ModelConfig
from farm_us.models.farm_model import FarmModel

T, H = 4, 28
N_SOIL, N_WEATHER = 3, 2


def _inputs(b=2, n_soil=0, n_weather=0):
    x = torch.randn(b, 6, T, H, H)
    tc = torch.zeros(b, T, 2)
    lc = torch.zeros(b, 2)
    soil = torch.randn(b, n_soil, H, H) if n_soil else None
    weather = torch.randn(b, n_weather, T, H, H) if n_weather else None
    return x, tc, lc, soil, weather


def _model(n_soil=0, n_weather=0, **overrides):
    cfg = ModelConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return FarmModel(cfg, n_timesteps=T, chip_size=H, use_dummy=True, dummy_embed_dim=16,
                     n_soil=n_soil, n_weather=n_weather)


def _lm(n_soil=0, n_weather=0, **model_overrides):
    from farm_us.training.lightning_module import FarmLightningModule

    cfg = FarmConfig()
    cfg.data.n_timesteps, cfg.data.chip_size = T, H
    cfg.data.soil_vars = [f"s{i}" for i in range(n_soil)]
    cfg.data.weather_vars = [f"w{i}" for i in range(n_weather)]
    for k, v in model_overrides.items():
        setattr(cfg.model, k, v)
    return FarmLightningModule(cfg, use_dummy=True, dummy_embed_dim=16)


def test_exo_refiner_off_by_default():
    m = _model()
    assert m.exo_refiner is None
    assert not any(n.startswith("exo_refiner.") for n, _ in m.named_parameters())


def test_exo_refiner_is_exact_noop_at_init_soil_and_weather():
    base = _model()
    refined = _model(N_SOIL, N_WEATHER, exogenous_refiner=True)
    missing, unexpected = refined.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and missing and all(k.startswith("exo_refiner.") for k in missing)
    base.eval()
    refined.eval()
    x, tc, lc, soil, weather = _inputs(n_soil=N_SOIL, n_weather=N_WEATHER)
    with torch.no_grad():
        out_base = base(x, tc, lc)["main"]
        out_refined = refined(x, tc, lc, soil=soil, weather=weather)["main"]
    assert torch.equal(out_base, out_refined)


def test_exo_refiner_noop_holds_for_arbitrary_soil_and_weather_values():
    """The zero-init contract must hold regardless of what soil/weather contain --
    at step 0 the branch's own `out` layer maps everything to exactly zero, so
    even wildly different covariate values must not change the prediction."""
    refined = _model(N_SOIL, N_WEATHER, exogenous_refiner=True)
    refined.eval()
    x, tc, lc, _, _ = _inputs(n_soil=N_SOIL, n_weather=N_WEATHER)
    soil_a = torch.zeros(2, N_SOIL, H, H)
    soil_b = torch.randn(2, N_SOIL, H, H) * 100
    weather_a = torch.zeros(2, N_WEATHER, T, H, H)
    weather_b = torch.randn(2, N_WEATHER, T, H, H) * 100
    with torch.no_grad():
        out_a = refined(x, tc, lc, soil=soil_a, weather=weather_a)["main"]
        out_b = refined(x, tc, lc, soil=soil_b, weather=weather_b)["main"]
    assert torch.equal(out_a, out_b)


def test_exo_refiner_only_soil_no_weather():
    m = _model(n_soil=N_SOIL, n_weather=0, exogenous_refiner=True)
    assert m.exo_refiner.weather_enc is None
    x, tc, lc, soil, _ = _inputs(n_soil=N_SOIL, n_weather=0)
    m.eval()
    with torch.no_grad():
        out = m(x, tc, lc, soil=soil, weather=None)["main"]
    assert out.shape == (2, 1, H, H)


def test_exo_refiner_only_weather_no_soil():
    m = _model(n_soil=0, n_weather=N_WEATHER, exogenous_refiner=True)
    assert m.exo_refiner.weather_enc is not None
    x, tc, lc, _, weather = _inputs(n_soil=0, n_weather=N_WEATHER)
    m.eval()
    with torch.no_grad():
        out = m(x, tc, lc, soil=None, weather=weather)["main"]
    assert out.shape == (2, 1, H, H)


def test_exo_refiner_requires_soil_when_configured():
    m = _model(n_soil=N_SOIL, n_weather=0, exogenous_refiner=True)
    x, tc, lc, _, _ = _inputs()
    with pytest.raises(ValueError, match="soil"):
        m(x, tc, lc, soil=None, weather=None)


def test_exo_refiner_requires_weather_when_configured():
    m = _model(n_soil=0, n_weather=N_WEATHER, exogenous_refiner=True)
    x, tc, lc, _, _ = _inputs()
    with pytest.raises(ValueError, match="weather"):
        m(x, tc, lc, soil=None, weather=None)


def test_exo_refiner_only_trains_only_the_exo_refiner():
    m = _model(N_SOIL, N_WEATHER, exogenous_refiner=True, finetune_mode="exo_refiner_only")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("exo_refiner.") for n in trainable)


def test_exo_refiner_only_requires_exogenous_refiner():
    with pytest.raises(ValueError, match="exogenous_refiner"):
        _model(finetune_mode="exo_refiner_only")


def test_exo_refiner_only_keeps_frozen_batchnorm_statistics():
    m = _model(N_SOIL, N_WEATHER, exogenous_refiner=True, finetune_mode="exo_refiner_only")
    keys = [k for k in m.state_dict() if "running_" in k or "num_batches_tracked" in k]
    before = {k: m.state_dict()[k].clone() for k in keys}
    m.train()
    x, tc, lc, soil, weather = _inputs(n_soil=N_SOIL, n_weather=N_WEATHER)
    m(x, tc, lc, soil=soil, weather=weather)
    assert keys and all(torch.equal(before[k], m.state_dict()[k]) for k in keys)
    assert m.exo_refiner.training and not m.head.training


def test_zero_init_exo_refiner_still_learns():
    m = _model(N_SOIL, N_WEATHER, exogenous_refiner=True, finetune_mode="exo_refiner_only")
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    m.train()
    x, tc, lc, soil, weather = _inputs(n_soil=N_SOIL, n_weather=N_WEATHER)
    target = torch.randn(2, 1, H, H)
    for _ in range(3):
        opt.zero_grad()
        out = m(x, tc, lc, soil=soil, weather=weather)["main"]
        ((out - target) ** 2).mean().backward()
        opt.step()
    assert m.exo_refiner.out.weight.abs().sum() > 0


def test_both_refiners_at_once_is_rejected():
    """ExogenousRefiner supersedes DetailRefiner -- it contains the same
    raw-imagery residual path -- so enabling both would apply two image-derived
    residuals to one coarse prediction. Must raise, and must raise BEFORE the
    encoder is built so a bad config fails fast."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _model(N_SOIL, N_WEATHER, detail_refiner=True, exogenous_refiner=True)


def test_exo_refiner_without_covariates_reproduces_detail_refiner():
    """The image-only ablation: with no soil and no weather configured,
    ExogenousRefiner must be structurally identical to DetailRefiner, so the two
    produce the same output from the same weights. This is what makes
    exogenous_refiner a true replacement rather than a different branch, and what
    makes the no-covariate run a valid control for measuring what soil/weather add."""
    detail = _model(detail_refiner=True)
    exo = _model(exogenous_refiner=True)  # n_soil=0, n_weather=0

    # Same parameter shapes under matching names => same architecture.
    d_shapes = {n.split(".", 1)[1]: p.shape for n, p in detail.named_parameters()
                if n.startswith("refiner.")}
    e_shapes = {n.split(".", 1)[1]: p.shape for n, p in exo.named_parameters()
                if n.startswith("exo_refiner.")}
    assert d_shapes and d_shapes == e_shapes

    # Copy the shared base, then give both branches the SAME non-zero weights --
    # zero-init alone would make them trivially agree at 0.
    base = _model()
    detail.load_state_dict(base.state_dict(), strict=False)
    exo.load_state_dict(base.state_dict(), strict=False)
    torch.manual_seed(0)
    branch_w = {k: torch.randn_like(v) for k, v in detail.refiner.state_dict().items()}
    detail.refiner.load_state_dict(branch_w)
    exo.exo_refiner.load_state_dict(branch_w)

    detail.eval()
    exo.eval()
    x, tc, lc, _, _ = _inputs()
    with torch.no_grad():
        out_detail = detail(x, tc, lc)["main"]
        out_exo = exo(x, tc, lc)["main"]
    assert torch.equal(out_detail, out_exo)
    # ...and non-zero weights really do move the prediction, so the equality above
    # is not comparing two untouched coarse predictions.
    with torch.no_grad():
        assert not torch.equal(out_detail, base.eval()(x, tc, lc)["main"])


def test_init_loader_accepts_missing_exo_refiner_keys(tmp_path):
    from farm_us.training.run import load_init_weights

    src = tmp_path / "base.ckpt"
    torch.save({"state_dict": _lm().state_dict()}, src)
    load_init_weights(_lm(N_SOIL, N_WEATHER, exogenous_refiner=True), str(src))


def test_lightning_module_derives_channel_counts_from_data_config():
    lm = _lm(N_SOIL, N_WEATHER, exogenous_refiner=True)
    assert lm.model.exo_refiner.n_soil == N_SOIL
    assert lm.model.exo_refiner.n_weather == N_WEATHER


def test_lightning_forward_passes_soil_and_weather_from_batch():
    lm = _lm(N_SOIL, N_WEATHER, exogenous_refiner=True)
    lm.eval()
    b = 2
    batch = {
        "image": torch.randn(b, 6, T, H, H),
        "temporal_coords": torch.zeros(b, T, 2),
        "location_coords": torch.zeros(b, 2),
        "soil": torch.randn(b, N_SOIL, H, H),
        "weather": torch.randn(b, N_WEATHER, T, H, H),
    }
    with torch.no_grad():
        out = lm(batch, return_aux=False)
    assert out["main"].shape == (b, 1, H, H)


def test_exo_refiner_only_frozen_batchnorm_survives_lightning_fit(tmp_path):
    """Regression, mirroring test_detail_refiner's equivalent: Lightning's
    evaluation loop restores each submodule's `training` flag by assignment,
    bypassing FarmModel.train(). Only a real Trainer.fit exercises this."""
    import lightning as L
    from torch.utils.data import DataLoader

    from farm_us.data.dataset import SyntheticFarmDataset
    from farm_us.training.lightning_module import FarmLightningModule

    cfg = FarmConfig()
    cfg.data.n_timesteps, cfg.data.chip_size = T, H
    cfg.data.soil_vars = [f"s{i}" for i in range(N_SOIL)]
    cfg.data.weather_vars = [f"w{i}" for i in range(N_WEATHER)]
    cfg.model.exogenous_refiner, cfg.model.finetune_mode = True, "exo_refiner_only"
    lm = FarmLightningModule(cfg, use_dummy=True, dummy_embed_dim=16)
    keys = [k for k in lm.state_dict() if "running_" in k or "num_batches_tracked" in k]
    before = {k: lm.state_dict()[k].clone() for k in keys}
    dl = DataLoader(
        SyntheticFarmDataset(n=4, n_timesteps=T, chip=H, n_soil=N_SOIL, n_weather=N_WEATHER),
        batch_size=2,
    )
    L.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False,
              enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=1,
              limit_train_batches=2, limit_val_batches=1, default_root_dir=tmp_path).fit(lm, dl, dl)
    after = lm.state_dict()
    assert keys and all(torch.equal(before[k], after[k]) for k in keys)


def test_weather_conditions_pixels_independently_and_backpropagates():
    """Two maps with the same chip mean must produce different spatial corrections.

    Set a transparent positive path through the branch so this tests the entire
    weather-to-output connection, not just a zero-init output or tensor shape.
    """
    from farm_us.models.exogenous_refiner import ExogenousRefiner

    branch = ExogenousRefiner(1, 1, n_weather=1, n_timesteps=1, hidden=1)
    with torch.no_grad():
        for p in branch.parameters():
            p.zero_()
        branch.weather_enc.net[0].weight.fill_(1)
        branch.weather_enc.to_shift.weight.fill_(1)
        branch.fuse[0].weight[0, 0, 1, 1] = 1  # weather-modulated feature, centre pixel
        branch.out.weight.fill_(1)
    weather = torch.ones(1, 1, 1, 4, 6)
    weather[..., 3:] = 3
    weather.requires_grad_()
    image, body = torch.zeros(1, 1, 1, 4, 6), torch.zeros(1, 1, 2, 3)
    pred = branch(image, body, weather=weather)
    torch.testing.assert_close(pred, weather[:, :, 0])
    flipped = branch(image, body, weather=weather.flip(-1))
    torch.testing.assert_close(flipped, pred.flip(-1))
    assert not torch.equal(pred, flipped)  # averaging would make these identical
    pred.sum().backward()
    assert weather.grad is not None and (weather.grad > 0).all()
    assert branch.weather_enc.net[0].weight.grad.abs().sum() > 0


def test_weather_rejects_legacy_vectors_and_wrong_spatial_grids():
    from farm_us.models.exogenous_refiner import ExogenousRefiner, WeatherEncoder

    enc = WeatherEncoder(2, 4, 2)
    with pytest.raises(ValueError, match="Legacy"):
        enc(torch.zeros(1, 2, 4))
    with pytest.raises(ValueError, match="expected"):
        enc(torch.zeros(1, 4, 2, 5, 6))  # same channel product, wrong variable/month axes
    branch = ExogenousRefiner(4, 1, n_weather=2, n_timesteps=4, hidden=2)
    with pytest.raises(ValueError, match="spatial grid"):
        branch(torch.zeros(1, 1, 4, 5, 6), torch.zeros(1, 1, 1, 1),
               weather=torch.zeros(1, 2, 4, 1, 1))


def test_legacy_weather_checkpoint_fails_instead_of_silent_partial_loading(tmp_path):
    from farm_us.training.run import load_init_weights

    lm = _lm(n_weather=1, exogenous_refiner=True)
    state = lm.state_dict()
    # Old MLP weights have the same names but 2-D Linear shapes.
    for key in list(state):
        if "weather_enc" in key and key.endswith("weight"):
            state[key] = state[key].squeeze(-1).squeeze(-1)
    checkpoint = tmp_path / "legacy_weather.ckpt"
    torch.save({"state_dict": state}, checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_init_weights(lm, str(checkpoint))
