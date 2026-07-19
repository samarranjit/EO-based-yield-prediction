import pytest
import torch

from farm_us.config import ModelConfig
from farm_us.models.farm_model import FarmModel


@pytest.mark.parametrize("T", [4, 8])
def test_farm_model_forward_main_and_aux(T):
    cfg = ModelConfig()
    m = FarmModel(cfg, n_timesteps=T, chip_size=224, use_dummy=True, dummy_embed_dim=32)
    m.train()
    x = torch.randn(2, 6, T, 224, 224)
    out = m(x, torch.zeros(2, T, 2), torch.zeros(2, 2))
    assert out["main"].shape == (2, 1, 224, 224)
    assert out["aux"].shape == (2, 1, 224, 224)


def test_farm_model_backward_runs():
    cfg = ModelConfig()
    m = FarmModel(cfg, n_timesteps=8, use_dummy=True, dummy_embed_dim=32)
    m.train()
    x = torch.randn(2, 6, 8, 224, 224)
    out = m(x, torch.zeros(2, 8, 2), torch.zeros(2, 2))
    (out["main"].mean() + out["aux"].mean()).backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert any(grads)


def test_aux_disabled_config():
    cfg = ModelConfig()
    cfg.use_auxiliary = False
    m = FarmModel(cfg, n_timesteps=8, use_dummy=True, dummy_embed_dim=32)
    out = m(torch.randn(2, 6, 8, 224, 224), torch.zeros(2, 8, 2), torch.zeros(2, 2))
    assert out["aux"] is None


def test_aux_block_must_be_selected():
    cfg = ModelConfig()
    cfg.aux_block_one_based = 20  # not in [8,16,24,32]
    with pytest.raises(ValueError):
        FarmModel(cfg, n_timesteps=8, use_dummy=True, dummy_embed_dim=32)


def test_inference_can_skip_aux():
    cfg = ModelConfig()
    m = FarmModel(cfg, n_timesteps=8, use_dummy=True, dummy_embed_dim=32)
    m.eval()
    out = m(torch.randn(2, 6, 8, 224, 224), torch.zeros(2, 8, 2), torch.zeros(2, 2), return_aux=False)
    assert out["aux"] is None
