import pytest
import torch

from farm_us.models.ppm import PyramidPoolingModule
from farm_us.models.temporal_reducer import TemporalFeatureReducer
from farm_us.models.upernet import PaperFaithfulUPerNetDecoder, SmallUPerNetDecoder


@pytest.mark.parametrize("mode", ["mean", "attention", "flatten_time"])
@pytest.mark.parametrize("T", [4, 8])
def test_temporal_reducer_shapes(mode, T):
    r = TemporalFeatureReducer(embed_dim=32, n_timesteps=T, mode=mode)
    x = torch.randn(2, 32, T, 16, 16)
    y = r(x)
    assert y.shape == (2, 32, 16, 16)


def test_ppm_preserves_spatial():
    ppm = PyramidPoolingModule(64, 64, bins=[1, 2, 3, 6])
    x = torch.randn(2, 64, 16, 16)
    assert ppm(x).shape == (2, 64, 16, 16)


@pytest.mark.parametrize("T", [4, 8])
def test_upernet_decoder_output(T):
    dec = PaperFaithfulUPerNetDecoder(encoder_dim=32, decoder_channels=48, n_levels=4)
    feats = [torch.randn(2, 32, 16, 16) for _ in range(4)]
    out = dec(feats)
    assert out.shape == (2, 48, 16, 16)


def test_small_upernet_decoder():
    dec = SmallUPerNetDecoder(encoder_dim=32, channels=24)
    out = dec(torch.randn(2, 32, 16, 16))
    assert out.shape == (2, 24, 16, 16)
