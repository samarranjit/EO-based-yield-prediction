"""Integration test for the REAL Prithvi-EO-2.0-600M-TL backbone.

Deselected by default (`-m "not integration"`). Run with the [prithvi] extra
installed and network access:  uv run --extra prithvi pytest -m integration
"""

import pytest
import torch

pytestmark = pytest.mark.integration


def test_real_prithvi_t8_forward_and_feature_shapes():
    from farm_us.models.prithvi_adapter import BackboneNotAvailable, PrithviAdapter

    try:
        ad = PrithviAdapter(
            backbone_id="ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
            pretrained=True, n_timesteps=8, out_blocks_one_based=[8, 16, 24, 32],
            use_dummy=False, expected_embed_dim=1280, depth=32,
        )
    except BackboneNotAvailable as e:
        pytest.skip(f"TerraTorch/Prithvi not available: {e}")

    assert ad.embed_dim == 1280
    assert ad.out_indices == [7, 15, 23, 31]
    x = torch.randn(1, 6, 8, 224, 224)
    tc = torch.zeros(1, 8, 2)
    lc = torch.zeros(1, 2)
    feats = ad(x, tc, lc)
    assert len(feats) == 4
    for f in feats:
        assert f.shape[0] == 1 and f.shape[1] == 1280 and f.shape[2] == 8
        assert f.shape[3] == 16 and f.shape[4] == 16


def test_real_farm_model_forward():
    from farm_us.config import ModelConfig
    from farm_us.models.farm_model import FarmModel
    from farm_us.models.prithvi_adapter import BackboneNotAvailable

    cfg = ModelConfig()
    try:
        m = FarmModel(cfg, n_timesteps=8, use_dummy=False)
    except BackboneNotAvailable as e:
        pytest.skip(f"TerraTorch/Prithvi not available: {e}")
    # batch=1 is an inference-shaped operating point: the PPM's global-pool bin
    # (spatial 1x1) + BatchNorm is undefined at batch=1 in train mode. Training
    # uses batch>=2 (paper: 8); inference uses eval() with running stats.
    m.eval()
    out = m(torch.randn(1, 6, 8, 224, 224), torch.zeros(1, 8, 2), torch.zeros(1, 2))
    assert out["main"].shape == (1, 1, 224, 224)
