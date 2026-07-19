import pytest
import torch

from farm_us.models.feature_extractor import one_based_to_index, tokens_to_feature_map
from farm_us.models.prithvi_adapter import PrithviAdapter


def test_block_index_mapping_is_one_based():
    # paper blocks 8,16,24,32 on a 32-block net → 0-based 7,15,23,31
    assert one_based_to_index([8, 16, 24, 32], 32) == [7, 15, 23, 31]
    # block 32 maps to the FINAL block (index 31), never 32
    assert one_based_to_index([32], 32)[0] == 31


def test_block_index_out_of_range():
    with pytest.raises(ValueError):
        one_based_to_index([33], 32)
    with pytest.raises(ValueError):
        one_based_to_index([0], 32)


def test_tokens_to_feature_map_drops_cls_and_reshapes():
    B, T, grid, D = 2, 8, 16, 12
    n = 1 + T * grid * grid
    tokens = torch.randn(B, n, D)
    f = tokens_to_feature_map(tokens, n_timesteps=T, has_cls_token=True)
    assert f.shape == (B, D, T, grid, grid)


def test_tokens_reshape_preserves_thw_order():
    # build tokens where value encodes (t,h,w) so we can verify layout
    _B, T, grid, _D = 1, 2, 3, 1
    idx = torch.arange(T * grid * grid).float().reshape(1, -1, 1)
    cls = torch.zeros(1, 1, 1)
    tokens = torch.cat([cls, idx], dim=1)
    f = tokens_to_feature_map(tokens, n_timesteps=T, has_cls_token=True)
    # first patch token (value 0) should land at t=0,h=0,w=0
    assert f[0, 0, 0, 0, 0] == 0
    # value 1 → t0,h0,w1 ; value grid → t0,h1,w0
    assert f[0, 0, 0, 0, 1] == 1
    assert f[0, 0, 0, 1, 0] == grid


@pytest.mark.parametrize("T", [4, 8])
def test_dummy_adapter_extracts_selected_blocks(T):
    ad = PrithviAdapter(use_dummy=True, dummy_embed_dim=16, n_timesteps=T,
                        out_blocks_one_based=[8, 16, 24, 32], depth=32)
    assert ad.out_indices == [7, 15, 23, 31]
    x = torch.randn(2, 6, T, 224, 224)
    feats = ad(x, torch.zeros(2, T, 2), torch.zeros(2, 2))
    assert len(feats) == 4
    for f in feats:
        assert f.shape == (2, 16, T, 16, 16)
