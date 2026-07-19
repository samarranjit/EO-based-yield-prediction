import numpy as np

from farm_us.evaluation.mosaic import MosaicAccumulator
from farm_us.utils.geospatial import cosine_blend_weights, tile_windows


def test_tile_windows_cover_and_are_full_size():
    ws = tile_windows(500, 500, chip=224, stride=224)
    for w in ws:
        assert w.height == 224 and w.width == 224
        assert w.row_off + 224 <= 500 and w.col_off + 224 <= 500


def test_mosaic_reconstructs_constant_field():
    H, W, chip, stride = 320, 320, 128, 64
    truth = np.full((H, W), 7.0, dtype=np.float32)
    acc = MosaicAccumulator(H, W)
    for win in tile_windows(H, W, chip, stride):
        rs, cs = win.as_slices()
        tile = truth[rs, cs]
        acc.add(tile, win)
    out, wsum = acc.finalize()
    assert np.allclose(out, 7.0, atol=1e-3)
    assert (wsum > 0).all()


def test_mosaic_reconstructs_gradient_with_blending():
    H, W, chip, stride = 256, 256, 128, 64
    yy, xx = np.mgrid[0:H, 0:W]
    truth = (yy + xx).astype(np.float32)
    acc = MosaicAccumulator(H, W)
    for win in tile_windows(H, W, chip, stride):
        rs, cs = win.as_slices()
        acc.add(truth[rs, cs], win)
    out, _ = acc.finalize()
    assert np.allclose(out, truth, atol=1e-2)


def test_cosine_weights_positive():
    w = cosine_blend_weights(64)
    assert w.shape == (64, 64)
    assert (w > 0).all()
