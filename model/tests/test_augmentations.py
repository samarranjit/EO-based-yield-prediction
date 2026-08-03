import numpy as np

from farm_us.config import AugmentConfig
from farm_us.data.transforms import AlignedAugment


def _sample():
    rng = np.random.default_rng(0)
    img = rng.random((6, 8, 10, 10)).astype(np.float32)
    label = rng.random((10, 10)).astype(np.float32)
    mask = (rng.random((10, 10)) > 0.5)
    return {"image": img.copy(), "label": label.copy(), "crop_mask": mask.copy()}


def test_hflip_aligned_image_and_label():
    cfg = AugmentConfig(hflip_p=1.0, vflip_p=0.0, noise_p=0.0)
    s0 = _sample()
    expect_img = np.flip(s0["image"], axis=-1)
    expect_lab = np.flip(s0["label"], axis=1)
    out = AlignedAugment(cfg, rng=np.random.default_rng(1))(_sample())
    assert np.allclose(out["image"], expect_img)
    assert np.allclose(out["label"], expect_lab)


def test_vflip_aligned():
    cfg = AugmentConfig(hflip_p=0.0, vflip_p=1.0, noise_p=0.0)
    s0 = _sample()
    expect_img = np.flip(s0["image"], axis=-2)
    expect_lab = np.flip(s0["label"], axis=0)
    out = AlignedAugment(cfg, rng=np.random.default_rng(1))(_sample())
    assert np.allclose(out["image"], expect_img)
    assert np.allclose(out["label"], expect_lab)


def test_all_timesteps_flipped_identically():
    cfg = AugmentConfig(hflip_p=1.0, vflip_p=0.0, noise_p=0.0)
    s = _sample()
    # make each timestep identical so misaligned flips would show
    s["image"][:] = s["image"][:, :1]
    out = AlignedAugment(cfg, rng=np.random.default_rng(2))(s)
    for t in range(1, 8):
        assert np.allclose(out["image"][:, t], out["image"][:, 0])


def test_noise_not_applied_to_label_or_mask():
    cfg = AugmentConfig(hflip_p=0.0, vflip_p=0.0, noise_p=1.0, noise_std=0.5)
    s0 = _sample()
    out = AlignedAugment(cfg, rng=np.random.default_rng(3))(_sample())
    assert np.allclose(out["label"], s0["label"])  # label untouched
    assert not np.allclose(out["image"], s0["image"])  # image noised


# --------------------------------------------------------------------------- #
# Production-shape regression tests.
#
# FarmChipDataset.__getitem__ emits the label as [1, H, W] (channel-first) while
# the masks stay [H, W]. The tests above use a 2-D label, so they passed even
# while a positive-axis flip was mis-flipping the real [1, H, W] label -- the
# label and its masks pointed at different parts of the chip, and the masked
# loss trained against nodata. These pin the real shapes.
# --------------------------------------------------------------------------- #

def _production_sample(h: int = 10, w: int = 10):
    """Mirror FarmChipDataset.__getitem__: label [1,H,W], masks [H,W]."""
    rng = np.random.default_rng(0)
    img = rng.random((6, 8, h, w)).astype(np.float32)
    label = rng.random((1, h, w)).astype(np.float32)  # <- leading channel axis
    return {
        "image": img,
        "label": label,
        "crop_mask": (rng.random((h, w)) > 0.5),
        "label_mask": (rng.random((h, w)) > 0.5),
        "valid_mask": (rng.random((h, w)) > 0.5),
    }


def test_channel_first_label_flips_on_spatial_axes_only():
    """h-flip must mirror width and v-flip must mirror height for a [1,H,W] label."""
    for hflip, vflip, axis in [(1.0, 0.0, -1), (0.0, 1.0, -2)]:
        cfg = AugmentConfig(hflip_p=hflip, vflip_p=vflip, noise_p=0.0)
        s0 = _production_sample()
        expected = np.flip(s0["label"], axis=axis)
        out = AlignedAugment(cfg, rng=np.random.default_rng(1))(_production_sample())
        assert out["label"].shape == s0["label"].shape
        assert np.allclose(out["label"], expected), f"label mis-flipped for axis {axis}"


def test_label_and_masks_stay_pixelwise_aligned_under_flips():
    """The core invariant: a label value and its mask must move together.

    Encodes each pixel's position in BOTH the label and the mask, then checks
    they still agree after augmentation. Under the axis bug the label mirrors on
    a different axis than the masks, so these disagree everywhere off-diagonal.
    """
    h = w = 10
    for hflip, vflip in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        cfg = AugmentConfig(hflip_p=hflip, vflip_p=vflip, noise_p=0.0)
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        position_id = (rows * w + cols).astype(np.float32)

        sample = {
            "image": np.zeros((6, 8, h, w), dtype=np.float32),
            "label": position_id[None].copy(),          # [1,H,W]
            "crop_mask": position_id.copy(),            # [H,W], same encoding
            "label_mask": np.ones((h, w), dtype=bool),
            "valid_mask": np.ones((h, w), dtype=bool),
        }
        out = AlignedAugment(cfg, rng=np.random.default_rng(7))(sample)
        assert np.array_equal(out["label"][0], out["crop_mask"]), (
            f"label/mask desynchronised for hflip={hflip} vflip={vflip}"
        )


def test_image_and_label_stay_aligned_under_flips():
    """Label must track the imagery it supervises, not just the masks."""
    h = w = 10
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    position_id = (rows * w + cols).astype(np.float32)
    for hflip, vflip in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        cfg = AugmentConfig(hflip_p=hflip, vflip_p=vflip, noise_p=0.0)
        sample = {
            "image": np.broadcast_to(position_id, (6, 8, h, w)).copy(),
            "label": position_id[None].copy(),
            "crop_mask": np.ones((h, w), dtype=bool),
        }
        out = AlignedAugment(cfg, rng=np.random.default_rng(11))(sample)
        assert np.array_equal(out["image"][0, 0], out["label"][0]), (
            f"image/label desynchronised for hflip={hflip} vflip={vflip}"
        )
