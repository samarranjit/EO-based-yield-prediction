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
