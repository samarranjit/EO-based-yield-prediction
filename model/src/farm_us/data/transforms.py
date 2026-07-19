"""Spatially-consistent augmentation.

The SAME spatial transform is applied to all 8 timesteps, all 6 bands, the yield
map, and every mask. Timesteps are NEVER flipped independently. Gaussian noise
is added AFTER normalization, only to the image, never to labels/masks.

Paper augmentations: H-flip p=0.2, V-flip p=0.2, Gaussian noise p=0.4.

Image tensor layout is ``[C, T, H, W]`` → height axis = -2, width axis = -1.
2-D companion arrays (label/masks/county_id) are ``[H, W]`` → axes (0, 1).
"""

from __future__ import annotations

import numpy as np

from ..config import AugmentConfig

_COMPANIONS = ("label", "crop_mask", "label_mask", "valid_mask", "county_id")


class AlignedAugment:
    def __init__(self, cfg: AugmentConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]
        flip_w = self.rng.random() < self.cfg.hflip_p  # horizontal flip = mirror width
        flip_h = self.rng.random() < self.cfg.vflip_p  # vertical flip = mirror height

        if flip_w:
            img = np.flip(img, axis=-1)
        if flip_h:
            img = np.flip(img, axis=-2)
        img = np.ascontiguousarray(img)

        for k in _COMPANIONS:
            v = sample.get(k)
            if v is None:
                continue
            if flip_w:
                v = np.flip(v, axis=1)
            if flip_h:
                v = np.flip(v, axis=0)
            sample[k] = np.ascontiguousarray(v)

        if self.rng.random() < self.cfg.noise_p:
            noise = self.rng.normal(0.0, self.cfg.noise_std, size=img.shape).astype(img.dtype)
            img = img + noise

        sample["image"] = img
        return sample
