"""Overlapping-tile mosaicking with smooth cosine blending.

Accumulate per-tile predictions weighted by a 2-D Hann window, plus the weight
sum, then normalize. This removes tile seams from overlapping windowed inference
and is independent of any model (unit-testable on synthetic arrays).
"""

from __future__ import annotations

import numpy as np

from ..utils.geospatial import ChipWindow, cosine_blend_weights


class MosaicAccumulator:
    def __init__(self, height: int, width: int, dtype=np.float32) -> None:
        self.height = height
        self.width = width
        self.acc = np.zeros((height, width), dtype=np.float64)
        self.wsum = np.zeros((height, width), dtype=np.float64)

    def add(self, tile: np.ndarray, window: ChipWindow, weights: np.ndarray | None = None) -> None:
        h, w = tile.shape[-2:]
        if weights is None:
            weights = cosine_blend_weights(h)[:h, :w]
        rs, cs = window.as_slices()
        rs = slice(rs.start, rs.start + h)
        cs = slice(cs.start, cs.start + w)
        self.acc[rs, cs] += np.asarray(tile, dtype=np.float64) * weights
        self.wsum[rs, cs] += weights

    def finalize(self, nodata: float = np.nan) -> tuple[np.ndarray, np.ndarray]:
        out = np.full((self.height, self.width), nodata, dtype=np.float32)
        valid = self.wsum > 1e-6
        out[valid] = (self.acc[valid] / self.wsum[valid]).astype(np.float32)
        return out, self.wsum.astype(np.float32)
