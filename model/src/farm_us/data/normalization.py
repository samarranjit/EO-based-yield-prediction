"""Train-fold-only normalization statistics (streaming).

Two input-normalization modes:
  - ``fold_training_statistics`` (default): per-band mean/std computed from the
    TRAINING years of the current LOYO fold only, via a streaming accumulator so
    statewide rasters never need to fit in memory.
  - ``official_prithvi_statistics``: the fixed Prithvi pretraining stats.

Target scaling (z-score / robust) statistics are ALSO computed from training
years only. Never touch validation or test-year pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import PRITHVI_BAND_MEAN, PRITHVI_BAND_STD
from ..training.metrics import TargetScaler


@dataclass
class StreamingBandStats:
    """Welford-style streaming per-band mean/std over [C, ...] arrays."""

    n_bands: int
    count: np.ndarray = field(default=None)  # type: ignore[assignment]
    mean: np.ndarray = field(default=None)  # type: ignore[assignment]
    m2: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        self.count = np.zeros(self.n_bands, dtype=np.float64)
        self.mean = np.zeros(self.n_bands, dtype=np.float64)
        self.m2 = np.zeros(self.n_bands, dtype=np.float64)

    def update(self, values_per_band: list[np.ndarray]) -> None:
        """``values_per_band[b]`` = 1-D array of valid reflectance values."""
        for b, vals in enumerate(values_per_band):
            vals = np.asarray(vals, dtype=np.float64).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            # Chan et al. parallel variance merge
            n_a = self.count[b]
            n_b = vals.size
            mean_b = vals.mean()
            m2_b = ((vals - mean_b) ** 2).sum()
            delta = mean_b - self.mean[b]
            tot = n_a + n_b
            self.mean[b] += delta * n_b / tot
            self.m2[b] += m2_b + delta**2 * n_a * n_b / tot
            self.count[b] = tot

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        std = np.sqrt(np.where(self.count > 1, self.m2 / np.maximum(self.count, 1), 1.0))
        std = np.where(std > 1e-6, std, 1.0)
        return self.mean.copy(), std


@dataclass
class NormStats:
    band_mean: list[float]
    band_std: list[float]
    target: dict
    mode: str
    train_years: list[int]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> NormStats:
        return cls(**json.loads(Path(path).read_text()))

    def target_scaler(self) -> TargetScaler:
        return TargetScaler.from_dict(self.target)


def find_norm_stats(checkpoint_path: str | Path) -> Path | None:
    """Locate the ``norm_stats.json`` belonging to a checkpoint.

    Searches upward from the checkpoint rather than assuming a fixed depth:
    ``train_fold`` writes the file at the run root, but checkpoints are often
    filed into subdirectories (``checkpoints/``, and in practice further
    subfolders per experiment variant), so a hard-coded ``parent.parent`` breaks
    as soon as anyone reorganises them.
    """
    start = Path(checkpoint_path).resolve()
    for directory in start.parents:
        candidate = directory / "norm_stats.json"
        if candidate.exists():
            return candidate
    return None


def official_prithvi_stats(train_years: list[int], target_scaler: TargetScaler) -> NormStats:
    return NormStats(
        band_mean=list(PRITHVI_BAND_MEAN),
        band_std=list(PRITHVI_BAND_STD),
        target=target_scaler.to_dict(),
        mode="official_prithvi_statistics",
        train_years=sorted(train_years),
    )


def target_scaler_from_values(values: np.ndarray, mode: str = "zscore") -> TargetScaler:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if mode == "none" or v.size == 0:
        return TargetScaler(mode="none")
    if mode == "robust":
        med = float(np.median(v))
        iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        return TargetScaler(mode="robust", center=med, scale=iqr if iqr > 0 else 1.0)
    return TargetScaler(mode="zscore", center=float(v.mean()), scale=float(v.std() or 1.0))


def normalize_image(image: np.ndarray, band_mean, band_std) -> np.ndarray:
    """``image`` [C, T, H, W]; z-score each band across all T/H/W."""
    mean = np.asarray(band_mean, dtype=image.dtype).reshape(-1, 1, 1, 1)
    std = np.asarray(band_std, dtype=image.dtype).reshape(-1, 1, 1, 1)
    return (image - mean) / std
