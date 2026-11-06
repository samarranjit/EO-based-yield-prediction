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

WEATHER_STATS_LAYOUT = "spatial_nearest_crop_label_v1"


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

    # Provenance of the statistics pass. These MUST keep defaults: `load` does
    # `cls(**json.loads(...))`, so any field without one would fail to read
    # every norm_stats.json written before they existed -- including the ones
    # sitting beside already-trained checkpoints.
    #: Chips actually read. None means "not recorded" (a pre-existing file).
    n_chips_used: int | None = None
    #: Chips available in the train split. Equal to n_chips_used for a full pass.
    n_chips_total: int | None = None
    #: Seed for the subsample draw; None when no subsampling occurred.
    stats_seed: int | None = None
    #: States the statistics were computed over. None means "not recorded" (a
    #: file written before this field existed) -- reuse can then only be checked
    #: against train_years, which does NOT detect a changed state list.
    states: list[str] | None = None

    #: Soil / weather channel statistics for models/exogenous_refiner.py. None
    #: means "not configured for this run" or "predates this field"; a run that
    #: configures data.soil_vars / weather_vars refuses statistics without them
    #: (data/exogenous_readers.py:require_exogenous_stats). Train-split only, from
    #: RAW values with missing values EXCLUDED (never zero-filled first) -- see
    #: ExogenousStatsAccumulator. Soil is masked to crop & valid-label pixels;
    #: weather is masked to the same pixels, pooled across months.
    soil_mean: list[float] | None = None
    soil_std: list[float] | None = None
    weather_mean: list[float] | None = None
    weather_std: list[float] | None = None
    #: The variable names, IN ORDER, the statistics above describe. Without them
    #: statistics for [clay, bd] would silently be applied to a config saying
    #: [bd, clay]; with them, a mismatch raises.
    soil_vars: list[str] | None = None
    weather_vars: list[str] | None = None
    #: Fraction of values that were finite (not missing/nodata) per variable in
    #: the pass that produced the statistics -- the provenance record for how
    #: much mean-imputation the model's inputs involve.
    soil_coverage: list[float] | None = None
    weather_coverage: list[float] | None = None
    # None identifies legacy chip-mean or unrecorded statistics, never spatial ones.
    weather_layout: str | None = None

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


class ExogenousStatsAccumulator:
    """Streaming soil/weather statistics from RAW values.

    Mean and std are over FINITE values only (StreamingBandStats drops
    non-finite values), so a missing value is excluded rather than counted as
    zero: [10, missing, 20] gives a mean of 15, not 10. Coverage -- the finite
    fraction per variable -- is tracked alongside, so the amount of missing data
    the model will see mean-imputed is recorded instead of hidden.

    Must be fed the raw arrays from data/exogenous_readers.py:read_exogenous_raw
    (NaN-preserving), never the normalized, NaN-filled model inputs.
    """

    def __init__(self, n_soil: int, n_weather: int) -> None:
        self.n_soil, self.n_weather = n_soil, n_weather
        self.soil = StreamingBandStats(n_soil) if n_soil else None
        self.weather = StreamingBandStats(n_weather) if n_weather else None
        self.soil_seen = np.zeros(n_soil); self.soil_finite = np.zeros(n_soil)
        self.weather_seen = np.zeros(n_weather); self.weather_finite = np.zeros(n_weather)

    def update(self, soil_raw: np.ndarray | None = None, weather_raw: np.ndarray | None = None,
               valid: np.ndarray | None = None) -> None:
        """``soil_raw`` [S,H,W] (masked by ``valid`` [H,W] if given); ``weather_raw`` [W,T,H,W], with the same spatial mask."""
        if self.soil is not None and soil_raw is not None:
            m = np.ones(soil_raw.shape[1:], dtype=bool) if valid is None else np.asarray(valid, bool)
            per = [soil_raw[b][m].ravel() for b in range(self.n_soil)]
            for b, v in enumerate(per):
                self.soil_seen[b] += v.size
                self.soil_finite[b] += np.isfinite(v).sum()
            self.soil.update(per)
        if self.weather is not None and weather_raw is not None:
            if weather_raw.ndim != 4 or weather_raw.shape[0] != self.n_weather:
                raise ValueError("raw weather statistics require [W,T,H,W] spatial maps")
            m = (np.ones(weather_raw.shape[-2:], dtype=bool)
                 if valid is None else np.asarray(valid, bool))
            per = [weather_raw[b][:, m].ravel() for b in range(self.n_weather)]
            for b, v in enumerate(per):
                self.weather_seen[b] += v.size
                self.weather_finite[b] += np.isfinite(v).sum()
            self.weather.update(per)

    def result(self) -> dict:
        out: dict = {}
        if self.soil is not None:
            m, s = self.soil.finalize()
            out["soil"] = {"mean": m.tolist(), "std": s.tolist(),
                           "coverage": (self.soil_finite / np.maximum(self.soil_seen, 1)).tolist()}
        if self.weather is not None:
            m, s = self.weather.finalize()
            out["weather"] = {"mean": m.tolist(), "std": s.tolist(),
                              "coverage": (self.weather_finite / np.maximum(self.weather_seen, 1)).tolist()}
        return out


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
