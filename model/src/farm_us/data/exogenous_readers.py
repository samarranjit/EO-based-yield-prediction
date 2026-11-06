"""Readers for the soil (POLARIS) and weather (PRISM) covariates.

Kept separate from raster_readers.py deliberately: GeotiffMonthlyReader is the
battle-tested HLS/label/CDL path used by every existing experiment, and this
module only ever ADDS to what the dataset builds -- it never touches that
reader's ``read_chip`` or its output contract.

Resolution, stated with numbers
-------------------------------
A chip is 224 px x 30 m = 6.72 km on a side.

* POLARIS is 1 arc-second: ~31 m N-S x ~23 m E-W at 42 N -- genuinely pixel
  resolution. It is reprojected onto the chip's own grid (bilinear) and enters
  the model as a [S, H, W] raster.
* PRISM "800 m" is 30 arc-seconds. Each monthly field is reprojected with
  nearest-neighbor sampling to [W,T,H,W], preserving distinct source-cell
  values across a chip. No averaging over the chip and no smoothing are used.
  Repeated values on the 30 m grid are not independent 30 m observations.

Everything returned here is RAW, with NaN for missing or nodata
----------------------------------------------------------------
Normalization statistics must be computed from these raw values, over finite
values only (data/normalization.py:ExogenousStatsAccumulator). Filling NaN is a
model-input concern and happens afterwards, in :func:`build_model_inputs`.
Filling first and computing statistics second silently biases them: the values
[10, missing, 20] would give a mean of 10 instead of 15.

File handles: opened per read, closed immediately
--------------------------------------------------
Measured on this machine: opening the 3 GB soil mosaic costs ~0.5 ms and a
PRISM month ~0.8 ms, against ~7 s of HLS reading per chip. Caching handles buys
nothing measurable, and a cached handle opened in the main process (e.g. by the
normalization pass) would be inherited by every forked DataLoader worker --
exactly what the "GDAL is not fork-safe" invariant forbids. So, like
GeotiffMonthlyReader, nothing here keeps a dataset open between calls: these
objects hold only paths and variable names, and are safe to construct anywhere,
pickle, and fork.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import DataConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

#: Month numbers for the 8 default timesteps (Apr..Nov), matching
#: config.DEFAULT_TIMESTEPS and GeotiffMonthlyReader.MONTHS exactly -- weather
#: timestep t must be the same calendar month as image timestep t, or the two
#: temporal axes silently desynchronise.
TIMESTEP_MONTHS: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11)


def soil_path(root: str | Path, var: str) -> Path:
    """``{soil_root}/{var}_0-30_weighted_average.tif`` -- the 0-30 cm summary
    from data_preparation/scripts/08, merged into one mosaic per property."""
    return Path(root) / f"{var}_0-30_weighted_average.tif"


def weather_path(root: str | Path, var: str, year: int, month: int) -> Path:
    """``{weather_root}/{var}/prism_{var}_{YYYYMM}.tif`` -- the layout
    data_preparation/scripts/06 (and 09 for derived vpdmean) writes."""
    return Path(root) / var / f"prism_{var}_{year}{month:02d}.tif"


# --------------------------------------------------------------------------- #
# Pre-flight: every required file must exist before any expensive pass starts
# --------------------------------------------------------------------------- #

def missing_exogenous_files(cfg: DataConfig, years) -> list[Path]:
    """Every configured soil/weather file that does not exist, for ``years``."""
    missing: list[Path] = []
    for var in cfg.soil_vars:
        p = soil_path(cfg.soil_root, var)
        if not p.exists():
            missing.append(p)
    months = TIMESTEP_MONTHS[: cfg.n_timesteps]
    for var in cfg.weather_vars:
        for year in years:
            for month in months:
                p = weather_path(cfg.weather_root, var, int(year), month)
                if not p.exists():
                    missing.append(p)
    return missing


def check_exogenous_files(cfg: DataConfig, years) -> None:
    """Raise if any configured soil/weather input is missing for any year.

    A missing weather month would otherwise reach the model as NaN -> 0 (mean
    imputation) for every chip of that year, which trains fine and looks fine
    and is quietly wrong. Refusing up front is cheaper than discovering it from
    a coverage report at the end of a multi-hour statistics pass.
    """
    missing = missing_exogenous_files(cfg, years)
    if missing:
        shown = "\n  ".join(str(p) for p in missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} exogenous input file(s) required by data.soil_vars / "
            f"data.weather_vars for years {sorted(int(y) for y in years)} do not exist:\n  "
            f"{shown}{more}\nComplete them (data_preparation scripts 06-09: e.g. "
            f"`09_preprocess_prism_monthly.py --derive-only` for vpdmean) or remove "
            f"the variable from the config."
        )


# --------------------------------------------------------------------------- #
# Low-level: reproject one raster onto a chip grid
# --------------------------------------------------------------------------- #

def _reproject_to_chip(path: Path, dst_transform, dst_crs, height: int, width: int,
                       resampling) -> np.ndarray:
    """Read ``path`` onto the chip grid; NaN where the source has no data.

    rasterio.warp.reproject reads only the source blocks the destination needs,
    so this never materialises a multi-GB mosaic. The dataset is closed before
    returning -- see the module docstring on handles.
    """
    import rasterio
    from rasterio.warp import reproject

    dst = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def chip_weighted_mean(field: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted mean of the finite pixels of ``field``.

    Falls back to the unweighted mean of finite pixels when the weights select
    nothing finite (a chip with no crop pixels), and to NaN when nothing is
    finite at all -- NaN, not 0, so the statistics pass counts it as missing.
    """
    finite = np.isfinite(field)
    if weights is not None:
        w = np.where(finite, np.asarray(weights, dtype=np.float64), 0.0)
        total = w.sum()
        if total > 0:
            return float((np.where(finite, field, 0.0) * w).sum() / total)
    if finite.any():
        return float(field[finite].mean())
    return float("nan")


# --------------------------------------------------------------------------- #
# Soil
# --------------------------------------------------------------------------- #

class SoilReader:
    """[S, H, W] soil chip, reprojected onto the caller's grid (static -- no year)."""

    def __init__(self, cfg: DataConfig) -> None:
        self.root = Path(cfg.soil_root)
        self.vars = list(cfg.soil_vars)

    def read_chip(self, dst_transform, dst_crs, height: int, width: int) -> np.ndarray:
        """``dst_transform``/``dst_crs`` are the CHIP's own (label-grid) transform
        and CRS, as returned by GeotiffMonthlyReader -- EPSG:5070 at 30 m, not the
        mosaic's native EPSG:4326. Bilinear: soil properties are continuous.
        Returns float32 [S, H, W], NaN where the mosaic has no data."""
        from rasterio.enums import Resampling

        out = np.full((len(self.vars), height, width), np.nan, dtype=np.float32)
        for i, var in enumerate(self.vars):
            p = soil_path(self.root, var)
            if not p.exists():
                raise FileNotFoundError(
                    f"soil variable {var!r} is configured in data.soil_vars but {p} "
                    f"does not exist. Run data_preparation scripts 07/08 and merge first."
                )
            out[i] = _reproject_to_chip(p, dst_transform, dst_crs, height, width,
                                        Resampling.bilinear)
        return out


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #

class WeatherSampler:
    """[W,T,H,W] monthly weather maps, nearest-sampled onto the chip grid."""

    def __init__(self, cfg: DataConfig) -> None:
        self.root = Path(cfg.weather_root)
        self.vars = list(cfg.weather_vars)
        self._warned_missing: set[tuple[str, int, int]] = set()

    def read_field(self, var: str, year: int, month: int, dst_transform, dst_crs,
                   height: int, width: int) -> np.ndarray | None:
        """One month of one variable on the chip grid, or None if the file is absent.

        Nearest-neighbor preserves source-cell values without blending adjacent
        cells. Nodata remains NaN until model-input standardization.
        """
        from rasterio.enums import Resampling

        p = weather_path(self.root, var, year, month)
        if not p.exists():
            key = (var, year, month)
            if key not in self._warned_missing:
                logger.warning("weather: no file for %s %d-%02d (%s) -- treated as missing",
                               var, year, month, p)
                self._warned_missing.add(key)
            return None
        return _reproject_to_chip(p, dst_transform, dst_crs, height, width, Resampling.nearest)

    def sample_chip(self, year: int, dst_transform, dst_crs, height: int, width: int,
                    n_timesteps: int) -> np.ndarray:
        """float32 [W,T,H,W]; missing cells/months stay NaN, never chip-averaged."""
        if not 1 <= n_timesteps <= len(TIMESTEP_MONTHS):
            raise ValueError("weather n_timesteps must select 1..8 months from April onward")
        months = TIMESTEP_MONTHS[:n_timesteps]
        out = np.full((len(self.vars), n_timesteps, height, width), np.nan, dtype=np.float32)
        for vi, var in enumerate(self.vars):
            for ti, month in enumerate(months):
                field = self.read_field(var, year, month, dst_transform, dst_crs, height, width)
                if field is not None:
                    out[vi, ti] = field
        return out


# --------------------------------------------------------------------------- #
# Shared by training (data/dataset.py) and raster inference (evaluation/inference.py)
# --------------------------------------------------------------------------- #

def read_exogenous_raw(cfg: DataConfig, year: int, transform, crs, crop_mask: np.ndarray,
                       soil_reader: SoilReader | None,
                       weather_sampler: WeatherSampler | None) -> dict:
    """RAW, NaN-preserving ``{"soil": [S,H,W] | None, "weather": [W,T,H,W] | None}``.

    The single place a chip's exogenous inputs are read. FarmChipDataset and
    evaluation/inference.py both call this, so training and raster prediction
    cannot drift apart in how soil/weather are sampled.
    """
    h, w = crop_mask.shape
    out: dict = {"soil": None, "weather": None}
    if cfg.soil_vars:
        if soil_reader is None:
            raise ValueError("data.soil_vars is configured but no SoilReader was supplied")
        out["soil"] = soil_reader.read_chip(transform, crs, h, w)
    if cfg.weather_vars:
        if weather_sampler is None:
            raise ValueError("data.weather_vars is configured but no WeatherSampler was supplied")
        out["weather"] = weather_sampler.sample_chip(
            year, transform, crs, h, w, cfg.n_timesteps,
        )
    return out


def require_exogenous_stats(cfg: DataConfig, stats) -> None:
    """Raise unless ``stats`` normalizes exactly the configured soil/weather variables.

    Checks presence, count, and the recorded variable NAMES AND ORDER. A
    count-only check would accept statistics computed for [clay, bd] when the
    config now says [bd, clay] -- each channel would then be scaled by the other
    variable's mean and std, with nothing downstream to notice.
    """
    for group in ("soil", "weather"):
        names = list(getattr(cfg, f"{group}_vars"))
        if not names:
            continue
        mean = getattr(stats, f"{group}_mean", None)
        std = getattr(stats, f"{group}_std", None)
        recorded = getattr(stats, f"{group}_vars", None)
        if mean is None or std is None:
            raise ValueError(
                f"data.{group}_vars={names} is configured, but the normalization "
                f"statistics have no {group}_mean/{group}_std. The model would receive "
                f"UN-normalized {group} values. Recompute statistics (drop "
                f"norm.reuse_stats_from) or use a norm_stats.json that includes them."
            )
        if recorded is None or list(recorded) != names or len(mean) != len(names) or len(std) != len(names):
            raise ValueError(
                f"Normalization statistics describe {group} variables {recorded!r}, but "
                f"data.{group}_vars={names}. Refusing to use them: with different "
                f"variables or a different order, channels would be scaled by another "
                f"variable's statistics."
            )

    if cfg.weather_vars:
        from .normalization import WEATHER_STATS_LAYOUT

        if stats.weather_layout != WEATHER_STATS_LAYOUT:
            raise ValueError(
                "Weather normalization is from legacy chip means or has unknown weather_layout. "
                "Spatial weather requires freshly computed training-pixel statistics. "
                "Initialize the new branch from farm-018, not a chip-mean exogenous checkpoint."
            )


def normalize_generic(arr: np.ndarray, mean, std) -> np.ndarray:
    """z-score each leading-axis channel, then fill NaN with 0.0.

    Shared by soil ([S,H,W]) and weather ([W,T,H,W]). After z-scoring, 0.0 is the
    TRAINING MEAN, so the fill is mean imputation: a missing value contributes a
    neutral input rather than propagating NaN into the loss.

    Only ever apply this with statistics computed from RAW values. Its output is
    a model input, not something to compute statistics from -- see the module
    docstring's [10, missing, 20] example.
    """
    mean = np.asarray(mean, dtype=np.float32).reshape(-1, *([1] * (arr.ndim - 1)))
    std = np.asarray(std, dtype=np.float32).reshape(-1, *([1] * (arr.ndim - 1)))
    std = np.where(std > 1e-6, std, 1.0)
    out = (arr.astype(np.float32) - mean) / std
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def build_model_inputs(cfg: DataConfig, stats, raw: dict) -> dict:
    """Normalized, NaN-filled model inputs ``{"soil": ..., "weather": ...}`` from raw.

    Raises instead of silently falling back to identity normalization when the
    statistics lack a configured group -- the earlier fallback fed raw clay
    percentages (~27) to a branch expecting z-scores (~0).
    """
    require_exogenous_stats(cfg, stats)
    out: dict = {}
    if raw.get("soil") is not None:
        if stats.soil_mean is None or stats.soil_std is None:
            raise ValueError("soil is configured but the normalization statistics have no soil_mean/soil_std")
        out["soil"] = normalize_generic(raw["soil"], stats.soil_mean, stats.soil_std)
    if raw.get("weather") is not None:
        if stats.weather_mean is None or stats.weather_std is None:
            raise ValueError("weather is configured but the normalization statistics have no weather_mean/weather_std")
        out["weather"] = normalize_generic(raw["weather"], stats.weather_mean, stats.weather_std)
    return out
