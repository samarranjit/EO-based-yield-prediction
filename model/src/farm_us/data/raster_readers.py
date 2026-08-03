"""Raster reader adapters with a common interface.

All readers expose ``read_chip(state, year, window) -> ChipArrays`` and use
**windowed** reads so statewide rasters never load fully into RAM.

Supported storage patterns (each its own adapter, one shared interface):
  1. ``GeotiffMonthlyReader``  — one 6-band GeoTIFF per (state, year, month).
  2. ``StateYearStackReader``  — one raster stack per (state, year) with all
     month×band planes.
  3. ``ChipNpzReader``         — pre-extracted chips in NPZ/Zarr.
  4. ``ManifestCogReader``     — a manifest of cloud-optimized GeoTIFFs.

Only #1 has a full windowed implementation here; the others declare the same
interface with clear extension points (the real imagery is not yet on disk —
see docs/DATA_INVENTORY.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rasterio.windows import Window

from ..config import DataConfig
from ..utils.geospatial import ChipWindow
from ..utils.logging import DataContractError, get_logger

logger = get_logger(__name__)


@dataclass
class ChipArrays:
    image: np.ndarray          # [C, T, H, W] reflectance (scaled), np.nan for missing
    month_valid: np.ndarray    # [T, H, W] bool per-month validity
    label: np.ndarray          # [H, W] float (nodata → nan)
    crop_mask: np.ndarray      # [H, W] bool
    county_id: np.ndarray | None  # [H, W] int or None
    transform: object
    crs: str
    bounds: tuple[float, float, float, float]


class RasterReader(Protocol):
    def raster_size(self, state: str, year: int) -> tuple[int, int]: ...
    def read_chip(self, state: str, year: int, window: ChipWindow) -> ChipArrays: ...


def _aligned_window(ds, bounds: tuple[float, float, float, float], name: str = ""):
    """Window into ``ds`` covering ``bounds``, for grids that share CRS + resolution.

    The label rasters emitted by data_preparation do NOT sit on the same grid as
    the CDL/HLS rasters. For Maryland the label origin is one 30 m pixel east of
    the CDL/HLS origin and its extent is smaller, so reading all three with the
    *same pixel window* silently pairs each label with imagery ~30 m away.
    Deriving each raster's own window from shared geographic bounds removes that
    skew exactly, with no resampling, as long as the grids differ by a whole
    number of pixels -- which is asserted here rather than assumed.
    """
    from rasterio.windows import from_bounds

    w = from_bounds(*bounds, transform=ds.transform)
    col_off, row_off = round(w.col_off), round(w.row_off)
    width, height = round(w.width), round(w.height)
    if abs(w.col_off - col_off) > 1e-6 or abs(w.row_off - row_off) > 1e-6:
        raise DataContractError(
            f"{name or ds.name}: grid is offset from the label grid by a fractional "
            f"pixel ({w.col_off - col_off:+.6f}, {w.row_off - row_off:+.6f}). "
            "Whole-pixel alignment is required; regenerate the rasters on a common "
            "grid or add explicit resampling."
        )
    return Window(col_off, row_off, width, height)


class GeotiffMonthlyReader:
    """One six-band GeoTIFF per (state, year, month) + label + CDL rasters.

    Expected layout (configurable via DataConfig roots):
        {imagery_root}/{year}/hls_{crop}_{STATE}_{YEAR}_{MON}.tif   (6 bands)
        {label_root}/{year}/nass_{crop}_yield_{STATE}_{YEAR}_{variant}_30m_{crop}_only.tif
        {cdl_root}/cdl_{crop}_{STATE}_{YEAR}.tif
    """

    MONTHS = ("APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV")

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.crop = cfg.crop.lower()

    # --- path helpers --- #
    def _img_path(self, state: str, year: int, mon: str) -> Path:
        return Path(self.cfg.imagery_root) / str(year) / f"hls_{self.crop}_{state}_{year}_{mon}.tif"

    def _label_path(self, state: str, year: int) -> Path:
        return (
            Path(self.cfg.label_root)
            / str(year)
            / f"nass_{self.crop}_yield_{state}_{year}_{self.cfg.label_variant}_30m_{self.crop}_only.tif"
        )

    def _cdl_path(self, state: str, year: int) -> Path:
        return Path(self.cfg.cdl_root) / f"cdl_{self.crop}_{state}_{year}.tif"

    def raster_size(self, state: str, year: int) -> tuple[int, int]:
        import rasterio

        with rasterio.open(self._label_path(state, year)) as ds:
            return ds.height, ds.width

    def read_chip(self, state: str, year: int, window: ChipWindow) -> ChipArrays:
        import rasterio
        from rasterio.windows import Window

        rw = Window(window.col_off, window.row_off, window.width, window.height)
        n_bands = len(self.cfg.band_order)
        T = self.cfg.n_timesteps

        # The label raster defines the canonical chip grid: manifest.py enumerates
        # windows from its height/width, so `window` is in LABEL pixel coordinates.
        # Read it first and use its geographic bounds to place every other raster
        # (see _aligned_window for why the same pixel window is not reusable).
        with rasterio.open(self._label_path(state, year)) as ds:
            lnod = ds.nodata if ds.nodata is not None else self.cfg.nodata
            label = ds.read(1, window=rw, boundless=True, fill_value=lnod).astype(np.float32)
            label[label == lnod] = np.nan
            transform = ds.window_transform(rw)
            crs = str(ds.crs)
            bounds = tuple(rasterio.windows.bounds(rw, ds.transform))  # type: ignore

        img = np.full((n_bands, T, window.height, window.width), np.nan, dtype=np.float32)
        month_valid = np.zeros((T, window.height, window.width), dtype=bool)

        for t, mon in enumerate(self.MONTHS[:T]):
            p = self._img_path(state, year, mon)
            if not p.exists():
                continue
            with rasterio.open(p) as ds:
                if ds.count < n_bands:
                    raise DataContractError(f"{p} has {ds.count} bands, expected {n_bands}")
                nod = ds.nodata if ds.nodata is not None else self.cfg.hls_nodata
                iw = _aligned_window(ds, bounds, str(p))
                arr = ds.read(window=iw, boundless=True, fill_value=nod).astype(np.float32)
                valid = np.all(arr != nod, axis=0) & np.all(np.isfinite(arr), axis=0)
                arr = arr * self.cfg.hls_scale
                arr[:, ~valid] = np.nan
                img[:, t] = arr[:n_bands]
                month_valid[t] = valid

        # crop mask (fill 0 = "not crop" outside the CDL footprint)
        with rasterio.open(self._cdl_path(state, year)) as ds:
            cw = _aligned_window(ds, bounds, str(self._cdl_path(state, year)))
            cdl = ds.read(1, window=cw, boundless=True, fill_value=0).astype(np.float32)
        crop_mask = cdl > 0.5

        return ChipArrays(
            image=img, month_valid=month_valid, label=label, crop_mask=crop_mask,
            county_id=None, transform=transform, crs=crs, bounds=bounds,  # type: ignore[arg-type]
        )


def build_reader(cfg: DataConfig) -> RasterReader:
    readers = {
        "geotiff_monthly": GeotiffMonthlyReader,
    }
    if cfg.reader not in readers:
        raise NotImplementedError(
            f"Reader {cfg.reader!r} is declared in the interface but not yet "
            f"implemented (imagery not on disk). Available: {list(readers)}"
        )
    return readers[cfg.reader](cfg)
