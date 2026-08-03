"""Georeferenced tiled inference over large state-year rasters.

Slides overlapping windows across a statewide raster (windowed reads), runs the
model per tile, de-standardizes, blends with :class:`MosaicAccumulator`, masks
non-crop pixels as no-data, and writes GeoTIFFs preserving CRS/transform/extent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import FarmConfig
from ..data.compositing import location_coords, temporal_coords
from ..data.normalization import NormStats, normalize_image
from ..utils.geospatial import pixel_to_lonlat, tile_windows
from .mosaic import MosaicAccumulator


@torch.no_grad()
def predict_state_year(
    model,
    reader,
    cfg: FarmConfig,
    norm: NormStats,
    state: str,
    year: int,
    device: str = "cpu",
    stride: int | None = None,
) -> dict[str, np.ndarray]:
    """Return dict with 'prediction' [H,W] and 'weight' [H,W] (physical units)."""
    model.eval().to(device)
    scaler = norm.target_scaler()
    h, w = reader.raster_size(state, year)
    chip = cfg.data.chip_size
    stride = stride or cfg.data.stride
    windows = tile_windows(h, w, chip, stride)
    acc = MosaicAccumulator(h, w)
    crop_acc = np.zeros((h, w), dtype=bool)

    for win in windows:
        arr = reader.read_chip(state, year, win)
        from ..data.dataset import apply_missing_month_policy

        img = apply_missing_month_policy(arr.image, arr.month_valid, cfg.data.missing_month_policy)
        img = normalize_image(img, norm.band_mean, norm.band_std).astype(np.float32)
        x = torch.from_numpy(img)[None].to(device)

        tc = lc = None
        if cfg.model.use_time_embed:
            tc = torch.from_numpy(temporal_coords(year, cfg.data.n_timesteps))[None].to(device)
        if cfg.model.use_location_embed:
            lon, lat = pixel_to_lonlat(arr.transform, chip / 2, chip / 2, arr.crs)
            lc = torch.from_numpy(location_coords(lat, lon))[None].to(device)

        out = model(x, tc, lc, return_aux=False)
        pred = out["main"][0, 0].float().cpu().numpy()
        pred = scaler.inverse(pred)
        acc.add(pred, win)
        rs, cs = win.as_slices()
        crop_acc[rs, cs] |= arr.crop_mask

    prediction, weight = acc.finalize(nodata=np.nan)
    prediction[~crop_acc] = np.nan  # never fill non-crop pixels
    return {"prediction": prediction, "weight": weight, "crop_mask": crop_acc}


@torch.no_grad()
def predict_and_compare_test_year(
    model,
    reader,
    cfg: FarmConfig,
    norm: NormStats,
    state: str,
    year: int,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Like predict_state_year, but restricted to the manifest's qualifying
    chips (real imagery only -- see filter_manifest_to_qualifying_chips), and
    also returns the true label + residual for spatial error mapping.

    predict_state_year tiles the ENTIRE state grid, which is wrong here: this
    project's imagery is chip-gated/sparse (only chips clearing
    min_crop_fraction inside the state boundary were ever downloaded), so
    windows outside that set have no real pixels and would silently predict
    from zero-filled input. Reusing the same manifest QC filter as training
    guarantees this map only shows chips the model actually saw real data for.

    Returns two DIFFERENT prediction surfaces, deliberately kept apart:

    ``prediction_full``  every crop pixel the model produced a value for. Useful
                         in its own right (the model can predict where no test
                         label exists) but NOT comparable to ``actual``.
    ``prediction``       restricted to ``comparison_mask`` -- the same
                         crop ∧ label ∧ HLS intersection the metrics use
                         (masks.target_valid_mask). ``prediction``, ``actual``
                         and ``residual`` all share this footprint exactly, so
                         the maps are directly comparable pixel-for-pixel.

    Conflating the two is what made the predicted map cover ~13% more pixels
    than the actual map.
    """
    import pandas as pd

    from ..data import masks as M
    from ..data.dataset import apply_missing_month_policy, filter_manifest_to_qualifying_chips
    from ..utils.geospatial import ChipWindow

    model.eval().to(device)
    scaler = norm.target_scaler()
    h, w = reader.raster_size(state, year)
    chip = cfg.data.chip_size
    n_months = M.min_valid_months(cfg.data)

    df = pd.read_parquet(cfg.data.manifest_path)
    df = df[(df["state"] == state) & (df["year"] == year)]
    df = filter_manifest_to_qualifying_chips(df, cfg)

    prediction_full = np.full((h, w), np.nan, dtype=np.float32)
    actual = np.full((h, w), np.nan, dtype=np.float32)
    # Mask accumulators use |= : tile_windows clamps the final row/column to
    # (raster_size - chip), so edge chips overlap. Plain assignment let a later
    # chip's mask erase an earlier one's on those seams.
    crop_acc = np.zeros((h, w), dtype=bool)
    comparison_mask = np.zeros((h, w), dtype=bool)

    for row in df.itertuples():
        win = ChipWindow(int(row.row_off), int(row.col_off), chip, chip)
        arr = reader.read_chip(state, year, win)

        img = apply_missing_month_policy(arr.image, arr.month_valid, cfg.data.missing_month_policy)
        img = normalize_image(img, norm.band_mean, norm.band_std).astype(np.float32)
        x = torch.from_numpy(img)[None].to(device)

        tc = lc = None
        if cfg.model.use_time_embed:
            tc = torch.from_numpy(temporal_coords(year, cfg.data.n_timesteps))[None].to(device)
        if cfg.model.use_location_embed:
            lon, lat = pixel_to_lonlat(arr.transform, chip / 2, chip / 2, arr.crs)
            lc = torch.from_numpy(location_coords(lat, lon))[None].to(device)

        out = model(x, tc, lc, return_aux=False)
        pred = scaler.inverse(out["main"][0, 0].float().cpu().numpy())

        rs, cs = win.as_slices()
        prediction_full[rs, cs] = pred
        actual[rs, cs] = arr.label  # raster_readers already maps label nodata -> NaN
        crop_acc[rs, cs] |= arr.crop_mask
        comparison_mask[rs, cs] |= M.target_valid_mask(
            arr.crop_mask, arr.label, arr.month_valid, cfg.data.nodata, n_months
        )

    prediction_full[~crop_acc] = np.nan
    # A pixel is only comparable if the model actually produced a finite value there.
    comparison_mask &= np.isfinite(prediction_full)

    prediction = np.where(comparison_mask, prediction_full, np.nan).astype(np.float32)
    actual = np.where(comparison_mask, actual, np.nan).astype(np.float32)
    residual = prediction - actual
    return {
        "prediction": prediction,
        "prediction_full": prediction_full,
        "actual": actual,
        "residual": residual,
        "comparison_mask": comparison_mask,
        "crop_mask": crop_acc,
    }


def write_geotiff(path: str | Path, array: np.ndarray, transform, crs, nodata: float = -9999.0) -> None:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.where(np.isfinite(array), array, nodata).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata,
        compress="deflate",
    ) as ds:
        ds.write(a, 1)
