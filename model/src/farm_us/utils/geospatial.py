"""Small geospatial helpers built on rasterio/pyproj.

Kept dependency-light so the model package does not need to reimplement CRS or
window math. Heavy statewide reads always go through windows (never full reads).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChipWindow:
    """A pixel window into a source raster grid."""

    row_off: int
    col_off: int
    height: int
    width: int

    def as_slices(self) -> tuple[slice, slice]:
        return (
            slice(self.row_off, self.row_off + self.height),
            slice(self.col_off, self.col_off + self.width),
        )


def tile_windows(
    raster_h: int,
    raster_w: int,
    chip: int,
    stride: int,
    drop_partial: bool = True,
) -> list[ChipWindow]:
    """Enumerate chip windows over a raster of size ``raster_h × raster_w``.

    With ``drop_partial`` the last partial window in each axis is clamped so a
    full-size chip is always returned (windows may overlap slightly at the edge),
    guaranteeing consistent tensor shapes for the model.
    """
    windows: list[ChipWindow] = []
    rows = list(range(0, max(raster_h - chip + 1, 1), stride))
    cols = list(range(0, max(raster_w - chip + 1, 1), stride))
    if drop_partial:
        if rows[-1] + chip < raster_h:
            rows.append(raster_h - chip)
        if cols[-1] + chip < raster_w:
            cols.append(raster_w - chip)
    for r in rows:
        for c in cols:
            r0 = min(r, max(raster_h - chip, 0))
            c0 = min(c, max(raster_w - chip, 0))
            windows.append(ChipWindow(r0, c0, min(chip, raster_h), min(chip, raster_w)))
    return windows


def pixel_to_lonlat(transform, row: float, col: float, src_crs: str) -> tuple[float, float]:
    """Center (lon, lat) in EPSG:4326 for a pixel (row, col) of a raster."""
    from pyproj import Transformer
    from rasterio.transform import xy

    x, y = xy(transform, row, col, offset="center")
    tr = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return float(lon), float(lat)


def cosine_blend_weights(chip: int, floor: float = 0.05) -> object:
    """2-D Hann/cosine weight window for seamless overlapping-tile mosaicking.

    A ``floor`` keeps weights strictly positive so raster corners (covered by a
    single tile's edge) are still reconstructed rather than dropping to no-data.
    """
    import numpy as np

    w1d = np.hanning(chip)
    w1d = np.clip(w1d, floor, None)
    return np.outer(w1d, w1d).astype("float32")
