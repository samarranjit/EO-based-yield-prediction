"""Deterministic chip-window enumeration over (state, year) rasters.

Windows are indexed, not materialized (no duplicate TIFF chips) unless the user
opts into materialization. Because splitting is by year, all windows of a given
(state, year) inherit that year's split — overlapping windows can never straddle
train/val/test.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.geospatial import ChipWindow, tile_windows


@dataclass
class ChipRef:
    state: str
    year: int
    window: ChipWindow


def enumerate_chips(
    state: str,
    year: int,
    raster_h: int,
    raster_w: int,
    chip: int,
    stride: int,
) -> list[ChipRef]:
    return [ChipRef(state, year, w) for w in tile_windows(raster_h, raster_w, chip, stride)]
