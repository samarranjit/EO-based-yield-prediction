"""Temporal-composite date convention and Prithvi metadata construction.

Eight composites: Apr, May, Jun, Jul, Aug, Sep, Oct, Nov(1-15). Each composite's
representative date defaults to the **midpoint** of its compositing interval
(documented convention); when real acquisition/composite metadata is available a
reader may override it. Prithvi consumes ``temporal_coords = [B, T, 2]`` as
(year, day-of-year) and ``location_coords = [B, 2]`` as (lat, lon).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from ..config import DEFAULT_TIMESTEPS


def representative_dates(year: int, timesteps=DEFAULT_TIMESTEPS) -> list[date]:
    """Midpoint date of each composite window."""
    out: list[date] = []
    for _label, month, (d0, d1) in timesteps:
        mid = (d0 + d1) // 2
        out.append(date(year, month, max(1, mid)))
    return out


def day_of_year(d: date) -> int:
    return d.timetuple().tm_yday


def temporal_coords(year: int, n_timesteps: int | None = None, timesteps=DEFAULT_TIMESTEPS) -> np.ndarray:
    """``[T, 2]`` array of (year, day-of-year).

    ``n_timesteps`` selects the first N composite windows (default: all 8).
    """
    if n_timesteps is not None:
        timesteps = timesteps[:n_timesteps]
    dates = representative_dates(year, timesteps)
    return np.array([[year, day_of_year(d)] for d in dates], dtype=np.float32)


def location_coords(lat: float, lon: float) -> np.ndarray:
    return np.array([lat, lon], dtype=np.float32)
