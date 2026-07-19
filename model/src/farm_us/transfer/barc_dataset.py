"""BARC high-resolution transfer dataset adapter.

BARC (Prince George's County, MD — GEOID 24033, excluded from national labels)
is the US analogue of the paper's 10 m yield-monitor set: **true / much more
direct 30 m yield observations**, kept strictly separate from the national
ridge-distributed pseudo-labels.

The BARC rasters are not on disk in this environment, so this adapter mirrors the
national :class:`FarmChipDataset` interface and documents the expected layout,
with a synthetic fallback for pipeline tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import FarmConfig
from ..data.dataset import SyntheticFarmDataset


@dataclass
class BarcConfig:
    root: str = "${oc.env:FARM_BARC_ROOT,../data_preparation/data/barc}"
    field_id_raster: str | None = None  # per-field id for field-level metrics
    label_kind: str = "measured_30m"   # NOT ridge pseudo-labels
    years: tuple[int, ...] = (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)


def build_barc_dataset(cfg: FarmConfig, barc: BarcConfig, year: int, synthetic: bool = True):
    if synthetic:
        return SyntheticFarmDataset(n=4, n_timesteps=cfg.data.n_timesteps, year=year, seed=year)
    raise NotImplementedError(
        "Real BARC rasters not available in this environment. Point BarcConfig.root "
        "at measured 30 m yield rasters and implement a windowed reader mirroring "
        "GeotiffMonthlyReader (see docs/BARC_TRANSFER.md)."
    )
