"""Datasets + Lightning DataModule.

- :class:`SyntheticFarmDataset` — tiny deterministic in-memory dataset for unit
  and integration tests (6 bands, T timesteps, 224×224, continuous labels, crop
  mask, invalid pixels, metadata, dates, lat/lon). No disk / backbone needed.
- :class:`FarmChipDataset` — manifest-driven real dataset using a windowed
  :class:`RasterReader`, normalization, masks, missing-month policy, augment.
- :class:`FarmDataModule` — wires train/val/test for one LOYO fold.

Every sample dict has:
    image [C,T,H,W] float32, label [H,W], mask [H,W] bool (crop∧label∧hls),
    temporal_coords [T,2], location_coords [2], plus scalar metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import STATE_FIPS, FarmConfig
from ..utils.logging import get_logger
from . import masks as M
from .compositing import location_coords, temporal_coords
from .normalization import NormStats, normalize_image
from .transforms import AlignedAugment

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Missing-month policy
# --------------------------------------------------------------------------- #

def apply_missing_month_policy(image: np.ndarray, month_valid: np.ndarray, policy: str) -> np.ndarray:
    """``image`` [C,T,H,W] with NaN for missing; return NaN-free image.

    Policies: drop (caller handles), zero_fill, temporal_interp, nearest_valid.
    'drop' here just zero-fills after normalization (validity recorded elsewhere).
    """
    img = image.copy()
    C, T, H, W = img.shape
    if policy in ("zero_fill", "drop"):
        return np.nan_to_num(img, nan=0.0)
    if policy in ("temporal_interp", "nearest_valid"):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    col = img[c, :, h, w]
                    if np.isnan(col).any() and not np.isnan(col).all():
                        idx = np.arange(T)
                        good = ~np.isnan(col)
                        if policy == "temporal_interp":
                            col[~good] = np.interp(idx[~good], idx[good], col[good])
                        else:
                            col[~good] = col[good][np.abs(idx[~good, None] - idx[good][None]).argmin(1)]
                        img[c, :, h, w] = col
        return np.nan_to_num(img, nan=0.0)
    raise ValueError(f"Unknown missing_month_policy {policy!r}")


def _to_tensor(sample: dict) -> dict:
    out = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(np.ascontiguousarray(v))
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #

class SyntheticFarmDataset(Dataset):
    """Deterministic synthetic geospatial chips for tests."""

    def __init__(
        self,
        n: int = 8,
        n_bands: int = 6,
        n_timesteps: int = 8,
        chip: int = 224,
        year: int = 2020,
        seed: int = 0,
        invalid_fraction: float = 0.2,
        missing_month: bool = False,
    ) -> None:
        self.n = n
        self.n_bands = n_bands
        self.T = n_timesteps
        self.chip = chip
        self.year = year
        self.invalid_fraction = invalid_fraction
        self.missing_month = missing_month
        self.rng = np.random.default_rng(seed)
        self._items = [self._make(i) for i in range(n)]

    def _make(self, i: int) -> dict:
        rng = np.random.default_rng(1000 + i)
        C, T, H, W = self.n_bands, self.T, self.chip, self.chip
        image = rng.normal(0.0, 1.0, size=(C, T, H, W)).astype(np.float32)
        # crop mask: a blob
        yy, xx = np.mgrid[0:H, 0:W]
        crop = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2) < (H * 0.4) ** 2
        # label correlated with band means (so a model can learn something)
        base = image.mean(axis=(0, 1))
        label = (3000 + 400 * base).astype(np.float32)
        label_mask = rng.random((H, W)) > self.invalid_fraction
        valid = crop & label_mask
        label[~valid] = np.nan
        month_valid = np.ones((T, H, W), dtype=bool)
        if self.missing_month:
            image[:, 1] = np.nan
            month_valid[1] = False
        return {
            "image": image, "label": label, "crop_mask": crop,
            "label_mask": label_mask, "month_valid": month_valid,
            "state": "IA", "year": self.year, "lat": 42.0, "lon": -93.5,
            "sample_id": f"synth_{i}",
        }

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        s = self._items[i]
        img = apply_missing_month_policy(s["image"], s["month_valid"], "temporal_interp")
        hls_valid = M.hls_valid_mask_from_months(s["month_valid"], 1)
        mask = M.combine_masks(s["crop_mask"], np.isfinite(s["label"]), hls_valid)
        label = np.nan_to_num(s["label"], nan=0.0)[None]  # [1,H,W]
        out = {
            "image": img.astype(np.float32),
            "label": label.astype(np.float32),
            "mask": mask[None].astype(np.float32),
            "temporal_coords": temporal_coords(s["year"], self.T),
            "location_coords": location_coords(s["lat"], s["lon"]),
            "year": s["year"], "state": s["state"], "sample_id": s["sample_id"],
        }
        return _to_tensor(out)


# --------------------------------------------------------------------------- #
# Real manifest-driven dataset
# --------------------------------------------------------------------------- #

@dataclass
class SampleRecord:
    sample_id: str
    state: str
    year: int
    row_off: int
    col_off: int
    lat: float
    lon: float


class FarmChipDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        cfg: FarmConfig,
        reader,
        norm_stats: NormStats,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.cfg = cfg
        self.reader = reader
        self.norm = norm_stats
        self.scaler = norm_stats.target_scaler()
        self.augment = AlignedAugment(cfg.augment) if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        from ..utils.geospatial import ChipWindow

        r = self.records[i]
        win = ChipWindow(r.row_off, r.col_off, cfg_chip(self.cfg), cfg_chip(self.cfg))
        chip = self.reader.read_chip(r.state, r.year, win)

        img = apply_missing_month_policy(chip.image, chip.month_valid, self.cfg.data.missing_month_policy)
        img = normalize_image(img, self.norm.band_mean, self.norm.band_std)

        n_months = M.min_valid_months(self.cfg.data)
        hls_valid = M.hls_valid_mask_from_months(chip.month_valid, n_months)
        label_valid = M.label_valid_mask(chip.label, self.cfg.data.nodata)
        # Equivalent to M.target_valid_mask(...); kept as separate components
        # because augmentation flips each one individually below.
        mask = M.combine_masks(chip.crop_mask, label_valid, hls_valid)

        label_std = self.scaler.transform(np.nan_to_num(chip.label, nan=0.0))

        sample = {
            "image": img.astype(np.float32),
            "label": label_std[None].astype(np.float32),
            "crop_mask": chip.crop_mask, "label_mask": label_valid, "valid_mask": hls_valid,
        }
        if self.augment is not None:
            sample = self.augment(sample)
            mask = M.combine_masks(sample["crop_mask"], sample["label_mask"], sample["valid_mask"])
            # label may have been flipped too — reload from sample if present
        out = {
            "image": sample["image"].astype(np.float32),
            "label": sample["label"].astype(np.float32),
            "mask": mask[None].astype(np.float32),
            "temporal_coords": temporal_coords(r.year, self.cfg.data.n_timesteps),
            "location_coords": location_coords(r.lat, r.lon),
            "year": r.year, "state": r.state, "sample_id": r.sample_id,
        }
        return _to_tensor(out)


def cfg_chip(cfg: FarmConfig) -> int:
    return cfg.data.chip_size


# --------------------------------------------------------------------------- #
# Manifest QC: which candidate chips actually have usable imagery.
#
# build_manifest() enumerates every chip window across the full raster grid
# with no crop-fraction filter (deferred by design -- see manifest.py). For a
# real training run that means most rows can point at chips outside a
# chip-gated imagery download's populated footprint (all-nodata reads,
# zero-weight in the loss, but a full wasted Prithvi forward/backward pass).
# This computes, per (state, year), exactly which chip windows clear
# min_crop_fraction *within the requested state's boundary* -- the same
# state-boundary-masked CDL logic used by the HLS download pipeline -- so
# training only ever touches chips that were actually fetched.
# --------------------------------------------------------------------------- #

def _qualifying_chip_set(
    cdl_path,
    label_path,
    county_gpkg: str,
    state_fips: str,
    chip_size: int,
    min_crop_fraction: float,
) -> set[tuple[int, int]]:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from .raster_readers import _aligned_window

    # Chip windows in the manifest are in LABEL pixel coordinates (manifest.py
    # enumerates them from the label raster's height/width), but the crop
    # fraction is measured on the CDL. Those two rasters are not on the same
    # grid -- for MD the label sits one 30 m pixel east -- so reading the CDL
    # at raw label coordinates would gate each chip on ground ~30 m from the
    # one actually supervised. Read the CDL onto the label grid instead.
    with rasterio.open(label_path) as lsrc:
        transform, crs = lsrc.transform, lsrc.crs
        lheight, lwidth = lsrc.height, lsrc.width
        bounds = tuple(lsrc.bounds)

    with rasterio.open(cdl_path) as src:
        cw = _aligned_window(src, bounds, str(cdl_path))
        crop = src.read(1, window=cw, boundless=True, fill_value=0) > 0.5
    if crop.shape != (lheight, lwidth):
        raise ValueError(f"aligned CDL {crop.shape} != label grid {(lheight, lwidth)}")

    counties = gpd.read_file(county_gpkg)
    state_counties = counties[counties["STATEFP"] == state_fips]
    if state_counties.empty:
        raise ValueError(f"No counties found for STATEFP={state_fips!r} in {county_gpkg}")
    if state_counties.crs != crs:
        state_counties = state_counties.to_crs(crs)

    state_mask = rasterize(
        [(geom, 1) for geom in state_counties.geometry],
        out_shape=crop.shape, transform=transform, fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)
    crop = crop & state_mask

    height, width = crop.shape
    qualifying: set[tuple[int, int]] = set()
    for row0 in range(0, height, chip_size):
        for col0 in range(0, width, chip_size):
            if row0 + chip_size > height or col0 + chip_size > width:
                continue
            block = crop[row0:row0 + chip_size, col0:col0 + chip_size]
            if float(block.mean()) >= min_crop_fraction:
                qualifying.add((row0, col0))
    return qualifying


def filter_manifest_to_qualifying_chips(df, cfg: FarmConfig):
    """Drop manifest rows whose window isn't in the state-masked qualifying set."""
    from pathlib import Path

    from .raster_readers import build_reader

    cache: dict[tuple[str, int], set[tuple[int, int]]] = {}
    keep = np.zeros(len(df), dtype=bool)
    # Reuse the reader's path convention rather than re-deriving the label
    # filename here; the two must not drift apart.
    reader = build_reader(cfg.data)

    for (state, year), group in df.groupby(["state", "year"]):
        key = (state, int(year))
        if key not in cache:
            cdl_path = Path(cfg.data.cdl_root) / f"cdl_{cfg.data.crop.lower()}_{state}_{year}.tif"
            label_path = reader._label_path(state, int(year))
            state_fips = STATE_FIPS.get(state)
            if not cdl_path.exists() or not label_path.exists() or state_fips is None:
                cache[key] = set()
            else:
                cache[key] = _qualifying_chip_set(
                    cdl_path, label_path, cfg.data.counties_path, state_fips,
                    cfg.data.chip_size, cfg.data.min_crop_fraction,
                )
        qualifying = cache[key]
        idx = group.index
        rows = group[["row_off", "col_off"]].itertuples(index=False)
        keep[df.index.get_indexer(idx)] = [
            (int(r.row_off), int(r.col_off)) in qualifying for r in rows
        ]

    return df[keep]


# --------------------------------------------------------------------------- #
# Lightning DataModule
# --------------------------------------------------------------------------- #

class FarmDataModule:
    """Minimal DataModule (works with plain PyTorch and Lightning Trainer).

    For the synthetic path (tests / smoke) it builds SyntheticFarmDatasets. The
    real path reads the manifest, restricts it to chips with actual imagery
    (see filter_manifest_to_qualifying_chips), and builds FarmChipDatasets for
    the fold's train/val/test years via a windowed RasterReader.
    """

    def __init__(self, cfg: FarmConfig, synthetic: bool = True, n_synth: int = 8) -> None:
        self.cfg = cfg
        self.synthetic = synthetic
        self.n_synth = n_synth
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self) -> None:
        if self.synthetic:
            T = self.cfg.data.n_timesteps
            self.train_ds = SyntheticFarmDataset(self.n_synth, n_timesteps=T, year=2019, seed=0)
            self.val_ds = SyntheticFarmDataset(max(2, self.n_synth // 2), n_timesteps=T, year=self.cfg.split.val_years[0], seed=1)
            self.test_ds = SyntheticFarmDataset(max(2, self.n_synth // 2), n_timesteps=T, year=self.cfg.split.test_year, seed=2)
        else:
            self._setup_real()

    def _setup_real(self) -> None:
        import pandas as pd

        from .raster_readers import build_reader
        from .splits import load_split_map, make_fold

        cfg = self.cfg
        df = pd.read_parquet(cfg.data.manifest_path)
        df = df[df["state"].isin(cfg.data.states)]

        before = len(df)
        df = filter_manifest_to_qualifying_chips(df, cfg)
        logger.info("manifest QC: %d/%d chips have qualifying imagery", len(df), before)

        smap = None
        if cfg.split.policy == "explicit_map":
            try:
                smap = load_split_map(cfg.split.split_map_path)
            except Exception:
                smap = None
        fold = make_fold(cfg.split.test_year, cfg.data.years, cfg.split.val_years,
                          policy=cfg.split.policy, split_map=smap)

        reader = build_reader(cfg.data)
        identity_stats = NormStats(
            band_mean=[0.0] * len(cfg.data.band_order),
            band_std=[1.0] * len(cfg.data.band_order),
            target={"mode": "none"}, mode="identity", train_years=[],
        )

        def records_for(years):
            sub = df[df["year"].isin(years)]
            return [
                SampleRecord(
                    sample_id=row.sample_id, state=row.state, year=int(row.year),
                    row_off=int(row.row_off), col_off=int(row.col_off),
                    lat=row.center_lat, lon=row.center_lon,
                )
                for row in sub.itertuples()
            ]

        self.train_ds = FarmChipDataset(records_for(fold.train_years), cfg, reader, identity_stats, augment=True)
        self.val_ds = FarmChipDataset(records_for(fold.val_years), cfg, reader, identity_stats, augment=False)
        self.test_ds = FarmChipDataset(records_for([fold.test_year]), cfg, reader, identity_stats, augment=False)

    def apply_norm_stats(self, stats: NormStats) -> None:
        """Re-inject train-fold-only stats into already-built real datasets.

        Datasets are first built with identity stats so compute_fold_stats()
        can measure real per-band/target statistics from *raw* pixel values;
        this applies the measured stats before any dataloader is actually
        iterated for training/eval. No-op for the synthetic path (those
        datasets never normalize internally).
        """
        if self.synthetic:
            return
        scaler = stats.target_scaler()
        for ds in (self.train_ds, self.val_ds, self.test_ds):
            if ds is not None:
                ds.norm = stats
                ds.scaler = scaler

    def _loader(self, ds, shuffle: bool, drop_last: bool = False):
        from torch.utils.data import DataLoader

        return DataLoader(
            ds, batch_size=self.cfg.train.batch_size, shuffle=shuffle,
            num_workers=self.cfg.train.num_workers, drop_last=drop_last,
        )

    def train_dataloader(self):
        # drop_last=True is REQUIRED, not an optimisation. The PPM pools to a 1x1
        # bin (cfg.model.ppm_bins starts at 1), so a trailing batch of one sample
        # reaches BatchNorm as [1, C, 1, 1] and torch raises
        # "Expected more than 1 value per channel when training".
        # This bites whenever len(train_ds) % batch_size == 1 -- it took down a
        # real run at step 1217/1217 of epoch 0 (2433 chips, batch_size 2) after
        # ~2h of compute. Cost is at most batch_size-1 chips per epoch, and
        # shuffle=True means a different chip is dropped each epoch.
        return self._loader(self.train_ds, True, drop_last=True)

    def val_dataloader(self):
        # NOT dropped: val/test run under model.eval(), where BatchNorm uses its
        # running statistics and a batch of 1 is fine. Dropping here would
        # silently discard samples from the reported metrics.
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        return self._loader(self.test_ds, False)
