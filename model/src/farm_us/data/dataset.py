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

from ..config import FarmConfig
from . import masks as M
from .compositing import location_coords, temporal_coords
from .normalization import NormStats, normalize_image
from .transforms import AlignedAugment

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

        hls_valid = M.hls_valid_mask_from_months(chip.month_valid, self.cfg.data.max_missing_months and 1 or 1)
        label_valid = M.label_valid_mask(chip.label, self.cfg.data.nodata)
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
# Lightning DataModule
# --------------------------------------------------------------------------- #

class FarmDataModule:
    """Minimal DataModule (works with plain PyTorch and Lightning Trainer).

    For the synthetic path (tests / smoke) it builds SyntheticFarmDatasets. The
    real path is wired via ``from_manifest`` (not exercised without imagery).
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
            raise NotImplementedError("Real manifest wiring lives in scripts/build_manifest + trainer.")

    def _loader(self, ds, shuffle: bool):
        from torch.utils.data import DataLoader

        return DataLoader(
            ds, batch_size=self.cfg.train.batch_size, shuffle=shuffle,
            num_workers=self.cfg.train.num_workers, drop_last=False,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        return self._loader(self.test_ds, False)
