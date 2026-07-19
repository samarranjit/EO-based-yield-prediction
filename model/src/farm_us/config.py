"""Structured configuration for FARM-US.

We use plain dataclasses that OmegaConf can (de)serialize and that
LightningCLI-style ``key=value`` overrides can patch. Every run resolves a
single :class:`FarmConfig`, which is saved verbatim next to each checkpoint so
experiments are reproducible.

Loading precedence (lowest → highest):
    dataclass defaults  <  YAML file(s)  <  CLI ``key=value`` overrides
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
# Canonical constants (single source of truth)
# --------------------------------------------------------------------------- #

#: Prithvi / HLS six-band order. DO NOT reorder — the encoder was pretrained on
#: exactly this sequence and every reader validates against it.
BAND_ORDER: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR1", "SWIR2")

#: Paper Table 1 channel-wise train stats (Canadian canola, HLS ×1e4 scale).
PAPER_BAND_MEAN: tuple[float, ...] = (493.94, 832.45, 901.06, 2927.87, 2427.47, 1658.56)
PAPER_BAND_STD: tuple[float, ...] = (250.38, 265.75, 481.92, 1038.83, 855.02, 855.37)

#: Official Prithvi-EO-2.0 pretraining stats (HF config.json).
PRITHVI_BAND_MEAN: tuple[float, ...] = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
PRITHVI_BAND_STD: tuple[float, ...] = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)

#: 17 initial study states → FIPS (from data_preparation/config.py).
STATE_FIPS: dict[str, str] = {
    "IL": "17", "IA": "19", "IN": "18", "MN": "27", "NE": "31", "MO": "29",
    "OH": "39", "SD": "46", "ND": "38", "KS": "20", "WI": "55", "MI": "26",
    "MD": "24", "DE": "10", "VA": "51", "NC": "37", "PA": "42",
}

#: CDL crop codes.
CDL_CODES: dict[str, int] = {"SOYBEANS": 5, "CORN": 1, "WHEAT": 22}

#: bu/ac → kg/ha conversion factors.
BU_AC_TO_KG_HA: dict[str, float] = {"SOYBEANS": 67.251, "CORN": 62.77, "WHEAT": 67.25}

#: Eight temporal composites: label → (month, day-window). Nov is 1–15 only.
DEFAULT_TIMESTEPS: tuple[tuple[str, int, tuple[int, int]], ...] = (
    ("APR", 4, (1, 30)), ("MAY", 5, (1, 31)), ("JUN", 6, (1, 30)),
    ("JUL", 7, (1, 31)), ("AUG", 8, (1, 31)), ("SEP", 9, (1, 30)),
    ("OCT", 10, (1, 31)), ("NOV", 11, (1, 15)),
)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class DataConfig:
    # Roots (override per machine via YAML / env, never hard-code in source).
    imagery_root: str = "${oc.env:FARM_IMAGERY_ROOT,../data_preparation/data/hls}"
    label_root: str = "../data_preparation/data/yield_labels/bilinear"
    cdl_root: str = "../data_preparation/data/cdl_masks"
    counties_path: str = "../data_preparation/data/counties/selected_states_counties_2023.gpkg"
    nass_csv: str = "../data_preparation/data/nass/nass_soybeans_county_yield_2014_2024.csv"
    manifest_path: str = "outputs/manifests/manifest.parquet"

    crop: str = "SOYBEANS"
    states: list[str] = field(default_factory=lambda: list(STATE_FIPS.keys()))
    years: list[int] = field(default_factory=lambda: list(range(2014, 2025)))

    # Storage adapter: geotiff_monthly | state_year_stack | chips_npz | manifest_cog
    reader: str = "geotiff_monthly"
    label_variant: str = "bilinear10km"  # or "county_values" (nearest)
    label_units: str = "kg_ha"

    band_order: list[str] = field(default_factory=lambda: list(BAND_ORDER))
    chip_size: int = 224
    stride: int = 224
    n_timesteps: int = 8

    target_crs: str = "EPSG:5070"
    resolution_m: float = 30.0
    nodata: float = -9999.0
    hls_scale: float = 1.0e-4  # DN → reflectance; set 1.0 if already reflectance
    hls_nodata: float = -9999.0

    # Missing-month policy: drop | zero_fill | temporal_interp | nearest_valid
    missing_month_policy: str = "temporal_interp"
    min_crop_fraction: float = 0.05
    min_label_fraction: float = 0.05
    min_month_valid_fraction: float = 0.10
    max_missing_months: int = 3

    exclude_geoids: list[str] = field(default_factory=lambda: ["24033"])  # BARC / PG County MD


@dataclass
class NormConfig:
    # fold_training_statistics | official_prithvi_statistics
    mode: str = "fold_training_statistics"
    # target scaling: none | zscore | robust
    target_scaling: str = "zscore"
    stats_path: str = "outputs/stats/norm_stats.json"


@dataclass
class SplitConfig:
    test_year: int = 2018
    val_years: list[int] = field(default_factory=lambda: [2017])
    # explicit_map | fixed_pool | previous_year
    policy: str = "explicit_map"
    split_map_path: str = "configs/splits/loyo_soybeans.yaml"


@dataclass
class ModelConfig:
    backbone_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL"
    pretrained: bool = True
    # full | frozen | decoder_only
    finetune_mode: str = "full"
    use_time_embed: bool = True
    use_location_embed: bool = True
    gradient_checkpointing: bool = False

    embed_dim: int = 1280
    depth: int = 32
    out_blocks_one_based: list[int] = field(default_factory=lambda: [8, 16, 24, 32])
    aux_block_one_based: int = 24

    # mean | attention | flatten_time
    temporal_reducer: str = "flatten_time"
    decoder_channels: int = 1024
    ppm_bins: list[int] = field(default_factory=lambda: [1, 2, 3, 6])
    multiscale_fpn: bool = False  # False = paper-faithful same-grid fusion
    head_dropout: float = 0.1
    use_auxiliary: bool = True
    aux_channels: int = 256
    final_activation: str = "linear"  # linear | relu (inference-only clip)


@dataclass
class LossConfig:
    # mse | huber | mae
    main: str = "mse"
    huber_delta: float = 1.0
    aux_weight: float = 0.2


@dataclass
class AugmentConfig:
    hflip_p: float = 0.2
    vflip_p: float = 0.2
    noise_p: float = 0.4
    noise_std: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 120
    batch_size: int = 8
    grad_accum: int = 1
    optimizer: str = "adamw"
    lr: float = 5e-6
    min_lr: float = 1e-8
    weight_decay: float = 0.1
    warmup_epochs: int = 20
    scheduler: str = "cosine"
    precision: str = "bf16-mixed"
    grad_clip: float | None = None
    num_workers: int = 4
    # single | ddp | fsdp
    strategy: str = "single"
    devices: int = 1
    accelerator: str = "auto"
    early_stopping: bool = False
    early_stopping_patience: int = 15
    detect_anomaly: bool = False
    limit_train_batches: float | None = None  # for smoke tests
    limit_val_batches: float | None = None
    log_wandb: bool = False
    output_dir: str = "outputs/runs"
    ckpt_monitor: str = "val/main_rmse_phys"
    ckpt_mode: str = "min"


@dataclass
class FarmConfig:
    data: DataConfig = field(default_factory=DataConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment_name: str = "us_soybeans"

    # ---- convenience ---- #
    def band_stats(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.norm.mode == "official_prithvi_statistics":
            return PRITHVI_BAND_MEAN, PRITHVI_BAND_STD
        return PAPER_BAND_MEAN, PAPER_BAND_STD

    def state_fips(self) -> dict[str, str]:
        return {s: STATE_FIPS[s] for s in self.data.states if s in STATE_FIPS}

    def to_container(self) -> dict[str, Any]:
        return OmegaConf.to_container(OmegaConf.structured(self), resolve=True)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #

def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> FarmConfig:
    """Merge dataclass defaults ← YAML file ← ``key=value`` CLI overrides."""
    base = OmegaConf.structured(FarmConfig())
    if config_path is not None:
        file_cfg = OmegaConf.load(Path(config_path))
        base = OmegaConf.merge(base, file_cfg)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    resolved = OmegaConf.to_container(base, resolve=True)
    return _from_container(resolved)  # type: ignore[arg-type]


def _from_container(d: dict[str, Any]) -> FarmConfig:
    return FarmConfig(
        data=DataConfig(**d.get("data", {})),
        norm=NormConfig(**d.get("norm", {})),
        split=SplitConfig(**d.get("split", {})),
        model=ModelConfig(**d.get("model", {})),
        loss=LossConfig(**d.get("loss", {})),
        augment=AugmentConfig(**d.get("augment", {})),
        train=TrainConfig(**d.get("train", {})),
        experiment_name=d.get("experiment_name", "us_soybeans"),
    )


def save_resolved_config(cfg: FarmConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(asdict(cfg)), path)
