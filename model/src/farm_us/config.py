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
#:
#: "BARC" is a pseudo-state, not a state: the USDA Beltsville Agricultural
#: Research Center transfer set, which is physically inside Prince George's
#: County, Maryland, hence FIPS 24. It is mapped here because
#: dataset.filter_manifest_to_qualifying_chips does
#: ``STATE_FIPS.get(state)`` and treats a None result as "no qualifying chips",
#: which would silently drop every BARC chip and evaluate on an empty test set
#: rather than raise. Giving it Maryland's FIPS makes the county-boundary
#: rasterization resolve to the MD polygons that genuinely contain BARC.
#: Inert for every other experiment -- no config lists BARC in data.states
#: except configs/experiments/barc_transfer.yaml.
STATE_FIPS: dict[str, str] = {
    "IL": "17", "IA": "19", "IN": "18", "MN": "27", "NE": "31", "MO": "29",
    "OH": "39", "SD": "46", "ND": "38", "KS": "20", "WI": "55", "MI": "26",
    "MD": "24", "DE": "10", "VA": "51", "NC": "37", "PA": "42",
    "BARC": "24",
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

    #: Manifest ``sample_id``s to drop before training/evaluation.
    #:
    #: Intended for chips whose imagery is physically unreadable -- a truncated
    #: DEFLATE tile raises only when decoded, inside a DataLoader worker, which
    #: kills the entire run hours in. Such chips cannot be used by any run, so
    #: excluding them is a statement of fact about the data, not a modelling
    #: choice, and it applies identically to train, val and test.
    #:
    #: Find them with ``scripts/scan_hls_corruption.py``. Keep this list SMALL
    #: and justified: it is silent data removal, so every entry should trace to
    #: a scan report. It is written into each run's resolved_config.yaml, so the
    #: exclusion stays on the record rather than living only in someone's shell
    #: history. Excluding chips for any reason other than unreadable data would
    #: be selection bias -- do not use this to drop inconvenient samples.
    exclude_sample_ids: list[str] = field(default_factory=list)


@dataclass
class NormConfig:
    # fold_training_statistics | official_prithvi_statistics
    mode: str = "fold_training_statistics"
    # target scaling: none | zscore | robust
    target_scaling: str = "zscore"
    stats_path: str = "outputs/stats/norm_stats.json"

    #: Cap on how many TRAIN chips the normalization pass reads. ``None`` (the
    #: default) reads every one, which is the historical behaviour and stays
    #: bit-for-bit unchanged -- set this only to opt in.
    #:
    #: The pass is single-threaded and NOT resumable, so on large folds it
    #: dominates wall-clock: the 4-state Corn Belt fold reads ~32k chips at
    #: ~2.1 s/chip = ~19 h before training starts, and a crash loses all of it.
    #: It exists only to estimate 6 band means/stds and a target scaler, which
    #: converge on a few million pixels; each chip contributes up to 50,176 per
    #: band, so ~2,000 chips is ~100M pixels per band -- far past convergence.
    #:
    #: Sampling is uniform WITHOUT replacement over the train split only (never
    #: val/test), seeded by ``stats_seed``, and the drawn indices are sorted so
    #: reads stay roughly sequential on disk. The chosen size and seed are
    #: recorded in norm_stats.json so a subsampled run stays reproducible.
    stats_max_chips: int | None = None
    stats_seed: int = 0

    #: Path to an existing ``norm_stats.json`` to LOAD instead of recomputing.
    #:
    #: The statistics pass is deterministic given (train split, seed, cap), so
    #: re-running it after a crash, a precision change, or any other edit that
    #: does not touch the data recomputes an answer already on disk -- ~60 min
    #: for the 4-state fold, ~19 h unsubsampled. ``evaluate_fold`` has always
    #: preferred the saved file for this reason; this gives ``train_fold`` the
    #: same option.
    #:
    #: LEAKAGE GUARD: ``train_fold`` refuses to load a file whose recorded
    #: ``train_years`` differ from the fold being trained. Statistics carry the
    #: fingerprint of the years they were computed on, so silently accepting a
    #: mismatched file could normalise using the test year -- the exact thing
    #: LOYO_PROTOCOL.md forbids. Reuse across a precision/batch/lr change is
    #: safe; reuse across a change of states, years, or test_year is not, and
    #: is rejected.
    #:
    #: Left None by default: reuse must be a deliberate act, or a stale file
    #: would quietly outlive the data it describes.
    reuse_stats_from: str | None = None


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
    batch_size: int = 24
    grad_accum: int = 1
    optimizer: str = "adamw"
    lr: float = 5e-6
    min_lr: float = 1e-8
    weight_decay: float = 0.1
    warmup_epochs: int = 20
    scheduler: str = "cosine"
    precision: str = "bf16-mixed"
    grad_clip: float | None = None
    num_workers: int = 32
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
    # Independent of the best/last checkpoints above: Lightning's save_last only
    # refreshes when the monitored metric ALSO improves that epoch (confirmed by
    # reading ModelCheckpoint's source directly), so a long plateau leaves no
    # recoverable weights past the last-best epoch. This periodic saver is keyed
    # on recency (monitor=None) instead, so it can't get stuck the same way -- a
    # single rolling snapshot, overwritten every N epochs (Lightning only allows
    # save_top_k in {0, 1, -1} when monitor=None; -1 keeps every one ever saved,
    # unbounded, so 1 is the efficient choice here).
    periodic_ckpt_every_n_epochs: int = 10


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
