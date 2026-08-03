# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repo has **two independent uv projects** plus notebooks/papers. There is no shared root environment for the pipeline code — always `cd` into the subproject before running Python.

```
data_preparation/   Builds yield-label rasters from NASS county yield + USDA CDL crop masks
model/               FARM-US: Prithvi-EO-2.0 fine-tuning for pixel-wise yield regression
notebooks/           Exploratory Jupyter notebooks (not part of either pipeline)
```

`model/` consumes `data_preparation/`'s output directly (relative paths like `../data_preparation/data/yield_labels/bilinear`) — the two are pipeline stages of one project, not decoupled services. When editing path conventions or file-naming in one, check `model/docs/DATA_CONTRACT.md` and `data_preparation/config.py` for the other side of the contract.

## data_preparation/

Produces model-ready 30 m yield-label GeoTIFFs from USDA NASS county yield stats and USDA CDL crop masks. Config-driven for one crop at a time via `CROP_NAME` in `config.py` (currently `SOYBEANS`).

```bash
cd data_preparation
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

python scripts/01_download_nass_yield.py          # needs NASS_API_KEY
python scripts/02_download_counties.py            # TIGER/Line county boundaries
python scripts/03_export_cdl_masks_gee.py         # needs `earthengine authenticate`; async GEE export
python scripts/04_make_yield_label_rasters.py     # nearest + bilinear10km label rasters
python scripts/05_qc_check_label_rasters.py       # QC CSV in data/qc/
```

Scripts must run in order 01→05; each depends on the previous step's output under `data/`. Missing (state, year) combos are skipped with a `SKIP` message, not a hard failure.

**Leakage protection**: Prince George's County, MD (GEOID `24033`) is excluded from all labels in `config.py` (`EXCLUDE_GEOIDS`) — it contains the BARC field site used as an external, measured-yield test set by `model/`. Do not remove this exclusion without checking `model/docs/BARC_TRANSFER.md`.

Target states: `IL IA IN MN NE MO OH SD ND KS WI MI MD DE VA NC PA`. Years: 2014–2024.

## model/ (FARM-US)

Fine-tunes **Prithvi-EO-2.0-600M-TL** (a geospatial ViT foundation model) into a dense pixel-wise 30 m crop-yield regressor: multi-temporal 6-band HLS imagery → `[B, 1, H, W]` yield map. This is a research replication of the FARM paper (Nejadshamsi et al. 2026), adapted from Canadian canola to US soybeans — see `model/README.md` for the full paper-vs-repo comparison table and `model/docs/PAPER_REPLICATION_NOTES.md` / `model/docs/DECISION_LOG.md` for the rationale behind every deviation.

### Setup and commands

Uses **uv**, Python 3.11–3.12. From `model/`:

```bash
uv sync --extra dev        # core + dev, CPU torch — enough for the full test suite
uv sync --extra prithvi    # ADD the real 600M-TL backbone (TerraTorch + weights, large)
```

`make help` lists shortcuts (wraps the same `uv run` commands below). Key targets: `make test`, `make lint`, `make fmt`, `make typecheck`, `make smoke`, `make train-smoke`, `make clean`.

```bash
# Tests
uv run pytest -m "not integration" -q     # default suite: synthetic data + dummy backbone, no network
uv run pytest -m integration -q           # real-Prithvi tests; needs --extra prithvi + network
uv run pytest tests/test_masked_losses.py -q -k some_test   # single file / test

# Lint / format / typecheck
uv run ruff check src tests scripts
uv run ruff format src tests scripts && uv run ruff check --fix src tests scripts
uv run mypy src

# CLI (also installed as the `farm-us` console script)
uv run python -m farm_us.cli <command> --config <yaml> [key=value ...]
```

CLI commands: `inventory`, `build-manifest`, `verify-splits`, `leakage-audit`, `inspect-batch`, `smoke-test-model`, `profile-memory`, `train`, `evaluate`, `run-loyo`, `predict-raster`, `band-importance`. All take `--config <path>` plus bare `key=value` overrides (e.g. `test_year=2018` — aliased in `cli.py` to `split.test_year`). Pass `--real` to swap the dummy backbone for the real Prithvi-EO-2.0-600M-TL (requires `--extra prithvi`); without it, everything (including `train`/`evaluate`) runs against a small dummy encoder for fast CPU iteration.

The synthetic smoke path (`configs/experiments/smoke_dummy.yaml`) runs train→checkpoint→evaluate on CPU with a dummy backbone in seconds — use it to sanity-check pipeline changes before touching real data or the real backbone.

### Config system

`src/farm_us/config.py` defines a single `FarmConfig` dataclass tree (`data`, `norm`, `split`, `model`, `loss`, `augment`, `train`) resolved via OmegaConf with precedence **dataclass defaults < YAML file < CLI `key=value` overrides**. YAML configs under `model/configs/` are partial overlays, not full copies of the schema — `configs/experiments/*.yaml` is the top-level entry point per experiment; `configs/data/`, `configs/model/`, `configs/training/`, `configs/splits/`, `configs/barc/` hold reusable fragments. Every run saves its fully-resolved config (`resolved_config.yaml`), `norm_stats.json`, and `provenance.json` (git commit, package versions, fold years, manifest fingerprint) next to the checkpoint under `outputs/runs/` — reproducibility depends on never hand-editing a resolved config after the fact.

Canonical constants live at the top of `config.py` and must not be duplicated or redefined elsewhere: `BAND_ORDER`, `PAPER_BAND_MEAN/STD`, `PRITHVI_BAND_MEAN/STD`, `STATE_FIPS`, `CDL_CODES`, `BU_AC_TO_KG_HA`, `DEFAULT_TIMESTEPS`.

### Architecture (see `model/docs/ARCHITECTURE.md` for the full diagram)

```
Prithvi-EO-2.0-600M-TL encoder (frozen/full, 32 transformer blocks)
  → select blocks {8,16,24,32} one-based == {7,15,23,31} zero-based
  → TemporalFeatureReducer per level (mean | attention | flatten_time[default])
  → PaperFaithfulUPerNetDecoder: lateral 1x1 → PPM(bins 1,2,3,6) on deepest → FPN top-down → concat+conv
  → main regression head (3x3 convs 1024→512→256→64, BN+ReLU+dropout, 1x1→1, bilinear to 224)
  → auxiliary head off block-24 level (deep supervision, weight 0.2, disableable)
```

Input tensor shape is `[B, 6, T=8, 224, 224]` plus `temporal_coords [B,T,2]` and `location_coords [B,2]`. **Never reshape to `[B, 48, 224, 224]`** — the 5-D shape is required by the Conv3D patch-embed and by `interpolate_pos_encoding` for T=8 (Prithvi pretrained with `num_frames=4`).

Package structure under `src/farm_us/`:
- `data/` — readers (`raster_readers.py`, `compositing.py`), `dataset.py` (`FarmDataModule`), `manifest.py`, `inventory.py`, `splits.py` (LOYO fold logic), `normalization.py`, `masks.py`, `transforms.py`
- `models/` — `farm_model.py` (`FarmModel`, top-level), `prithvi_adapter.py` (real/dummy backbone switch), `feature_extractor.py`, `temporal_reducer.py`, `upernet.py`, `fpn.py`, `ppm.py`, `regression_head.py`, `auxiliary_head.py`
- `training/` — `run.py` (`train_fold`, `evaluate_fold`, `run_loyo`, `compute_fold_stats`), `trainer.py`, `lightning_module.py`, `losses.py`, `metrics.py`, `callbacks.py`
- `evaluation/` — `evaluator.py`, `inference.py` (tiled inference), `aggregation.py`, `mosaic.py`, `plots.py`
- `interpretability/` — `spectral.py` (band importance), `attention.py` (temporal attention capture), `occlusion.py`, `plotting.py`
- `transfer/` — BARC high-resolution transfer experiments (`barc_dataset.py`, `barc_experiments.py`)
- `utils/` — `geospatial.py`, `distributed.py`, `reproducibility.py`, `logging.py`

### Non-obvious invariants (violate these only with a very good reason)

- **Band order is fixed**: `[BLUE, GREEN, RED, NIR_NARROW, SWIR1, SWIR2]` — the encoder was pretrained on exactly this sequence. Validated against raster metadata, not filenames. See `model/docs/DATA_CONTRACT.md`.
- **LOYO discipline**: for any fold, the test year is never used for normalization, target scaling, model/checkpoint selection, early stopping, or threshold tuning — train-only statistics, enforced by `data/normalization.py` + `leakage-audit`. See `model/docs/LOYO_PROTOCOL.md`.
- **Label provenance caveat**: US labels are ridge-distributed *pseudo-pixel* county yield, not measured 30 m ground truth. The neural-net split is strictly LOYO, but ridge-model provenance may not be — don't present LOYO results as proof of intra-field accuracy; that's what BARC transfer (measured labels) is for.
- **GDAL is not fork-safe**: raster readers open datasets inside `__getitem__` (per-worker), never in `__init__`. If you see GDAL crashes with `num_workers>0`, keep this pattern.
- **Nodata semantics**: label nodata is `-9999.0` → NaN, excluded from loss/metrics. Validity is never inferred from `yield == 0` (real yields can be low/zero). `valid = crop_mask ∧ label_valid ∧ hls_valid`.
- **Apple MPS**: the PPM's bins `[1,2,3,6]` on a 16×16 grid break MPS adaptive pooling — use `train.accelerator: cpu` on Macs (as in `smoke_dummy.yaml`); CUDA is unaffected.
- Model checkpoints, chips, rasters, caches, and `outputs/` are gitignored — never assume they're present; regenerate via the CLI pipeline (`inventory` → `build-manifest` → `verify-splits`/`leakage-audit` → `train`/`run-loyo`).
- Real backbone weights are fetched by TerraTorch from HF Hub on first build, not pre-downloaded during scaffolding — don't add eager download logic.

For anything unclear beyond this, check `model/docs/` first: `ARCHITECTURE.md`, `DATA_CONTRACT.md`, `LOYO_PROTOCOL.md`, `BARC_TRANSFER.md`, `TRAINING_GUIDE.md`, `TROUBLESHOOTING.md`, `LOCATION_EMBEDDING_RISK.md`, `PAPER_REPLICATION_NOTES.md`, `DECISION_LOG.md`.
