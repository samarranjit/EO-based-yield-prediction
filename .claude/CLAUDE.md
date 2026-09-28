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
  → [optional] DetailRefiner residual: main = coarse + refiner(raw image, head body)
```

Input tensor shape is `[B, 6, T=8, 224, 224]` plus `temporal_coords [B,T,2]` and `location_coords [B,2]`. **Never reshape to `[B, 48, 224, 224]`** — the 5-D shape is required by the Conv3D patch-embed and by `interpolate_pos_encoding` for T=8 (Prithvi pretrained with `num_frames=4`).

**The 256-DOF ceiling.** Prithvi tokenises 14×14 px, so a 224×224 chip yields a 16×16 token grid and the head emits **256 independent values** bilinearly upsampled to 50,176 pixels. One token covers 420 m at 30 m resolution. Nothing finer than a token can be expressed by the encoder→decoder→head path, however well trained — upsampling adds pixels, not information. This is why BARC predictions were near-flat (predicted std 1.09 vs measured 13.56 bu/ac) and why `DetailRefiner` exists: it reads the **raw full-resolution imagery**, which still carries the 196 native pixels per token that tokenisation averages away. Any future attempt to recover sub-token detail must inject new full-resolution information (refiner, upsampled input, sub-token-shifted inference) — rearranging decoder features cannot help.

Package structure under `src/farm_us/`:
- `data/` — readers (`raster_readers.py`, `compositing.py`), `dataset.py` (`FarmDataModule`), `manifest.py`, `inventory.py`, `splits.py` (LOYO fold logic), `normalization.py`, `masks.py`, `transforms.py`
- `models/` — `farm_model.py` (`FarmModel`, top-level), `prithvi_adapter.py` (real/dummy backbone switch), `feature_extractor.py`, `temporal_reducer.py`, `upernet.py`, `fpn.py`, `ppm.py`, `regression_head.py`, `auxiliary_head.py`, `detail_refiner.py` (sub-token residual branch)
- `training/` — `run.py` (`train_fold`, `evaluate_fold`, `run_loyo`, `compute_fold_stats`), `trainer.py`, `lightning_module.py`, `losses.py`, `metrics.py`, `callbacks.py`
- `evaluation/` — `evaluator.py`, `inference.py` (tiled inference), `aggregation.py`, `mosaic.py`, `plots.py`
- `interpretability/` — `spectral.py` (band importance), `attention.py` (temporal attention capture), `occlusion.py`, `plotting.py`
- `transfer/` — BARC high-resolution transfer experiments (`barc_dataset.py`, `barc_experiments.py`)
- `utils/` — `geospatial.py`, `distributed.py`, `reproducibility.py`, `logging.py`

### Non-obvious invariants (violate these only with a very good reason)

- **Band order is fixed**: `[BLUE, GREEN, RED, NIR_NARROW, SWIR1, SWIR2]` — the encoder was pretrained on exactly this sequence. Validated against raster metadata, not filenames. See `model/docs/DATA_CONTRACT.md`.
- **LOYO discipline**: for any fold, the test year is never used for normalization, target scaling, model/checkpoint selection, early stopping, or threshold tuning — train-only statistics, enforced by `data/normalization.py` + `leakage-audit`. See `model/docs/LOYO_PROTOCOL.md`.
- **Label provenance caveat**: US labels are ridge-distributed *pseudo-pixel* county yield, not measured 30 m ground truth. The neural-net split is strictly LOYO, but ridge-model provenance may not be — don't present LOYO results as proof of intra-field accuracy; that's what BARC transfer (measured labels) is for.
- **What the pseudo-labels contain, exactly**: `data_preparation/scripts/fit_ridge_to_county_vi_and_yield.py` fits Ridge on **county-mean** VI (NDVI, EVI, GCVI, NDWI) → NASS county yield, i.e. a *between-county* relationship. `distribute_county_yield_to_pixels.py` then applies those 4 coefficients to **within-county VI z-scores**, bounds and renormalises them, and multiplies by county yield (mean-preserving). So the within-county structure of every label is a monotone function of one fixed 4-VI linear projection — **one degree of freedom per pixel, with no soil, drainage or management term in it**, and a cross-scale extrapolation (between-county coefficients applied within county) that was never validated at pixel scale. Consequences: (a) a 30 m covariate such as soil has no within-county signal to learn from these labels; (b) any model component with sub-token expressivity trained on them learns to imitate the ridge function; (c) the measured variance split is 84% between-county (std 6.911) / 16% within-county (std 2.983), so county-scale metrics are dominated by the level, not the detail.
- **`init_from` vs `resume_from` are opposites** (`training/run.py`): `resume_from` continues an interrupted run (Lightning restores optimiser moments, LR-scheduler position, epoch counter) — correct for "the job died at epoch 13". `init_from` transfers **weights only** into a new run with a fresh optimiser, LR schedule and epoch counter — correct for "fine-tune the Corn Belt model on BARC". Using `resume_from` for transfer is the trap: a checkpoint from epoch 18 of a 30-epoch cosine schedule resumes with LR already decayed to near `min_lr` and stops almost immediately. Passing both raises.
- **`init_from` is strict except for one whitelisted prefix.** `load_init_weights` calls `load_state_dict(strict=False)` but then raises on any missing key not starting with `model.refiner.`, and on any unexpected key. So **the architecture must match the source checkpoint exactly** — `temporal_reducer: attention` vs `flatten_time` alone will fail the load. Copy the whole `model:` block from the config the checkpoint was trained with. If you add a new branch, zero-initialise its output (so step 0 reproduces the source checkpoint) and extend that prefix list rather than loosening the check.
- **`norm.inherit_init_stats` is mandatory with `finetune_mode: refiner_only`** (enforced in `train_fold`): a frozen backbone must be fed the normalisation it was trained with. Recomputing band/target stats on a small fine-tuning set changes the frozen encoder's inputs and destroys the step-0 equivalence the refiner design depends on. Note `norm.reuse_stats_from`'s guard validates **train_years only** — it will NOT catch a change of `data.states`, so never reuse stats across a states change.
- **`refiner_only` freezes weights *and* holds BatchNorm/dropout in eval** for every child module except `refiner` (`farm_model.py`). `requires_grad=False` alone would still let BN running statistics drift toward the small fine-tuning set, so the coarse prediction would stop matching the loaded checkpoint.
- **GDAL is not fork-safe**: raster readers open datasets inside `__getitem__` (per-worker), never in `__init__`. If you see GDAL crashes with `num_workers>0`, keep this pattern.
- **Nodata semantics**: label nodata is `-9999.0` → NaN, excluded from loss/metrics. Validity is never inferred from `yield == 0` (real yields can be low/zero). `valid = crop_mask ∧ label_valid ∧ hls_valid`.
- **Apple MPS**: the PPM's bins `[1,2,3,6]` on a 16×16 grid break MPS adaptive pooling — use `train.accelerator: cpu` on Macs (as in `smoke_dummy.yaml`); CUDA is unaffected.
- Model checkpoints, chips, rasters, caches, and `outputs/` are gitignored — never assume they're present; regenerate via the CLI pipeline (`inventory` → `build-manifest` → `verify-splits`/`leakage-audit` → `train`/`run-loyo`).
- Real backbone weights are fetched by TerraTorch from HF Hub on first build, not pre-downloaded during scaffolding — don't add eager download logic.

### Current status (last updated 2026-09-26)

Real HLS imagery, real pseudo-labels and **real measured BARC rasters are all on disk** — the pipeline has been run end-to-end on the real 600M backbone many times. Docs written during scaffolding that say otherwise are stale.

**`farm-018` is the reference county-scale checkpoint** — `outputs/runs/cornbelt5_soybeans_longer_time/test2024/checkpoints/farm-018-0.0000.ckpt`, from `configs/experiments/cornbelt5_soybeans_longer.yaml` (5 states IA/IL/IN/MD/MN, 2014–2024, test 2024, `decoder_only`, `temporal_reducer: attention`). Everything downstream `init_from`s it.

| Experiment | Labels | Metric |
|---|---|---|
| farm-018, county test 2024 | pseudo | pixel r² **0.799**, MAE 3.63 bu/ac; per-state (chip-mean) r² IA 0.54 / IL 0.76 / IN 0.48 / MD 0.35 / MN 0.61 |
| BARC, decoder fine-tune | measured | pixel `pearson_r2` 0.001–0.25 across folds — **essentially no sub-field skill** |
| BARC, zero-shot from farm-018 | measured | r² −0.63, `pearson_r2` 0.023, bias +10.9 bu/ac |
| **BARC, `refiner_only` (11-fold LOYO)** | measured | mean pixel `pearson_r2` **0.465**, MAE 11.91 bu/ac (`outputs/comparisons/val1_vs_val2.md`) |
| Spatial holdout MO (zero-shot) | pseudo | r² −0.09…0.43, `pearson_r2` 0.26–0.49, bias +1.8…+4.6 |
| Spatial holdout NE (zero-shot) | pseudo | r² **−0.35**, `pearson_r2` 0.164, bias +5.31 |

Three things to carry into any new work:

1. **The refiner is the main positive result.** A 20,801-parameter branch on a fully frozen 775M base took BARC from no skill to mean `pearson_r2` 0.465. That is direct evidence the 256-DOF ceiling — not the encoder's representation — was the binding constraint on sub-field accuracy. It needed `lr 1e-2, weight_decay 0.0` (`barc_refiner_hilr.yaml`); at the original `lr 1e-4, wd 0.01` the decay pulled the zero-initialised output layer back toward zero and it barely moved.
2. **Spatial generalisation is weak, and is the live risk.** Zero-shot on unseen states gives negative-to-modest r² with a consistent **positive bias** (over-prediction), against 0.799 in-sample. LOYO holds out a year but trains on all states, and 99.6% of test locations were seen in other years, so LOYO alone measures very little about new ground. Any new *static* spatial covariate (soil, elevation, static rasters) is a high-resolution location fingerprint and must be validated under `configs/experiments/spatial_holdout.yaml`, not just LOYO — report the LOYO-vs-spatial-holdout gap, since that gap *is* the memorisation measurement. See `docs/LOCATION_EMBEDDING_RISK.md`.
3. **Two-validation-year folds were tested and are roughly neutral** — mean `pearson_r2` 0.453 (2 val) vs 0.465 (1 val) over 11 folds, trading fold-level noise for one less training year. `configs/splits/loyo_soybeans_2val.yaml`. Use 1 val year unless a specific fold's early stopping is the problem.

Report `r2` (against 1:1) as the accuracy number and `pearson_r2` (refitted slope/intercept) as correlation — they diverge a lot here because of bias, and the metrics JSON says so explicitly. Pool BARC field-level results across folds (`scripts/barc_field_metrics.py`); per-fold BARC sets are 2–4 chips and individually noisy.

For anything unclear beyond this, check `model/docs/` first: `ARCHITECTURE.md`, `DATA_CONTRACT.md`, `LOYO_PROTOCOL.md`, `BARC_TRANSFER.md`, `TRAINING_GUIDE.md`, `TROUBLESHOOTING.md`, `LOCATION_EMBEDDING_RISK.md`, `PAPER_REPLICATION_NOTES.md`, `DECISION_LOG.md`.
