# FARM-US

A professional, reproducible implementation of the **FARM** crop-yield regression
framework (Nejadshamsi et al., *AgriEngineering* 2026 — "FARM: Crop Yield
Prediction via Regression on Prithvi's Encoder for Satellite Sensing"), **adapted
to a United States soybean dataset**.

FARM fine-tunes the **Prithvi-EO-2.0-600M-TL** geospatial foundation model into a
dense, pixel-wise 30 m yield-regression model: multi-temporal HLS imagery →
continuous single-channel yield map `[B, 1, H, W]`.

> **Research-integrity note.** The national US labels here are **ridge-distributed
> county yield** ("pseudo-pixel labels"), *not* measured 30 m yield. Good numbers
> against pseudo-labels do **not** by themselves prove intra-field accuracy —
> county-level aggregation and BARC transfer are the complementary evaluations.
> See [docs/LOYO_PROTOCOL.md](docs/LOYO_PROTOCOL.md) and
> [docs/BARC_TRANSFER.md](docs/BARC_TRANSFER.md).

---

## Relationship to the FARM paper

| | Paper (FARM) | This repo (FARM-US) |
|---|---|---|
| Crop / region | Canola, Canadian Prairies | Soybeans (config-driven), 17 US states |
| Years | 2018–2020, 2022–2023 | 2014–2024 |
| Timesteps | T=5 (May–Sep) | **T=8** (Apr–Nov 1–15) |
| Backbone | Prithvi-EO-2.0-600M | **Prithvi-EO-2.0-600M-TL** (time+location) |
| Split | temporal hold-out | strict **LOYO** (leave-one-year-out) |
| Labels | upsampled county yield | ridge-distributed county yield (external) |
| Input embedding | fig: Conv3D `[B,C,T,H,W]` / prose: `[B,C·T,H,W]` | official Conv3D `[B,6,8,224,224]` (see notes) |

Every deviation and every ambiguity resolution is documented in
[docs/PAPER_REPLICATION_NOTES.md](docs/PAPER_REPLICATION_NOTES.md) and
[docs/DECISION_LOG.md](docs/DECISION_LOG.md).

## Directory structure

```
model/
├── configs/         data / model / training / splits / experiments / barc YAMLs
├── src/farm_us/     package: data, models, training, evaluation, interpretability, transfer, utils
├── scripts/         thin CLI wrappers
├── tests/           pytest suite (synthetic + dummy backbone; marked integration for real Prithvi)
├── docs/            all documentation
└── outputs/         runs, manifests, inventory (gitignored)
```

## Environment installation

Uses **uv**. From `model/`:

```bash
uv sync --extra dev            # core + dev (CPU torch) — runs the whole test suite
uv sync --extra prithvi        # ADD the real Prithvi-EO-2.0-600M-TL backbone (TerraTorch, large)
```

The core install intentionally does **not** pull TerraTorch/model weights, so the
full synthetic pipeline and tests run anywhere. The real 600M backbone is the
`[prithvi]` extra. **No checkpoints are downloaded during scaffolding** — they are
fetched by TerraTorch the first time you build the real model.

## Required data

Produced by the sibling `../data_preparation` pipeline (not part of this repo):

- **Yield labels** (ridge/bilinear): `../data_preparation/data/yield_labels/bilinear/<year>/…tif` (EPSG:5070, 30 m, kg/ha, nodata −9999).
- **CDL crop masks**: `../data_preparation/data/cdl_masks/cdl_soybeans_<STATE>_<YEAR>.tif` (0/1).
- **HLS imagery** (6-band monthly composites): **not yet on disk** — set `FARM_IMAGERY_ROOT` when available. See [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md).

## Quick start

```bash
# 1. Inspect what's on disk
uv run python -m farm_us.cli inventory      --config configs/data/us_soybeans_hls.yaml

# 2. Build the deterministic chip manifest (Parquet + fingerprint)
uv run python -m farm_us.cli build-manifest --config configs/experiments/us_soybeans.yaml

# 3. Resolve + audit a LOYO fold
uv run python -m farm_us.cli verify-splits  --config configs/experiments/us_soybeans.yaml test_year=2018
uv run python -m farm_us.cli leakage-audit  --config configs/experiments/us_soybeans.yaml test_year=2018

# 4. Model smoke test (dummy backbone, CPU) and one batch
uv run python -m farm_us.cli smoke-test-model --config configs/experiments/us_soybeans.yaml
uv run python -m farm_us.cli inspect-batch    --config configs/experiments/smoke_dummy.yaml

# 5. Train one fold  (real 600M: add `--real`, use --extra prithvi + a GPU)
uv run python -m farm_us.cli train    --config configs/experiments/us_soybeans.yaml test_year=2018
uv run python -m farm_us.cli evaluate --config configs/experiments/us_soybeans.yaml test_year=2018 checkpoint=<ckpt>

# 6. All folds
uv run python -m farm_us.cli run-loyo --config configs/experiments/us_soybeans.yaml

# 7. Georeferenced tiled inference → GeoTIFF
uv run python scripts/predict_raster.py --config configs/experiments/us_soybeans.yaml \
    checkpoint=<ckpt> state=IL year=2018
```

`make help` lists developer shortcuts. The synthetic smoke path
(`configs/experiments/smoke_dummy.yaml`) runs train→checkpoint→evaluate on CPU
with a dummy backbone in seconds.

## LOYO protocol
One test year, ≥1 validation years, all remaining years train — mutually
exclusive. Normalization + target scaling use **training years only**. Test data
is never used for normalization, model/checkpoint selection, or thresholds. Full
rules and the leakage audit: [docs/LOYO_PROTOCOL.md](docs/LOYO_PROTOCOL.md).

## Interpretability
- **Spectral band importance** from Conv3D patch-embed weight magnitudes (`farm_us.interpretability.spectral`).
- **Temporal attention** month×month heatmaps + incoming-attention curve, blocks 8 & 16 (`…attention`).
- **Occlusion** sensitivity by month/band (`…occlusion`).

## BARC high-resolution transfer
Three experiments (zero-shot / fine-tune-from-FARM / train-from-Prithvi) on the
BARC field site (measured 30 m yield, kept separate from pseudo-labels), LOYO.
See [docs/BARC_TRANSFER.md](docs/BARC_TRANSFER.md).

## Model checkpoint storage
Weights, checkpoints, chips, rasters, caches, and run outputs are **gitignored**.
Each run writes `resolved_config.yaml`, `norm_stats.json`, and `provenance.json`
(config, fold years, package versions, git commit, param counts, manifest
fingerprint, label provenance) next to the checkpoint.

## Expected hardware
Real 600M-TL, T=8, dense 224² output ≈ the paper's 48 GB-class GPU at batch 8.
Use the memory ladder (bf16 → micro-batch + grad-accum → checkpointing → DDP/FSDP)
in [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) and the `profile-memory`
command. The 300M-TL fallback config exists for smaller GPUs.

## OOM troubleshooting
See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) (checkpoint download,
TerraTorch API drift, T=8 pos-embedding, raster misalignment, NaN loss, OOM,
rasterio+multiprocessing, MPS adaptive-pool, slow network storage).

## Reproducibility
Seed 0; resolved config + norm stats + provenance saved per run; deterministic
manifest with a fingerprint; train-only statistics.

## Known limitations
- HLS imagery is not yet on disk here, so **no real-data batch has been run**;
  everything real-data-facing is verified against real label/CDL rasters and has
  a documented code path, but end-to-end real training is pending imagery.
- Labels are ridge-distributed pseudo-pixel targets (see integrity note above).
- The real 600M forward pass is exercised by a **marked integration test**, not
  in the default CPU suite.
- Location embeddings risk a spatial shortcut — see
  [docs/LOCATION_EMBEDDING_RISK.md](docs/LOCATION_EMBEDDING_RISK.md).
