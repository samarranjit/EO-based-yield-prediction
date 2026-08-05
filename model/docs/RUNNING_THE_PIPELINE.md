# Running the FARM-US Pipeline End-to-End

Practical runbook for training, validating and testing the model on a single
state-year experiment (Maryland soybeans, LOYO test year 2024). Everything here
is copy-pasteable. For *why* the architecture looks the way it does see
[PAPER_REPLICATION_NOTES.md](PAPER_REPLICATION_NOTES.md); for the data contract
see [DATA_CONTRACT.md](DATA_CONTRACT.md).

All commands run from the `model/` directory unless stated otherwise.

---

## 0. What the experiment does

Strict **leave-one-year-out (LOYO)** on Maryland:

| Split      | Years                | Used for                              |
| ---------- | -------------------- | ------------------------------------- |
| train      | 2014–2022 (9 years) | fitting weights                       |
| validation | 2023                 | checkpoint selection**only**    |
| test       | 2024                 | touched exactly once, at the very end |

The test year is never used for normalization, target scaling, checkpoint
selection or early stopping. Normalization statistics come from **training years
only** — this is enforced in `training/run.py::compute_fold_stats`.

Config: [`configs/experiments/maryland_soybeans.yaml`](../configs/experiments/maryland_soybeans.yaml).
That single file is the source of truth for an experiment. Every command takes
`--config <path>` plus optional bare `key=value` overrides.

---

## 1. One-time setup

### 1.1 Environment (CUDA GPU required for real runs)

```bash
cd model
rm -rf .venv                      # never reuse a .venv built on another machine
uv sync --extra dev               # core + dev; resolves CUDA torch on Linux
uv sync --extra prithvi           # adds TerraTorch + the real 600M backbone (large)

uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `cuda.is_available()` is `False`, stop and fix the torch build first.

### 1.2 Pre-cache the Prithvi weights

```bash
export HF_HOME="$PWD/.hf_cache"   # keeps the 2.5 GB checkpoint out of ~/.cache
uv run hf download ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL
```

Public repo, no token needed. Set `HF_HOME` to the same path in every later
shell, or TerraTorch re-downloads to the default location.

### 1.3 Confirm the test suite passes

```bash
uv run pytest -m "not integration" -q     # expect 67 passed, 1 skipped
```

### 1.4 Required data on disk

| What            | Location                                                                                                                    | Notes                                       |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| HLS imagery     | `../data_preparation/data/Maryland/HLS_bands/{year}/hls_soybeans_MD_{year}_{MON}.tif`                                     | 8 months APR–NOV, 6 bands + OBS_COUNT      |
| Yield labels    | `../data_preparation/data/yield_labels/maryland_pseudo/{year}/nass_soybeans_yield_MD_{year}_pseudo_30m_soybeans_only.tif` | symlinks to`Maryland/pseudo_yield_files/` |
| CDL crop masks  | `../data_preparation/data/cdl_masks/cdl_soybeans_MD_{year}.tif`                                                           | 0/1                                         |
| County polygons | `../data_preparation/data/county_maps/selected_states_counties_2023.gpkg`                                                 | restricts CDL to the state boundary         |

**Units caveat:** the Maryland pseudo-yield rasters are **bu/ac**, not kg/ha
(the file's own band description says `pseudo_yield_bu_ac`). `label_units` in the
config records this. Multiply by 67.251 for kg/ha.

---

## 2. Pre-flight checks

Fast, cheap, and each catches a different class of error. Run them in order —
never jump straight to training.

```bash
# 1. Is the data where the config thinks it is?
uv run python -m farm_us.cli inventory --config configs/experiments/maryland_soybeans.yaml

# 2. Enumerate every candidate 224x224 chip -> outputs/manifests/manifest.parquet
uv run python -m farm_us.cli build-manifest --config configs/experiments/maryland_soybeans.yaml

# 3. Confirm the train/val/test year split resolves as intended
uv run python -m farm_us.cli verify-splits --config configs/experiments/maryland_soybeans.yaml

# 4. Confirm the manifest respects that split (no year/chip leakage)
uv run python -m farm_us.cli leakage-audit --config configs/experiments/maryland_soybeans.yaml

# 5. Read one REAL batch end-to-end (slow: ~1h, see note below)
uv run python -m farm_us.cli inspect-batch --config configs/experiments/maryland_soybeans.yaml --real
```

Expected from step 5 — shapes and physically sensible values:

```
manifest QC: ~2951/19800 chips have qualifying imagery
image           shape=(8, 6, 8, 224, 224)
label           shape=(8, 1, 224, 224)
mask            shape=(8, 1, 224, 224)
location_coords min=-77.3  max=39.6        <- real Maryland lat/lon
year            min=2015   max=2022        <- train years only, no leakage
```

Optional GPU sanity checks:

```bash
uv run python -m farm_us.cli smoke-test-model --config configs/experiments/maryland_soybeans.yaml --real
uv run python -m farm_us.cli profile-memory  --config configs/experiments/maryland_soybeans.yaml --real
```

`profile-memory` builds a real AdamW optimizer and takes a real step, so its peak
figure reflects actual training. On a 47 GiB GPU: `batch_size=2, grad_accum=4`
profiles at **33.8 GB** (effective batch 8, matching the paper). `batch_size=8`
OOMs at ~49 GB.

---

## 3. Training

### 3.1 Launch detached (survives disconnection)

```bash
cd /home/cholab/LabMembers/Samar/EO-based-yield-prediction/model
export HF_HOME="$PWD/.hf_cache"

LOGFILE="outputs/logs/maryland_train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/logs

setsid nohup uv run python -m farm_us.cli train \
  --config configs/experiments/maryland_soybeans.yaml --real \
  > "$LOGFILE" 2>&1 < /dev/null &
disown

echo "launched. log: $LOGFILE"
```

Why each piece:

| Piece           | Purpose                                                   |
| --------------- | --------------------------------------------------------- |
| `setsid`      | own session leader — detached from the terminal entirely |
| `nohup`       | ignores`SIGHUP` sent when a terminal closes             |
| `> LOG 2>&1`  | output to a real file, not a pipe that can disappear      |
| `< /dev/null` | never blocks on terminal stdin                            |
| `& disown`    | background it, drop it from the shell's job table         |

A plain `&` is **not** enough — the job dies with the terminal.

### 3.2 Resuming an interrupted run

```bash
uv run python -m farm_us.cli train \
  --config configs/experiments/maryland_soybeans.yaml --real \
  resume_from=outputs/runs/maryland_soybeans/test2024/checkpoints/last-v1.ckpt
```

Restores weights, optimizer state and LR-scheduler position, and continues the
cosine schedule rather than restarting warmup.

### 3.3 Useful overrides

```bash
train.epochs=60                       # shorter run
train.batch_size=1 train.grad_accum=8 # smaller GPU, same effective batch
train.periodic_ckpt_every_n_epochs=5  # more frequent recovery snapshots
test_year=2023 val_years=[2022]       # a different LOYO fold
```

---

## 4. Monitoring

```bash
# alive? GPU busy?
ps aux | grep "farm_us.cli train" | grep -v grep
nvidia-smi

# live log
tail -f outputs/logs/maryland_train_*.log

# per-epoch validation metrics (the real numbers)
CSV=$(ls -t outputs/runs/maryland_soybeans/test2024/csv/version_*/metrics.csv | head -1)
awk -F',' 'NR>1 && $16!="" {printf "epoch=%-4s rmse=%-8.4f mae=%-8.4f r2=%-8.4f r=%.4f\n", $1,$16,$13,$15,$14}' "$CSV"
```

TensorBoard (live charts). Run on the server:

```bash
uv run tensorboard --logdir outputs/runs/maryland_soybeans/test2024/tb --port 6006
```

Then from **your local machine**, tunnel and open `http://localhost:6006`:

```bash
ssh -L 6006:localhost:6006 cholab@100.125.210.36
```

**Expect ~1 hour of silence with the GPU at 0% after launch.** That is
`compute_fold_stats` reading every training chip once to measure normalization
statistics. It is not a hang, and it is not resumable — it re-runs on every
launch. GPU activity begins only after it finishes.

### Reading the metrics

Logged every step, for both `train/` and `val/`:

| Metric                         | Meaning                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| `{stage}/loss`               | `main_loss + 0.2*aux_loss` — the optimized quantity, in **z-scored** units |
| `{stage}/aux_loss`           | auxiliary head (block 24) loss alone, before weighting                              |
| `{stage}/valid_px`           | valid pixels in that batch (crop ∧ label ∧ HLS)                                   |
| `{stage}/skipped_zero_valid` | flag: batch had zero valid pixels                                                   |

Logged once per epoch, pooled over the whole validation set, in **physical
units** (`_phys`):

| Metric                 | Meaning                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `val/main_rmse_phys` | RMSE in bu/ac —**this drives checkpoint selection** |
| `val/main_mae_phys`  | MAE in bu/ac                                               |
| `val/main_r2`        | R² — negative means worse than predicting the mean       |
| `val/main_pearson_r` | correlation — can be good even when R² is poor           |

Only RMSE/MAE carry `_phys` because they are scale-dependent; R² and Pearson r
are scale-invariant.

---

## 5. Checkpoints

Written to `outputs/runs/maryland_soybeans/test2024/checkpoints/`:

| File                           | Meaning                                                  |
| ------------------------------ | -------------------------------------------------------- |
| `farm-{epoch}-{rmse}.ckpt`   | **best** validation RMSE so far (`save_top_k=1`) |
| `farm-periodic-{epoch}.ckpt` | rolling snapshot every 10 epochs, regardless of quality  |
| `last.ckpt`                  | Lightning's "last" — see caveat                         |

**Caveat on `last.ckpt`:** Lightning only refreshes it when the *monitored metric
also improves*. During a long plateau it goes stale, which is why the periodic
checkpoint exists — use `farm-periodic-*.ckpt` for resuming after a crash.

Each checkpoint is ~9.9 GB. Tune frequency with
`train.periodic_ckpt_every_n_epochs`.

Alongside the checkpoints, each run writes `resolved_config.yaml` (fully resolved
settings), `norm_stats.json` (fold normalization) and `provenance.json` (git
commit, package versions, fold years). These are **overwritten** on each launch.

---

## 6. Evaluating on the held-out test year

```bash
export HF_HOME="$PWD/.hf_cache"
CKPT=outputs/runs/maryland_soybeans/test2024/checkpoints/farm-032-0.0000.ckpt   # your best checkpoint

LOGFILE="outputs/logs/maryland_eval_$(date +%Y%m%d_%H%M%S).log"
setsid nohup uv run python -m farm_us.cli evaluate \
  --config configs/experiments/maryland_soybeans.yaml --real \
  checkpoint="$CKPT" \
  > "$LOGFILE" 2>&1 < /dev/null &
disown
echo "launched. log: $LOGFILE"
```

Results land in `outputs/runs/maryland_soybeans/test2024/eval/<checkpoint-name>/`
— the directory is keyed on the checkpoint filename, so evaluating a second
checkpoint never overwrites the first.

| Output                   | Contents                                                    |
| ------------------------ | ----------------------------------------------------------- |
| `scatter_obs_pred.png` | predicted vs observed, one dot per pixel, 1:1 line          |
| `residual_hist.png`    | error distribution — centred on 0 = unbiased               |
| `residual_vs_obs.png`  | error vs true yield — a slope means regression-to-the-mean |
| `per_chip_metrics.csv` | per-chip MAE/RMSE/R²/bias — find*which* chips fail      |
| `eval_pixels.npz`      | raw pixel arrays, so plots can be regenerated cheaply       |

Headline numbers appear in the log:
`Eval (test year 2024, checkpoint ...): {'mae': ..., 'rmse': ..., 'r2': ..., 'pearson_r': ...}`

All metrics are computed **only** over valid pixels (crop ∧ valid label ∧ valid
HLS), never over the whole 224×224 frame.

### Re-plotting without re-running (seconds, no GPU)

```bash
uv run python scripts/replot_eval.py \
  --eval-dir outputs/runs/maryland_soybeans/test2024/eval/<checkpoint-name> \
  --alpha 0.02
```

Lower `--alpha` reveals density in dense scatters; ~0.02 suits ~1M points.

---

## 7. Spatial error maps

```bash
uv run python scripts/map_test_errors.py \
  --config configs/experiments/maryland_soybeans.yaml \
  checkpoint="$CKPT" state=MD year=2024
```

Fast (minutes) — loads the saved `norm_stats.json` instead of recomputing.
Writes to `outputs/predictions/`:

| File                                              | Footprint                                                                    |
| ------------------------------------------------- | ---------------------------------------------------------------------------- |
| `_pred.tif`, `_actual.tif`, `_residual.tif` | **identical** — the comparison set                                    |
| `_pred_full.tif`                                | every crop pixel the model predicted;**not** comparable to `_actual` |
| `_comparison_mask.tif`                          | 1 where the three above are valid                                            |
| `_error_map.png`                                | actual yield (green) with error overlaid (red/blue)                          |

`_pred.tif` and `_pred_full.tif` differ because the model can predict where no
test label exists (~194k pixels for MD 2024). Only ever compare `_pred.tif`
against `_actual.tif`; comparing `_pred_full.tif` will show phantom coverage.

Open the GeoTIFFs in QGIS to inspect interactively.

---

## 8. Full LOYO across all folds

```bash
setsid nohup uv run python -m farm_us.cli run-loyo \
  --config configs/experiments/maryland_soybeans.yaml --real \
  > outputs/logs/loyo_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
disown
```

Trains and evaluates one fold per year. Very long — 11 folds × (1 h stats +
training). Prefer individual `test_year=` runs unless you need the full sweep.

---

## 9. Gotchas

**The ~1 hour startup pass.** Every `train` / `evaluate` / `inspect-batch --real`
recomputes fold normalization statistics before any GPU work. Not resumable.
`map_test_errors.py` and `replot_eval.py` skip it by loading saved stats.

**Checkpoints trained before commit `b5f4707` are invalid.** Three bugs were
fixed that changed what the model was trained on:

1. augmentation flipped the `[1,H,W]` label on the wrong axis, desynchronising
   labels from masks on ~36% of samples and corrupting the target scaler;
2. label rasters sit one 30 m pixel east of the CDL/HLS grid, so every label was
   paired with imagery 30 m away;
3. the predicted map covered ~13% more pixels than the actual map.

Anything trained earlier must be retrained from scratch — resuming carries the
same corruption in the optimizer state.

**Stale checkpoints occupy ~47 GB.** They are not deleted automatically and
`save_top_k=1` only tracks within a single Trainer instance. Remove old ones
manually before a fresh run if disk matters.

**`min_valid_months` is effectively 1, not 5.** `dataset.py` computed it as
`max_missing_months and 1 or 1`, which is the constant `1` for every input, so
the configured `max_missing_months: 3` has never applied. A pixel with only 1 of
8 usable months currently counts as valid for training *and* metrics. Behaviour
is preserved deliberately in `masks.min_valid_months()` — changing it alters
which pixels train and score, a scientific decision, not a refactor.

**`train_years` in `norm_stats.json` is hardcoded to `[2019]`.** A provenance
label bug in `compute_fold_stats`; the statistics themselves are computed from
all training years correctly.

**Apple MPS is unsupported** — the PPM's `[1,2,3,6]` bins break MPS adaptive
pooling. Use `train.accelerator: cpu` on Macs; CUDA is unaffected.

---

## 10. Quick reference

```bash
cd model
export HF_HOME="$PWD/.hf_cache"
CFG=configs/experiments/maryland_soybeans.yaml

uv run python -m farm_us.cli inventory      --config $CFG
uv run python -m farm_us.cli build-manifest  --config $CFG
uv run python -m farm_us.cli verify-splits   --config $CFG
uv run python -m farm_us.cli leakage-audit   --config $CFG
uv run python -m farm_us.cli inspect-batch   --config $CFG --real

setsid nohup uv run python -m farm_us.cli train --config $CFG --real \
  > outputs/logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & disown

setsid nohup uv run python -m farm_us.cli evaluate --config $CFG --real \
  checkpoint=<path> > outputs/logs/eval_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & disown

uv run python scripts/map_test_errors.py --config $CFG checkpoint=<path> state=MD year=2024
uv run python scripts/replot_eval.py --eval-dir <eval-dir> --alpha 0.02
```
