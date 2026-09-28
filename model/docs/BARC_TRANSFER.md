# BARC High-Resolution Transfer

BARC (USDA Beltsville Agricultural Research Center, Prince George's County MD —
GEOID 24033) is the US analogue of the paper's 10 m yield-monitor dataset:
**measured / much-more-direct 30 m yield**, kept strictly separate from the
national ridge-distributed pseudo-labels. BARC is **excluded from national
training** (upstream `EXCLUDE_GEOIDS`, plus `data.exclude_geoids`).

## Three experiments (`transfer/barc_experiments.py`, mirror paper §4.4)
| Name | Init | Weights updated | Loss | Paper result (canola) |
|---|---|---|---|---|
| `zero_shot` | national FARM ckpt | none | — | R²=0.508 |
| `finetune_from_farm` | national FARM ckpt | yes | MSE | **R²=0.768 (best)** |
| `train_from_prithvi` | original Prithvi | yes (fresh decoder/head) | Huber/heteroscedastic | R²=0.675 |

Configs: `configs/barc/barc_zero_shot.yaml`, `barc_finetune.yaml`,
`barc_from_prithvi.yaml`.

```bash
uv run python scripts/run_barc_transfer.py --config configs/barc/barc_finetune.yaml \
    experiment=finetune_from_farm national_checkpoint=<farm.ckpt>
```

## Protocol
- **LOYO on BARC years** with strict year separation (same rules as national).
- Never mix BARC into national training unless a config explicitly defines that.
- The zero-shot experiment applies the county-trained checkpoint **without weight
  updates**; fine-tune selects checkpoints on a BARC validation year and tests on
  a held-out BARC year.
- `train_from_prithvi` may use a **heteroscedastic** loss (implemented,
  `losses.masked_heteroscedastic`) and extra brightness/contrast augmentation, per
  the paper.

## Metrics
Pixel · **field-level** · field-year · annual · mapped residuals. Field-level
needs a per-field id raster (`BarcConfig.field_id_raster`) — aggregate predicted
pixels per field before scoring, analogous to county aggregation.

## Status (2026-09-26)

Real BARC rasters **are on disk** and all experiments below have been run against
the real 600M backbone. Earlier text in this file saying otherwise was from
scaffolding.

```
data_preparation/data/barc_data/
  yield_dataset/   barc_soybeans_yield_{2014..2024}_30m.tif   measured 30 m yield
                   barc_field_id_map.{csv,json}               per-field ids
  yield_labels/{year}/                                        reader-pattern tree
  cdl_masks/       cdl_soybeans_BARC_{2014..2024}.tif
  HLS_Composites/
  BARC_region_shapefile/
```

`data.states: [BARC]` selects it; `min_crop_fraction: 0.0001` is **mandatory**
(BARC chips are mostly non-crop). About **36 qualifying chips over 11 years, 2–4
per fold** — per-fold metrics are noisy, so pool field-level results across folds
with `scripts/barc_field_metrics.py`.

## Fourth experiment: `refiner_only` (the one that worked)

`configs/experiments/barc_refiner.yaml` / `barc_refiner_hilr.yaml` /
`barc_refiner_hilr_val2.yaml`

```bash
uv run python -m farm_us.cli run-loyo --config configs/experiments/barc_refiner_hilr.yaml --real \
  init_from=outputs/runs/cornbelt5_soybeans_longer_time/test2024/checkpoints/farm-018-0.0000.ckpt
```

The 775M base stays **exactly** farm-018 (frozen, BN/dropout in eval); only the
20,801-parameter `DetailRefiner` trains. Requires `norm.inherit_init_stats: true`
— a frozen model must see the normalisation it was trained with, and recomputing
stats on ~36 chips would break the step-0 equivalence.

### Measured results (pixel `pearson_r2`, measured labels)

| Approach | Result |
|---|---|
| Zero-shot from farm-018 | r² −0.63, `pearson_r2` 0.023, bias **+10.9 bu/ac** |
| Decoder fine-tune (`lr 2e-7`) | `pearson_r2` 0.001–0.25 across folds — no sub-field skill |
| **`refiner_only`, 11-fold LOYO** | mean `pearson_r2` **0.465**, MAE 11.91 bu/ac |

Per-fold range 0.125 (2014) to 0.682 (2017). Full table and per-year scatter plots:
`outputs/comparisons/val1_vs_val2.md`.

This is the project's main positive result: it says the binding constraint on
sub-field accuracy was the **256-DOF tokenisation ceiling** (see ARCHITECTURE.md),
not the encoder's representation. Predicted std was 1.09 vs measured 13.56 bu/ac
before the branch existed.

**Hyperparameters are load-bearing.** `lr 1e-2, weight_decay 0.0`. At the original
`lr 1e-4, wd 0.01` weight decay pulled the zero-initialised output layer back
toward zero as fast as it learned and the branch barely moved.

### Validation-split variant
`configs/splits/loyo_soybeans_2val.yaml` validates on the years before **and**
after the test year (8 training years instead of 9), which fixed 2015 — a fold
that validated on 2014 alone and stopped after 1 epoch. Over 11 folds it is
roughly neutral: mean `pearson_r2` 0.453 (2 val) vs 0.465 (1 val). Prefer 1 val
year unless a specific fold's early stopping is the problem.

### Caveat carried into every BARC number
farm-018 was trained on 2014–2022 Corn Belt pseudo-labels **with MD included**, and
BARC is in MD, so folds 2014–2022 are not held out from the *base* model. The
measured BARC labels are unseen in every fold, but the base model's exposure to
MD pseudo-labels is not year-clean. State this when reporting.

## Resolution note (paper)
The paper upsampled 10 m monitor data 10 m→5 m (112→224 px) to match the
county-model input spec. For BARC at native 30 m the chips already match; if a
higher-resolution BARC product is used, apply the same extent-preserving
upsampling before inference.
