# FARM-US: soil/weather refiner — implementation and status

**Reviewed and updated for spatial weather: 28 September 2026.** Remote repository: `/home/cholab/LabMembers/Samar/EO-based-yield-prediction`. This describes the current working tree, including uncommitted changes, rather than GitHub alone. Base commit: `db2d533`.

**Spatial weather revision:** the former chip-average weather MLP has been replaced by nearest-sampled weather maps and 1×1 convolutions producing per-pixel conditioning. See [EXOGENOUS_REFINER.md](EXOGENOUS_REFINER.md) for the current contract and checkpoint compatibility.

## 1. Where the project stands

**The ExogenousRefiner is implemented and connected to data loading, training, evaluation, and raster inference. Its improvement in yield prediction has not yet been demonstrated.**

| Item                           | Current status                                                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| Existing FARM model            | Trained five-state`farm-018` checkpoint exists.                                                                                    |
| Earlier BARC work              | Existing DetailRefiner experiments exist; their results are not results for the new soil/weather branch.                             |
| New architecture               | Small spatial CNN + pointwise weather CNN, with an additive yield correction.                                                        |
| Soil files                     | Four merged 0–30 cm mosaics: clay, bulk density, saturated hydraulic conductivity, organic matter.                                  |
| Weather files                  | Precipitation, mean temperature, derived VPD; 88 monthly files each, April–November 2014–2024. All required filenames are present. |
| Standardization                | Implemented for both soil and weather, with train-only statistics and transfer safeguards.                                           |
| Training configuration         | `model/configs/experiments/cornbelt5_soil_weather.yaml` implements branch-only Stage 1.                                            |
| Completed exogenous runs found | `smoke_dummy_exo`: synthetic data/dummy encoder. No completed real five-state exogenous run found under `model/outputs/runs`.    |
| Supervision                    | Existing redistributed pixel labels and masked loss remain; no county-aggregate loss has been added.                                 |

Review covered the repository structure, relevant preprocessing scripts, model/data/training/evaluation code, configurations, tests, documentation, raster metadata, and run artifacts. Real raster reads were sampled; this was not an exhaustive scan of every raster pixel or checkpoint.

## 2. How the architecture connects

```text
Standardized HLS [6 bands × 8 months × 224 × 224]
    │
    ├── Existing FARM: Prithvi → temporal reducers → UPerNet → regression head
    │                                                        ├─ coarse yield map
    │                                                        └─ 64-channel features
    │
    └── Stack bands/months into 48 spatial channels
                  + 4 standardized soil channels
                  ↓
             Small CNN → 16 feature maps ← weather CNN scales/shifts these maps per pixel
                  ↓                              ↑
        Combine with upsampled FARM features     3 variables × 8 monthly maps
                  ↓
             CNN → one correction per 30 m pixel
                  ↓
        Final yield = coarse FARM yield + correction
```

The current configuration has **22,321 trainable branch parameters**. The existing FARM weights are frozen during Stage 1. Soil and weather do not change Prithvi's six-band input.

`ExogenousRefiner` is an **alternative to** the old `DetailRefiner`, not another branch stacked onto it. Enabling both raises an error. With both covariate lists empty, ExogenousRefiner has the same image-only structure as DetailRefiner; this is the intended control experiment.

### Inside the branch

OperationShape or purposeReshape HLS`[B,6,8,224,224]` → `[B,48,224,224]`; only the branch makes this reshape.Concatenate soilFour maps → 52 total input channels.Spatial CNNTwo 3×3 convolutions:`52 → 16 → 16`, each with ReLU; no pooling.Weather network`[B,3,8,224,224]` → 24 ordered channels → 1×1 convolution/ReLU → separate 1×1 scale and shift heads, each `[B,16,224,224]`.Weather conditioning`features × (1 + scale_map) + shift_map`, independently conditioned at each pixel. This is called FiLM.FARM feature connectionRegression-head body has 64 channels on the current 16×16 grid; bilinearly resize to 224×224.Fusion/outputConcatenate 16 + 64 channels; 3×3 convolution`80 → 16`, ReLU, then 1×1 convolution `16 → 1`.

There is no transformer in the new branch and no BatchNorm. Weather changes how the branch responds to local soil/imagery patterns. Its pointwise network retains differences between PRISM cells; it does not blend neighboring weather values.

The final convolution starts with zero weights and bias; weather scale/shift outputs also start at zero. Thus, with matching weights, normalization, and evaluation behavior, the initial prediction equals FARM's prediction. **This guarantees the starting point, not improvement after training.** Corrections can increase or decrease yield and change both its average level and spatial variation.

The current FARM head predicts on a coarse grid and interpolates to 30 m; the branch provides a direct full-resolution path. This does not mean Prithvi necessarily discards all within-patch information, or that every flat prediction has an architectural cause.

## 3. Soil: static maps with spatial detail

Storage, relative to the repository root:

```text
data_preparation/data/soil/polaris/weightedaverage_merged/
    clay_0-30_weighted_average.tif
    bd_0-30_weighted_average.tif
    ksat_0-30_weighted_average.tif
    om_0-30_weighted_average.tif
```

| Channel order | Variable | Stored physical units |
| ------------- | -------- | --------------------- |
| 1             | `clay` | Percent               |
| 2             | `bd`   | g/cm³                |
| 3             | `ksat` | cm/hour               |
| 4             | `om`   | Percent               |

These are **one static file per property**, reused from 2014 through 2024. The model receives four depth-summary channels, not separate depth layers or eight repetitions of soil for eight months.

The preprocessing script combines 0–5, 5–15, and 15–30 cm using thickness weights:

```text
0–30 cm summary = (5 × shallow + 10 × middle + 15 × deeper) / 30
```

For `ksat` and `om`, source log10 values are converted using `10**value` **before** depth averaging. The merged inputs are already linear: do not exponentiate them again. The script renormalizes weights over available valid depths; a finite output alone does not prove all three depths were present. The ksat average is a model feature, not a calculation of effective whole-profile water flow.

Checked mosaics use EPSG:4326 at **1 arc-second**, nominally 30 m; their ground dimensions vary with latitude. At each read, `SoilReader` bilinearly reprojects them onto the exact 224×224 training chip grid. Soil is never averaged into Prithvi tokens. Standardization changes values, not spatial resolution.

## 4. Weather: monthly maps for every pixel

```text
data_preparation/data/prism/monthly/<variable>/prism_<variable>_<YYYYMM>.tif
```

| Channel order | Variable    | Meaning                                                                                                             |
| ------------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 1             | `ppt`     | Monthly precipitation total, mm                                                                                     |
| 2             | `tmean`   | Monthly mean temperature, °C                                                                                       |
| 3             | `vpdmean` | Derived`(vpdmin + vpdmax) / 2`, hPa; a midpoint proxy, not a separately downloaded true monthly mean VPD product. |

The checked PRISM files use EPSG:4269 at **30 arc-seconds**, nominally 800 m. A 6.72 km chip spans many weather cells; weather is not physically constant across it.

For each variable and month, the reader:

1. Selects the raster for the sample's year and calendar month.
2. Reprojects it with **nearest-neighbor** sampling onto the chip grid.
3. Retains the complete map: each target pixel receives its source PRISM cell value, including differences between cells.
4. Stacks April through November, producing `[3 variables, 8 months, 224, 224]`.

There is no chip averaging or fallback chip mean. Missing source cells remain NaN until standardization/imputation. Nearest-neighbor alignment repeats observed source-cell values; it does not invent 30 m weather information.

The pointwise weather CNN receives all 24 values at each pixel in fixed variable/month order, so it can distinguish April rain from August rain. It does not run separately each month or model daily events. The monthly sequence is fixed in the current readers; changing its length/calendar requires coordinated changes.

**Temporal detail:** weather uses whole calendar months. The HLS November composite uses a first-half-of-November window. Month order matches, but these November observation windows are not identical. Full-November weather is suitable only when available at the intended prediction date; it would look ahead for a November 15 forecast.

## 5. Normalization: already implemented

Both new input groups use **z-score standardization**, not min–max scaling:

```text
standardized value = (physical value − training mean) / training standard deviation
```

| Input   | Statistics used                                                                                                                                                       |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HLS     | Existing`farm-018` band means/stds, inherited unchanged. Reflectance conversion precedes standardization.                                                           |
| Yield   | Existing target center/scale, inherited unchanged: approximately 55.4985 and 9.1882 bu/ac.                                                                            |
| Soil    | One mean/std per property, from finite crop pixels with valid labels in sampled training chips.                                                                       |
| Weather | One mean/std per variable, pooling finite crop/valid-label pixels in sampled training chips across all eight months.**Not separate statistics for each month.** |

The current setup samples up to **2,000 training chips, seed 0**, for new covariate statistics. Missing values are excluded before calculating statistics. Soil may be observed repeatedly at the same location across training years; these are statistics of sampled chip-years, not unique locations.

Since farm-018 has no covariate statistics, a dedicated pass reads label/CDL grids, soil, and weather **without loading HLS**. The resulting `norm_stats.json` records variable order, means/stds, finite coverage, and `weather_layout=spatial_nearest_crop_label_v1`. Coverage below the configured default 95% raises an error during this statistics pass.

At model-input time, missing values become zero **after** standardization—equivalent to mean imputation. Zero rainfall is a real value and remains distinct from missing data while statistics are computed. There is no explicit soil/weather missingness channel.

Validation, test, and raster inference reuse the saved statistics. During later transfer, matching soil/weather statistics are preserved; different variable names/order or a legacy weather layout cause an error, and only absent groups are computed. Old chip-mean weather statistics must not be reused. These safeguards prevent silently rescaling a trained branch.

The branch predicts in standardized yield units:

```text
physical yield = 55.4985 + 9.1882 × (FARM standardized prediction + correction)
```

Thus, a correction of `+1` means approximately **+9.19 bu/ac**, not +1 bu/ac. Current preprocessing does not apply an additional log transform to the linear ksat/OM inputs; standardization does not remove skew or outliers.

## 6. What happens for each training sample

1. The manifest supplies state, year, chip location, and coordinates; imagery/QC filters determine eligible chips.
2. The label window defines the geographic grid. HLS and CDL are read in geographic alignment.
3. HLS missing-month handling and inherited standardization are applied.
4. Four static soil maps are read for that location; 24 weather rasters are read for that year and retained as spatial maps.
5. Soil/weather are standardized with saved statistics. Soil and weather are flipped together with HLS, labels, and masks during spatial augmentation. Variable/month axes remain fixed; image noise is not added to soil/weather.
6. The batch contains `image`, `soil`, `weather`, scaled `label`, validity `mask`, and temporal/location coordinates.
7. FARM produces coarse predictions/features; ExogenousRefiner produces corrections; the existing masked loss trains the permitted parameters.

Readers open/close files within each operation rather than sharing open GDAL handles across worker processes. Static means stored once, **not cached in memory**: soil is read again when that location is sampled. Covariates are assembled on demand rather than saved as per-chip files. Frozen FARM also still performs forward computation; the tiny trainable parameter count does not imply negligible runtime.

## 7. What is optimized now—and what is not yet configured

Current experiment: IA, IL, IN, MD, MN; **training 2014–2022, validation 2023, test 2024**. There are 82 explicit corrupt-chip exclusions.

| Setting           | Current value/behavior                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Mode              | `exo_refiner_only`: train spatial CNN, pointwise weather CNN/FiLM, fusion/output together.                                                                       |
| Frozen components | Encoder, temporal reducers, decoder, main head, auxiliary branch. Frozen modules also stay in evaluation mode to protect BatchNorm statistics and disable dropout. |
| Initialization    | `init_from=farm-018`: weights only; fresh optimizer, schedule, and epoch counter.                                                                                |
| Optimizer         | AdamW on parameters with`requires_grad=True`; LR `0.01`, weight decay `0`.                                                                                   |
| Schedule          | Two warm-up epochs, cosine decay to`1e-6`; maximum 60 epochs.                                                                                                    |
| Batch/precision   | Batch 8, 16-bit mixed precision, gradient clipping 1.0.                                                                                                            |
| Selection         | Lowest`val/main_rmse_phys`; early-stopping patience 15.                                                                                                          |
| Loss              | Masked pixel MSE + 0.2 × auxiliary MSE. Frozen auxiliary output contributes no gradient to the branch.                                                            |

Old chip-mean weather checkpoints have incompatible Linear weights; initialize this spatial branch from farm-018. Weather inputs now require about 4.6 MiB per chip, plus activations and prefetched batches.

Use `resume_from` only to continue an interrupted run with the same architecture. A legacy BARC DetailRefiner checkpoint is not directly interchangeable with ExogenousRefiner: old branch keys are unexpected. Start this experiment from the branch-free farm-018 checkpoint.

**Later stages need deliberate configuration:** `decoder_only` can train decoder/reducers/heads and the new branch with Prithvi frozen, but the optimizer currently uses one LR group. Separate low decoder/high branch rates are not implemented. Likewise, `exo_refiner_only` trains the weather network too; it does **not** implement the proposed BARC option of freezing weather while training only the spatial refiner.

## 8. Verification and remaining interpretation limits

### Spatial weather revision verification

- Full regression suite on the modified source: **154 passed, 1 skipped** (142 seconds).
- The 49 focused tests cover spatial weather conditioning, reader values, monthly ordering, standardization, augmentation, inference parity, and legacy checkpoint/statistics rejection.
- The updated real-data reader check samples 40 chip-years across five states and reports pixel-month coverage. No model training experiment was launched by this change.

### Checks performed for the initial review (before the spatial revision)

- Full repository pytest suite: **147 passed, 1 skipped** in 135.78 seconds. The skipped alignment test could not find its expected real-label fixture; this does not mean the project's actual label data are absent. These checks do not replace a real exogenous training pilot.
- File preflight: zero missing configured soil/weather files for all 11 years; 4 soil mosaics and 264 weather files.
- Real-data reader check: 40 chip-years, two locations per state across 2015/2018/2021/2023. All sampled soil crop pixels were finite; no missing weather summaries. This is a sample, not a complete coverage certificate.
- Sampled median within-chip weather spread was approximately 5–8% of between-year spread: precipitation 3.36 versus 40.13 mm, temperature 0.058 versus 1.189 °C, VPD 0.079 versus 1.039 hPa. These were diagnostics of the previous chip-summary design; the spatial implementation now retains this variation.

### Important limits

1. **No demonstrated covariate gain yet.** Compare unchanged FARM, an HLS-only ExogenousRefiner, and the full soil/weather branch under matching splits/training conditions. Evaluate measured BARC yield separately from pseudo-label fit.
2. **The supervision is still redistributed county yield.** Better agreement with those pixel labels does not establish true soil-driven within-field accuracy. Loss changes remain outside this experiment as requested.
3. **Static soil can identify locations.** Temporal holdout does not establish transfer to new regions. A strict spatial holdout also requires checking that the starting FARM checkpoint did not train on the held-out region.
4. **Existing normalization caveat:** the fresh HLS statistics path iterates an augmented training dataset; image noise can affect newly computed band statistics. Stage 1 inherits farm-018 statistics and uses the separate unaugmented covariate pass, so do not silently change the inherited HLS scaling for this experiment.
5. **Coverage/provenance are partial.** File existence does not guarantee full spatial/depth coverage. The 95% check concerns sampled training statistics, and spatial weather coverage counts finite pixel-months over the selected crop/valid-label mask. Inherited normalization metadata also does not separately record a complete new covariate sampling fingerprint.

**Next practical phase:** run a bounded real-model training pilot to check memory, throughput, finite losses, and LR stability; then run the matched HLS-only/full-covariate comparison. Keep the 2024 test year for final comparison rather than tuning. Do not interpret the synthetic smoke checkpoint as an accuracy result.

## 9. Code map

Paths below are relative to the repository root.

| Question                                          | Source                                                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| How does the branch compute its correction?       | `model/src/farm_us/models/exogenous_refiner.py`                                                       |
| Where is it added and what stays frozen?          | `model/src/farm_us/models/farm_model.py`                                                              |
| How are soil/weather read and standardized?       | `model/src/farm_us/data/exogenous_readers.py`                                                         |
| How are training statistics accumulated?          | `model/src/farm_us/data/normalization.py`                                                             |
| How are stats inherited and weights initialized?  | `model/src/farm_us/training/run.py`                                                                   |
| How is a batch assembled/aligned?                 | `model/src/farm_us/data/dataset.py`, `raster_readers.py`, `transforms.py`                         |
| Which loss, optimizer, and metrics run?           | `model/src/farm_us/training/lightning_module.py`                                                      |
| How are inputs used during evaluation/maps?       | `model/src/farm_us/evaluation/evaluator.py`, `inference.py`                                         |
| Which variables and hyperparameters are selected? | `model/configs/experiments/cornbelt5_soil_weather.yaml`                                               |
| How are soil depths and PRISM variables prepared? | `data_preparation/scripts/08_preprocess_polaris_depth_summary.py`, `09_preprocess_prism_monthly.py` |
| Which tests cover the addition?                   | `model/tests/test_exogenous_refiner.py`, `test_exogenous_stats.py`                                  |
| How can real reads be checked?                    | `model/scripts/smoke_test_exogenous_readers.py`                                                       |

Existing `model/docs/EXOGENOUS_REFINER.md` provides additional context; this reviewed guide clarifies that branch HLS is standardized, FARM body features are initially coarse, November windows differ, and zero initialization is not a guarantee against later degradation.
