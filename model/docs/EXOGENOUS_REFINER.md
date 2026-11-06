# ExogenousRefiner: spatial soil and monthly weather

Updated 28 September 2026: **weather maps replace the previous chip-average
weather vector.** No new downloads or manifest rebuild are required.

## Architecture

```text
HLS [B,6,8,H,W] ── existing FARM ── coarse yield + 64-channel head features
     │                                                  │
     └─ reshape [B,48,H,W] + soil [B,4,H,W]               │
                  ↓                                     │
          Conv3×3 → ReLU → Conv3×3 → ReLU                │
                  ↓                                     │
       spatial FiLM: h × (1 + scale_map) + shift_map      │
                  ↑                                     │
weather [B,3,8,H,W]                                      │
   → reshape [B,24,H,W]                                  │
   → Conv1×1 → ReLU → separate Conv1×1 scale/shift heads  │
                  ↓                                     ↓
             concatenate with upsampled FARM head features
                  ↓
          Conv3×3 → ReLU → Conv1×1 → yield correction
                  ↓
         final yield = FARM yield + correction
```

Default H=W=224, hidden width 16, branch size **22,321 parameters**. Changing
Linear layers to 1×1 convolutions preserves parameter count, but increases
activation/input memory. The weather encoder operates independently at each
pixel; it mixes monthly variables but not neighboring pixels. Soil/HLS/fusion
3×3 convolutions learn neighborhood effects at full resolution without pooling.
The model can smooth or change detail; preserving inputs is not an accuracy
guarantee.

The output layer and FiLM heads start at zero. Thus the branch initially leaves
FARM predictions unchanged. Later corrections can be positive or negative.
Prithvi still receives six HLS bands, and its architecture is unchanged.
`ExogenousRefiner` and legacy `DetailRefiner` remain mutually exclusive.

## Input files and spatial handling

| Input | Files | Model handling |
|---|---|---|
| HLS | Existing monthly composites | Inherited reflectance scaling and band standardization; full 30 m grid |
| Soil | `data_preparation/data/soil/polaris/weightedaverage_merged/<var>_0-30_weighted_average.tif` | Four static maps (`clay,bd,ksat,om`); bilinear alignment to the chip; reused across years |
| Weather | `data_preparation/data/prism/monthly/<var>/prism_<var>_<YYYYMM>.tif` | `ppt,tmean,vpdmean`, April–November; nearest-neighbor alignment to each 30 m pixel, **no chip averaging** |

Weather shape before batching is `[variables, months, height, width]`.
Every target pixel receives its source PRISM cell value, so nearby pixels in
the same cell repeat the same value. Differences between cells remain.
Repeating nominal 800 m observations does not create independent 30 m weather.
There is no bilinear weather interpolation, crop-weighted chip mean, or fallback
chip mean in the model-input path. The crop mask selects statistics/loss pixels;
it does not collapse weather maps.

Month order is fixed April–November. All 24 variable/month combinations are
available at every pixel. PRISM November is a whole calendar month; HLS uses a
first-half-of-November window. Account for data availability if forecasting
before the end of November. `vpdmean` is the midpoint of monthly vpdmin/vpdmax,
not an independently supplied mean-VPD product.

Soil remains a thickness-weighted 0–30 cm summary of three source layers.
Ksat/OM have already been back-transformed from log10 to linear values before
depth averaging. Neither averaging depths again nor exponentiating these
merged inputs again is appropriate.

## Standardization and missing values

1. Inherit FARM's original HLS and target scaling with `inherit_init_stats`.
2. Compute absent soil/weather means and standard deviations from sampled
   **training** chips only (up to 2,000, seed 0 in the current experiment).
3. For each weather variable, pool finite values over crop pixels with valid
   labels, across all months. One mean/std per variable, not per month or pixel.
   These are target-grid, area-weighted pixel observations; repeated PRISM cells
   are not counted as independent measurements for scientific inference.
4. Standardize maps with `(value - mean) / std`. Fill missing cells with zero
   afterward, representing the training mean. Keep raw NaNs for statistics.
5. Record finite coverage and reject coverage below 95% in the stats pass.
   Weather coverage now measures finite pixel-months, not finite chip summaries.
6. Save/reuse exact variable order and `weather_layout=spatial_nearest_crop_label_v1`.
   Legacy chip-average statistics are rejected; their spread is not that of the
   new pixel-level inputs.

The dedicated covariate statistics pass reads label/CDL/soil/weather without
HLS. Static soil is read on demand, not duplicated into yearly files. Readers
open/close files within each operation, keeping DataLoader worker handles safe.
Training, evaluation, and raster inference use the same input preparation.

Soil and weather both undergo the same spatial flips as HLS, labels, and masks.
Only spatial axes flip: month/variable order stays fixed. Image noise does not
affect soil/weather. No new missingness channel has been added.

## Training and checkpoint compatibility

`cornbelt5_soil_weather.yaml` uses `exo_refiner_only`: train the branch while
FARM weights, BatchNorm statistics, and dropout behavior remain frozen.
The existing pixel labels/loss, splits, and FARM architecture are unchanged.
Use a fresh run initialized from the branch-free farm-018 checkpoint:

```bash
# Run from model/; this starts real training when explicitly launched.
uv run python -m farm_us.cli train --config configs/experiments/cornbelt5_soil_weather.yaml --real \
  init_from=outputs/runs/cornbelt5_soybeans_longer_time/test2024/checkpoints/farm-018-0.0000.ckpt
```

Old exogenous checkpoints containing a **Linear chip-mean weather MLP** are not
compatible with the new convolutional weather network. Loading fails rather
than dropping those weights. Their weather normalization files are also
rejected. Farm-018 has neither weather weights nor weather statistics, so its
normal initialization path remains supported. Legacy DetailRefiner reproduction
is unchanged. Use `resume_from` only for an interrupted spatial-weather run of
the same architecture.

For the HLS-only control, set `data.soil_vars=[] data.weather_vars=[]` and use a
different experiment name. The current optimizer still has one learning-rate
group; separate decoder/branch rates and BARC spatial-only training with frozen
weather remain later work.

## Memory and checks

The default float32 weather input uses `3×8×224×224×4` bytes ≈ **4.6 MiB per chip**
(≈36.8 MiB per batch of eight). Worker prefetching and autograd activations add
more memory. Training still performs the frozen FARM forward pass. Test actual
throughput/memory before a full training run; no accuracy improvement is claimed
from this implementation alone.

Tests cover exact nearest-cell values and nodata, variable/month/year order,
spatial statistics and legacy-stat rejection, synchronized augmentation,
training/inference input agreement, spatially different learned corrections,
weather gradients, zero initialization, frozen FARM, and checkpoint loading.
`scripts/smoke_test_exogenous_readers.py` checks real maps and coverage; averages
printed there are diagnostic summaries only, never model inputs.

See [EXOGENOUS_REFINER_STATUS_GUIDE.md](EXOGENOUS_REFINER_STATUS_GUIDE.md) for the
full workflow and code map, and [DECISION_LOG.md](DECISION_LOG.md), decision 26.
