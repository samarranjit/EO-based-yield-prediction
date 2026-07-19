# FARM Paper Replication Notes

Source paper: **Nejadshamsi, Zhang, Porth, Zaki, Porth, Khoshdel (2026).
"FARM: Crop Yield Prediction via Regression on Prithvi's Encoder for Satellite
Sensing." *AgriEngineering* 2026, 8, 2.** DOI: 10.3390/agriengineering8010002.

This document records (1) a detailed reading of the paper, (2) the architecture
as we implement it, (3) every tensor shape we can pin down, (4) the paper's
hyper-parameters and preprocessing, (5) the three high-resolution transfer
experiments, and (6) every ambiguity together with the decision we made. Each
decision is tagged with a provenance marker:

- `[PAPER]` — an explicit detail stated in the paper text or figure.
- `[PRITHVI]` — a fact from the official Prithvi-EO-2.0 / TerraTorch source or
  the HF `config.json`, used where the paper is silent.
- `[INFERENCE]` — our own reasoned choice to fill a gap the paper leaves open.
- `[US-ADAPT]` — a deliberate change required by the US soybean dataset (which
  differs from the paper's Canadian canola dataset).

---

## 1. Summary of the paper

FARM (Fine-tuning Agricultural Regression Models) adapts the pretrained
**Prithvi-EO-2.0-600M** geospatial foundation model into a **dense, pixel-wise
crop-yield regression** model. The task: turn a multi-temporal HLS satellite
image chip into a continuous 30 m yield map (one value per pixel).

- **Region / crop**: canola, Canadian Prairies (Saskatchewan + Manitoba).
- **Imagery**: Harmonized Landsat Sentinel-2 (HLS), 6 bands
  (Blue, Green, Red, Narrow-NIR, SWIR-1, SWIR-2). Monthly composites,
  **May–September** = **T = 5** timesteps. Years 2018, 2019, 2020, 2022, 2023.
  Scenes with >10% cloud cover excluded before compositing.
- **Chips**: 224 × 224 px at 30 m (≈6.72 × 6.72 km).
- **Labels (main dataset)**: county-level NASS-equivalent yield **bilinearly
  upsampled** to 30 m. The paper is explicit that these are *upsampled
  county-level supervised targets, not true pixel measurements* — i.e.
  **pseudo-pixel labels**. Units kg/ha (also reported kg/ac, bu/ac).
- **Independent validation labels**: a separate **10 m yield-monitor** dataset
  (2013–2024), true intra-field measurements, used only as an external test set.
- **Split**: temporal hold-out. 6079 train chips / 1749 validation chips.
- **Encoder**: Prithvi-EO-2.0-600M ViT, 32 transformer blocks, patch size 14,
  fully unfrozen during fine-tuning.
- **Decoder**: ViT-adapted **UPerNet** consuming transformer blocks
  **8, 16, 24, 32**; FPN top-down fusion + **PSP/PPM** on the deepest feature +
  multi-level fusion + 3×3 conv bottleneck. Decoder width **1024**.
- **Auxiliary decoder**: a "small UPerNet" (256 channels) attached to **block
  24**, with its own 1-channel regression output, used for deep supervision.
- **Main regression head**: 3×3 convs 512 → 256 → 64, each with BatchNorm+ReLU,
  then a final 1-channel projection, then bilinear upsample H/14 → H.
- **Loss**: masked MSE (eq. 1) or Huber (eq. 2); total `L = L_main + 0.2·L_aux`
  (eq. 3). Best reported config = **MSE + Aux** (R² = 0.8105, RMSE 0.4368 std).
- **Metrics**: MAE, RMSE, R², Pearson r — reported standardized and in physical
  units.
- **Interpretability**: temporal attention (blocks 8 & 16, month×month
  heatmaps + incoming-attention curve) and spectral-band importance from the
  patch-embedding weight magnitudes (NIR 31.2%, SWIR1 24.5%, SWIR2 21.8%,
  Red 14.2%, Green 5.1%, Blue 3.2%).
- **High-res transfer (Section 4.4)**: three experiments on the 10 m monitor
  data — zero-shot (R²=0.508), fine-tune-from-FARM (R²=0.768, best),
  train-from-Prithvi (R²=0.675).
- **Stated limitations**: single crop, single region, imagery-only (no
  weather/soil), acquisition→inference latency, no uncertainty quantification.

---

## 2. Layer-by-layer architecture specification (as implemented)

```
INPUT  x                      [B, 6, T, 224, 224]     T=8 (US) / T=5 (paper)      [US-ADAPT]
       temporal_coords        [B, T, 2]  (year, day-of-year)                      [PRITHVI]
       location_coords        [B, 2]     (lat, lon)                               [PRITHVI]

ENCODER  Prithvi-EO-2.0-600M(-TL)
  patch_embed  Conv3d(6→1280, kernel=(1,14,14), stride=(1,14,14))                 [PRITHVI]
  + pos_embed + temporal_embed + location_embed  (cls token prepended)
  32 transformer blocks (dim=1280, heads=16, mlp_ratio=4)
  forward_features -> list of 32 per-block token tensors, each
        [B, 1 + T*16*16, 1280]   (1 cls token + T*256 patch tokens)               [PRITHVI]

FEATURE EXTRACTION  select 1-based blocks {8,16,24,32} = 0-based {7,15,23,31}     [PAPER]
  drop cls token, rearrange (t h w) -> reshape to
        f_l  [B, 1280, T, 16, 16]                                                 [INFERENCE]

TEMPORAL REDUCER  (per level)  [B, 1280, T, 16, 16] -> [B, 1280, 16, 16]
  primary = flatten-time + 1x1 conv projection (paper treats time as channels)   [INFERENCE]

MAIN DECODER  PaperFaithfulUPerNetDecoder
  lateral 1x1 conv per level:   1280 -> 1024                                      [INFERENCE proj]
  PPM on deepest (block 32):    bins [1,2,3,6], out 1024, concat+conv -> 1024     [PAPER/INFERENCE]
  FPN top-down fusion (same 16x16 grid): add + 3x3 smooth per level               [PAPER]
  multi-level fusion: concat 4 levels -> 3x3 conv bottleneck -> 1024              [PAPER]
  fused feature                 [B, 1024, 16, 16]                                 [PAPER]

MAIN HEAD
  3x3 conv 1024->512, BN, ReLU, (dropout 0.1)
  3x3 conv 512->256,  BN, ReLU, (dropout 0.1)
  3x3 conv 256->64,   BN, ReLU
  1x1 conv 64->1
  bilinear upsample 16x16 -> 224x224
  main_pred                     [B, 1, 224, 224]                                  [PAPER]

AUXILIARY DECODER  (from block 24 level)  small UPerNet, 256 ch
  lateral 1x1 1280->256; PPM(256); 3x3 conv -> 256
  1x1 conv 256->1
  bilinear upsample -> 224
  aux_pred                      [B, 1, 224, 224]                                  [PAPER]

LOSS  L = L_main + 0.2 * L_aux   (both masked; aux only in train/val)             [PAPER]
```

### Selected transformer blocks — indexing
The paper says "layers 8, 16, 24, and 32" of a 32-block encoder ("selecting
these specific layers divides the network into four evenly spaced stages"). We
interpret these as **1-based** block numbers → **0-based indices 7, 15, 23, 31**.
Block 32 (0-based 31) is the **last** block. A unit test
(`tests/test_prithvi_features.py`) asserts the mapping and that index 31 is the
final block (never 32, which would be out of range).

---

## 3. Tensor shapes that can be determined

| Stage | Shape | Source |
|---|---|---|
| Input image | `[B, 6, T, 224, 224]` (fig: `[B,C,T,H,W]`) | [PAPER fig] / [US-ADAPT T=8] |
| Paper "data structuring" prose | `[B, C·T, 224, 224]` = `[B, 30, 224, 224]` (T=5) | [PAPER text] (see §7 discrepancy) |
| Patch grid | 16 × 16 (224/14) per frame | [PRITHVI] |
| Tokens per block | `[B, 1 + T·256, 1280]` | [PRITHVI] |
| Reshaped level feature | `[B, 1280, T, 16, 16]` | [INFERENCE] |
| Fig. neck label | `[B, 1024, T, 14, 14]` (they write H/14=16 but label "14") | [PAPER fig] |
| After temporal reduce | `[B, 1280, 16, 16]` | [INFERENCE] |
| After lateral proj | `[B, 1024, 16, 16]` | [PAPER width] |
| Main decoder output | `[B, 1024, 16, 16]` | [PAPER fig] |
| Aux decoder output | `[B, 256, 16, 16]` | [PAPER fig] |
| Main / aux prediction | `[B, 1, 224, 224]` | [PAPER] |

Note: the figure labels the token grid as `H/14 × W/14` but writes the literal
number "14"; with H=224 and patch 14 the true grid is **16 × 16**. We use 16.

---

## 4. Original paper hyper-parameters

| Item | Value | Source |
|---|---|---|
| Framework | PyTorch 2.6.0 + Lightning | [PAPER] |
| Seed | 0 | [PAPER] |
| Epochs | 120 | [PAPER] |
| Batch size | 8 | [PAPER] |
| Optimizer | AdamW | [PAPER] |
| Weight decay | 0.1 | [PAPER] |
| LR schedule | cosine annealing after 20-epoch linear warm-up | [PAPER] |
| LR start / min | 5e-6 / 1e-8 | [PAPER] |
| Dropout | 0.1 (regression head) | [PAPER] |
| Precision | bf16 mixed | [PAPER] |
| Augmentation | H-flip p=0.2, V-flip p=0.2, Gaussian noise p=0.4 | [PAPER] |
| Encoder | fully unfrozen | [PAPER] |
| Decoder width | 1024 | [PAPER] |
| Aux loss weight | 0.2 | [PAPER] |
| Hardware | NVIDIA GPU, 48 GB | [PAPER] |

## 5. Original paper preprocessing

- Exclude scenes with >10% cloud cover, then **monthly composite**. [PAPER]
- Per-band **z-score** normalization using **training-set** channel mean/std.
  Table 1 values (native HLS reflectance ×10⁴ scale):

  | Band | Mean | Std |
  |---|---|---|
  | Blue | 493.94 | 250.38 |
  | Green | 832.45 | 265.75 |
  | Red | 901.06 | 481.92 |
  | NIR-narrow | 2927.87 | 1038.83 |
  | SWIR-1 | 2427.47 | 855.02 |
  | SWIR-2 | 1658.56 | 855.37 |

- Labels standardized (z-score); metrics de-standardized back to physical units.
- These paper stats are Canadian-canola specific; **for US soy we recompute
  per-LOYO-fold training statistics** (`normalization.py`). [US-ADAPT]

For comparison, the **official Prithvi** pretraining stats (HF config) are far
larger because they are on a different scale/region:
Blue 1087/2248, Green 1342/2179, Red 1433/2178, NIR 2734/1850, SWIR1 1958/1242,
SWIR2 1363/1049. We expose both via the `normalization.mode` switch. [PRITHVI]

## 6. High-resolution transfer experiments (Section 4.4)

1. **Zero-shot**: apply the county-trained FARM checkpoint directly to the 10 m
   monitor data (upsampled 10 m → 5 m so 112→224 px keeps the same extent). No
   weight updates. Reported R²=0.508, RMSE 0.921.
2. **Fine-tune from FARM**: init from the national FARM checkpoint, fine-tune on
   monitor training years, select on monitor val year. R²=0.768 (best), RMSE 0.628.
3. **Train from Prithvi**: init encoder from original Prithvi weights (not FARM),
   train same decoder/head on monitor data; used a **heteroscedastic loss** and
   extra brightness/contrast augmentation. R²=0.675, RMSE 0.557.

Our `transfer/barc_experiments.py` mirrors these three as `zero_shot`,
`finetune_from_farm`, `train_from_prithvi`, using **BARC** (Prince George's Co.,
MD) as the US analogue of the monitor dataset. [US-ADAPT]

---

## 7. Ambiguities / inconsistencies and our decisions

### 7.1 `[C·T,H,W]` prose vs `[B,C,T,H,W]` + Conv3D figure — **the central discrepancy**
- **Paper text (§2.1.2 & §2.2.1)**: "Both dimensions are flattened along the
  channel and time dimensions … `[C×T, H, W]` … 30-channel tensor `[B,30,H,W]`
  … a 2D convolution is then applied." → time folded into channels, 2D patch
  embed.
- **Paper figure 3**: input `[B, C, T, H, W]`, **"Conv3D + flatten", patch size
  14** — i.e. true 3D spatiotemporal embedding.
- **Official Prithvi-EO-2.0**: `patch_embed` is `Conv3d` with kernel `(1,14,14)`;
  input is `(B, C, T, H, W)`. [PRITHVI]
- **Decision** `[INFERENCE]`: follow the **figure + official model** — feed
  `[B, 6, T, 224, 224]` through the real Prithvi Conv3D. We do **not** invent a
  30-channel Conv2D. The prose's "treat time as channels" idea is instead
  honored *downstream* in the default **TemporalFeatureReducer**
  (`flatten_time`: stack time into channels then 1×1-project), which is the
  faithful place for that operation. A `flatten_channels` ablation reproduces the
  literal prose interpretation as a non-default experiment.

### 7.2 T=8 vs pretrained num_frames=4
- Prithvi-EO-2.0 pretrained with `num_frames=4`; the paper used T=5. [PAPER/PRITHVI]
- The official `PrithviViT.interpolate_pos_encoding()` **re-computes the
  sinusoidal temporal/positional embedding for the actual T** at runtime, so
  arbitrary T (5, 8, …) is supported natively. [PRITHVI]
- **Decision** `[US-ADAPT]`: use **T=8** (Apr–Nov) directly through the official
  interpolation path. No reduction to 4; no splitting 8 months into two 4-month
  samples. A `T=4` chunked variant exists only as a named ablation.
- Shape tests cover **both T=4 and T=8** (`tests/test_decoder_shapes.py`,
  `tests/test_model_forward.py`).

### 7.3 Temporal → 2D reduction is undocumented
- The paper's decoder is "2D UPerNet" but never states how the T tokens collapse
  to a 2D map. [PAPER gap]
- **Decision** `[INFERENCE]`: explicit, swappable `TemporalFeatureReducer` with
  `mean`, `attention` (learned pooling), and `flatten_time` (default) modes. All
  emit `[B, D, 16, 16]` so the decoder interface is identical regardless of mode.

### 7.4 Encoder embed dim 1280 vs decoder width 1024
- Figure labels neck features "1024"; the real 600M encoder emits **1280**. [PAPER vs PRITHVI]
- **Decision** `[INFERENCE]`: keep decoder width **1024** (paper) and add an
  explicit `1280 → 1024` lateral 1×1 projection per level.

### 7.5 All four ViT levels share one spatial resolution (16×16)
- Standard CNN-UPerNet expects a resolution pyramid; an isotropic ViT gives the
  **same** 16×16 grid at every selected block. [PRITHVI]
- **Decision** `[INFERENCE]`: implement **same-grid hierarchical semantic
  fusion** (lateral proj + top-down add without spatial rescaling) as the
  paper-faithful default. An explicit multi-scale variant (per-level up/down
  adapters) is available as a separate, non-default experiment.

### 7.6 Auxiliary branch source level
- Figure arrow reads "From block 24". [PAPER]
- **Decision**: aux decoder consumes the **block-24 (0-based 23)** level. Configurable.

### 7.7 PPM pooling bins
- Not given numerically. [PAPER gap]
- **Decision** `[INFERENCE]`: standard UPerNet bins `[1, 2, 3, 6]`; configurable.

### 7.8 Final projection conv kernel
- Head uses "series of three 3×3 convolutions"; final projection unspecified. [PAPER]
- **Decision** `[INFERENCE]`: intermediate layers 3×3; **final 1×1** projection.

### 7.9 Gaussian-noise scale
- p=0.4 given; magnitude not. [PAPER gap]
- **Decision** `[INFERENCE]`: additive Gaussian applied **after normalization**,
  default σ=0.1 (config `augment.noise_std`), never to labels/masks.

### 7.10 Location-embedding leakage risk
- `-TL` location embeddings let the model learn regional yield baselines — a
  spatial shortcut that LOYO does not control for. [INFERENCE]
- **Decision**: mandatory planned ablation "TL + time, no location";
  see `docs/LOCATION_EMBEDDING_RISK.md`.

### 7.11 Label provenance / LOYO leakage
- US labels are **ridge-distributed** county yield (external pipeline). If the
  ridge weights for a held-out year used that year's NASS yield, the *labels*
  are not strictly LOYO even though the *NN split* is. [US-ADAPT]
- **Decision**: we never refit the ridge model; we record `label_source` /
  `ridge_provenance` in the manifest and the leakage audit flags it. See
  `docs/LOYO_PROTOCOL.md`.

### 7.12 `-TL` vs plain 600M
- Paper says "Prithvi-EO-2.0-600M"; the task specifies the **`-TL`** (temporal +
  location) variant. [PAPER vs task]
- **Decision** `[US-ADAPT]`: default backbone `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`,
  with time/location embedding toggles; plain 600M available via config.

---

## 8. What is NOT reproduced / out of scope
- The **ridge county→pixel redistribution** is external and never fit here.
- Real 10 m yield-monitor data is replaced by the **BARC** field site interface.
- The paper's exact chip inventory (6079/1749) is dataset-specific; our chip
  count follows the US manifest builder.
