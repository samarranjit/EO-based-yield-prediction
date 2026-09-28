# Implementation Decision Log

Condensed index of every ambiguity and the decision taken. Full reasoning +
provenance tags ([PAPER]/[PRITHVI]/[INFERENCE]/[US-ADAPT]) are in
[PAPER_REPLICATION_NOTES.md](PAPER_REPLICATION_NOTES.md) §7.

| # | Ambiguity | Decision | Tag |
|---|---|---|---|
| 1 | `[C·T,H,W]` prose vs `[B,C,T,H,W]`+Conv3D figure | Use official Prithvi Conv3D `[B,6,T,224,224]`; honor "time as channels" in the default `flatten_time` reducer | INFERENCE |
| 2 | T=8 vs pretrained num_frames=4 | Use T=8 via official `interpolate_pos_encoding`; no reduction to 4; 4-frame only as ablation | US-ADAPT |
| 3 | Temporal→2D reduction unspecified | Explicit `TemporalFeatureReducer` (mean/attention/flatten_time; default flatten_time) | INFERENCE |
| 4 | Encoder dim 1280 vs decoder width 1024 | Keep decoder 1024; explicit 1280→1024 lateral 1×1 | INFERENCE |
| 5 | 4 ViT levels share one 16×16 grid | Same-grid hierarchical FPN fusion (default); multiscale variant optional | INFERENCE |
| 6 | Aux branch source | Block 24 (0-based 23), configurable | PAPER |
| 7 | PPM bins | [1,2,3,6], configurable | INFERENCE |
| 8 | Head final conv kernel | 3×3 intermediates, 1×1 final projection | INFERENCE |
| 9 | Gaussian noise scale | After normalization, σ=0.1 (config), never labels/masks | INFERENCE |
| 10 | Location-embedding shortcut | Mandatory no-location ablation; documented | INFERENCE |
| 11 | Ridge label LOYO provenance | Never refit; record provenance; audit flags | US-ADAPT |
| 12 | 600M vs 600M-TL | Default `-TL` (time+location), toggles + non-TL available | US-ADAPT |
| 13 | Block indexing 8/16/24/32 | 0-based 7/15/23/31; 32→final block; unit-tested | PAPER |
| 14 | Target/label standardization | z-score from training years only; metrics de-standardized | PAPER/US-ADAPT |
| 15 | Normalization stats | Per-LOYO-fold training stats (default) or official Prithvi stats (switch) | US-ADAPT |
| 16 | Missing-month handling | Explicit policy (default temporal_interp); never silent substitution | INFERENCE |
| 17 | BARC = monitor analogue | Prince George's Co. MD as measured-label transfer site | US-ADAPT |
| 18 | Sub-token detail unreachable (256-DOF ceiling) | `DetailRefiner` residual branch on raw full-res pixels; zero-init output so step 0 == source checkpoint | US-ADAPT |
| 19 | Transfer vs resume conflated by one `ckpt_path` | Split into `init_from` (weights only, fresh optimiser) and `resume_from` (Lightning state); passing both raises | INFERENCE |
| 20 | Norm stats for a frozen base | `norm.inherit_init_stats` reuses the source checkpoint's `norm_stats.json`; **required** with `finetune_mode: refiner_only` | US-ADAPT |
| 21 | Weight decay vs zero-init output layer | `wd 0.0` + `lr 1e-2` for refiner-only; decay otherwise cancels a zero-initialised head | INFERENCE |
| 22 | One validation year can be pathological | `loyo_soybeans_2val.yaml` (year before + after) available; measured ~neutral, 1 val year stays default | US-ADAPT |
| 23 | LOYO measures nothing about new ground | `spatial_holdout.yaml` — zero-shot on unseen states, reported **per state**, never pooled | US-ADAPT |

## Findings that constrain future design

**The pseudo-labels carry one within-county degree of freedom.**
`fit_ridge_to_county_vi_and_yield.py` fits Ridge on *county-mean* VI (NDVI, EVI,
GCVI, NDWI) → NASS county yield — a **between-county** relationship.
`distribute_county_yield_to_pixels.py` then applies those 4 coefficients to
**within-county VI z-scores**, bounds, renormalises and multiplies by county yield
(mean-preserving). So within-county label structure is a monotone function of one
fixed 4-VI linear projection, with no soil, drainage or management term, and it
rests on a cross-scale extrapolation never validated at pixel scale. Three
consequences:

1. A 30 m covariate (soil, elevation) has **no within-county signal to learn** from
   these labels — at best it is a redundant proxy for VI.
2. Any component with sub-token expressivity trained on them learns to **imitate
   the ridge function**. The coarse path is safe only because the 256-DOF ceiling
   prevents it from expressing that function.
3. The target itself depends on a **county-wide** normalisation the model cannot
   observe (a chip is 6.72 km, a county ~50 km), so part of the within-county
   training signal is structurally unpredictable from the input.

**Spatial generalisation is the weak axis, and static rasters are the risk.**
Zero-shot on unseen states: NE r² −0.35 (`pearson_r2` 0.164, bias +5.31), MO r²
−0.09…0.43, against 0.799 in-sample. LOYO holds out a year but trains on all
states, and 99.6% of test locations appear in training in other years. A *static*
high-resolution covariate is therefore a location fingerprint with far more
capacity than the lat/lon embedding that `LOCATION_EMBEDDING_RISK.md` already
flags — and removing that embedding measurably *helped* (0.4978 vs 0.4829).
Validate any such input under **both** LOYO and `spatial_holdout.yaml`; the gap
between them is the memorisation measurement.

**Report both r² and `pearson_r2`.** They diverge substantially here because of
bias. `r2` (against 1:1) is the accuracy number; `pearson_r2` allows a refitted
slope and intercept. The metrics JSON says this inline — quoting only the second
overstates results, especially on the spatial holdouts.

## Verified in this environment (updated 2026-09-26)
- Model forward+backward (dummy backbone) for **T=4 and T=8** → `[B,1,224,224]`.
- Full Lightning train→checkpoint→provenance→evaluate→plots (synthetic, CPU).
- Unit tests pass (incl. `tests/test_detail_refiner.py`); ruff clean.
- Real label/CDL raster metadata + alignment; inventory (17 states × 11 yr, 0 gaps).
- **Real Prithvi-EO-2.0-600M-TL training at scale** — `cornbelt5_soybeans_longer_time`
  (5 states, 2014–2024, test 2024) is the reference run; `farm-018` its checkpoint.
- **Real measured-label BARC transfer**, 11-fold LOYO, including `refiner_only`.
- **Zero-shot spatial holdout** on unseen states (MO, NE).

## Superseded scaffolding claims
Earlier revisions of this file and of `BARC_TRANSFER.md` stated that the real
600M forward pass was untested, that HLS imagery was not on disk, and that BARC
rasters were absent. All three are stale — real HLS composites, pseudo-labels and
measured 30 m BARC yield (2014–2024) are on disk and have been trained on.

## Open, not implemented
Proposed but **not built** as of this date — do not document as behaviour:
weather (PRISM/gridMET) and soil (POLARIS) covariates, FiLM conditioning on the
head, a progressive-upsampling decoder with multi-level fusion, and sub-token
shifted-window inference. Each is discussed in ARCHITECTURE.md's DOF section or in
the findings above; none has code.
