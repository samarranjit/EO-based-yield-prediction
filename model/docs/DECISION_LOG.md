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

## Verified in this environment
- Model forward+backward (dummy backbone) for **T=4 and T=8** → `[B,1,224,224]`.
- Full Lightning train→checkpoint→provenance→evaluate→plots (synthetic, CPU).
- 59/59 unit tests pass; ruff clean.
- Real label/CDL raster metadata + alignment; inventory (17 states × 11 yr, 0 gaps);
  manifest build on real DE 2018 (345 chips + fingerprint).

## NOT verified (documented, code path present)
- Real Prithvi-600M forward pass — marked integration test, needs `[prithvi]` + network + GPU.
- Real-imagery end-to-end training — HLS imagery not yet on disk.
