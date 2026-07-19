# Troubleshooting

## Model checkpoint download errors
The real backbone is fetched by TerraTorch from the HF Hub on first build. If it
fails: check network / `HF_TOKEN`; pre-cache with `huggingface-cli download
ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`; set `HF_HOME` to a writable cache.
Scaffolding does **not** download weights — only the real path does.

## TerraTorch API changes
`prithvi_adapter._load_terratorch` uses `terratorch.registry.BACKBONE_REGISTRY`
with keys `prithvi_eo_v2_600_tl` / `prithvi_eo_v2_600`. If TerraTorch renames
these, update the key mapping there. The adapter raises a clear
`BackboneNotAvailable` with install guidance if TerraTorch is missing; the rest
of the package runs on the dummy backbone regardless.

## T=8 shape / positional-embedding errors
Prithvi pretrained with `num_frames=4`; T=8 works via the official
`interpolate_pos_encoding` (re-computed sinusoidal temporal embeddings). If you
see a pos-embed size mismatch, confirm you pass `num_frames`/`n_timesteps=8` to
the adapter and that `x` is `[B,6,8,224,224]` (5-D). Never reshape to
`[B,48,224,224]`.

## Intermediate-feature / hook errors
Feature extraction uses `forward_features` (returns per-block tokens) — preferred
over hooks. Block indices are **0-based 7,15,23,31**; `one_based_to_index` raises
on out-of-range (e.g. 33). Attention capture (`AttentionCapturer`) hooks
`.blocks[i].attn`; if the block layout differs, adjust the attribute path.

## Attention weights are None
Fused/flash attention does not return weights. Use an "interpretation mode" that
selects the non-fused attention path before capturing, or compute month×month
reduction from an explicitly returned attention tensor via
`temporal_from_attention`. Never fabricate attention values.

## Raster misalignment
The validation/inventory tooling checks CRS, resolution, size, transform, band
count/order, nodata. Label and CDL for a (state, year) must share size+transform
(EPSG:5070, 30 m). Reproject/regrid upstream if they differ.

## "No valid pixels"
A chip with zero valid crop∧label∧HLS pixels contributes 0 loss (finite, with
grad) and is flagged (`*/skipped_zero_valid`). Tune `min_crop_fraction` /
`min_label_fraction` in the manifest/QC step to drop such chips up front.

## NaN / Inf loss
Losses compute in fp32 even under bf16 and clamp the denominator. Enable
`train.detect_anomaly: true` and `train.grad_clip` to localize. Check input
scaling (`hls_scale`) and that nodata was converted to NaN, not left as −9999.

## OOM
Follow the memory ladder in TRAINING_GUIDE.md; use `profile-memory`. Do not
change the scientific model to fit — reduce micro-batch + add grad-accum, enable
gradient checkpointing, or shard (DDP/FSDP).

## rasterio + multiprocessing
GDAL objects are not fork-safe. Readers open datasets **inside** `__getitem__`
(per worker), not in `__init__`. If you see GDAL crashes with
`num_workers>0`, keep dataset handles worker-local (as implemented) or set
`num_workers=0` to isolate.

## Apple MPS: adaptive-pool error
`Adaptive pool MPS: input sizes must be divisible by output sizes` — the PPM uses
bins [1,2,3,6] on a 16×16 grid (16 not divisible by 3/6), unsupported on MPS. Use
`train.accelerator: cpu` on Macs (as in `smoke_dummy.yaml`); CUDA is unaffected.

## Slow network-mounted storage
Label rasters live on OneDrive here. Windowed reads minimize I/O, but for
training copy the needed state-years to local SSD, or materialize chips
(NPZ/Zarr) once via the `chips_npz` reader path.
