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

## Corrupted HLS tile crashes a DataLoader worker
`rasterio.errors.RasterioIOError: ... TIFFReadEncodedTile() failed` /
`ZIPDecode: Decoding error at scanline N` means a specific GDAL block in an
imagery file is genuinely corrupt (bit rot or a truncated write), not a
transient I/O blip -- it reproduces at the exact same block offset every time.
`scripts/scan_hls_corruption.py` finds every bad block for a config up front
and writes `outputs/qc/hls_corruption_<name>.json`.

`data.exclude_sample_ids` alone does not fully cover this: it excludes chips
on the LABEL pixel grid, but `GeotiffMonthlyReader.read_chip` re-aligns each
imagery read into the IMAGERY raster's own grid via `_aligned_window` (see
that function's docstring -- label and HLS/CDL rasters do not share an
origin). A neighboring, non-excluded chip's aligned window can still land on
the same corrupted block that its excluded neighbor also touches.

Because of that gap, `read_chip` catches `RasterioIOError` per month-read and
treats that timestep as missing (same as a genuinely absent file), logging a
warning and falling through to the configured `missing_month_policy` instead
of propagating and killing the whole training run. This is deliberately at
the single-month granularity, not "skip the whole chip" or "skip the whole
file" -- one corrupted block in one month of one chip should not discard the
other 7 valid months of real data for that chip.
