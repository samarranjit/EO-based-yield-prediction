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

## Status
Real BARC rasters are **not on disk** in this environment. `barc_dataset.py`
mirrors the national dataset interface with a synthetic fallback for pipeline
tests; the experiment builder is verified on the dummy backbone. Point
`BarcConfig.root` at measured 30 m BARC yield + matching HLS composites and
implement a windowed reader mirroring `GeotiffMonthlyReader` to run for real.

## Resolution note (paper)
The paper upsampled 10 m monitor data 10 m→5 m (112→224 px) to match the
county-model input spec. For BARC at native 30 m the chips already match; if a
higher-resolution BARC product is used, apply the same extent-preserving
upsampling before inference.
