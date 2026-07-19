# Location-Embedding Leakage Risk

## The concern
Prithvi-EO-2.0-**TL** adds a **location embedding** derived from each chip's
`(lat, lon)`. Crop yield has strong, persistent spatial structure (soils,
climate, management). A model given location can learn a **regional yield
baseline** — a spatial shortcut — rather than reading the spectral-temporal
signal.

This is **not direct target leakage** (no label is fed in), but it inflates
LOYO metrics in a way that does **not** reflect the ability to map yield from
imagery, and it will not transfer to unseen geographies.

## Why LOYO does not control for it
LOYO holds out *years*, not *places*. The same counties appear in train and test
across years, so a location-conditioned baseline learned on training years
applies almost unchanged to the test year.

## Mandatory ablation
Always run, alongside the primary model:

```bash
uv run python -m farm_us.cli train \
    --config configs/experiments/us_soybeans.yaml \
    --config configs/experiments/ablation_no_location.yaml \
    test_year=2018
```

`ablation_no_location.yaml` sets `model.use_location_embed: false` (time
embeddings stay on). Compare its LOYO metrics to the full `-TL` model:

- Small gap → the model is genuinely using imagery.
- Large gap → the full model leans on the location shortcut; report both and
  prefer the no-location number as the honest imagery-only estimate.

## Config switches
`model.use_time_embed`, `model.use_location_embed` (both/either/neither), and a
non-TL backbone via `model.backbone_id`. The Lightning module passes
`temporal_coords`/`location_coords` only when the corresponding switch is on.

## Reporting
State clearly in any results whether location embeddings were enabled, and always
report the no-location ablation. Geographic generalization additionally requires
a **leave-one-region-out** style evaluation (future work) — LOYO alone is
insufficient.
