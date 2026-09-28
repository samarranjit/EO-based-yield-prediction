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
report the no-location ablation.

## Measured: the ablation, and the spatial holdout (2026-09)

**The embedding is not load-bearing.** Removing it made results slightly *better*
— `pearson_r2` 0.4978 (no location) vs 0.4829 (full) on cornbelt4. Caveat: that run
also carried new regularisation, so the two were not perfectly isolated.

**But spatial generalisation is weak regardless.** `configs/experiments/spatial_holdout.yaml`
applies `farm-018` zero-shot to states it never saw, in a year it never saw:

| State | r² (vs 1:1) | `pearson_r2` | bias (bu/ac) |
|---|---|---|---|
| MO | −0.09 … 0.43 | 0.26 … 0.49 | +1.8 … +4.6 |
| NE | **−0.35** | 0.164 | +5.31 |

Against **0.799** in-sample. Report the states **separately** — the pooled metric
is dominated by whichever state has the most soybean pixels. NE is the informative
one: heavy centre-pivot irrigation keeps canopies green under stress, so it probes
whether the model learned reflectance→yield or just "greener = more". The
consistent positive bias says it over-predicts on new ground.

## Implication for any new static covariate

LOYO holds out a year but trains on all states, and 99.6% of test locations appear
in training in other years. A **static** high-resolution raster — soil (POLARIS),
elevation, any time-invariant layer — is therefore a location fingerprint with far
more capacity than the lat/lon embedding this document was written about, and
county yield is strongly persistent year to year.

So: validate any such input under **both** LOYO and `spatial_holdout.yaml`.

- Helps under both → real signal.
- Helps under LOYO only → location memorisation, and you have quantified it.

The gap between the two *is* the measurement. Without it the two cases are
indistinguishable. This supersedes the earlier note calling region-holdout
evaluation "future work" — the config exists and has been run.
