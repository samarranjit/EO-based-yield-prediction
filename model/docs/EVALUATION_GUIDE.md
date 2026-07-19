# Evaluation Guide

All metrics are **masked** (valid crop∧label∧HLS pixels) and reported
**de-standardized** in physical units (kg/ha; also kg/ac, bu/ac) in addition to
standardized diagnostics. Metrics are never computed on standardized values
alone.

## Metrics (`training/metrics.py`)
MAE · RMSE · R² (sklearn-style `1 − SS_res/SS_tot`) · Pearson r · r² · mean bias ·
%MAE (where a valid denominator exists). Verified against sklearn in
`tests/test_metrics.py`.

## Levels of aggregation (`evaluation/aggregation.py`)
1. all valid crop pixels (global, pixel-weighted)
2. per chip
3. per county-year
4. per state-year
5. per test year
6. macro average across counties
7. macro average across states
8. global pixel-weighted average

Never rely on the single global pixel metric alone — big states/counties dominate
it. Macro averages give each county/state equal weight.

## County aggregation vs original NASS
`aggregate_to_county` averages predicted crop pixels per county;
`compare_county_to_nass` joins to the **original NASS county yield**. Keep this
comparison **separate** from the pseudo-label comparison — recovering county
means is a weaker claim than intra-field accuracy.

## Bootstrap confidence intervals
`bootstrap_ci` resamples **groups (counties/chips)**, not pixels — neighboring
pixels are not independent, so pixel bootstrap would understate uncertainty.

## Plots (`evaluation/plots.py`)
observed-vs-predicted scatter · residual histogram · residual-vs-observed ·
predicted/residual maps · per-state / per-year metric tables (CSV). Saved under
`outputs/runs/<exp>/test<year>/eval/`.

## Run it
```bash
uv run python -m farm_us.cli evaluate --config configs/experiments/us_soybeans.yaml \
    test_year=2018 checkpoint=<ckpt>
```
Outputs: `per_chip_metrics.csv/.parquet`, `metrics_by_{state,year}.csv`, and PNGs.
`summarize()` returns global-pixel + macro-by-state/year metrics.

## Georeferenced products (`evaluation/inference.py` + `mosaic.py`)
Windowed inference with cosine-blended overlapping tiles; non-crop pixels masked
as no-data (never filled). Exports predicted-yield GeoTIFF (+ residual / weight /
county table) preserving CRS, transform, extent, pixel size. Tiling round-trips
are tested in `tests/test_tiled_inference.py`.
