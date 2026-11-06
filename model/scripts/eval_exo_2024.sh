#!/usr/bin/env bash
# Evaluate the soil+weather (ExogenousRefiner) checkpoint on the 2024 TEST year.
# Nothing here trains or changes the model; it only reads the checkpoint.
#
#     cd model && bash scripts/eval_exo_2024.sh 2>&1 | tee outputs/logs/eval_exo_2024.log
#
# Three stages, each answering a different question:
#   1. evaluate        -> pixel-level metrics on the test-year chips (vs the pseudo-labels)
#   2. map_test_errors -> per-state prediction GeoTIFFs for 2024 (needed for county numbers)
#   3. county_*        -> average each state's prediction per county, compare with NASS,
#                         pool all states into one scatter + metrics
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> model/

CONFIG=configs/experiments/cornbelt5_soil_weather.yaml     # must match how the model was built
NAME=Cornbelt5_spatial_soil_weather_anneal_20261001_091136 # the run that produced the checkpoint
CKPT=outputs/runs/$NAME/test2024/checkpoints/farm-003-0.0000.ckpt   # its best checkpoint
YEAR=2024
STATES="IA IL IN MD MN"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Pixel-level evaluation on the test-year chips.
#    - --config  : rebuilds the SAME architecture (ExogenousRefiner + 4 soil + 3 weather
#                  variables). A config without them could not load this checkpoint.
#    - --real    : real Prithvi backbone, not the small dummy one.
#    - checkpoint: the weights to score. The matching norm_stats.json (band stats from
#                  farm-018, soil/weather stats from this run) is found automatically by
#                  walking up from the checkpoint, so nothing is recomputed.
#    - experiment_name: only decides where results are written:
#                  outputs/runs/$NAME/test2024/eval/farm-003-0.0000/
#    Compare with farm-018's file: outputs/runs/cornbelt5_soybeans_longer_time/test2024/
#                  eval/farm-018-0.0000/metrics_global_pixel.json
# ---------------------------------------------------------------------------
echo "=== 1/3 pixel-level evaluation ==="
uv run python -m farm_us.cli evaluate --config "$CONFIG" --real \
    checkpoint="$CKPT" experiment_name="$NAME" train.output_dir="$PWD/outputs/runs"

# ---------------------------------------------------------------------------
# 2. Write the 2024 prediction maps, one state at a time.
#    Writes outputs/predictions/$NAME/soybeans_<ST>_2024_{pred,actual,residual,...}.tif
#    Only chips with real imagery are predicted, and pred/actual share one footprint,
#    so they can be compared pixel for pixel.
# ---------------------------------------------------------------------------
echo "=== 2/3 prediction rasters ==="
for ST in $STATES; do
    uv run python scripts/map_test_errors.py --config "$CONFIG" \
        checkpoint="$CKPT" experiment_name="$NAME" state="$ST" year="$YEAR"
done

# ---------------------------------------------------------------------------
# 3. County level. Per state: average the predicted raster inside each county and
#    join NASS's reported county yield. Writes
#    outputs/county_aggregation/$NAME/soybeans_<ST>_2024_pred_county_agg.csv
#    (columns: mean = predicted, nass = actual). Then pool all states.
#    Counties with < 100 pixels or no NASS value are excluded, as before.
# ---------------------------------------------------------------------------
echo "=== 3/3 county aggregation ==="
for ST in $STATES; do
    uv run python scripts/county_aggregate.py \
        --raster "outputs/predictions/$NAME/soybeans_${ST}_${YEAR}_pred.tif" \
        --state "$ST" --year "$YEAR"
done
uv run python scripts/county_scatter_all.py \
    --dir "outputs/county_aggregation/$NAME" --year "$YEAR"

echo
echo "pixel metrics : outputs/runs/$NAME/test2024/eval/farm-003-0.0000/metrics_global_pixel.json"
echo "county scatter: outputs/county_aggregation/$NAME/all_states_${YEAR}_pred_vs_nass.png"
