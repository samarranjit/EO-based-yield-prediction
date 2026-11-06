#!/usr/bin/env bash
# Zero-shot spatial holdout: farm-018 on Ohio soybeans, 2020-2024 (the years OH has imagery).
#   cd model && bash scripts/run_spatial_holdout_oh.sh 2>&1 | tee outputs/logs/spatial_holdout/ohio_farm018.log
# Results: outputs/runs/spatial_holdout_oh/test<YEAR>/eval/farm-018-0.0000/
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/experiments/spatial_holdout_oh.yaml
CKPT=outputs/runs/cornbelt5_soybeans_longer_time/test2024/checkpoints/farm-018-0.0000.ckpt
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
mkdir -p outputs/logs/spatial_holdout

echo "=== building Ohio manifest (once) ==="
uv run python -m farm_us.cli build-manifest --config "$CONFIG"

for YEAR in 2020 2021 2022 2023 2024; do
    echo "==================== OH $YEAR ===================="
    uv run python -m farm_us.cli evaluate --config "$CONFIG" --real \
        checkpoint="$CKPT" test_year="$YEAR" train.output_dir="$PWD/outputs/runs"
done
echo "done -> outputs/runs/spatial_holdout_oh/test*/eval/farm-018-0.0000/metrics_global_pixel.json"
