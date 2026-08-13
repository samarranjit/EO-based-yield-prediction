#!/usr/bin/env bash
# Launch the 4-state Corn Belt training run, detached and crash-resistant.
#
# Run from the model/ directory:
#     bash scripts/run_cornbelt4_training.sh
#
# To resume from a checkpoint instead of starting fresh:
#     RESUME=outputs/runs/cornbelt4_soybeans/test2024/checkpoints/last.ckpt \
#         bash scripts/run_cornbelt4_training.sh
#
# WHY THE PREFLIGHT CHECKS EXIST
# ------------------------------
# build_trainer runs AFTER the ~60 min normalization stats pass, so a bad
# train.precision value does not surface until an hour of work has been thrown
# away. That happened: `fp16-mixed` is not a Lightning token (it wants
# `16-mixed`) and the run died an hour in, leaving the GPU idle overnight.
# Every check below is cheap and runs before anything is committed.
set -euo pipefail

CONFIG=configs/experiments/cornbelt4_soybeans.yaml
STATS_MAX_CHIPS=2000
LOGDIR=outputs/logs/cornbelt4

cd "$(dirname "$0")/.."

echo "=== preflight ==="

# 1. Refuse to start a second trainer. Two runs would fight over the GPU and
#    both would be slower; worse, they write to the same run directory.
if pgrep -f "farm_us\.cli train" | grep -qv "^$$\$"; then
    if ps -eo pid,cmd | grep "farm_us\.cli train" | grep -qv "grep\|bash"; then
        echo "ABORT: a training process is already running:" >&2
        ps -eo pid,etimes,cmd | grep "farm_us\.cli train" | grep -v "grep\|bash" >&2
        exit 1
    fi
fi
echo "  no training process running          OK"

# 2. GPU must be free. A leftover process from a half-killed run holds VRAM and
#    causes a confusing OOM several minutes in.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -gt 2000 ]; then
    echo "ABORT: GPU already has ${USED} MiB in use -- check for an orphaned process" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    exit 1
fi
echo "  GPU free (${USED} MiB used)              OK"

# 3. Config resolves, and precision is a token Lightning actually accepts.
#    This is the check whose absence cost ~7 hours.
uv run python - "$CONFIG" <<'PY'
import sys
from lightning.fabric.connector import _convert_precision_to_unified_args
from farm_us.config import load_config

cfg = load_config(sys.argv[1])
_convert_precision_to_unified_args(cfg.train.precision)   # raises on a bad token
print(f"  precision {cfg.train.precision!r} accepted by Lightning   OK")
print(f"  states={cfg.data.states} test={cfg.split.test_year} val={cfg.split.val_years}")
print(f"  finetune={cfg.model.finetune_mode} batch={cfg.train.batch_size} epochs={cfg.train.epochs}")
print(f"  excluded corrupt chips: {len(cfg.data.exclude_sample_ids)}")
PY

echo
echo "=== launching ==="
mkdir -p "$LOGDIR"
LOG="$LOGDIR/train_$(date +%Y%m%d_%H%M%S).log"

# setsid  -> new session, survives SSH disconnect / terminal close
# nohup   -> ignore SIGHUP as a second layer
# </dev/null -> no stdin, so it can never block waiting for input
# &  disown  -> background and drop from the shell's job table
setsid nohup uv run python -m farm_us.cli train \
    --config "$CONFIG" --real \
    norm.stats_max_chips=$STATS_MAX_CHIPS \
    ${RESUME:+resume_from="$RESUME"} \
    > "$LOG" 2>&1 </dev/null &
disown || true

sleep 20
echo "  log: $LOG"
ps -eo pid,pgid,etimes,cmd | grep "farm_us\.cli train" | grep -v "grep\|bash" | head -2 || true
echo
echo "=== monitor ==="
echo "  tail -f $LOG"
echo "  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
echo
echo "=== stop (kills the whole process group, not just one pid) ==="
echo '  PGID=$(ps -o pgid= -p $(pgrep -f "farm_us.cli train" | head -1) | tr -d " ")'
echo '  kill -TERM -- -$PGID'
echo
echo "Expect: manifest QC ~2 min, stats pass ~60 min, then epochs at ~3.5 s/step"
echo "        (1,347 steps/epoch => ~1.3 h/epoch, ~26 h for 20 epochs)"
