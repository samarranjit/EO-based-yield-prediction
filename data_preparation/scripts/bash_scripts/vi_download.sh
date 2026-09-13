#!/usr/bin/env bash
# Download HLS scenes once per state-year, process counties locally, and restart
# automatically if real progress stops. Existing CSV rows, cached downloads, and
# county TIFFs are reused on every restart.

set -u

export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV

PROJECT_DIR="/home/cholab/LabMembers/Samar/EO-based-yield-prediction"
SCRIPT="$PROJECT_DIR/data_preparation/scripts/download_county_wise_summary_table_from_azure_hls.py"
CDL_DIR="$PROJECT_DIR/data_preparation/data/cdl_masks"
OUT_DIR="$PROJECT_DIR/data_preparation/outputs/county_vi_summary"
CACHE_DIR="/home/cholab/LabMembers/Samar/EO-based-yield-prediction/temp"
LOG_DIR="$OUT_DIR/_watchdog_logs"

STATES=(IN)
YEARS=(2022)

DOWNLOAD_WORKERS=9
PROCESS_WORKERS=8
STALL_TIMEOUT=900
CHECK_INTERVAL=15
RETRY_SLEEP=30
MAX_RESTARTS=100

mkdir -p "$OUT_DIR" "$CACHE_DIR" "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

ACTIVE_JOB=""

kill_group() {
    local pgid="$1"
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..15}; do
        kill -0 -- "-$pgid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
}

cleanup() {
    if [[ -n "$ACTIVE_JOB" ]]; then
        echo "Stopping active downloader..."
        kill_group "$ACTIVE_JOB"
    fi
}
trap cleanup INT TERM

run_once() {
    local state="$1"
    local year="$2"
    local attempt="$3"
    local cdl="$CDL_DIR/cdl_soybeans_${state}_${year}.tif"

    if [[ ! -f "$cdl" ]]; then
        echo "[WATCHDOG] Missing CDL, skipping: $cdl"
        return 2
    fi

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local runlog="$LOG_DIR/${state}_${year}_attempt${attempt}_${timestamp}.log"

    local command=(
        uv run python "$SCRIPT"
        --state "$state"
        --year "$year"
        --cdl "$cdl"
        --cdl-mode binary
        --out-dir "$OUT_DIR"
        --cache-dir "$CACHE_DIR"
        --save-pixel-tif
        --download-workers "$DOWNLOAD_WORKERS"
        --process-workers "$PROCESS_WORKERS"
        --gdal-cache-mb 256
        --max-scene-cloud 70
        --min-obs-season 3
        --cmr-page-limit 100
        --cmr-max-items 5000
        --max-retries 8
        --download-read-timeout 180
        --earthdata-strategy netrc
        --netrc-file "$HOME/.netrc"
    )

    echo ""
    echo "============================================================"
    echo "[WATCHDOG] Starting $state $year, attempt $attempt"
    echo "[WATCHDOG] Log: $runlog"
    echo "============================================================"

    setsid bash -c '
        set -o pipefail
        project_dir="$1"
        runlog="$2"
        shift 2
        cd "$project_dir" || exit 97
        "$@" 2>&1 | tee "$runlog"
        exit "${PIPESTATUS[0]}"
    ' bash "$PROJECT_DIR" "$runlog" "${command[@]}" &

    local job=$!
    ACTIVE_JOB="$job"

    local previous_progress=0
    local idle_seconds=0

    while kill -0 "$job" 2>/dev/null; do
        sleep "$CHECK_INTERVAL"

        # Count actual completed downloads/counties. Heartbeat text does not reset
        # this timer, so a wedged process is still detected.
        local current_progress
        current_progress=$(grep -Ec '\[DOWNLOAD DONE\]|\[COUNTY DONE\]' "$runlog" 2>/dev/null || true)

        if (( current_progress > previous_progress )); then
            previous_progress=$current_progress
            idle_seconds=0
        else
            idle_seconds=$((idle_seconds + CHECK_INTERVAL))
        fi

        if (( idle_seconds >= STALL_TIMEOUT )); then
            echo "[WATCHDOG] No completed download/county for ${idle_seconds}s; restarting."
            kill_group "$job"
            wait "$job" 2>/dev/null || true
            ACTIVE_JOB=""
            return 1
        fi
    done

    wait "$job"
    local exit_code=$?
    ACTIVE_JOB=""
    return "$exit_code"
}

for STATE in "${STATES[@]}"; do
    for YEAR in "${YEARS[@]}"; do
        ATTEMPT=0

        while (( ATTEMPT < MAX_RESTARTS )); do
            ATTEMPT=$((ATTEMPT + 1))
            run_once "$STATE" "$YEAR" "$ATTEMPT"
            CODE=$?

            case "$CODE" in
                0)
                    echo "[WATCHDOG] Completed $STATE $YEAR."
                    break
                    ;;
                2)
                    # Missing CDL is also returned as 2 by run_once before Python.
                    if [[ ! -f "$CDL_DIR/cdl_soybeans_${STATE}_${YEAR}.tif" ]]; then
                        break
                    fi
                    echo "[WATCHDOG] $STATE $YEAR is incomplete; resuming in ${RETRY_SLEEP}s."
                    ;;
                *)
                    echo "[WATCHDOG] $STATE $YEAR failed/stalled with code $CODE."
                    ;;
            esac

            if (( ATTEMPT >= MAX_RESTARTS )); then
                echo "[WATCHDOG] Gave up on $STATE $YEAR after $MAX_RESTARTS attempts."
                exit 1
            fi
            sleep "$RETRY_SLEEP"
        done
    done
done

echo "[WATCHDOG] All requested state-years are complete."
