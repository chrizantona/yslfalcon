#!/usr/bin/env bash
# Pod job for the zero-shot reference (EXP-000): setup -> protocol -> crops (CPU, background) + eval -> submit -> check.
# Run from a run dir holding a copy of the repo, detached from the ssh session:
#   setsid nohup bash scripts/pod_job.sh > status/job.log 2>&1 < /dev/null &
# Progress: status/stage and status/stages; result: status/DONE or status/FAILED; per-step logs in logs/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/exp000_clip_zeroshot.yaml}
source scripts/pod_setup.sh

stage protocol
if [ ! -f protocol/val_v1/folds.csv ]; then
  $PY -m reid.protocol --data-config "$DATA_CFG" --config "$VAL_CFG" > logs/protocol.log 2>&1
fi

$PY -m reid.crops --data-config "$DATA_CFG" > logs/crops.log 2>&1 &
CROPS_PID=$!
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > m {m = $2; i = $1} END {print i}')
echo "$(date +%T) crops running in background; GPU $CUDA_VISIBLE_DEVICES" >> status/stages

stage eval
$PY -m reid.evaluate --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$CONFIG" > logs/eval.log 2>&1

stage submit
$PY -m reid.submit --data-config "$DATA_CFG" --config "$CONFIG" > logs/submit.log 2>&1

stage check
$PY -m reid.check_submission --data-config "$DATA_CFG" --config "$CONFIG" > logs/check.log 2>&1

stage wait_crops
wait "$CROPS_PID"

stage done
touch status/DONE
