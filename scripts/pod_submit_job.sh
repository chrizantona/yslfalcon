#!/usr/bin/env bash
# Detached submission from a pinned trained checkpoint and a dev-calibrated threshold.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${CONFIG:?set CONFIG}" "${GPU:?set GPU}" "${PREPARED_ENV:?set PREPARED_ENV}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
PY="$PREPARED_ENV/bin/python"
mkdir -p status logs
echo $$ > status/submit_pid
trap 'echo "submission failed ($(date -Is))" > status/FAILED' ERR
echo submission > status/stage
"$PY" -m reid.submit --data-config configs/data.yaml --config "$CONFIG" > logs/submission.log 2>&1
echo format_check > status/stage
"$PY" -m reid.check_submission --data-config configs/data.yaml --config "$CONFIG" > logs/submission_check.log 2>&1
echo done > status/stage
touch status/DONE
