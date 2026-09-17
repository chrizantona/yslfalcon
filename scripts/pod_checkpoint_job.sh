#!/usr/bin/env bash
# Trained checkpoint evaluation, slices and the complete inference benchmark on one idle GPU.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${CONFIG:?set CONFIG}" "${GPU:?set GPU}"
source scripts/pod_setup.sh
EXP=$(sed -n 's/^exp_id: *\([^ #]*\).*/\1/p' "$CONFIG")
CHECKPOINT=$(sed -n 's/^checkpoint: *\([^ #]*\).*/\1/p' "$CONFIG")
CHECKPOINT_DIR=$(sed -n 's/^checkpoint_dir: *\([^ #]*\).*/\1/p' "$DATA_CFG")
stage evaluate
$PY -m reid.checkpoint_eval --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$CONFIG" > logs/evaluate.log 2>&1
stage slices
$PY -m reid.slices --exp-dir "artifacts/$EXP" --val-config "$VAL_CFG" --attributes protocol/attributes.csv > logs/slices.log 2>&1
stage benchmark
$PY -m reid.benchmark --data-config "$DATA_CFG" --config "$CONFIG" --checkpoint "$CHECKPOINT_DIR/$CHECKPOINT" > logs/benchmark.log 2>&1
stage done
touch status/DONE
