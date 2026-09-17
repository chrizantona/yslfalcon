#!/usr/bin/env bash
# Pod job for a weight soup: build and score the soup (reid.soup) -> mAP by slice -> the same soup scored in fp16
# (reid.precision_eval) -> speed of the soup with its benchmark precision (reid.benchmark).
# Run from a run dir holding a copy of the repo, detached:
#   SOUP=configs/exp043_soup_exp042_seeds.yaml CHECK=configs/check_exp043_fp16.yaml GPU=0 PREPARED_ENV=... \
#     setsid nohup bash scripts/pod_soup_job.sh > status/job.log 2>&1 < /dev/null &
# Progress: status/stage, status/stages; result: status/DONE or status/FAILED; logs in logs/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${SOUP:?set SOUP}" "${CHECK:?set CHECK}" "${GPU:?set GPU}"
source scripts/pod_setup.sh
EXP=$(sed -n 's/^exp_id: *\([^ #]*\).*/\1/p' "$SOUP")

stage soup
$PY -m reid.soup --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$SOUP" > logs/soup.log 2>&1

stage slices
$PY -m reid.slices --exp-dir "artifacts/$EXP" --val-config "$VAL_CFG" --attributes protocol/attributes.csv > logs/slices.log 2>&1

stage precision  # writes artifacts/<EXP>/fp16_check/
$PY -m reid.precision_eval --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$CHECK" > logs/precision.log 2>&1

stage benchmark
CHECKPOINT_DIR=$(sed -n 's/^checkpoint_dir: *\([^ #]*\).*/\1/p' "$DATA_CFG")
CHECKPOINT=$(sed -n 's/^checkpoint_name: *\([^ #]*\).*/\1/p' "$SOUP")
$PY -m reid.benchmark --data-config "$DATA_CFG" --config "$SOUP" --checkpoint "$CHECKPOINT_DIR/$CHECKPOINT" > logs/benchmark.log 2>&1

stage done
touch status/DONE
