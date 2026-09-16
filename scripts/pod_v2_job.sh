#!/usr/bin/env bash
# Pod job for a protocol version switch: build the protocol, re-score finished experiments from their saved dev-fold
# embeddings (reid.recompute), mAP by slice, then the organisers'-method speed benchmark of the baseline.
# Run from a run dir holding a copy of the repo plus artifacts/<exp>/ embeddings, detached:
#   EXPS="EXP-000 EXP-001" BASELINE_CONFIG=configs/exp016_soup_exp010_seeds.yaml BASELINE_CHECKPOINT=<file name> GPU=2 \
#     setsid nohup bash scripts/pod_v2_job.sh > status/job.log 2>&1 < /dev/null &
# Progress: status/stage, status/stages; result: status/DONE or status/FAILED; logs in logs/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${EXPS:?set EXPS}" "${GPU:?set GPU}"
source scripts/pod_setup.sh
export CUDA_VISIBLE_DEVICES=$GPU
VERSION=$(sed -n 's/^version: *\([^ #]*\).*/\1/p' "$VAL_CFG")

stage protocol
if [ ! -f "protocol/$VERSION/folds.csv" ]; then
  $PY -m reid.protocol --data-config "$DATA_CFG" --config "$VAL_CFG" > logs/protocol.log 2>&1
fi

stage recompute
$PY -m reid.recompute --val-config "$VAL_CFG" --exps $EXPS > logs/recompute.log 2>&1

stage slices
for exp in $EXPS; do
  $PY -m reid.slices --exp-dir "artifacts/$exp/$VERSION" --val-config "$VAL_CFG" --attributes protocol/attributes.csv \
    > "logs/slices_$exp.log" 2>&1
done

if [ -n "${BASELINE_CONFIG:-}" ]; then
  CHECKPOINT_DIR=$(sed -n 's/^checkpoint_dir: *\([^ #]*\).*/\1/p' "$DATA_CFG")
  stage benchmark
  $PY -m reid.benchmark --data-config "$DATA_CFG" --config "$BASELINE_CONFIG" \
    --checkpoint "$CHECKPOINT_DIR/$BASELINE_CHECKPOINT" > logs/benchmark.log 2>&1
fi

stage done
touch status/DONE
