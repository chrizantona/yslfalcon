#!/usr/bin/env bash
# Pod job for a training experiment: setup -> crops -> external set (pretraining configs with train.source) ->
# one training process per seed on one GPU -> aggregate ->
# mAP by slice -> speed and size with the seed-0 checkpoint.
# Run from a run dir holding a copy of the repo, detached from the ssh session:
#   CONFIG=configs/exp001_clip_vitb16_bot.yaml GPU=2 setsid nohup bash scripts/pod_train_job.sh > status/job.log 2>&1 < /dev/null &
# PARALLEL_SEEDS=0 trains the seeds one after another. Progress: status/stage, status/stages; result: status/DONE or
# status/FAILED; logs in logs/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${CONFIG:?set CONFIG}" "${GPU:?set GPU}"
PARALLEL_SEEDS=${PARALLEL_SEEDS:-1}
source scripts/pod_setup.sh

stage crops  # derived copy of the dataset in /dev/shm; existing crops are skipped
$PY -m reid.crops --data-config "$DATA_CFG" > logs/crops.log 2>&1

if grep -q '^  source:' "$CONFIG"; then
  stage external  # pretraining set from data.yaml unpacked into /dev/shm; unpacking is skipped when already done
  $PY -m reid.external --data-config "$DATA_CFG" --config "$CONFIG" > logs/external.log 2>&1
fi

stage train
export CUDA_VISIBLE_DEVICES=$GPU
SEEDS=$(sed -n 's/^seeds: *\[\(.*\)\].*/\1/p' "$CONFIG" | tr ',' ' ')
PIDS=()
for seed in $SEEDS; do
  $PY -m reid.train --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$CONFIG" --seed "$seed" \
    > "logs/train_seed$seed.log" 2>&1 &
  PIDS+=($!)
  if [ "$PARALLEL_SEEDS" = 0 ]; then wait "$!"; fi
done
for pid in "${PIDS[@]}"; do wait "$pid"; done

stage aggregate
$PY -m reid.aggregate --config "$CONFIG" > logs/aggregate.log 2>&1

EXP=$(sed -n 's/^exp_id: *\([^ #]*\).*/\1/p' "$CONFIG")
stage slices
$PY -m reid.slices --exp-dir "artifacts/$EXP" --val-config "$VAL_CFG" --attributes protocol/attributes.csv \
  > logs/slices.log 2>&1

stage benchmark  # after training, so the seeds don't share the GPU with the measurement
CHECKPOINT_DIR=$(sed -n 's/^checkpoint_dir: *\([^ #]*\).*/\1/p' "$DATA_CFG")
CHECKPOINT=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "artifacts/$EXP/seed0/metrics.json")
$PY -m reid.benchmark --data-config "$DATA_CFG" --config "$CONFIG" --checkpoint "$CHECKPOINT_DIR/$CHECKPOINT" \
  > logs/benchmark.log 2>&1

stage done
touch status/DONE
