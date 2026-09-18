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
flock 8
$PY -m reid.crops --data-config "$DATA_CFG" > logs/crops.log 2>&1
flock -u 8

if grep -q '^  source:' "$CONFIG"; then
  stage external  # pretraining set from data.yaml unpacked into /dev/shm; unpacking is skipped when already done
  $PY -m reid.external --data-config "$DATA_CFG" --config "$CONFIG" > logs/external.log 2>&1
fi

stage train
SEEDS=$(sed -n 's/^seeds: *\[\(.*\)\].*/\1/p' "$CONFIG" | tr ',' ' ')
TRAIN_MODULE=reid.train
if grep -q '^  arch:' "$CONFIG"; then TRAIN_MODULE=reid.train_paper; fi  # reproduced architectures, reid.paper_models
PIDS=()
cleanup_seeds() {
  local rc=$?
  if [ "$rc" != 0 ] && [ "${#PIDS[@]}" -gt 0 ]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
  fi
}
trap cleanup_seeds EXIT
IFS=',' read -ra SEED_GPU_LIST <<< "${SEED_GPUS:-$GPU}"
seed_index=0
for seed in $SEEDS; do
  seed_gpu=${SEED_GPU_LIST[$((seed_index % ${#SEED_GPU_LIST[@]}))]}
  if [ "$seed_gpu" != "$GPU" ] && [ -n "${BENCHMARK_RUN:-}" ]; then
    stage wait_benchmark
    while [ ! -f "$BENCHMARK_RUN/status/DONE" ]; do
      if [ -f "$BENCHMARK_RUN/status/FAILED" ]; then cat "$BENCHMARK_RUN/status/FAILED"; exit 1; fi
      sleep 5
    done
    stage train
  fi
  seed_uuid=$(nvidia-smi -i "$seed_gpu" --query-gpu=uuid --format=csv,noheader)
  CUDA_VISIBLE_DEVICES=$seed_uuid $PY -m "$TRAIN_MODULE" --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$CONFIG" --seed "$seed" \
    > "logs/train_seed$seed.log" 2>&1 &
  PIDS+=($!)
  if [ "$PARALLEL_SEEDS" = 0 ]; then wait "$!"; fi
  seed_index=$((seed_index + 1))
done
for pid in "${PIDS[@]}"; do wait "$pid"; done

stage aggregate
$PY -m reid.aggregate --config "$CONFIG" > logs/aggregate.log 2>&1

EXP=$(sed -n 's/^exp_id: *\([^ #]*\).*/\1/p' "$CONFIG")
stage slices
$PY -m reid.slices --exp-dir "artifacts/$EXP" --val-config "$VAL_CFG" --attributes protocol/attributes.csv \
  > logs/slices.log 2>&1

if grep -Eq '^  backbone: (siglip_afem|osnet_ain|siglip_osnet)' "$CONFIG"; then
  stage precision
  $PY -m reid.precision_seeds --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$CONFIG" \
    > logs/precision.log 2>&1
fi

stage benchmark  # after training, so the seeds don't share the GPU with the measurement
CHECKPOINT_DIR=$(sed -n 's/^checkpoint_dir: *\([^ #]*\).*/\1/p' "$DATA_CFG")
CHECKPOINT=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "artifacts/$EXP/seed0/metrics.json")
$PY -m reid.benchmark --data-config "$DATA_CFG" --config "$CONFIG" --checkpoint "$CHECKPOINT_DIR/$CHECKPOINT" \
  > logs/benchmark.log 2>&1

stage done
touch status/DONE
