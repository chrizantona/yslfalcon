#!/usr/bin/env bash
# exp1: plain ViT-L; exp2: independent fp16 benchmark; improvement unlocks exp3 -> exp4 -> exp5.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
RUNS=$(dirname "$PWD")
export PREPARED_ENV="$RUNS/exp1/.venv"
echo $$ > status/night_pid
echo exp1_exp2 > status/night_stage
trap 'echo "night controller failed ($(date -Is))" > status/night_FAILED' ERR

(cd "$RUNS/exp2"; CONFIG=configs/exp2.yaml GPU=3 bash scripts/pod_benchmark_job.sh > status/job.log 2>&1) &
benchmark_pid=$!
CONFIG=configs/exp1.yaml GPU=1 SEED_GPUS=1,3,2 BENCHMARK_RUN="$RUNS/exp2" bash scripts/pod_train_job.sh
wait "$benchmark_pid"

# Check the real trained encoder in fp16, including its BNNeck, after all training processes exit.
(cd "$RUNS/exp2"; CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) PYTHONPATH=src "$PREPARED_ENV/bin/python" -m reid.benchmark \
  --data-config configs/data.yaml --config configs/exp2.yaml --checkpoint /workspace/checkpoints/exp1_seed0.pt \
  --out artifacts/EXP-021/benchmark_trained.json > logs/benchmark_trained.log 2>&1)
(cd "$RUNS/exp2"; CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) PYTHONPATH=src \
  "$PREPARED_ENV/bin/python" -m reid.precision_eval --data-config configs/data.yaml --val-config configs/val_v2.yaml \
  --config configs/exp2.yaml > logs/precision_eval.log 2>&1)
PYTHONPATH=src "$PREPARED_ENV/bin/python" -m reid.night_gate --config configs/exp1.yaml --out status/gate.json
if ! "$PREPARED_ENV/bin/python" -c 'import json; assert json.load(open("status/gate.json"))["continue"]'; then
  echo "No gain beyond seed spread; external pretraining skipped" > status/night_SKIPPED
  echo skipped > status/night_stage
  touch status/night_DONE
  exit 0
fi

echo exp3 > status/night_stage
(cd "$RUNS/exp3"; CONFIG=configs/exp3.yaml GPU=3 PARALLEL_SEEDS=0 bash scripts/pod_train_job.sh > status/job.log 2>&1)
echo exp4 > status/night_stage
(cd "$RUNS/exp4"; CONFIG=configs/exp4.yaml GPU=1 SEED_GPUS=1,3,2 bash scripts/pod_train_job.sh > status/job.log 2>&1)
echo exp5 > status/night_stage
(cd "$RUNS/exp5"; GPU=3 SOUPS=configs/exp5.yaml bash scripts/pod_eval_job.sh > status/job.log 2>&1)
(cd "$RUNS/exp5"; CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) PYTHONPATH=src "$PREPARED_ENV/bin/python" -m reid.benchmark \
  --data-config configs/data.yaml --config configs/exp5.yaml --checkpoint /workspace/checkpoints/exp5.pt \
  > logs/benchmark.log 2>&1)
echo done > status/night_stage
touch status/night_DONE
