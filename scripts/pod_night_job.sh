#!/usr/bin/env bash
# EXP-020: plain ViT-L; EXP-021: independent fp16 benchmark; improvement unlocks EXP-022 -> EXP-023 -> EXP-024.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
RUNS=$(dirname "$PWD")
export PREPARED_ENV="$RUNS/EXP-020/.venv"
echo $$ > status/night_pid
echo EXP-020_EXP-021 > status/night_stage
trap 'echo "night controller failed ($(date -Is))" > status/night_FAILED' ERR

(cd "$RUNS/EXP-021"; CONFIG=configs/exp021_clip_vitl14_fp16.yaml GPU=3 bash scripts/pod_benchmark_job.sh > status/job.log 2>&1) &
benchmark_pid=$!
CONFIG=configs/exp020_clip_vitl14_bot_30ep.yaml GPU=1 SEED_GPUS=1,3,2 BENCHMARK_RUN="$RUNS/EXP-021" bash scripts/pod_train_job.sh
wait "$benchmark_pid"

# Check the real trained encoder in fp16, including its BNNeck, after all training processes exit.
(cd "$RUNS/EXP-021"; CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) PYTHONPATH=src "$PREPARED_ENV/bin/python" -m reid.benchmark \
  --data-config configs/data.yaml --config configs/exp021_clip_vitl14_fp16.yaml --checkpoint /workspace/checkpoints/exp020_seed0.pt \
  --out artifacts/EXP-021/benchmark_trained.json > logs/benchmark_trained.log 2>&1)
(cd "$RUNS/EXP-021"; CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) PYTHONPATH=src \
  "$PREPARED_ENV/bin/python" -m reid.precision_eval --data-config configs/data.yaml --val-config configs/val_v2.yaml \
  --config configs/exp021_clip_vitl14_fp16.yaml > logs/precision_eval.log 2>&1)
PYTHONPATH=src "$PREPARED_ENV/bin/python" -m reid.night_gate --config configs/exp020_clip_vitl14_bot_30ep.yaml --out status/gate.json
if ! "$PREPARED_ENV/bin/python" -c 'import json; assert json.load(open("status/gate.json"))["continue"]'; then
  echo "No gain beyond seed spread; external pretraining skipped" > status/night_SKIPPED
  echo skipped > status/night_stage
  touch status/night_DONE
  exit 0
fi

echo EXP-022 > status/night_stage
(cd "$RUNS/EXP-022"; CONFIG=configs/exp022_pretrain_veriwild_test_vitl14.yaml GPU=3 PARALLEL_SEEDS=0 bash scripts/pod_train_job.sh > status/job.log 2>&1)
echo EXP-023 > status/night_stage
(cd "$RUNS/EXP-023"; CONFIG=configs/exp023_finetune_veriwild_init_vitl14.yaml GPU=1 SEED_GPUS=1,3,2 bash scripts/pod_train_job.sh > status/job.log 2>&1)
echo EXP-024 > status/night_stage
(cd "$RUNS/EXP-024"; GPU=3 SOUPS=configs/exp024_soup_exp023_seeds.yaml bash scripts/pod_eval_job.sh > status/job.log 2>&1)
(cd "$RUNS/EXP-024"; CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) PYTHONPATH=src "$PREPARED_ENV/bin/python" -m reid.benchmark \
  --data-config configs/data.yaml --config configs/exp024_soup_exp023_seeds.yaml --checkpoint /workspace/checkpoints/exp024.pt \
  > logs/benchmark.log 2>&1)
echo done > status/night_stage
touch status/night_DONE
