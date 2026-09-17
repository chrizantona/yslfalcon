#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
MASTER=/workspace/yslfalcon_runs/EXP-026
export PREPARED_ENV="$MASTER/.venv" PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
PY="$PREPARED_ENV/bin/python"
mkdir -p status logs
echo $$ > status/soup_followup_pid
trap 'echo "soup follow-up failed ($(date -Is))" > status/FAILED' ERR
echo wait_followup > status/stage
while [ ! -f "$MASTER/status/followup_DONE" ]; do
  if [ -f "$MASTER/status/followup_FAILED" ]; then cat "$MASTER/status/followup_FAILED"; exit 1; fi
  sleep 5
done
"$PY" -c 'import json; assert json.load(open("/workspace/yslfalcon_runs/EXP-028/artifacts/EXP-028/bootstrap_vs_EXP-023.json"))["decision"] == "accepted"'
SOUPS=configs/exp037_soup_exp028_seeds.yaml GPU="$GPU" bash scripts/pod_eval_job.sh > logs/soup_job.log 2>&1
mv status/DONE status/SOUP_DONE
echo compare > status/stage
for base in EXP-028 EXP-024; do
  if [ "$base" = EXP-024 ]; then path="$MASTER/artifacts/$base"; else path="$(dirname "$MASTER")/$base/artifacts/$base"; fi
  "$PY" -m reid.research_compare --base "$path" --new artifacts/EXP-037 \
    --out "artifacts/EXP-037/bootstrap_vs_$base.json" > "logs/compare_$base.log" 2>&1
done
echo precision > status/stage
"$PY" -m reid.precision_eval --data-config configs/data.yaml --val-config configs/val_v2.yaml \
  --config configs/check_exp037_fp16.yaml > logs/precision.log 2>&1
echo benchmark > status/stage
"$PY" -m reid.benchmark --data-config configs/data.yaml --config configs/exp037_soup_exp028_seeds.yaml \
  --checkpoint /workspace/checkpoints/exp037.pt > logs/benchmark.log 2>&1
echo done > status/stage
touch status/DONE status/soup_followup_DONE
