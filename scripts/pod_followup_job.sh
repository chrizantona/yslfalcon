#!/usr/bin/env bash
# EXP-025…029, three seeds per training change; optional EXP-036 if LLRD 0.9 does not help.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
MASTER=$PWD
RUNS=$(dirname "$PWD")
export PREPARED_ENV="$MASTER/.venv" PYTHONPATH="$MASTER/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="$PREPARED_ENV/bin/python"
echo $$ > status/followup_pid
trap 'echo "follow-up failed ($(date -Is))" > "$MASTER/status/followup_FAILED"' ERR
while [ ! -f status/CODE_PUSHED ]; do sleep 5; done
test -f status/preflight_DONE

echo submission > status/followup_stage
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader) "$PY" -m reid.submit \
  --data-config configs/data.yaml --config configs/submit_exp024.yaml > logs/submission.log 2>&1
"$PY" -m reid.check_submission --data-config configs/data.yaml --config configs/submit_exp024.yaml > logs/submission_check.log 2>&1

echo EXP-025 > status/followup_stage
(cd "$RUNS/EXP-025"; CONFIG=configs/exp025_flip_tta_exp024.yaml GPU=3 bash scripts/pod_checkpoint_job.sh > status/job.log 2>&1)
"$PY" -m reid.research_compare --base "$MASTER/artifacts/EXP-024" --new "$RUNS/EXP-025/artifacts/EXP-025" \
  --out "$RUNS/EXP-025/artifacts/EXP-025/bootstrap_vs_EXP-024.json" > logs/compare_EXP-025.log 2>&1

train() {
  local exp=$1 config=$2 base=$3
  echo "$exp" > "$MASTER/status/followup_stage"
  (cd "$RUNS/$exp"; CONFIG="$config" GPU=1 SEED_GPUS=1,3,2 bash scripts/pod_train_job.sh > status/job.log 2>&1)
  "$PY" -m reid.research_compare --base "$base" --new "$RUNS/$exp/artifacts/$exp" \
    --out "$RUNS/$exp/artifacts/$exp/bootstrap_vs_$(basename "$base").json" > "$MASTER/logs/compare_$exp.log" 2>&1
}
train EXP-026 configs/exp026_vitl14_llrd09.yaml "$MASTER/artifacts/EXP-023"
train EXP-027 configs/exp027_vitl14_arcface.yaml "$MASTER/artifacts/EXP-023"
train EXP-028 configs/exp028_vitl14_arcface_imprinting.yaml "$RUNS/EXP-027/artifacts/EXP-027"
"$PY" -m reid.research_compare --base "$MASTER/artifacts/EXP-023" --new "$RUNS/EXP-028/artifacts/EXP-028" \
  --out "$RUNS/EXP-028/artifacts/EXP-028/bootstrap_vs_EXP-023.json" > logs/compare_EXP-028_vs_EXP-023.log 2>&1
train EXP-029 configs/exp029_vitl14_280.yaml "$MASTER/artifacts/EXP-023"

if "$PY" -c 'import json; assert json.load(open("artifacts/EXP-026/bootstrap_vs_EXP-023.json"))["decision"] == "accepted"'; then
  echo "LLRD 0.9 improved quality; higher-lr fallback skipped" > "$RUNS/EXP-036/status/SKIPPED"
else
  train EXP-036 configs/exp036_vitl14_llrd09_lr3e-5.yaml "$MASTER/artifacts/EXP-026"
  "$PY" -m reid.research_compare --base "$MASTER/artifacts/EXP-023" --new "$RUNS/EXP-036/artifacts/EXP-036" \
    --out "$RUNS/EXP-036/artifacts/EXP-036/bootstrap_vs_EXP-023.json" > logs/compare_EXP-036_vs_EXP-023.log 2>&1
fi
echo done > status/followup_stage
touch status/followup_DONE
