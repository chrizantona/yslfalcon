#!/usr/bin/env bash
# Pod job for the analysis tools: image attributes -> mAP by slice per experiment -> speed and size per config.
# Run from a run dir holding a copy of the repo, detached from the ssh session:
#   GPU=2 setsid nohup bash scripts/pod_tools_job.sh > status/job.log 2>&1 < /dev/null &
# Experiment artifacts (per_query_ap*.csv) must already be in artifacts/<EXP-ID>/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/pod_setup.sh

EXPS=${EXPS-"EXP-000 EXP-001 EXP-002 EXP-003 EXP-004"}
CONFIGS=${CONFIGS-"configs/exp000_clip_zeroshot.yaml configs/exp001_clip_vitb16_bot.yaml configs/exp002_resnet50_bot.yaml configs/exp003_clip_vitb16_bot_30ep.yaml configs/exp004_clip_vitb16_bot_lr5e-6.yaml"}
export CUDA_VISIBLE_DEVICES=${GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > m {m = $2; i = $1} END {print i}')}

stage attributes
if [ ! -f protocol/attributes.csv ]; then
  $PY -m reid.attributes --data-config "$DATA_CFG" --config configs/exp000_clip_zeroshot.yaml > logs/attributes.log 2>&1
fi

stage slices
for exp in $EXPS; do
  $PY -m reid.slices --exp-dir "artifacts/$exp" --val-config "$VAL_CFG" --attributes protocol/attributes.csv \
    > "logs/slices_$exp.log" 2>&1
done

stage benchmark
for config in $CONFIGS; do
  $PY -m reid.benchmark --data-config "$DATA_CFG" --config "$config" \
    > "logs/benchmark_$(basename "$config" .yaml).log" 2>&1
done

stage done
touch status/DONE
