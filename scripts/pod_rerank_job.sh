#!/usr/bin/env bash
# Pod job for a re-ranking experiment on saved embeddings: setup -> re-rank every seed -> aggregate -> slices.
# Run from a run dir holding a copy of the repo and artifacts/<source_exp>/seed<k>/, detached from the ssh session:
#   CONFIG=configs/exp006_rerank_exp003.yaml setsid nohup bash scripts/pod_rerank_job.sh > status/job.log 2>&1 < /dev/null &
# CPU only. Progress: status/stage, status/stages; result: status/DONE or status/FAILED; logs in logs/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${CONFIG:?set CONFIG}"
source scripts/pod_setup.sh
EXP=$(sed -n 's/^exp_id: *\([^ #]*\).*/\1/p' "$CONFIG")

stage rerank
$PY -m reid.rerank --val-config "$VAL_CFG" --config "$CONFIG" > logs/rerank.log 2>&1

stage aggregate
$PY -m reid.aggregate --config "$CONFIG" > logs/aggregate.log 2>&1

stage slices
$PY -m reid.slices --exp-dir "artifacts/$EXP" --val-config "$VAL_CFG" --attributes protocol/attributes.csv \
  > logs/slices.log 2>&1

stage done
touch status/DONE
