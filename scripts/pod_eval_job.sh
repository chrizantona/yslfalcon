#!/usr/bin/env bash
# Pod job for experiments without training, scored on the dev fold: public ReID checkpoints (reid.evaluate), seed
# ensembles on saved embeddings (reid.ensemble) and weight soups (reid.soup), each followed by mAP by slice.
# Run from a run dir holding a copy of the repo (plus artifacts/<source_exp>/seed<k>/ for ensembles), detached:
#   EVALS="configs/a.yaml ..." ENSEMBLES="..." SOUPS="..." GPU=2 setsid nohup bash scripts/pod_eval_job.sh > status/job.log 2>&1 < /dev/null &
# Progress: status/stage, status/stages; result: status/DONE or status/FAILED; logs in logs/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${GPU:?set GPU}"
source scripts/pod_setup.sh
export CUDA_VISIBLE_DEVICES=$GPU

run() {  # module, space-separated configs
  for config in $2; do
    exp=$(sed -n 's/^exp_id: *\([^ #]*\).*/\1/p' "$config")
    stage "$exp"
    $PY -m "$1" --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$config" > "logs/$exp.log" 2>&1
    $PY -m reid.slices --exp-dir "artifacts/$exp" --val-config "$VAL_CFG" --attributes protocol/attributes.csv \
      > "logs/slices_$exp.log" 2>&1
  done
}
run reid.evaluate "${EVALS:-}"
run reid.ensemble "${ENSEMBLES:-}"
run reid.soup "${SOUPS:-}"

stage done
touch status/DONE
