#!/usr/bin/env bash
# Run a queue of EXPs one after another, each in its own copy of this repo under /workspace/yslfalcon_runs/<EXP>/,
# detached from ssh. Several queues can run at once on different GPUs.
# Queue file: one EXP per line, "<EXP-ID> <config> <benchmark GPU> <seed GPUs> <seeds in parallel: 1|0> [train|soup]";
# empty lines and # comments are skipped. GPU numbers belong to the pod and may be edited; configs may not.
# Run from a copy of the repo; PREPARED_ENV points to a venv built once (scripts/pod_env_job.sh or pod_setup.sh):
#   QUEUE=scripts/queues/pod_b_dfn2b.txt PREPARED_ENV=/tmp/EXP-044/.venv \
#     setsid nohup bash scripts/pod_queue_job.sh > status/queue_pod_b_dfn2b.log 2>&1 < /dev/null &
# Progress: status/queue_<name>_stage; result: status/queue_<name>_DONE or status/queue_<name>_FAILED.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${QUEUE:?set QUEUE}" "${PREPARED_ENV:?set PREPARED_ENV}"
MASTER=$PWD
RUNS=${RUNS:-/workspace/yslfalcon_runs}
NAME=$(basename "$QUEUE" .txt)
export PREPARED_ENV OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p status
rm -f "status/queue_${NAME}_DONE" "status/queue_${NAME}_FAILED"
echo $$ > "status/queue_${NAME}_pid"
trap 'echo "queue $NAME failed ($(date -Is))" > "$MASTER/status/queue_${NAME}_FAILED"' ERR

while read -r exp config gpu seed_gpus parallel kind; do
  case "$exp" in ""|\#*) continue;; esac
  echo "$(date +%T) $exp start" >> "status/queue_${NAME}_stage"
  run="$RUNS/$exp"
  mkdir -p "$run/status"  # job.log below is opened before the job creates status/
  tar -C "$MASTER" --exclude=./.venv --exclude=./.python --exclude=./.uv-cache --exclude=./status --exclude=./logs \
    --exclude=./artifacts -cf - . | tar -C "$run" -xf -
  # < /dev/null: the job must not read the rest of the queue file from stdin
  if [ "${kind:-train}" = soup ]; then
    (cd "$run"; SOUPS="$config" GPU="$gpu" bash scripts/pod_eval_job.sh > status/job.log 2>&1 < /dev/null)
  else
    (cd "$run"; CONFIG="$config" GPU="$gpu" SEED_GPUS="$seed_gpus" PARALLEL_SEEDS="$parallel" \
      bash scripts/pod_train_job.sh > status/job.log 2>&1 < /dev/null)
  fi
  echo "$(date +%T) $exp done" >> "status/queue_${NAME}_stage"
done < "$QUEUE"
touch "status/queue_${NAME}_DONE"
