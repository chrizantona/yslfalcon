#!/usr/bin/env bash
# EXP-030…034 on one pod whose GPU memory is shared with other containers. The VeRi-776 pretraining EXP-033 runs on
# GPU 2 in the background; meanwhile the backbone screens EXP-030 and EXP-032 train their three seeds in parallel on
# GPUs 1/3/2. Once EXP-033 is done, its fine-tune EXP-034 trains on GPUs 1/3/2, then EVA02 EXP-031 trains one seed at a
# time on GPU 2: it needs ~55 GB per seed, more than GPUs 1 and 3 have free.
# Run from /workspace/yslfalcon_runs/EXP-030 (a copy of the repo with the prepared venv), detached:
#   setsid nohup bash scripts/pod_series_job.sh > status/series.log 2>&1 < /dev/null &
# Every EXP runs in its own copy /workspace/yslfalcon_runs/<EXP>/ with its own status/ and logs/.
# Progress: status/series_stage; result: status/series_DONE or status/series_FAILED.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
MASTER=$PWD
RUNS=$(dirname "$PWD")
export PREPARED_ENV="$MASTER/.venv" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p status
rm -f status/series_DONE status/series_FAILED
echo $$ > status/series_pid
trap 'echo "series failed ($(date -Is))" > "$MASTER/status/series_FAILED"' ERR

run() {  # EXP id, config, benchmark GPU, seed GPUs, seeds in parallel (1) or one after another (0)
  echo "$(date +%T) $1 start" >> "$MASTER/status/series_stage"
  if [ "$RUNS/$1" != "$MASTER" ]; then
    mkdir -p "$RUNS/$1/status"  # job.log below is opened before pod_train_job.sh creates status/
    tar -C "$MASTER" --exclude=./.venv --exclude=./.python --exclude=./.uv-cache --exclude=./status --exclude=./logs \
      --exclude=./artifacts -cf - . | tar -C "$RUNS/$1" -xf -
  fi
  (cd "$RUNS/$1"; CONFIG="$2" GPU="$3" SEED_GPUS="$4" PARALLEL_SEEDS="$5" bash scripts/pod_train_job.sh > status/job.log 2>&1)
  echo "$(date +%T) $1 done" >> "$MASTER/status/series_stage"
}

run EXP-033 configs/exp033_pretrain_veriwild_veri776_vitl14.yaml 2 2 1 &
PRETRAIN=$!
run EXP-030 configs/exp030_dfn2b_vitl14.yaml 1 1,3,2 1
run EXP-032 configs/exp032_siglip2_so400m14.yaml 1 1,3,2 1
wait "$PRETRAIN"
run EXP-034 configs/exp034_arcface_imprinting_init_exp033.yaml 1 1,3,2 1
run EXP-031 configs/exp031_eva02_l14.yaml 2 2 0
touch status/series_DONE
