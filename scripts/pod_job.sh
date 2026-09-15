#!/usr/bin/env bash
# Pod job for steps 1-2: dataset -> venv -> tests -> protocol -> crops (CPU, background) + eval -> submit -> check.
# Run from a run dir holding a copy of the repo, detached from the ssh session:
#   setsid nohup bash scripts/pod_job.sh > status/job.log 2>&1 < /dev/null &
# Progress: status/stage and status/stages; result: status/DONE or status/FAILED; per-step logs in logs/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/exp000_clip_zeroshot.yaml}
DATA_ZIP=${DATA_ZIP:-/workspace/lct2026_data/dataset.zip}
DATA_CFG=configs/data.yaml
VAL_CFG=configs/val_v1.yaml

mkdir -p status logs
rm -f status/DONE status/FAILED
# everything the job downloads stays inside the run dir, so deleting the run dir leaves only the dataset on the pod
export UV_CACHE_DIR="$PWD/.uv-cache" UV_LINK_MODE=copy HF_HOME="$PWD/.hf" PYTHONPATH="$PWD/src" CUDA_DEVICE_ORDER=PCI_BUS_ID
trap 'echo "failed at stage $(cat status/stage) ($(date +%T))" > status/FAILED' ERR

stage() {
  echo "$1" > status/stage
  echo "$(date +%T) $1" >> status/stages
}

stage dataset
DATA_DIR=$(sed -n 's/^dataset_dir: *\([^ #]*\).*/\1/p' "$DATA_CFG")
if [ ! -f "$DATA_DIR/train.csv" ]; then
  python3 - "$DATA_ZIP" "$DATA_DIR" <<'EOF'
import sys, zipfile
archive = zipfile.ZipFile(sys.argv[1])
assert archive.testzip() is None, "corrupt dataset.zip"
archive.extractall(sys.argv[2])
EOF
fi

stage venv
uv venv -q --system-site-packages --python /usr/bin/python3 .venv
# torch comes from the system; open-clip and timm are installed without deps so they don't pull a second torch
grep -vE '^(open-clip-torch|timm)==' requirements.txt > .venv/requirements-rest.txt
VIRTUAL_ENV=.venv uv pip install -q -r .venv/requirements-rest.txt pytest
VIRTUAL_ENV=.venv uv pip install -q --no-deps $(grep -E '^(open-clip-torch|timm)==' requirements.txt)
PY=.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)" > logs/env.txt
VIRTUAL_ENV=.venv uv pip freeze >> logs/env.txt

stage tests
$PY -m pytest -q tests > logs/tests.log 2>&1

stage protocol
if [ ! -f protocol/val_v1/folds.csv ]; then
  $PY -m reid.protocol --data-config "$DATA_CFG" --config "$VAL_CFG" > logs/protocol.log 2>&1
fi

$PY -m reid.crops --data-config "$DATA_CFG" > logs/crops.log 2>&1 &
CROPS_PID=$!
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > m {m = $2; i = $1} END {print i}')
echo "$(date +%T) crops running in background; GPU $CUDA_VISIBLE_DEVICES" >> status/stages

stage eval
$PY -m reid.evaluate --data-config "$DATA_CFG" --val-config "$VAL_CFG" --config "$CONFIG" > logs/eval.log 2>&1

stage submit
$PY -m reid.submit --data-config "$DATA_CFG" --config "$CONFIG" > logs/submit.log 2>&1

stage check
$PY -m reid.check_submission --data-config "$DATA_CFG" --config "$CONFIG" > logs/check.log 2>&1

stage wait_crops
wait "$CROPS_PID"

stage done
touch status/DONE
