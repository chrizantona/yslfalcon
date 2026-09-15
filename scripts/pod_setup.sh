# Shared setup for pod jobs (sourced, not run): stage bookkeeping, dataset unpacking, venv, tests.
# Everything the job downloads stays inside the run dir, so deleting the run dir leaves only the dataset on the pod.
DATA_CFG=configs/data.yaml
VAL_CFG=configs/val_v1.yaml
DATA_ZIP=${DATA_ZIP:-/workspace/lct2026_data/dataset.zip}
PY=.venv/bin/python

mkdir -p status logs
rm -f status/DONE status/FAILED
echo $$ > status/pid
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
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)" > logs/env.txt
VIRTUAL_ENV=.venv uv pip freeze >> logs/env.txt

stage tests
$PY -m pytest -q tests > logs/tests.log 2>&1
