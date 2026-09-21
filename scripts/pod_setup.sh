# Shared setup for pod jobs (sourced, not run): stage bookkeeping, dataset unpacking, venv, tests.
# Everything the job downloads stays inside the run dir, so deleting the run dir leaves only the dataset on the pod.
DATA_CFG=${DATA_CFG:-configs/data.yaml}  # configs/data_sam.yaml: the masked copy of the dataset
VAL_CFG=configs/val_v2.yaml
DATA_ZIP=${DATA_ZIP:-/workspace/lct2026_data/dataset.zip}
PY=${PREPARED_ENV:-$PWD/.venv}/bin/python

mkdir -p status logs
rm -f status/DONE status/FAILED
echo $$ > status/pid
export UV_CACHE_DIR="$PWD/.uv-cache" UV_LINK_MODE=copy HF_HOME="$PWD/.hf" PYTHONPATH="$PWD/src" CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_UUID=$(nvidia-smi -i "${GPU:?set GPU before setup}" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES=$GPU_UUID
trap 'echo "failed at stage $(cat status/stage) ($(date +%T))" > status/FAILED' ERR

stage() {
  echo "$1" > status/stage
  echo "$(date +%T) $1" >> status/stages
}

stage dataset
exec 8>/workspace/lct2026_data/prepare.lock
flock 8
DATA_DIR=$(sed -n 's/^dataset_dir: *\([^ #]*\).*/\1/p' "$DATA_CFG")
if [ ! -f "$DATA_DIR/train.csv" ]; then
  python3 - "$DATA_ZIP" "$DATA_DIR" <<'EOF'
import sys, zipfile
import hashlib
with open(sys.argv[1], "rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == "a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613", "dataset checksum mismatch"
archive = zipfile.ZipFile(sys.argv[1])
assert archive.testzip() is None, "corrupt dataset.zip"
archive.extractall(sys.argv[2])
EOF
fi
flock -u 8

stage venv
if [ -n "${PREPARED_ENV:-}" ]; then
  test -x "$PY"
else
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
uv python install 3.13.7
uv venv -q --python 3.13.7 .venv
uv pip install -q --python "$PY" -r requirements-pod.txt --index-url https://download.pytorch.org/whl/cu130
uv pip install -q --python "$PY" -r requirements.txt pytest==9.1.1
fi
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)" > logs/env.txt
uv pip freeze --python "$PY" >> logs/env.txt

stage tests
$PY -m pytest -q tests > logs/tests.log 2>&1
