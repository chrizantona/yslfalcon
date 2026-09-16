#!/usr/bin/env bash
# Isolated runtime on the fast local disk; /dev/shm is mounted noexec and is only used for the cache.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p status logs /dev/shm/exp1 /tmp/exp1
rm -f status/env_FAILED status/env_DONE
trap 'echo "environment preparation failed ($(date -Is))" > status/env_FAILED' ERR
export UV_CACHE_DIR=/dev/shm/exp1/.uv-cache UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR=/tmp/exp1/python
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
export PYTHONPATH="$PWD/src"
uv python install 3.13.7 > logs/install_python.log 2>&1
uv venv -q --clear --python 3.13.7 /tmp/exp1/.venv
uv pip install --python /tmp/exp1/.venv/bin/python -r requirements-pod.txt \
  --index-url https://download.pytorch.org/whl/cu130 > logs/install_torch.log 2>&1
uv pip install --python /tmp/exp1/.venv/bin/python -r requirements.txt pytest==9.1.1 > logs/install.log 2>&1
uv pip check --python /tmp/exp1/.venv/bin/python > logs/pip_check.log 2>&1
if [ -d .venv ] && [ ! -L .venv ]; then mv .venv .venv_old; fi
ln -s /tmp/exp1/.venv .venv
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /tmp/exp1/.venv/bin/python -m pytest -q tests > logs/tests_bootstrap.log 2>&1
touch status/env_DONE
