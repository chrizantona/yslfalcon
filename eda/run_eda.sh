#!/usr/bin/env bash
# Pod-side pipeline: verify + unpack data -> extract features -> execute the notebook.
# Usage (from the project root on the pod):  GPU=2 bash eda/run_eda.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="$PWD/hf" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
PY=.venv/bin/python
SHM=/dev/shm/lct2026_dataset

if [ ! -f "$SHM/train.csv" ]; then
  echo "[1/4] verifying and unpacking dataset.zip -> $SHM"
  $PY - "$SHM" <<'EOF'
import sys, zipfile
z = zipfile.ZipFile("BIG_DATA/dataset.zip")
bad = z.testzip()
assert bad is None, f"corrupt member: {bad}"
z.extractall(sys.argv[1])
print("unpacked", len(z.namelist()), "members")
EOF
fi
ln -sfn "$SHM" BIG_DATA/dataset

echo "[2/4] feature extraction"
[ -f eda/cache/emb_dino_scene.npy ] || $PY eda/extract_features.py

echo "[3/4] building notebook"
$PY -m ipykernel install --prefix .venv --name lct-eda --display-name "lct-eda" >/dev/null
$PY -m jupytext --to ipynb --set-kernel lct-eda eda/eda.py

echo "[4/4] executing notebook"
$PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 eda/eda.ipynb
echo "done"
