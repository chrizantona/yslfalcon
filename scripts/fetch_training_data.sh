#!/usr/bin/env bash
# Download the two external files the submitted model was trained from into EXT_DIR and verify their sha256:
# the public SigLIP 2 So400m/14 weights (the starting point) and the VERI-Wild test split (the pretraining set).
# Usage: bash scripts/fetch_training_data.sh   (EXT_DIR defaults to /workspace/ext_data; set KAGGLE_API_TOKEN if the
# Kaggle download asks for it). Files that already exist with the right checksum are skipped. The competition
# dataset is not public: it comes from the participant's account (README.md, "Внешние данные").
set -Eeuo pipefail
EXT_DIR=${EXT_DIR:-/workspace/ext_data}
KAGGLE="https://www.kaggle.com/api/v1/datasets/download"

fetch() {  # relative path, url, sha256, [kaggle]
  local path="$EXT_DIR/$1" url=$2 sha=$3 auth=()
  if [ -f "$path" ] && echo "$sha  $path" | sha256sum -c --status; then
    echo "ok (cached) $1"; return
  fi
  if [ "${4:-}" = kaggle ]; then
    if [ -n "${KAGGLE_API_TOKEN:-}" ]; then auth=(-H "Authorization: Bearer $KAGGLE_API_TOKEN"); fi
  fi
  mkdir -p "$(dirname "$path")"
  curl -sSfL --retry 5 -C - "${auth[@]}" -o "$path" "$url"
  echo "$sha  $path" | sha256sum -c --quiet && echo "ok $1"
}

fetch weights/clip/siglip2_so400m14_webli.safetensors \
  https://huggingface.co/timm/ViT-SO400M-14-SigLIP2/resolve/f9e2bfdc4807cd0fbadf67301e45193d6f330c20/open_clip_model.safetensors \
  bd5884118e72b5cc24c37cff23b8d202e2c3dac723b8a150901b022785235626
fetch kaggle/veriwild_test.zip "$KAGGLE/mrkdagods/veriwild-test?datasetVersionNumber=1" \
  34f4ba6bf73302145ce85ab2477cb2761972d053706cdb5932e8ab44a39c3b56 kaggle
