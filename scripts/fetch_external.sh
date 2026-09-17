#!/usr/bin/env bash
# Download the external data and public weights listed in DATA.md into EXT_DIR and verify every sha256.
# Usage on the pod: KAGGLE_API_TOKEN=<token> bash scripts/fetch_external.sh   (EXT_DIR defaults to /workspace/ext_data)
# Files that already exist with the right checksum are skipped. The competition dataset is not public: see DATA.md.
set -Eeuo pipefail
EXT_DIR=${EXT_DIR:-/workspace/ext_data}
TARGET=${1:-all}  # all | clip_l | backbones | veriwild
case "$TARGET" in all|clip_l|backbones|veriwild) ;; *) echo "unknown target: $TARGET" >&2; exit 2;; esac
GDRIVE="https://drive.usercontent.google.com/download?export=download&confirm=t&id="
FASTREID="https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1"
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

if [ "$TARGET" = all ] || [ "$TARGET" = clip_l ]; then
  fetch weights/clip/exp020.safetensors \
    https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/1627032197142fbe2a7cfec626f4ced3ae60d07a/open_clip_pytorch_model.safetensors \
    7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019
fi
[ "$TARGET" != clip_l ] || exit 0

if [ "$TARGET" = all ] || [ "$TARGET" = backbones ]; then  # backbone screens EXP-030…032, EXP-048 and their chains
  fetch weights/clip/dfn2b_vitl14.bin \
    https://huggingface.co/apple/DFN2B-CLIP-ViT-L-14/resolve/13d68e40771108ec1812f566404e8c6f7de85f67/open_clip_pytorch_model.bin \
    c02c84198ede77ad80df0803224022f1e9e95ce42551baa91a1b5d87e229e2bb
  fetch weights/clip/eva02_l14_merged2b.safetensors \
    https://huggingface.co/timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k/resolve/bf4190eb65dd5204ffb03e980108beb1200e0873/open_clip_model.safetensors \
    00af04296f09f24dcc69559440a80b7a44daf4855a72827e016067e6e571b851
  fetch weights/clip/siglip2_so400m14_webli.safetensors \
    https://huggingface.co/timm/ViT-SO400M-14-SigLIP2/resolve/f9e2bfdc4807cd0fbadf67301e45193d6f330c20/open_clip_model.safetensors \
    bd5884118e72b5cc24c37cff23b8d202e2c3dac723b8a150901b022785235626
  fetch weights/clip/convnext_large_d_320_ft_soup.safetensors \
    https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/resolve/654d0f80ff73c58e7281a3ca7dc425589049e2e1/open_clip_model.safetensors \
    1a8e47bdefd6d36202ec66a8975edda9e3c89560bcd1ecefe6f676311637a671
fi
[ "$TARGET" != backbones ] || exit 0

# VERI-Wild large test split (10,000 vehicles), used for the EXP-009 pretraining
fetch kaggle/veriwild_test.zip "$KAGGLE/mrkdagods/veriwild-test?datasetVersionNumber=1" \
  34f4ba6bf73302145ce85ab2477cb2761972d053706cdb5932e8ab44a39c3b56 kaggle
[ "$TARGET" != veriwild ] || exit 0
# VeRi-776, used by the EXP-033 pretraining together with VERI-Wild test
fetch kaggle/veri776.zip "$KAGGLE/abhyudaya12/veri-vehicle-re-identification-dataset" \
  9d45b9ad779d04d58c14abc09a0f681427145464c0aa9faa3ac19b3fea4d0694 kaggle
# FastReID (Apache-2.0) release v0.1.1: EXP-011, EXP-012, EXP-017, EXP-018
fetch weights/fast-reid/veriwild_bot_R50-ibn.pth "$FASTREID/veriwild_bot_R50-ibn.pth" \
  f591e4f7f31ce0be2f4079171cfc03fcbecab5ea5b40f9148c753b9c6f061e8e
fetch weights/fast-reid/vehicleid_bot_R50-ibn.pth "$FASTREID/vehicleid_bot_R50-ibn.pth" \
  48992dbfdb99b93c9a0c2f15dd27b7a738f8b8a7e49dd231391ed211f88d759a
fetch weights/fast-reid/veri_sbs_R50-ibn.pth "$FASTREID/veri_sbs_R50-ibn.pth" \
  57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd
# CLIP-ReID (MIT), links from the README of github.com/Syliz517/CLIP-ReID: EXP-013, EXP-014
fetch weights/clip-reid/VehicleID_clipreid_ViT-B-16_60.pth "${GDRIVE}168BLegHHxNqatW5wx1YyL2REaThWoof5" \
  538e8288741b45deb1dd8225e47b57fc0702fde79156ca673599e53c1905b27b
fetch weights/clip-reid/VeRi_clipreid_ViT-B-16_60.pth "${GDRIVE}1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e" \
  71f6f2723be4d563f1df201133daa53f57bee2a67266486559d5ce453a60f805
curl -sSfL -o "$EXT_DIR/weights/fast-reid/LICENSE" https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/LICENSE
curl -sSfL -o "$EXT_DIR/weights/clip-reid/LICENSE" https://raw.githubusercontent.com/Syliz517/CLIP-ReID/master/LICENSE
