#!/usr/bin/env bash
# Download the external data and public weights listed in DATA.md into EXT_DIR and verify every sha256.
# Usage on the pod: KAGGLE_API_TOKEN=<token> bash scripts/fetch_external.sh   (EXT_DIR defaults to /workspace/ext_data)
# Files that already exist with the right checksum are skipped. The competition dataset is not public: see DATA.md.
set -Eeuo pipefail
EXT_DIR=${EXT_DIR:-/workspace/ext_data}
TARGET=${1:-all}  # all | clip_l | backbones | paper | osnet | veriwild
case "$TARGET" in all|clip_l|backbones|paper|osnet|veriwild) ;; *) echo "unknown target: $TARGET" >&2; exit 2;; esac
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
  fetch weights/clip/siglip2_so400m16_256_webli.safetensors \
    https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-256/resolve/8b4c2efdfd48e0d72644df99ed5dc20c110fc352/open_clip_model.safetensors \
    c824f8bb65616a30b57f32e62ca3f4a10287b00f263c989706093cd20daaa4f0
  fetch weights/clip/siglip2_so400m16_384_webli.safetensors \
    https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-384/resolve/fec784dabb3081a5f101fc74eefaf9d1ed08237b/open_clip_model.safetensors \
    8ceec00df3c462d155d172933339cd9178a86bfab72f4458dbe84bd54f5ad026
  fetch weights/clip/convnext_large_d_320_ft_soup.safetensors \
    https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/resolve/654d0f80ff73c58e7281a3ca7dc425589049e2e1/open_clip_model.safetensors \
    1a8e47bdefd6d36202ec66a8975edda9e3c89560bcd1ecefe6f676311637a671
fi
[ "$TARGET" != backbones ] || exit 0

if [ "$TARGET" = all ] || [ "$TARGET" = paper ]; then  # reproduced architectures EXP-055…058 (reid.paper_models)
  fetch weights/paper/resnext101_ibn_a-6ace051d.pth \
    https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth \
    6ace051dc2d2de2aace20cc327d9f2f64892696eaf13d8b1bbf0accf45684ac8
  fetch weights/paper/tinyclip_vit61m_32_laion400m.safetensors \
    https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M/resolve/94cbbea5c7949cfe7bdafde64bcea5e403f59852/model.safetensors \
    0c0670dfa6fbf32c3158ef57b10772d5c9668093bda676d78e1452268585d230
  fetch weights/paper/tinyclip_vit61m_32_laion400m_config.json \
    https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M/resolve/94cbbea5c7949cfe7bdafde64bcea5e403f59852/config.json \
    40a59f3db33a1df844687e94f25b186bf187bec1d8d1389f0f88b1ab013e233b
  fetch weights/paper/tinyclip_vit40m_32_laion400m.safetensors \
    https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/resolve/95ec8197b3f2fe7f747865c61ca556cf0768b2f7/model.safetensors \
    1b08131ac0ac54429aab216c397c550527d31f5682afb5d21d35091ad39538ae
  fetch weights/paper/tinyclip_vit40m_32_laion400m_config.json \
    https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/resolve/95ec8197b3f2fe7f747865c61ca556cf0768b2f7/config.json \
    b786ac49b5cdcb16496a33c8ab4884de036b2f64de27d0b73caf8391f94c9bee
  fetch weights/paper/convnext_large_fb_in22k.safetensors \
    https://huggingface.co/timm/convnext_large.fb_in22k/resolve/be6de86d9928c6cc2a8986963243142f403f382d/model.safetensors \
    7e02ea57d6024d94f5523e9b58cdddd81a1fb4f2e5128b4a879b76f9847058bf
  fetch weights/paper/best_convnext_large_rerank.pth "${GDRIVE}1DO5RejFjM-b7rwMfryX0BZLWJNHDt14m" \
    35ddfd1ea7b73b90a66ef41d652bf6600cef1ce9260f1d7aa8f2f84ac3f90206
fi
[ "$TARGET" != paper ] || exit 0

if [ "$TARGET" = all ] || [ "$TARGET" = osnet ]; then
  fetch weights/osnet/osnet_ain_x1_0_imagenet.pth \
    https://huggingface.co/kaiyangzhou/osnet/resolve/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_ain_x1_0_imagenet.pth \
    b8a578eacc122c16496c8835c96a0157db62963b3de05b5a5a4b1ff5bad7546a
fi
[ "$TARGET" != osnet ] || exit 0

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
