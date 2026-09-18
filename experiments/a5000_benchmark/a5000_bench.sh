#!/usr/bin/env bash
# One-off speed benchmark on the rented RTX A5000 pod; not part of the repo.
# Speed depends on architecture and precision only, so each model is a public backbone + fresh BNNeck saved as a
# checkpoint of the trained shape and measured with reid.benchmark --checkpoint, exactly like the trained soups.
set -uo pipefail
cd /root/yslfalcon
export PYTHONPATH=$PWD/src HF_HOME=/root/.hf OMP_NUM_THREADS=8
PY=/root/benv/bin/python
OUT=/root/bench_out; CK=/root/speed_ckpt; CFG=/root/bench_cfg
mkdir -p $OUT $CK $CFG status logs
while [ ! -f status/setup_DONE ]; do [ -f status/setup_FAILED ] && exit 1; sleep 5; done
cat > /root/bench_data.yaml <<EOF
dataset_dir: /dev/shm/lct2026/dataset
crop_max_side: 512
external_weights_dir: /root/ext_data/weights
checkpoint_dir: $CK
EOF
fp16() { sed 's/^  precision: fp32/  precision: fp16/' "$1" > "$CFG/$(basename "$1" .yaml)_fp16.yaml"; echo "$CFG/$(basename "$1" .yaml)_fp16.yaml"; }
fp32() { sed 's/^  precision: fp16/  precision: fp32/' "$1" > "$CFG/$(basename "$1" .yaml)_fp32.yaml"; echo "$CFG/$(basename "$1" .yaml)_fp32.yaml"; }

echo checkpoints > status/bench_stage
$PY - > logs/speed_ckpt.log 2>&1 <<'EOF'
import torch
from reid.data import load_yaml
from reid.models import ReIDModel, build_backbone
models = {"siglip2": "configs/exp042_siglip2_arcface_imprinting.yaml", "laion_vitl": "configs/exp028_vitl14_arcface_imprinting.yaml",
          "dfn2b": "configs/exp045_dfn2b_init_exp044.yaml", "convnext": "configs/exp048_convnext_large_d_320.yaml",
          "eva02": "configs/exp031_eva02_l14.yaml", "vitb": "configs/exp016_soup_exp010_seeds.yaml"}
for name, path in models.items():
    cfg = load_yaml(path)
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "init_checkpoint"}
    backbone, dim, _, _ = build_backbone(model_cfg, "/root/ext_data/weights")
    torch.save({"state_dict": ReIDModel(backbone, dim, 924).state_dict()}, f"/root/speed_ckpt/{name}.pt")
    print(name, dim, flush=True)
EOF

run() {  # label, config, checkpoint
  echo "$1" > status/bench_stage
  $PY -m reid.benchmark --data-config /root/bench_data.yaml --config "$2" --checkpoint "$CK/$3.pt" --out "$OUT/$1.json" \
    > "logs/bench_$1.log" 2>&1 || echo "$1 failed" >> status/bench_errors
}
run exp043_siglip2_fp16 configs/exp043_soup_exp042_seeds.yaml siglip2
run exp037_laion_vitl_fp16 configs/exp037_soup_exp028_seeds.yaml laion_vitl
run exp047_dfn2b_fp16 configs/exp047_soup_exp045_seeds.yaml dfn2b
run exp048_convnext_fp16 "$(fp16 configs/exp048_convnext_large_d_320.yaml)" convnext
run exp048_convnext_fp32 configs/exp048_convnext_large_d_320.yaml convnext
run exp042_siglip2_fp32 configs/exp042_siglip2_arcface_imprinting.yaml siglip2
run exp031_eva02_fp16 "$(fp16 configs/exp031_eva02_l14.yaml)" eva02
run exp037_laion_vitl_fp32 "$(fp32 configs/exp037_soup_exp028_seeds.yaml)" laion_vitl
run exp047_dfn2b_fp32 "$(fp32 configs/exp047_soup_exp045_seeds.yaml)" dfn2b
run exp016_vitb_fp32 configs/exp016_soup_exp010_seeds.yaml vitb
run exp031_eva02_fp32 configs/exp031_eva02_l14.yaml eva02
echo done > status/bench_stage
touch status/bench_DONE
