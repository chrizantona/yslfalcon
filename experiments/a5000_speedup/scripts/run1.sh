set -uo pipefail
cd /root/yslfalcon
export PYTHONPATH=$PWD/src HF_HOME=/root/.hf
PY=/root/benv/bin/python
mkdir -p /root/speed_ckpt /root/bench_out
cat > /root/bench_data.yaml <<EOT
dataset_dir: /dev/shm/lct2026/dataset
crop_max_side: 512
external_weights_dir: /root/ext_data/weights
checkpoint_dir: /root/speed_ckpt
EOT
nvidia-smi dmon -s pc -d 1 > logs/dmon.log 2>&1 &
DMON=$!
(while true; do echo "$(date +%T) $(cat /proc/loadavg) $(grep MHz /proc/cpuinfo | awk "{s+=\$4; if(\$4>m)m=\$4} END {printf \"avg %d max %d\", s/NR, m}")"; sleep 2; done) > logs/cpu.log 2>&1 &
CPUMON=$!
echo ckpt > status/run1
$PY - > logs/speed_ckpt.log 2>&1 <<"EOT"
import torch
from reid.data import load_yaml
from reid.models import ReIDModel, build_backbone
cfg = load_yaml("configs/exp042_siglip2_arcface_imprinting.yaml")
model_cfg = {k: v for k, v in cfg["model"].items() if k != "init_checkpoint"}
backbone, dim, _, _ = build_backbone(model_cfg, "/root/ext_data/weights")
torch.save({"state_dict": ReIDModel(backbone, dim, 924).state_dict()}, "/root/speed_ckpt/siglip2.pt")
print("siglip2", dim, flush=True)
EOT
echo baseline > status/run1
$PY -m reid.benchmark --data-config /root/bench_data.yaml --config configs/exp043_soup_exp042_seeds.yaml --checkpoint /root/speed_ckpt/siglip2.pt --out /root/bench_out/baseline_fp16.json > logs/bench_baseline.log 2>&1 || echo baseline >> status/run1_errors
echo profile > status/run1
$PY /root/prof.py /root/bench_out/profile.json > logs/profile.log 2>&1 || echo profile >> status/run1_errors
kill $DMON $CPUMON
echo done > status/run1
