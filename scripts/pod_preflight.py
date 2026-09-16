"""Read-only checks before the overnight ViT-L jobs; uses the experiment's isolated environment."""
import importlib.metadata
from pathlib import Path
import subprocess
import sys

import cv2
import pandas as pd
import torch

from reid.data import load_yaml, crop_box
from reid.models import build_backbone, ReIDModel, checkpoint_name


base, first = load_yaml("configs/exp003_clip_vitb16_bot_30ep.yaml"), load_yaml("configs/exp1.yaml")
for key in ("train", "embed", "eval_fold", "refusal", "seeds"):
    assert first[key] == base[key], f"recipe changed: {key}"
assert first["model"]["input_hw"] == base["model"]["input_hw"]
assert first["model"]["name"] == "ViT-L-14"
assert sys.version_info[:3] == (3, 13, 7)
assert "init_checkpoint" not in first["model"]
assert torch.__version__ == "2.10.0+cu130", torch.__version__
assert importlib.metadata.version("torchvision") == "0.25.0+cu130"
assert importlib.metadata.version("open-clip-torch") == "3.3.0"
assert cv2.__version__ == "5.0.0"
assert torch.cuda.is_available()

data, val = load_yaml("configs/data.yaml"), load_yaml("configs/val_v2.yaml")
for filename in ("exp1.yaml", "exp3.yaml", "exp4.yaml", "exp5.yaml"):
    config = load_yaml(Path("configs") / filename)
    for seed in config.get("seeds", [None]):
        assert not (Path(data["checkpoint_dir"]) / checkpoint_name(config, seed)).exists(), "checkpoint name already used"
dataset, protocol = Path(data["dataset_dir"]), Path(val["out_dir"])
folds = pd.read_csv(protocol / "folds.csv")
assert folds.groupby("vehicle_id").fold.nunique().max() == 1
train = folds[folds.fold.isin(first["train"]["folds"])]
assert set(train.vehicle_id).isdisjoint(folds[folds.fold.isin([0, 4])].vehicle_id)
assert train.vehicle_id.nunique() == 924 and len(train) == 5725
assert all((Path(data["crops_dir"]) / f"{i}.jpg").is_file() for i in train.image_id)

query = pd.read_csv(protocol / "fold0_query.csv")
gallery = pd.read_csv(protocol / "fold0_gallery.csv")
benchmark = pd.read_csv(dataset / "test_query.csv")
assert len(query) == 1137 and len(gallery) == 789
for rows in (query, gallery, benchmark):
    assert all((dataset / "images" / f"{i}.jpg").is_file() for i in rows.image_id)

# Check cross-platform staging against fresh pod preprocessing for a dev-frame sample.
for row in pd.concat([query.iloc[:25], gallery.iloc[:25]]).itertuples():
    crop = crop_box(cv2.imread(str(dataset / "images" / f"{row.image_id}.jpg")),
                    (row.x, row.y, row.w, row.h), data["crop_max_side"])
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, data["crop_jpeg_quality"]])
    assert ok
    assert encoded.tobytes() == (Path(data["crops_dir"]) / f"{row.image_id}.jpg").read_bytes(), "staged crop differs"

for gpu in (1, 3, 2):
    fields = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,utilization.gpu,uuid",
                                      "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    assert int(fields[0]) >= 65536, f"GPU {gpu}: insufficient free VRAM"
    assert int(fields[1]) < 20, f"GPU {gpu}: busy; inspect before starting"
    print("GPU", gpu, "free MiB", fields[0], "util", fields[1], "uuid", fields[2], flush=True)

backbone, dim, mean, std = build_backbone(first["model"], data["external_weights_dir"])
assert dim == 768 and tuple(mean) == tuple(std) == (0.5, 0.5, 0.5)
model = ReIDModel(backbone, dim, 924).cuda()
batch_size = first["train"]["vehicles_per_batch"] * first["train"]["images_per_vehicle"]
x = torch.zeros(batch_size, 3, *first["model"]["input_hw"], device="cuda")
labels = torch.arange(batch_size, device="cuda") // first["train"]["images_per_vehicle"]
with torch.autocast("cuda", dtype=torch.bfloat16):
    feature, logits = model(x)
loss = torch.nn.functional.cross_entropy(logits.float(), labels) + feature.float().square().mean()
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print("Synthetic batch", batch_size, "BF16 backward finite; peak VRAM MiB",
      round(torch.cuda.max_memory_allocated() / 2**20), flush=True)

print("Ready: exact EXP-003 recipe, val_v2, 924 train vehicles, isolated runtime, unchanged dev JPEGs and crops.", flush=True)
