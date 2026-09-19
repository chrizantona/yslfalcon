"""One real LCT batch check for the frozen EXP-048 + GLPE hybrid."""
import argparse
import json
from pathlib import Path

import torch

from reid.data import load_yaml
from reid.models import checkpoint_name
from reid.paper_models import build_paper_model
from reid.train import CropDataset, PKSampler, training_table
from reid.train_paper import build_paper_optimizer, load_paper_init, paper_loss


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/exp132_clip_convnext_glpe_frozen.yaml")
args = parser.parse_args()
cfg = load_yaml(args.config)
data, val, tcfg = load_yaml("configs/data.yaml"), load_yaml("configs/val_v2.yaml"), cfg["train"]
assert cfg["parent"] == "EXP-048" and cfg["eval_fold"] == 0 and tcfg["folds"] == [1, 2, 3]
assert val["holdout_fold"] == 4
assert not (Path(data["checkpoint_dir"]) / checkpoint_name(cfg, 0)).exists()

table, crops, classes = training_table(cfg, data, val)
assert len(table) == 5725 and len(classes) == 924
model, dim, mean, std = build_paper_model(cfg["model"], len(classes), data["external_weights_dir"],
                                          load_pretrained=False)
parent = Path(data["checkpoint_dir"]) / cfg["model"]["init_checkpoint"]
load_paper_init(model, parent, cfg["model"]["init_checkpoint_sha256"])
assert dim == 768 + 5 * 1536
assert model.visual.trunk.num_features == 1536 and model.global_dim == 768
assert not any(parameter.requires_grad for parameter in model.visual.parameters())
assert not any(parameter.requires_grad for parameter in model.global_bn.parameters())

sampler = PKSampler(table, tcfg["vehicles_per_batch"], tcfg["images_per_vehicle"], seed=0)
sampler.set_epoch(1)
indices = next(iter(sampler))
dataset = CropDataset(table, crops, cfg["model"]["input_hw"], mean, std, tcfg)
batch = [dataset[index] for index in indices]
x = torch.stack([item[0] for item in batch]).cuda()
y = torch.as_tensor([item[1] for item in batch], device="cuda")

model = model.cuda().train()
assert not model.visual.training and not model.global_bn.training
optimizer = build_paper_optimizer(model, tcfg)
cross_entropy = torch.nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    embeddings, logits = model(x)
loss, branches = paper_loss(embeddings, logits, y, tcfg["loss"], cross_entropy)
optimizer.zero_grad(set_to_none=True)
loss.backward()
bad = [name for name, parameter in model.named_parameters()
       if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
assert torch.isfinite(loss) and not bad, bad
assert all(parameter.grad is None for parameter in model.visual.parameters())
optimizer.step()

model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    descriptor = model(x[:8]).float()
    parent_global = torch.nn.functional.normalize(model.global_bn(model.visual(x[:8])).float(), dim=1)
assert torch.allclose(descriptor[:, :768], parent_global, atol=2e-3, rtol=2e-3)
assert torch.allclose(descriptor[:, :768].norm(dim=1), torch.ones(8, device="cuda"), atol=2e-3)
local = descriptor[:, 768:].reshape(8, 5, 1536)
expected_local_norm = cfg["model"]["local_scale"] / 5 ** 0.5
assert torch.allclose(local.norm(dim=2), torch.full((8, 5), expected_local_norm, device="cuda"), atol=2e-3)

print(json.dumps({
    "exp_id": cfg["exp_id"],
    "parent_checkpoint": parent.name,
    "train_images": len(table),
    "train_vehicles": len(classes),
    "batch": len(x),
    "descriptor_dim": descriptor.shape[1],
    "loss": loss.item(),
    "loss_branches": [branch.item() for branch in branches],
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "all_trainable_gradients_finite": True,
    "parent_global_reproduced": True,
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "holdout_fold_untouched": True,
}, indent=2))
