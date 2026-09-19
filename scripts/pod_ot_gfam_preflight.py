"""Real LCT batch check for the stabilised OT-GFAM + GLPE experiment."""
import json
from pathlib import Path

import torch

from reid.data import load_yaml
from reid.models import checkpoint_name
from reid.paper_models import build_paper_model
from reid.train import CropDataset, PKSampler, training_table
from reid.train_paper import build_paper_optimizer, paper_loss


cfg = load_yaml("configs/exp123_ot_gfam_stable.yaml")
data = load_yaml("configs/data.yaml")
val = load_yaml("configs/val_v2.yaml")
tcfg = cfg["train"]
assert cfg["eval_fold"] == 0 and tcfg["folds"] == [1, 2, 3]
assert val["holdout_fold"] == 4
assert not (Path(data["checkpoint_dir"]) / checkpoint_name(cfg, 0)).exists()

table, crops, classes = training_table(cfg, data, val)
assert len(table) == 5725 and len(classes) == 924
model, dim, mean, std = build_paper_model(cfg["model"], len(classes), data["external_weights_dir"])
assert dim == 6 * 768
assert all(pool.input_activation == "softplus" and pool.min_p == 1.0 for pool in model.pools)

sampler = PKSampler(table, tcfg["vehicles_per_batch"], tcfg["images_per_vehicle"], seed=0)
sampler.set_epoch(1)
indices = next(iter(sampler))
dataset = CropDataset(table, crops, cfg["model"]["input_hw"], mean, std, tcfg)
batch = [dataset[i] for i in indices]
x = torch.stack([item[0] for item in batch]).cuda()
y = torch.as_tensor([item[1] for item in batch], device="cuda")

model = model.cuda().train()
optimizer = build_paper_optimizer(model, tcfg)
assert {group["branch"]: group["lr"] for group in optimizer.param_groups} == tcfg["lr_groups"]
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
assert all(pool.p.grad is not None and torch.isfinite(pool.p.grad) for pool in model.pools)
optimizer.step()

model.eval()
with torch.no_grad():
    descriptor = torch.nn.functional.normalize(model(x[:8]).float(), dim=1)
similarity = descriptor @ descriptor.T
different = y[:8, None] != y[None, :8]
assert torch.isfinite(descriptor).all()
assert similarity[different].max() < 0.999, "descriptors of different vehicles collapsed"

print(json.dumps({
    "train_images": len(table),
    "train_vehicles": len(classes),
    "batch": len(x),
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "loss": loss.item(),
    "loss_branches": [branch.item() for branch in branches],
    "gem_p_after_step": [pool.p.item() for pool in model.pools],
    "max_cosine_between_different_vehicles": similarity[different].max().item(),
    "all_trainable_gradients_finite": True,
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "holdout_fold_untouched": True,
}, indent=2))
