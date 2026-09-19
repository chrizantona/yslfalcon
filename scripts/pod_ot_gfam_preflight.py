"""Real LCT or external-data batch check for a stabilised OT-GFAM + GLPE experiment."""
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
parser.add_argument("--config", default="configs/exp123_ot_gfam_stable.yaml")
args = parser.parse_args()
cfg = load_yaml(args.config)
data = load_yaml("configs/data.yaml")
val = load_yaml("configs/val_v2.yaml")
tcfg = cfg["train"]
assert val["holdout_fold"] == 4
external_source = tcfg.get("source")
if external_source:
    assert external_source in {"veriwild_test", "vehicleid"}
else:
    assert cfg["eval_fold"] == 0 and tcfg["folds"] == [1, 2, 3]
assert not (Path(data["checkpoint_dir"]) / checkpoint_name(cfg, 0)).exists()

table, crops, classes = training_table(cfg, data, val)
if external_source:
    expected = {"veriwild_test": (138517, 10000), "vehicleid": (221567, 26328)}[external_source]
    assert (len(table), len(classes)) == expected
else:
    assert len(table) == 5725 and len(classes) == 924
init = cfg["model"].get("init_checkpoint")
model, dim, mean, std = build_paper_model(cfg["model"], len(classes), data["external_weights_dir"],
                                          load_pretrained=not init)
if init:
    load_paper_init(model, Path(data["checkpoint_dir"]) / init,
                    cfg["model"].get("init_checkpoint_sha256"))
assert dim == 6 * 768
activation = cfg["model"]["gem_input_activation"]
assert all(pool.input_activation == activation and pool.min_p == 1.0 for pool in model.pools)

sampler = PKSampler(table, tcfg["vehicles_per_batch"], tcfg["images_per_vehicle"], seed=0)
sampler.set_epoch(1)
indices = next(iter(sampler))
dataset = CropDataset(table, crops, cfg["model"]["input_hw"], mean, std, tcfg)
batch = [dataset[i] for i in indices]
x = torch.stack([item[0] for item in batch]).cuda()
y = torch.as_tensor([item[1] for item in batch], device="cuda")

model = model.cuda().train()
optimizer = build_paper_optimizer(model, tcfg)
if "lr_groups" in tcfg:
    assert {group["branch"]: group["lr"] for group in optimizer.param_groups} == tcfg["lr_groups"]
else:
    assert len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["lr"] == tcfg["lr"]
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
    "exp_id": cfg["exp_id"],
    "gem_input_activation": activation,
    "init_checkpoint": init,
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
