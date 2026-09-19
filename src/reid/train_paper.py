"""Train the reproduced architectures of reid.paper_models with their papers' recipes (EXP-055…058).

Data, PK sampler, dev-fold evaluation and outputs are those of reid.train; the model (model.arch), the losses and the
optimiser differ. For every supervised embedding i the loss is ce * CE(logits_i) with label smoothing plus a metric
loss on embedding i: train.loss = {ce, supcon: weight, temperature} or {ce, triplet: weight, margin}; the losses are
summed over the embeddings and logged per branch (loss_branches; for GLPE: global, 2 parts, 3 parts). Adam with L2
weight decay. The large-dataset recipe warms up and drops x0.1 at milestones; train.scheduler: cosine selects the
paper's 24-epoch small-dataset cosine schedule without warm-up.
model.init_checkpoint copies everything but the classifiers from a pretraining run.

Usage: PYTHONPATH=src python -m reid.train_paper --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp055_clip_senet_vehicleid.yaml --seed 0
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from reid.data import load_yaml
from reid.embed import embed_boxes
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.models import checkpoint_name
from reid.paper_models import build_paper_model
from reid.train import (CropDataset, PKSampler, _worker_init, batch_hard_triplet, supervised_contrastive,
                        training_table)


def warmup_multistep(warmup_epochs, milestones):
    def factor(epoch):  # epochs completed so far
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / warmup_epochs
        return 0.1 ** sum(epoch >= milestone for milestone in milestones)
    return factor


def build_paper_scheduler(optimizer, train_cfg):
    """The two schedules stated by CLIP-SENet: cosine for small sets, warm-up/multistep for large sets."""
    kind = train_cfg.get("scheduler", "multistep")
    if kind == "multistep":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_multistep(train_cfg["warmup_epochs"], train_cfg["milestones"]))
    if kind == "cosine":
        t_max = train_cfg.get("scheduler_t_max", train_cfg["epochs"])
        if t_max < train_cfg["epochs"]:
            raise ValueError("cosine scheduler_t_max cannot be shorter than the training run")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.0)
    raise ValueError(f"unknown paper scheduler: {kind}")


def build_paper_optimizer(model, train_cfg):
    """Adam from the paper, optionally with separate rates for the pretrained CNN/CLIP towers and new layers."""
    rates = train_cfg.get("lr_groups")
    if not rates:
        return torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    if all(hasattr(model, name) for name in ("cnn", "clip", "clip_proj", "semantic_norm", "fuse", "afem",
                                             "classifiers")):
        modules = {
            "cnn": [model.cnn],
            "clip": [model.clip, model.clip_proj],
            "new": [model.semantic_norm, model.fuse, model.afem, model.classifiers],
        }
    elif all(hasattr(model, name) for name in ("features", "stage_norm", "units", "pools", "classifiers")):
        modules = {
            "backbone": [model.features],
            "new": [model.stage_norm, model.units, model.pools, model.classifiers],
        }
    else:
        raise ValueError("lr_groups are unsupported for this paper model")
    if set(rates) != set(modules):
        raise ValueError(f"lr_groups must contain exactly {sorted(modules)}")
    groups, seen = [], set()
    for branch, branch_modules in modules.items():
        parameters = [p for module in branch_modules for p in module.parameters() if p.requires_grad]
        if any(id(p) in seen for p in parameters):
            raise ValueError(f"duplicate parameter in paper optimiser group {branch}")
        seen.update(map(id, parameters))
        groups.append({"params": parameters, "lr": float(rates[branch]), "branch": branch})
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected:
        raise ValueError(f"paper optimiser groups miss {len(expected - seen)} trainable parameters")
    return torch.optim.Adam(groups, weight_decay=train_cfg["weight_decay"])


def paper_loss(embeddings, logits, labels, loss_cfg, cross_entropy):
    """(sum over the supervised embeddings, the loss of every branch)."""
    branches = []
    for embedding, logit in zip(embeddings, logits):
        loss = loss_cfg["ce"] * cross_entropy(logit.float(), labels)
        if "supcon" in loss_cfg:
            loss = loss + loss_cfg["supcon"] * supervised_contrastive(embedding.float(), labels,
                                                                      loss_cfg["temperature"])
        if "triplet" in loss_cfg:
            loss = loss + loss_cfg["triplet"] * batch_hard_triplet(embedding.float(), labels, loss_cfg["margin"])
        branches.append(loss)
    return sum(branches), branches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = load_yaml(args.data_config), load_yaml(args.val_config), load_yaml(args.config)
    tcfg, seed, device = cfg["train"], args.seed, "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_df, crops_dir, classes = training_table(cfg, data_cfg, val_cfg)
    query, gallery, no_match = load_fold(Path(val_cfg["out_dir"]), cfg["eval_fold"])
    init = cfg["model"].get("init_checkpoint")
    model, _, mean, std = build_paper_model(cfg["model"], len(classes), data_cfg["external_weights_dir"],
                                            load_pretrained=not init)
    if init:  # the pretraining run's classifiers cover other vehicles
        state = torch.load(Path(data_cfg["checkpoint_dir"]) / init, map_location="cpu", weights_only=True)["state_dict"]
        missing, unexpected = model.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith("classifiers.")}, strict=False)
        if unexpected or any(not k.startswith("classifiers.") for k in missing):
            raise ValueError(f"checkpoint does not match the model: {missing} {unexpected}")
    model = model.to(device)
    optimizer = build_paper_optimizer(model, tcfg)
    scheduler = build_paper_scheduler(optimizer, tcfg)

    P, K = tcfg["vehicles_per_batch"], tcfg["images_per_vehicle"]
    sampler = PKSampler(train_df, P, K, seed)
    dataset = CropDataset(train_df, crops_dir, cfg["model"]["input_hw"], mean, std, tcfg)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=tcfg["num_workers"], pin_memory=True,
                        persistent_workers=True, worker_init_fn=_worker_init,
                        generator=torch.Generator().manual_seed(seed))
    cross_entropy = torch.nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    images_dir = Path(data_cfg["dataset_dir"]) / "images"

    def embed(df):
        model.eval()
        return embed_boxes(model, df, images_dir, cfg["model"]["input_hw"], mean, std, data_cfg["crop_max_side"],
                           device=device, **cfg["embed"])

    out = Path("artifacts") / cfg["exp_id"] / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("w")
    print(f"{len(classes)} vehicles, {len(train_df)} images, {len(sampler)} batches/epoch, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)

    start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals, branch_totals, steps, epoch_start = np.zeros(2), 0.0, 0, time.time()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings, logits = model(x)
            loss, branches = paper_loss(embeddings, logits, y, tcfg["loss"], cross_entropy)
            loss_value = loss.item()
            if not math.isfinite(loss_value):  # stop at once rather than train on NaN for hours (EXP-061)
                raise FloatingPointError(f"loss {loss_value} at epoch {epoch}, step {steps}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += [loss_value, (logits[0].argmax(1) == y).float().mean().item()]
            branch_totals = branch_totals + torch.stack(branches).detach()
            steps += 1
        record = {"epoch": epoch, "loss": totals[0] / steps, "train_acc": totals[1] / steps,
                  "loss_branches": [round(v / steps, 4) for v in branch_totals.tolist()],
                  "lr": optimizer.param_groups[0]["lr"], "img_per_s": steps * P * K / (time.time() - epoch_start),
                  "elapsed_min": (time.time() - start) / 60}
        if tcfg.get("lr_groups"):
            record["lr_groups"] = {group["branch"]: group["lr"] for group in optimizer.param_groups}
        if hasattr(model, "pools"):
            record["gem_p"] = [round(pool.p.item(), 3) for pool in model.pools]
        scheduler.step()
        if epoch % tcfg["eval_every"] == 0 or epoch == tcfg["epochs"]:
            result = evaluate_fold(embed, query, gallery, no_match)
            record["dev"] = {**result["retrieval"], "refusal_F1": result["refusal"]["F1"],
                             "refusal_TNR": result["refusal"]["TNR"]}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    header = {"exp_id": cfg["exp_id"], "seed": seed, "protocol": val_cfg["version"], "fold": cfg["eval_fold"],
              "n_query": len(query), "n_gallery": len(gallery), "train_minutes": (time.time() - start) / 60}
    # weights stay on the pod; the random file name is recorded here, in the card and in the registry
    checkpoints = Path(data_cfg["checkpoint_dir"])
    checkpoints.mkdir(parents=True, exist_ok=True)
    name = checkpoint_name(cfg, seed)
    if (checkpoints / name).exists():
        raise FileExistsError(checkpoints / name)
    torch.save({"exp_id": cfg["exp_id"], "seed": seed, "config": cfg, "state_dict": model.state_dict()},
               checkpoints / name)
    header["checkpoint"] = name
    header["checkpoint_mb"] = round((checkpoints / name).stat().st_size / 1e6, 1)
    save_fold_results(out, result, query, header, emb_dtype=np.float16)


if __name__ == "__main__":
    main()
