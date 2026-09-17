"""Train a ReID model (ID + batch-hard triplet loss, BNNeck, PK sampler) on protocol folds, evaluate on the dev fold.

train.source switches training to an external set or a list of them (pretraining); model.init_checkpoint starts from a pretrained backbone.

One seed per call. Writes to artifacts/<exp_id>/seed<k>/: train_log.jsonl (losses, lr, speed, periodic dev metrics),
metrics.json, threshold.json, refusal_curve.csv, per_query_ap.csv, query_emb.npy and gallery_emb.npy (float16).

Usage: PYTHONPATH=src python -m reid.train --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp001_clip_vitb16_bot.yaml --seed 0
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import v2

from reid.data import load_yaml, to_model_input
from reid.embed import embed_boxes
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.external import source_names
from reid.models import ReIDModel, MarginClassifier, build_backbone, load_backbone, checkpoint_name
from reid.optim import optimizer_groups


class CropDataset(Dataset):
    """Training crops with flip, pad-and-crop and random erasing; returns (image, class index)."""

    def __init__(self, df, crops_dir, input_hw, mean, std, cfg):
        dirs = df.crops_dir if "crops_dir" in df else [crops_dir] * len(df)  # external sets carry a dir per row
        self.paths = [str(Path(d) / f"{image_id}.jpg") for d, image_id in zip(dirs, df.image_id)]
        self.labels = df.label.to_numpy()
        self.input_hw = tuple(input_hw)
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.flip_p, self.pad = cfg["flip_p"], cfg["pad"]
        self.erase = v2.RandomErasing(p=cfg["erasing_p"], value=0)  # 0 = mean colour after normalisation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = to_model_input(cv2.imread(self.paths[i]), self.input_hw, self.mean, self.std)
        if random.random() < self.flip_p:
            x = x.flip(-1)
        if self.pad:
            h, w = self.input_hw
            x = F.pad(x, (self.pad,) * 4)
            top, left = random.randint(0, 2 * self.pad), random.randint(0, 2 * self.pad)
            x = x[:, top: top + h, left: left + w]
        return self.erase(x), self.labels[i]


class PKSampler(Sampler):
    """Batches of P vehicles x K images; every training vehicle appears once per epoch.

    A vehicle's frames are taken round-robin over its cameras, so a batch holds one frame per (vehicle, camera) burst
    before repeating a burst, and positives come from other cameras. Needs df with a fresh 0..n-1 index."""

    def __init__(self, df, P, K, seed):
        self.P, self.K, self.seed, self.epoch = P, K, seed, 0
        self.views = {vehicle: [frames.index.to_numpy() for _, frames in rows.groupby("camera_id")]
                      for vehicle, rows in df.groupby("vehicle_id")}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.views) // self.P

    def _pick(self, views, rng):
        queues = [list(rng.permutation(frames)) for frames in views]
        rng.shuffle(queues)
        picked = []
        while len(picked) < self.K and any(queues):
            for queue in queues:
                if queue and len(picked) < self.K:
                    picked.append(queue.pop())
        while len(picked) < self.K:  # fewer frames than K: repeat some
            picked.append(rng.choice(picked))
        return picked

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        vehicles = rng.permutation(list(self.views))
        for b in range(len(self)):
            yield [int(i) for vehicle in vehicles[b * self.P: (b + 1) * self.P] for i in self._pick(self.views[vehicle], rng)]


def batch_hard_triplet(feat, labels, margin):
    """Mean over anchors of relu(farthest positive - closest negative + margin), Euclidean distance."""
    sq = (feat ** 2).sum(1)
    dist = (sq[:, None] + sq[None, :] - 2 * feat @ feat.T).clamp(min=1e-12).sqrt()
    same = labels[:, None] == labels[None, :]
    hardest_pos = dist.masked_fill(~same, 0).max(1).values
    hardest_neg = dist.masked_fill(same, float("inf")).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def warmup_cosine(total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def _worker_init(_):
    cv2.setNumThreads(1)


@torch.no_grad()
def imprint_classifier(model, loader, device):
    """Initialise ArcFace directions from train-only mean normalised embeddings; leave BN buffers unchanged."""
    if not isinstance(model.classifier, MarginClassifier) or model.classifier.kind != "arcface":
        raise ValueError("imprinting requires an ArcFace head")
    was_training = model.training
    model.eval()
    sums = torch.zeros_like(model.classifier.weight, device=device)
    counts = torch.zeros(sums.shape[0], device=device, dtype=torch.long)
    try:
        for x, labels in loader:
            labels = labels.to(device)
            embeddings = F.normalize(model(x.to(device)).float(), dim=1)
            sums.index_add_(0, labels, embeddings)
            counts += torch.bincount(labels, minlength=len(counts))
        if (counts == 0).any() or (sums.norm(dim=1) == 0).any():
            raise ValueError("every training class must have a nonzero imprinting centre")
        model.classifier.weight.copy_(F.normalize(sums / counts[:, None], dim=1))
        return {"classes": len(counts), "images": int(counts.sum()),
                "min_images_per_class": int(counts.min()), "max_images_per_class": int(counts.max())}
    finally:
        model.train(was_training)


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

    protocol_dir = Path(val_cfg["out_dir"])
    if tcfg.get("source"):  # pretraining on external sets prepared by reid.external; our folds are not used
        tables = [pd.read_csv(Path(data_cfg["external"][name]["dir"]) / "train.csv")
                  .assign(crops_dir=str(Path(data_cfg["external"][name]["dir"]) / "images"))
                  for name in source_names(tcfg)]
        train_df, crops_dir = pd.concat(tables, ignore_index=True), None
    else:
        assert cfg["eval_fold"] not in tcfg["folds"] and val_cfg["holdout_fold"] not in tcfg["folds"]
        folds = pd.read_csv(protocol_dir / "folds.csv")
        train_df, crops_dir = folds[folds.fold.isin(tcfg["folds"])].reset_index(drop=True), data_cfg["crops_dir"]
    classes = {vehicle: k for k, vehicle in enumerate(sorted(train_df.vehicle_id.unique()))}
    train_df["label"] = train_df.vehicle_id.map(classes)
    query, gallery, no_match = load_fold(protocol_dir, cfg["eval_fold"])

    backbone, feat_dim, mean, std = build_backbone(cfg["model"], data_cfg.get("external_weights_dir"),
                                                load_pretrained=not bool(cfg["model"].get("init_checkpoint")))
    model = ReIDModel(backbone, feat_dim, len(classes), tcfg.get("id_head"))
    if cfg["model"].get("init_checkpoint"):  # backbone from a pretraining run; BNNeck and classifier start fresh
        load_backbone(model, Path(data_cfg["checkpoint_dir"]) / cfg["model"]["init_checkpoint"],
                      interpolate_position=cfg["model"].get("interpolate_positional_embedding", False))
    model = model.to(device)
    groups = optimizer_groups(model, tcfg)
    optimizer = torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])

    P, K = tcfg["vehicles_per_batch"], tcfg["images_per_vehicle"]
    sampler = PKSampler(train_df, P, K, seed)
    dataset = CropDataset(train_df, crops_dir, cfg["model"]["input_hw"], mean, std, tcfg)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=tcfg["num_workers"], pin_memory=True,
                        persistent_workers=True, worker_init_fn=_worker_init,
                        generator=torch.Generator().manual_seed(seed))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(tcfg["epochs"] * len(sampler), tcfg["warmup_epochs"] * len(sampler)))
    id_loss = torch.nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])

    images_dir = Path(data_cfg["dataset_dir"]) / "images"

    def embed(df):
        model.eval()
        return embed_boxes(model, df, images_dir, cfg["model"]["input_hw"], mean, std, data_cfg["crop_max_side"],
                           device=device, **cfg["embed"])

    out = Path("artifacts") / cfg["exp_id"] / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "optimizer_groups.json").write_text(json.dumps([
        {k: v for k, v in group.items() if k != "params"} | {"n_parameters": sum(p.numel() for p in group["params"])}
        for group in groups], indent=2))
    imprinting = None
    if tcfg.get("id_head", {}).get("imprinting"):
        initial_dataset = CropDataset(train_df, crops_dir, cfg["model"]["input_hw"], mean, std,
                                      {**tcfg, "flip_p": 0, "pad": 0, "erasing_p": 0})
        initial_loader = DataLoader(initial_dataset, batch_size=tcfg["id_head"]["imprint_batch_size"],
                                    num_workers=tcfg["num_workers"], worker_init_fn=_worker_init,
                                    generator=torch.Generator().manual_seed(seed))
        imprinting = imprint_classifier(model, initial_loader, device)
        print(json.dumps({"imprinting": imprinting}), flush=True)
    log = (out / "train_log.jsonl").open("w")
    print(f"{len(classes)} vehicles, {len(train_df)} images, {len(sampler)} batches/epoch", flush=True)

    start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals, steps, epoch_start = np.zeros(3), 0, time.time()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feat, logits = model(x, y)
            loss_id = id_loss(logits.float(), y)
            loss_triplet = batch_hard_triplet(feat.float(), y, tcfg["triplet_margin"])
            optimizer.zero_grad(set_to_none=True)
            (loss_id + loss_triplet).backward()
            optimizer.step()
            scheduler.step()
            totals += [loss_id.item(), loss_triplet.item(), (logits.argmax(1) == y).float().mean().item()]
            steps += 1

        record = {
            "epoch": epoch,
            "loss_id": totals[0] / steps,
            "loss_triplet": totals[1] / steps,
            "train_acc": totals[2] / steps,
            "lr_backbone": max(g["lr"] for g in optimizer.param_groups[:-1]),
            "img_per_s": steps * P * K / (time.time() - epoch_start),
            "elapsed_min": (time.time() - start) / 60,
        }
        if tcfg.get("layer_decay"):
            record["lr_backbone_min"] = min(g["lr"] for g in optimizer.param_groups[:-1])
        if epoch % tcfg["eval_every"] == 0 or epoch == tcfg["epochs"]:
            result = evaluate_fold(embed, query, gallery, no_match)
            record["dev"] = {**result["retrieval"], "refusal_F1": result["refusal"]["F1"], "refusal_TNR": result["refusal"]["TNR"]}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    header = {"exp_id": cfg["exp_id"], "seed": seed, "protocol": val_cfg["version"], "fold": cfg["eval_fold"],
              "n_query": len(query), "n_gallery": len(gallery), "train_minutes": (time.time() - start) / 60}
    if imprinting is not None:
        header["imprinting"] = imprinting
    if tcfg.get("save_checkpoint", True):
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
