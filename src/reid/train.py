"""Train a ReID model (ID + batch-hard triplet loss, BNNeck, PK sampler) on protocol folds, evaluate on the dev fold.

train.metric_loss: {type: supcon, temperature: t} replaces the triplet loss with supervised contrastive loss;
{type: supcon+triplet, temperature: t} adds both (triplet with train.triplet_margin).

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
from reid.external import load_table, source_names
from reid.models import ReIDModel, MarginClassifier, build_backbone, load_backbone, checkpoint_name
from reid.optim import optimizer_groups
from reid.afem_models import KINDS as AFEM_KINDS, build_extended_model


class CropDataset(Dataset):
    """Training crops with flip, pad-and-crop, random erasing and optional colour jitter; returns (image, class index).

    cfg.color_jitter: {p, brightness, contrast, saturation, hue} - torchvision ColorJitter with probability p."""

    def __init__(self, df, crops_dir, input_hw, mean, std, cfg):
        dirs = df.crops_dir if "crops_dir" in df else [crops_dir] * len(df)  # external sets carry a dir per row
        self.paths = [str(Path(d) / f"{image_id}.jpg") for d, image_id in zip(dirs, df.image_id)]
        self.labels = df.label.to_numpy()
        self.input_hw = tuple(input_hw)
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.flip_p, self.pad = cfg["flip_p"], cfg["pad"]
        erase = {"p": cfg["erasing_p"], "value": 0}  # 0 = mean colour after normalisation
        if "erasing_scale" in cfg:
            erase["scale"] = tuple(cfg["erasing_scale"])
        if "erasing_ratio" in cfg:
            erase["ratio"] = tuple(cfg["erasing_ratio"])
        self.erase = v2.RandomErasing(**erase)
        jitter = dict(cfg.get("color_jitter") or {})
        p = jitter.pop("p", 0.0)
        self.jitter = v2.RandomApply([v2.ColorJitter(**jitter)], p=p) if jitter else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = cv2.imread(self.paths[i])
        if self.jitter is not None:  # on the RGB uint8 image, before normalisation
            rgb = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).permute(2, 0, 1)
            image = np.ascontiguousarray(self.jitter(rgb).permute(1, 2, 0).numpy()[:, :, ::-1])
        x = to_model_input(image, self.input_hw, self.mean, self.std)
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
    before repeating a burst, and positives come from other cameras. Needs df with a fresh 0..n-1 index.

    hard_share > 0 with a neighbour map (set_neighbours) grows every batch around one anchor vehicle from the vehicles
    the model currently finds most similar to it, so the hardest negative of a batch is a real lookalike rather than
    whichever vehicle happened to be drawn. The rest of the batch stays random, and every vehicle still appears at
    most once per epoch."""

    def __init__(self, df, P, K, seed, hard_share=0.0):
        self.P, self.K, self.seed, self.epoch = P, K, seed, 0
        self.hard_share, self.neighbours = hard_share, None
        self.views = {vehicle: [frames.index.to_numpy() for _, frames in rows.groupby("camera_id")]
                      for vehicle, rows in df.groupby("vehicle_id")}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_neighbours(self, neighbours):
        """{vehicle: vehicles ranked by similarity}; None goes back to purely random batches."""
        self.neighbours = neighbours

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

    def _vehicle_batches(self, rng):
        """The P vehicles of every batch: a shuffled partition, or one grown around an anchor from its lookalikes."""
        order = [v for v in rng.permutation(list(self.views))]
        if not self.neighbours or self.hard_share <= 0:
            return [order[b * self.P: (b + 1) * self.P] for b in range(len(self))]
        n_hard = max(1, round(self.hard_share * self.P))
        remaining = dict.fromkeys(order)  # an ordered set: the shuffled order is the random part of every batch
        batches = []
        for _ in range(len(self)):
            anchor = next(iter(remaining))
            del remaining[anchor]
            batch = [anchor]
            for neighbour in self.neighbours.get(anchor, ()):
                if len(batch) >= n_hard:
                    break
                if neighbour in remaining:
                    del remaining[neighbour]
                    batch.append(neighbour)
            while len(batch) < self.P and remaining:
                batch.append(next(iter(remaining)))
                del remaining[batch[-1]]
            batches.append(batch)
        return batches

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        for batch in self._vehicle_batches(rng):
            yield [int(i) for vehicle in batch for i in self._pick(self.views[vehicle], rng)]


def batch_hard_triplet(feat, labels, margin):
    """Mean over anchors of relu(farthest positive - closest negative + margin), Euclidean distance."""
    sq = (feat ** 2).sum(1)
    dist = (sq[:, None] + sq[None, :] - 2 * feat @ feat.T).clamp(min=1e-12).sqrt()
    same = labels[:, None] == labels[None, :]
    hardest_pos = dist.masked_fill(~same, 0).max(1).values
    hardest_neg = dist.masked_fill(same, float("inf")).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def supervised_contrastive(feat, labels, temperature):
    """SupCon (Khosla et al., 2020): minus the mean over anchors of the mean log-probability of their positives among
    all other images of the batch, cosine similarity / temperature on L2-normalised features. Anchors without a
    positive are skipped (the PK sampler never produces them)."""
    z = F.normalize(feat, dim=1)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = (z @ z.T / temperature).masked_fill(eye, float("-inf"))
    log_prob = logits - logits.logsumexp(1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    per_anchor = log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1).clamp(min=1)
    return -per_anchor[positives.any(1)].mean()


def similarity_distillation(feat, teacher, temperature):
    """KL(teacher || student) between the rows of the two batch similarity matrices; the teacher is a constant.

    The teacher's coordinates are never copied, only its judgement of which images are alike: the softmax over a row
    ignores a shift of all similarities, so the student keeps its own basis. The self-similarity of 1 would take
    almost the whole softmax mass, so each row keeps only its n - 1 other images - dropped, not masked with -inf,
    because a zero target against an infinite logit makes F.kl_div return NaN."""
    student = F.normalize(feat, dim=1)
    n = len(student)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=student.device)
    student_sim = (student @ student.T / temperature)[off_diagonal].view(n, n - 1)
    teacher_sim = (teacher @ teacher.T / temperature)[off_diagonal].view(n, n - 1)
    return F.kl_div(student_sim.log_softmax(1), teacher_sim.softmax(1), reduction="batchmean")


class GradientReversal(torch.autograd.Function):
    """Identity forward, gradient multiplied by -lambda backward (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


@torch.no_grad()
def lookalike_vehicles(classifier, vehicles, pool):
    """{vehicle: its `pool` most similar vehicles} from the ArcFace class centres.

    The centres are what the model itself uses to tell vehicles apart, so centres that sit close together are exactly
    the vehicles it confuses - and they come free, without another pass over the crops."""
    centres = F.normalize(classifier.weight.detach().float(), dim=1)
    similarity = centres @ centres.T
    similarity.fill_diagonal_(float("-inf"))
    nearest = similarity.topk(min(pool, len(centres) - 1), dim=1).indices.cpu().numpy()
    return {vehicles[i]: [vehicles[j] for j in row] for i, row in enumerate(nearest)}


def camera_codes(df):
    """Camera ids as int64 class indices: pandas category codes are int8, which cross_entropy cannot take."""
    return df.camera_id.astype("category").cat.codes.to_numpy().astype(np.int64)


def camera_lambda(cfg, epoch):
    """The adversarial weight, ramped linearly from 0 over the first ramp_epochs so the head is useful before it
    starts pushing back on the backbone."""
    return cfg["lambda"] * min(1.0, epoch / max(cfg.get("ramp_epochs", 1), 1))


def external_table(data_cfg, name):
    """An external set as a table for CropDataset; its label column carries the row index, which is what the
    distillation needs to look up teacher embeddings. The labels themselves are never used: this stream is unlabelled."""
    spec = data_cfg["external"][name]
    table = load_table(spec["dir"], spec.get("table", "train.csv")).reset_index(drop=True)
    table["crops_dir"] = str(Path(spec["dir"]) / "images")
    table["label"] = np.arange(len(table))
    return table


def cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def teacher_embeddings(distill, data_cfg, tcfg, train_df, crops_dir, device):
    """Mean of the L2-normalised embeddings of the distillation teachers over the training crops, in train_df order.

    Every teacher carries its own config, so members with different towers and input sizes can be mixed; they are
    loaded one at a time and freed, and the augmentations are off - the teacher sees the plain crop."""
    from reid.inference import load_trained_encoder  # local: reid.inference imports the model builders
    parts = []
    for spec in distill["teachers"]:
        model_cfg = load_yaml(spec["config"])["model"]
        encoder, _, mean, std = load_trained_encoder(model_cfg, Path(data_cfg["checkpoint_dir"]) / spec["checkpoint"],
                                                     data_cfg.get("external_weights_dir"), spec.get("sha256"))
        encoder = encoder.eval().to(device)
        loader = DataLoader(CropDataset(train_df, crops_dir, model_cfg["input_hw"], mean, std,
                                        {**tcfg, "flip_p": 0, "pad": 0, "erasing_p": 0}),
                            batch_size=tcfg["id_head"]["imprint_batch_size"], num_workers=tcfg["num_workers"],
                            worker_init_fn=_worker_init)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            parts.append(torch.cat([F.normalize(encoder(x.to(device)).float(), dim=1) for x, _ in loader]))
        del encoder
        torch.cuda.empty_cache()
    return F.normalize(torch.stack(parts).mean(0), dim=1)


def warmup_cosine(total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def _worker_init(_):
    cv2.setNumThreads(1)


def enable_gradient_checkpointing(model, enabled):
    """Enable activation checkpointing explicitly and fail if a requested backbone cannot provide it."""
    if not enabled:
        return False
    setter = getattr(model.backbone, "set_grad_checkpointing", None)
    if setter is None:
        raise ValueError("gradient_checkpointing requested but the backbone does not support it")
    setter(True)
    return True


def disable_fused_attention_blocks(model, indices):
    """Use timm's explicit attention math in selected blocks while leaving their weights and graph unchanged.

    Some large ViTs produce non-finite gradients in the fused BF16 SDPA backward when activation checkpointing
    recomputes the first block.  The unfused path implements the same scaled QK softmax and AV product, but avoids
    that kernel.  Fail loudly if a config requests it for an incompatible backbone.
    """
    indices = list(indices or [])
    if not indices:
        return []
    blocks = getattr(getattr(getattr(model, "backbone", None), "trunk", None), "blocks", None)
    if blocks is None:
        raise ValueError("unfused_attention_blocks requires a timm-backed OpenCLIP transformer")
    if len(set(indices)) != len(indices) or any(not isinstance(index, int) for index in indices):
        raise ValueError("unfused_attention_blocks must contain unique integer block indices")
    for index in indices:
        if not 0 <= index < len(blocks):
            raise ValueError(f"unfused attention block {index} is outside 0..{len(blocks) - 1}")
        attention = getattr(blocks[index], "attn", None)
        if attention is None or not hasattr(attention, "fused_attn"):
            raise ValueError(f"block {index} does not expose timm fused_attn")
        attention.fused_attn = False
    return indices


def select_best_dev(previous, model, result, epoch):
    """Keep a CPU snapshot when dev mAP@10 improves, without writing one giant checkpoint per epoch.

    The returned result and weights come from the same evaluation point.  At most the current and previous best
    snapshots coexist while an improvement is copied; this is preferable to accumulating checkpoints on disk.
    """
    score = float(result["retrieval"]["mAP10"])
    if previous is not None and score <= previous["mAP10"]:
        return previous, False
    state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {"epoch": int(epoch), "mAP10": score, "state_dict": state_dict, "result": result}, True


def training_table(cfg, data_cfg, val_cfg):
    """(training rows with class labels and a 0..n-1 index, crops dir or None when rows carry their own, classes)."""
    tcfg = cfg["train"]
    if tcfg.get("source"):  # pretraining on external sets prepared by reid.external; our folds are not used
        tables = []
        for name in source_names(tcfg):
            spec = data_cfg["external"][name]
            tables.append(load_table(spec["dir"], spec.get("table", "train.csv"))
                          .assign(crops_dir=str(Path(spec["dir"]) / "images")))
        train_df, crops_dir = pd.concat(tables, ignore_index=True), None
    else:
        assert cfg["eval_fold"] not in tcfg["folds"] and val_cfg["holdout_fold"] not in tcfg["folds"]
        folds = pd.read_csv(Path(val_cfg["out_dir"]) / "folds.csv")
        train_df, crops_dir = folds[folds.fold.isin(tcfg["folds"])].reset_index(drop=True), data_cfg["crops_dir"]
        if tcfg.get("pseudo_labels"):  # extra vehicles clustered out of unlabelled images (reid.pseudo_labels)
            pseudo = pd.read_csv(tcfg["pseudo_labels"])
            train_df = pd.concat([train_df, pseudo[train_df.columns.intersection(pseudo.columns)]],
                                 ignore_index=True)
            train_df["vehicle_id"] = train_df.vehicle_id.astype(str)  # our ids are ints and pseudo ones are "pl_*"
    classes = {vehicle: k for k, vehicle in enumerate(sorted(train_df.vehicle_id.unique()))}
    train_df["label"] = train_df.vehicle_id.map(classes)
    return train_df, crops_dir, classes


def class_centres(embeddings, k, iterations=10):
    """k unit centres of one class: farthest-point init, then Lloyd iterations on cosine similarity.

    Sub-centre ArcFace needs its centres to differ: the max over identical centres keeps them tied forever. A class
    with fewer images than centres repeats one; those duplicates stay tied, which is the single-centre behaviour."""
    chosen = [int((embeddings @ F.normalize(embeddings.mean(0), dim=0)).argmin())]  # the least typical image first
    while len(chosen) < min(k, len(embeddings)):
        chosen.append(int((embeddings @ embeddings[chosen].T).max(1).values.argmin()))
    centres = embeddings[chosen]
    for _ in range(iterations):
        nearest = (embeddings @ centres.T).argmax(1)
        centres = torch.stack([F.normalize(embeddings[nearest == c].mean(0), dim=0) if (nearest == c).any()
                               else centres[c] for c in range(len(centres))])
    return centres[torch.arange(k) % len(centres)]  # fewer images than centres: the centres repeat cyclically


@torch.no_grad()
def imprint_sub_centers(model, loader, device):
    """Sub-centre ArcFace imprinting: k centres per class from a small k-means over that class's train embeddings."""
    head = model.classifier
    was_training = model.training
    model.eval()
    try:
        embeddings, labels = [], []
        for x, y in loader:
            embeddings.append(F.normalize(model(x.to(device)).float(), dim=1))
            labels.append(y.to(device))
        embeddings, labels = torch.cat(embeddings), torch.cat(labels)
        counts = torch.bincount(labels, minlength=head.n_classes)
        if (counts == 0).any():
            raise ValueError("every training class must have a nonzero imprinting centre")
        centres = torch.stack([class_centres(embeddings[labels == c], head.sub_centers) for c in range(head.n_classes)])
        head.weight.copy_(centres.reshape_as(head.weight))
        return {"classes": head.n_classes, "images": len(labels), "sub_centers": head.sub_centers,
                "min_images_per_class": int(counts.min()), "max_images_per_class": int(counts.max())}
    finally:
        model.train(was_training)


@torch.no_grad()
def imprint_classifier(model, loader, device):
    """Initialise ArcFace directions from train-only mean normalised embeddings; leave BN buffers unchanged."""
    if not isinstance(model.classifier, MarginClassifier) or model.classifier.kind != "arcface":
        raise ValueError("imprinting requires an ArcFace head")
    was_training = model.training
    model.eval()
    classifiers = [model.classifier]
    if hasattr(model, "auxiliary_classifier"):
        classifiers.append(model.auxiliary_classifier)
    sums = [torch.zeros_like(head.weight, device=device) for head in classifiers]
    counts = torch.zeros(sums[0].shape[0], device=device, dtype=torch.long)
    try:
        for x, labels in loader:
            labels = labels.to(device)
            x = x.to(device)
            features = model.imprinting_features(x) if hasattr(model, "imprinting_features") else [model(x)]
            for total, embeddings in zip(sums, features):
                total.index_add_(0, labels, F.normalize(embeddings.float(), dim=1))
            counts += torch.bincount(labels, minlength=len(counts))
        if (counts == 0).any() or any((total.norm(dim=1) == 0).any() for total in sums):
            raise ValueError("every training class must have a nonzero imprinting centre")
        for head, total in zip(classifiers, sums):
            head.weight.copy_(F.normalize(total / counts[:, None], dim=1))
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
    train_df, crops_dir, classes = training_table(cfg, data_cfg, val_cfg)
    query, gallery, no_match = load_fold(protocol_dir, cfg["eval_fold"])

    if cfg["model"]["backbone"] in AFEM_KINDS:
        model, feat_dim, mean, std = build_extended_model(cfg["model"], len(classes),
                                                        data_cfg.get("external_weights_dir"), head=tcfg.get("id_head"))
    else:
        backbone, feat_dim, mean, std = build_backbone(cfg["model"], data_cfg.get("external_weights_dir"),
                                                    load_pretrained=not bool(cfg["model"].get("init_checkpoint")))
        model = ReIDModel(backbone, feat_dim, len(classes), tcfg.get("id_head"))
    if cfg["model"].get("init_checkpoint"):  # backbone from a pretraining run; BNNeck and classifier start fresh
        load_backbone(model, Path(data_cfg["checkpoint_dir"]) / cfg["model"]["init_checkpoint"],
                      interpolate_position=cfg["model"].get("interpolate_positional_embedding", False))
    gradient_checkpointing = enable_gradient_checkpointing(model, tcfg.get("gradient_checkpointing", False))
    unfused_attention_blocks = disable_fused_attention_blocks(model, tcfg.get("unfused_attention_blocks"))
    model = model.to(device)
    groups = optimizer_groups(model, tcfg)
    camera_cfg, camera_head, camera_targets = tcfg.get("camera_adversarial"), None, None
    if camera_cfg:  # camera_id is a target of an auxiliary loss here, never an input: inference is unchanged
        codes = camera_codes(train_df)
        camera_targets = torch.as_tensor(codes, device=device)
        camera_head = torch.nn.Linear(feat_dim, int(codes.max()) + 1).to(device)
        groups.append({"params": list(camera_head.parameters()), "lr": tcfg["lr_head"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])

    P, K = tcfg["vehicles_per_batch"], tcfg["images_per_vehicle"]
    hard_cfg = tcfg.get("hard_identity")
    sampler = PKSampler(train_df, P, K, seed, hard_share=hard_cfg["share"] if hard_cfg else 0.0)
    dataset = CropDataset(train_df, crops_dir, cfg["model"]["input_hw"], mean, std, tcfg)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=tcfg["num_workers"], pin_memory=True,
                        persistent_workers=True, worker_init_fn=_worker_init,
                        generator=torch.Generator().manual_seed(seed))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(tcfg["epochs"] * len(sampler), tcfg["warmup_epochs"] * len(sampler)))
    id_loss = torch.nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    metric_loss = tcfg.get("metric_loss", {"type": "triplet"})
    if metric_loss["type"] not in ("triplet", "supcon", "supcon+triplet"):
        raise ValueError(f"unknown metric loss {metric_loss['type']}")

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
        imprint = imprint_sub_centers if tcfg["id_head"].get("sub_centers", 1) > 1 else imprint_classifier
        imprinting = imprint(model, initial_loader, device)
        print(json.dumps({"imprinting": imprinting}), flush=True)
    distill, teacher, extra_loader, extra_teacher = tcfg.get("distill"), None, None, None
    if distill:  # embeddings of the teachers over the plain training crops, computed once and then frozen
        teacher = teacher_embeddings(distill, data_cfg, tcfg, train_df, crops_dir, device)
        header_distill = {"teachers": [spec["checkpoint"] for spec in distill["teachers"]],
                          "weight": distill["weight"], "temperature": distill["temperature"],
                          "images": len(teacher), "dim": teacher.shape[1]}
        if distill.get("extra"):  # unlabelled images distilled next to our batches; the seeds share the cached teacher
            extra = distill["extra"]
            extra_df = external_table(data_cfg, extra["source"])
            cache = Path("artifacts") / cfg["exp_id"] / f"teacher_{extra['source']}.npy"
            if cache.exists():
                extra_teacher = torch.from_numpy(np.load(cache)).to(device)
            else:
                extra_teacher = teacher_embeddings(distill, data_cfg, tcfg, extra_df, None, device)
                np.save(cache, extra_teacher.cpu().numpy())
            extra_loader = cycle(DataLoader(
                CropDataset(extra_df, None, cfg["model"]["input_hw"], mean, std, tcfg),
                batch_size=extra["batch_size"], shuffle=True, drop_last=True, num_workers=tcfg["num_workers"],
                worker_init_fn=_worker_init, generator=torch.Generator().manual_seed(seed)))
            header_distill["extra"] = {**extra, "images": len(extra_teacher)}
        print(json.dumps({"distill": header_distill}), flush=True)
    log = (out / "train_log.jsonl").open("w")
    print(f"{len(classes)} vehicles, {len(train_df)} images, {len(sampler)} batches/epoch", flush=True)

    start = time.time()
    best_dev = None
    for epoch in range(1, tcfg["epochs"] + 1):
        sampler.set_epoch(epoch)
        if hard_cfg and epoch >= hard_cfg["start_epoch"]:  # batches of lookalikes once the class centres mean something
            sampler.set_neighbours(lookalike_vehicles(model.classifier, sorted(classes), hard_cfg["pool"]))
        model.train()
        totals, steps, epoch_start = np.zeros(3), 0, time.time()
        auxiliary_total = distill_total = extra_total = camera_total = 0.0
        # the sampler is deterministic given (seed, epoch), so iterating it here reproduces the batches the loader
        # is about to yield; distillation needs the row indices to look up its teacher embeddings
        for (x, y), rows in zip(loader, sampler):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(x, y)
                feat, logits = outputs[:2]
            loss_id = id_loss(logits.float(), y)
            if metric_loss["type"] == "supcon":
                loss_metric = supervised_contrastive(feat.float(), y, metric_loss["temperature"])
            elif metric_loss["type"] == "supcon+triplet":
                loss_metric = (supervised_contrastive(feat.float(), y, metric_loss["temperature"])
                               + batch_hard_triplet(feat.float(), y, tcfg["triplet_margin"]))
            else:
                loss_metric = batch_hard_triplet(feat.float(), y, tcfg["triplet_margin"])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_id + loss_metric
            if teacher is not None:
                loss_distill = similarity_distillation(feat.float(), teacher[rows], distill["temperature"])
                distill_total += loss_distill.item()
                loss = loss + distill["weight"] * loss_distill
            if camera_head is not None:  # the head learns the camera, the reversed gradient unlearns it in the backbone
                camera_logits = camera_head(GradientReversal.apply(feat.float(), camera_lambda(camera_cfg, epoch)))
                loss_camera = id_loss(camera_logits, camera_targets[rows])
                camera_total += loss_camera.item()
                loss = loss + loss_camera
            if extra_loader is not None:  # the same loss on unlabelled external images: only the teacher speaks there
                x_extra, rows_extra = next(extra_loader)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feat_extra = model.backbone(x_extra.to(device, non_blocking=True))
                loss_extra = similarity_distillation(feat_extra.float(), extra_teacher[rows_extra.to(device)],
                                                     distill["temperature"])
                extra_total += loss_extra.item()
                loss = loss + distill["extra"].get("weight", distill["weight"]) * loss_extra
            if len(outputs) == 4:
                app_feat, app_logits = outputs[2:]
                if metric_loss["type"] != "supcon":
                    raise ValueError("AFEM/OSNet auxiliary recipe requires EXP-054 SupCon")
                auxiliary = id_loss(app_logits.float(), y) + supervised_contrastive(app_feat.float(), y, metric_loss["temperature"])
                auxiliary_total += auxiliary.item()
                loss = loss + tcfg["auxiliary_weight"] * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {steps}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            totals += [loss_id.item(), loss_metric.item(), (logits.argmax(1) == y).float().mean().item()]
            steps += 1

        record = {
            "epoch": epoch,
            "loss_id": totals[0] / steps,
            f"loss_{metric_loss['type']}": totals[1] / steps,
            "train_acc": totals[2] / steps,
            "lr_backbone": max(g["lr"] for g in optimizer.param_groups[:-1]),
            "img_per_s": steps * P * K / (time.time() - epoch_start),
            "elapsed_min": (time.time() - start) / 60,
        }
        if hasattr(model.backbone, "optimizer_groups"):
            record["lr_groups"] = {g["branch"]: g["lr"] for g in optimizer.param_groups}
            record["lr_backbone"] = max(g["lr"] for g in optimizer.param_groups if g["branch"] in {"siglip", "osnet"})
        if hasattr(model, "auxiliary_classifier"):
            record["loss_auxiliary"] = auxiliary_total / max(steps, 1)
        if teacher is not None:
            record["loss_distill"] = distill_total / max(steps, 1)
            if extra_loader is not None:
                record["loss_distill_extra"] = extra_total / max(steps, 1)
        if camera_head is not None:
            record["loss_camera"] = camera_total / max(steps, 1)
            record["camera_lambda"] = camera_lambda(camera_cfg, epoch)
        afem = getattr(model.backbone, "refinement", None)
        if hasattr(afem, "group_logits"):
            record["afem_residual_scale"] = float(afem.residual_scale.detach())
            record["afem_group_weights"] = afem.group_logits.detach().softmax(0).tolist()
            record["afem_residual_ratio"] = float(afem.last_residual_ratio)
        if hasattr(model.backbone, "last_branch_norms"):
            record["branch_norms_before_l2"] = {k: float(v) for k, v in model.backbone.last_branch_norms.items()}
        appearance = getattr(model.backbone, "appearance", model.backbone)
        pool = getattr(appearance, "pool", None)
        if hasattr(pool, "p"):
            record["gem_p"] = float(pool.p.detach())
        if tcfg.get("layer_decay"):
            record["lr_backbone_min"] = min(g["lr"] for g in optimizer.param_groups[:-1])
        if epoch % tcfg["eval_every"] == 0 or epoch == tcfg["epochs"]:
            result = evaluate_fold(embed, query, gallery, no_match)
            record["dev"] = {**result["retrieval"], "refusal_F1": result["refusal"]["F1"], "refusal_TNR": result["refusal"]["TNR"]}
            if tcfg.get("select_best_dev", False):
                best_dev, improved = select_best_dev(best_dev, model, result, epoch)
                record["selected_best"] = improved
                record["best_epoch"] = best_dev["epoch"]
                record["best_mAP10"] = best_dev["mAP10"]
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    header = {"exp_id": cfg["exp_id"], "seed": seed, "protocol": val_cfg["version"], "fold": cfg["eval_fold"],
              "n_query": len(query), "n_gallery": len(gallery), "train_minutes": (time.time() - start) / 60}
    header["gradient_checkpointing"] = gradient_checkpointing
    if unfused_attention_blocks:
        header["unfused_attention_blocks"] = unfused_attention_blocks
    if imprinting is not None:
        header["imprinting"] = imprinting
    if teacher is not None:
        header["distill"] = header_distill
    checkpoint_state = model.state_dict()
    if tcfg.get("select_best_dev", False):
        if best_dev is None:
            raise RuntimeError("select_best_dev requires at least one dev evaluation")
        result = best_dev["result"]
        checkpoint_state = best_dev["state_dict"]
        header["checkpoint_selection"] = "max_dev_mAP10"
        header["selected_epoch"] = best_dev["epoch"]
        header["selected_mAP10"] = best_dev["mAP10"]
    if tcfg.get("save_checkpoint", True):
        # weights stay on the pod; the random file name is recorded here, in the card and in the registry
        checkpoints = Path(data_cfg["checkpoint_dir"])
        checkpoints.mkdir(parents=True, exist_ok=True)
        name = checkpoint_name(cfg, seed)
        if (checkpoints / name).exists():
            raise FileExistsError(checkpoints / name)
        torch.save({"exp_id": cfg["exp_id"], "seed": seed, "config": cfg, "state_dict": checkpoint_state},
                   checkpoints / name)
        header["checkpoint"] = name
        header["checkpoint_mb"] = round((checkpoints / name).stat().st_size / 1e6, 1)
    save_fold_results(out, result, query, header, emb_dtype=np.float16)


if __name__ == "__main__":
    main()
