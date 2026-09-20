"""Pairwise cross-attention re-ranker over the top candidates of a frozen encoder (EXP-168).

The encoder describes every vehicle on its own and then compares two finished vectors. This module looks at two
vehicles at once: patch tokens of the query and of one candidate are projected to a common width and passed through
blocks of mutual cross-attention, so what the query looks like depends on which candidate it is compared with. The
head scores the pair and the score reorders the encoder's top-N window - the window keeps its similarity values and
only their order changes, so row statistics, the refusal confidence and everything outside the window stay as they
were. The encoder is frozen.

Measured ceiling on fold 0 (EXP-165 embeddings): a perfect reordering of the top-20 would give mAP@10 0.9897 against
0.8995 today, because 98.8% of the positives already sit inside that window - this is an ordering problem.

Usage: PYTHONPATH=src python -m reid.rerank_pairs --data-config configs/data.yaml \
           --val-config configs/val_v2.yaml --config configs/exp168_pair_reranker.yaml --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from reid.data import load_yaml
from reid.embed import pick_device
from reid.evaluate import load_fold
from reid.gallery_transforms import normalise
from reid.inference import load_trained_encoder
from reid.metrics import pick_threshold, refusal_curve, reid_eval, reid_eval_top10
from reid.train import CropDataset, _worker_init


class CrossBlock(nn.Module):
    """One block of mutual attention: each side attends to the other through the same weights, so the block - and
    with it the whole score - is symmetric in the pair."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, a, b):
        qa, qb, ka, kb = self.norm_q(a), self.norm_q(b), self.norm_kv(a), self.norm_kv(b)
        a = a + self.attn(qa, kb, kb, need_weights=False)[0]
        b = b + self.attn(qb, ka, ka, need_weights=False)[0]
        return a + self.ff(self.norm_ff(a)), b + self.ff(self.norm_ff(b))


class PairReranker(nn.Module):
    """Logit of "the same vehicle, seen from another camera" for a pair of patch-token sets."""

    def __init__(self, token_dim, width=256, blocks=2, heads=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(token_dim, width), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([CrossBlock(width, heads) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(),
                                  nn.Linear(width, 1))

    def forward(self, tokens_a, tokens_b):
        a, b = self.proj(tokens_a), self.proj(tokens_b)
        for block in self.blocks:
            a, b = block(a, b)
        u, v = a.mean(1), b.mean(1)
        # symmetric in (u, v): neither the sum nor the absolute difference depends on the order of the pair
        return self.head(torch.cat([u + v, (u - v).abs()], dim=1)).squeeze(1)


def plain_loader(df, crops_dir, input_hw, mean, std, batch_size, num_workers=8):
    frame = df.assign(label=0).reset_index(drop=True)
    plain = {"flip_p": 0.0, "pad": 0, "erasing_p": 0.0}
    return DataLoader(CropDataset(frame, crops_dir, input_hw, mean, std, plain), batch_size=batch_size,
                      num_workers=num_workers, worker_init_fn=_worker_init)


@torch.no_grad()
def token_bank(tower, loader, device):
    """float16 patch tokens (n_rows, n_tokens, dim) of the frozen tower, aligned with the rows of the loader."""
    chunks = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            chunks.append(tower.trunk.forward_features(x.to(device)).half().cpu())
    return torch.cat(chunks)


@torch.no_grad()
def crop_embeddings(encoder, loader, device):
    """L2-normalised embeddings of the frozen encoder, used only to mine hard negatives for training."""
    chunks = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            chunks.append(F.normalize(encoder(x.to(device)).float(), dim=-1).cpu().numpy())
    return np.concatenate(chunks)


def training_groups(df, embeddings, n_negatives, hard_share, rng):
    """(anchor, positive, negatives) per anchor: the positive is the same vehicle from another camera, part of the
    negatives are the nearest other vehicles under the frozen encoder and the rest are drawn at random."""
    vehicle, camera = df.vehicle_id.to_numpy(), df.camera_id.to_numpy()
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    ranked = np.argsort(-similarity, axis=1)
    n_hard = int(round(hard_share * n_negatives))
    groups = []
    for anchor in range(len(df)):
        positives = np.flatnonzero((vehicle == vehicle[anchor]) & (camera != camera[anchor]))
        if not len(positives):  # a vehicle seen from one camera teaches nothing about cross-camera matching
            continue
        others = ranked[anchor][vehicle[ranked[anchor]] != vehicle[anchor]]
        hard = others[:n_hard]
        pool = np.setdiff1d(np.flatnonzero(vehicle != vehicle[anchor]), hard)
        groups.append((anchor, positives, np.concatenate(
            [hard, rng.choice(pool, n_negatives - n_hard, replace=False)])))
    return groups


def group_loss(logits):
    """logits: (n_groups, 1 + n_negatives), the positive first. The listwise term lifts the positive above the
    candidates it competes with; the binary term keeps the logit meaningful on its own, for the blend."""
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    binary = torch.zeros_like(logits)
    binary[:, 0] = 1.0
    return F.cross_entropy(logits, target) + F.binary_cross_entropy_with_logits(logits, binary)


def train(model, tokens, groups, cfg, device, seed, log):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    per_epoch = len(groups) // cfg["groups_per_batch"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"],
                                                    total_steps=max(cfg["epochs"] * per_epoch, 1), pct_start=0.1)
    rng = np.random.default_rng(seed)
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(len(groups))
        total, correct, seen = 0.0, 0, 0
        for step in range(per_epoch):
            chosen = [groups[i] for i in order[step * cfg["groups_per_batch"]: (step + 1) * cfg["groups_per_batch"]]]
            candidates = np.array([[rng.choice(pos), *neg] for _, pos, neg in chosen])
            anchors = np.repeat([a for a, _, _ in chosen], candidates.shape[1])
            a = tokens[anchors].to(device, non_blocking=True).float()
            b = tokens[candidates.ravel()].to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(a, b)
            logits = logits.float().view(len(chosen), -1)
            loss = group_loss(logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
            correct += int((logits.argmax(1) == 0).sum())
            seen += len(chosen)
        record = {"epoch": epoch, "loss": total / max(per_epoch, 1), "positive_on_top": correct / max(seen, 1),
                  "elapsed_min": (time.time() - start) / 60}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)
    return model


@torch.no_grad()
def candidate_scores(model, query_tokens, gallery_tokens, candidates, device, batch_size=128):
    """Re-ranker logit for every (query, candidate) pair of the window."""
    model.eval()
    n_query, window = candidates.shape
    left = np.repeat(np.arange(n_query), window)
    right = candidates.ravel()
    out = torch.empty(len(left))
    for start in range(0, len(left), batch_size):
        stop = min(start + batch_size, len(left))
        a = query_tokens[left[start:stop]].to(device).float()
        b = gallery_tokens[right[start:stop]].to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out[start:stop] = model(a, b).float().cpu()
    return out.view(n_query, window).numpy()


def reorder(similarity, candidates, scores, weight):
    """Permute the similarity values inside the window by (1 - w) * rank_encoder + w * rank_reranker.

    The window keeps its own values and only redistributes them, so the row's statistics - and with them the refusal
    confidence - are exactly what the encoder produced; what changes is which gallery item carries which value."""
    rows = np.arange(len(candidates))[:, None]
    inside = similarity[rows, candidates]
    window = candidates.shape[1]
    rank_encoder = np.argsort(np.argsort(-inside, axis=1), axis=1)      # 0 = best under the encoder
    rank_reranker = np.argsort(np.argsort(-scores, axis=1), axis=1)
    mixed = (1 - weight) * rank_encoder + weight * rank_reranker
    place = np.argsort(np.argsort(mixed, axis=1, kind="stable"), axis=1)  # 0 = gets the largest value
    values = np.sort(inside, axis=1)[:, ::-1]                            # the window's own values, best first
    out = similarity.copy()
    out[rows, candidates] = values[rows, place]
    return out


def evaluate(ranked, reduced, query, gallery, keep):
    """Retrieval on the whole gallery and refusal on the gallery the protocol reduces, both after reordering."""
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    top10, _ = reid_eval_top10(ranked, q_pid, q_cam, g_pid, g_cam)
    full, _ = reid_eval(ranked, q_pid, q_cam, g_pid, g_cam)
    best = pick_threshold(refusal_curve(reduced, q_pid, q_cam, g_pid[keep], g_cam[keep]))
    return {"mAP10": float(top10["mAP10"]), "R1": float(top10["R1"]), "R5": float(top10["R5"]),
            "mAP": float(full["mAP"]), "mINP": float(full["mINP"]), "refusal": float(best["score"]),
            "F1": float(best["F1"]), "TNR": float(best["TNR"]), "threshold": float(best["threshold"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    torch.manual_seed(args.seed)
    device = pick_device()
    out = Path("artifacts") / cfg["exp_id"] / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    encoder, _, mean, std = load_trained_encoder(cfg["model"], Path(data_cfg["checkpoint_dir"]) / cfg["checkpoint"],
                                                 data_cfg.get("external_weights_dir"))
    encoder = encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    tower = encoder[0]  # Sequential(backbone, bnneck): patch tokens come from the backbone's trunk

    folds = pd.read_csv(Path(val_cfg["out_dir"]) / "folds.csv")
    train_df = folds[folds.fold.isin(cfg["train"]["folds"])].reset_index(drop=True)
    query, gallery, no_match = load_fold(Path(val_cfg["out_dir"]), cfg["eval_fold"])
    loaders = {name: plain_loader(frame, data_cfg["crops_dir"], cfg["model"]["input_hw"], mean, std, 64)
               for name, frame in (("train", train_df), ("query", query), ("gallery", gallery))}

    print(json.dumps({"stage": "tokens", "train": len(train_df), "query": len(query), "gallery": len(gallery)}),
          flush=True)
    train_tokens = token_bank(tower, loaders["train"], device)
    query_tokens = token_bank(tower, loaders["query"], device)
    gallery_tokens = token_bank(tower, loaders["gallery"], device)
    mining = crop_embeddings(encoder, loaders["train"], device)

    groups = training_groups(train_df, mining, cfg["train"]["negatives"], cfg["train"]["hard_share"],
                             np.random.default_rng(args.seed))
    print(json.dumps({"stage": "train", "groups": len(groups), "of_rows": len(train_df),
                      "tokens_gb": round(train_tokens.numel() * 2 / 1e9, 2)}), flush=True)

    model = PairReranker(train_tokens.shape[-1], cfg["width"], cfg["blocks"], cfg["heads"]).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    with (out / "train_log.jsonl").open("w") as log:
        train(model, train_tokens, groups, cfg["train"], device, args.seed, log)

    source = Path(cfg["source_dir"])
    query_emb = normalise(np.load(source / "query_emb.npy").astype(np.float32))
    gallery_emb = normalise(np.load(source / "gallery_emb.npy").astype(np.float32))
    similarity = query_emb @ gallery_emb.T
    keep = ~gallery.vehicle_id.isin(no_match).to_numpy()
    reduced_similarity = similarity[:, keep]

    def window_of(matrix, tokens):
        picked = np.argsort(-matrix, axis=1, kind="stable")[:, :cfg["window"]]
        return picked, candidate_scores(model, query_tokens, tokens, picked, device)

    full_window = window_of(similarity, gallery_tokens)
    # the refusal gallery drops whole vehicles, so its window is a different set of candidates and is scored again
    reduced_window = window_of(reduced_similarity, gallery_tokens[torch.as_tensor(keep)])

    results = {}
    for weight in cfg["weights"]:
        ranked = similarity if weight == 0 else reorder(similarity, *full_window, weight)
        reduced = reduced_similarity if weight == 0 else reorder(reduced_similarity, *reduced_window, weight)
        results[f"w{weight}"] = evaluate(ranked, reduced, query, gallery, keep)
        print(json.dumps({"weight": weight, **{k: round(v, 6) for k, v in results[f"w{weight}"].items()}}), flush=True)

    summary = {"exp_id": cfg["exp_id"], "protocol": val_cfg["version"], "fold": cfg["eval_fold"], "seed": args.seed,
               "window": cfg["window"], "reranker_parameters": parameters,
               "reranker_mb": round(parameters * 4 / 1e6, 2), "variants": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out / "reranker.pt")
    print(json.dumps({"stage": "done", **{k: summary[k] for k in ("reranker_parameters", "reranker_mb")}}), flush=True)


if __name__ == "__main__":
    main()
