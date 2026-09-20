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
from reid.train import CropDataset, _worker_init, external_table


class CrossBlock(nn.Module):
    """One block of mutual attention: each side attends to the other through the same weights, so the block - and
    with it the whole score - is symmetric in the pair."""

    def __init__(self, width, heads, dropout=0.0):
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * width, width))

    def forward(self, a, b):
        qa, qb, ka, kb = self.norm_q(a), self.norm_q(b), self.norm_kv(a), self.norm_kv(b)
        a = a + self.drop(self.attn(qa, kb, kb, need_weights=False)[0])
        b = b + self.drop(self.attn(qb, ka, ka, need_weights=False)[0])
        return a + self.drop(self.ff(self.norm_ff(a))), b + self.drop(self.ff(self.norm_ff(b)))


class PairReranker(nn.Module):
    """Logit of "the same vehicle, seen from another camera" for a pair of patch-token sets."""

    def __init__(self, token_dim, width=256, blocks=2, heads=4, dropout=0.0):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(token_dim, width), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([CrossBlock(width, heads, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, tokens_a, tokens_b):
        a, b = self.proj(tokens_a), self.proj(tokens_b)
        for block in self.blocks:
            a, b = block(a, b)
        u, v = a.mean(1), b.mean(1)
        # symmetric in (u, v): neither the sum nor the absolute difference depends on the order of the pair
        return self.head(torch.cat([u + v, (u - v).abs()], dim=1)).squeeze(1)


def plain_loader(df, crops_dir, input_hw, mean, std, batch_size, num_workers=8, flip=False):
    frame = df if "label" in df else df.assign(label=0)
    plain = {"flip_p": 1.0 if flip else 0.0, "pad": 0, "erasing_p": 0.0}  # flip_p 1.0 = the mirrored view, not a coin
    return DataLoader(CropDataset(frame.reset_index(drop=True), crops_dir, input_hw, mean, std, plain),
                      batch_size=batch_size, num_workers=num_workers, worker_init_fn=_worker_init)


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


@torch.no_grad()
def mine_neighbours(embeddings, vehicle_codes, n_hard, device, chunk=2048):
    """For every row, the nearest images of other vehicles under the frozen encoder."""
    emb = torch.as_tensor(embeddings, device=device)
    codes = torch.as_tensor(np.asarray(vehicle_codes), device=device)
    out = torch.empty(len(emb), n_hard, dtype=torch.int64)
    for start in range(0, len(emb), chunk):
        stop = min(start + chunk, len(emb))
        similarity = emb[start:stop] @ emb.T
        similarity[codes[start:stop, None] == codes[None, :]] = -float("inf")
        out[start:stop] = similarity.topk(n_hard, dim=1).indices.cpu()
    return out.numpy()


def take_tokens(banks, indices, rng, keep, device):
    """One stored view per row - upright or mirrored - with a random subset of its tokens kept."""
    bank = banks[rng.integers(len(banks))]
    tokens = bank[torch.as_tensor(np.asarray(indices))].to(device, non_blocking=True).float()
    if keep < 1.0:
        n = tokens.shape[1]
        wanted = max(1, int(round(keep * n)))
        chosen = torch.rand(len(tokens), n, device=tokens.device).argsort(1)[:, :wanted]
        tokens = tokens.gather(1, chosen[:, :, None].expand(-1, -1, tokens.shape[2]))
    return tokens


def window_groups(embeddings, vehicle, camera, window, device):
    """(anchor, candidates, positives mask, junk mask) shaped like the window the re-ranker sees at inference.

    The candidates are the anchor's nearest images under the frozen encoder, exactly as the encoder's top-N is built
    at query time, so training sees the composition it will face: a few positives, some junk and hard negatives. When
    no cross-camera positive made the window one replaces the weakest candidate - those are the cases the re-ranker
    exists to fix."""
    # mine_neighbours already masks the anchor out through the codes, so every column is a real neighbour
    nearest = mine_neighbours(embeddings, np.arange(len(embeddings)), window, device)
    rows_of = {}
    for row, code in enumerate(vehicle):
        rows_of.setdefault(code, []).append(row)
    groups = []
    for anchor_row in range(len(vehicle)):
        elsewhere = [r for r in rows_of[vehicle[anchor_row]] if camera[r] != camera[anchor_row]]
        if not elsewhere:
            continue
        candidates = nearest[anchor_row].copy()
        same = vehicle[candidates] == vehicle[anchor_row]
        positive = same & (camera[candidates] != camera[anchor_row])
        if not positive.any():
            candidates[-1] = elsewhere[0]
            same = vehicle[candidates] == vehicle[anchor_row]
            positive = same & (camera[candidates] != camera[anchor_row])
        groups.append((anchor_row, candidates, positive, same & ~positive))
    return groups


def window_loss(logits, positive, junk):
    """Listwise over the window with several positives allowed, junk masked out of both terms.

    Junk is the same vehicle from the same camera: the scorer drops it from the submitted list, so where it lands
    carries no reward and teaching the module to like or dislike it only adds noise."""
    usable = ~junk
    scores = logits.masked_fill(junk, float("-inf"))
    log_prob = scores - scores.logsumexp(1, keepdim=True)
    listwise = -(log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp(min=1))
    binary = F.binary_cross_entropy_with_logits(logits, positive.float(), reduction="none")
    return listwise.mean() + (binary * usable).sum() / usable.sum().clamp(min=1)


def cached(path, build):
    """Token banks are the same for every seed - the tower is frozen - so they are built once and kept in /dev/shm."""
    path = Path(path)
    if path.exists():
        return torch.from_numpy(np.load(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    bank = build()
    np.save(path, bank.numpy())
    return bank


@torch.no_grad()
def validation_gain(model, banks, frames, embeddings, cfg, device):
    """What the re-ranker does to mAP@10 on a held-out fold, scored through the protocol's own query/gallery split.

    Using the fold's raw table instead - every image a query against the rest - is a different task: denser in
    positives and with a gallery twice the size, and it disagrees with the dev fold for that reason alone."""
    model.eval()
    (query_banks, gallery_banks), (query, gallery), (query_emb, gallery_emb) = banks, frames, embeddings
    similarity = query_emb @ gallery_emb.T
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    picked = np.argsort(-similarity, axis=1, kind="stable")[:, :cfg["window"]]
    scores = candidate_scores(model, query_banks, gallery_banks, picked, device)
    before, _ = reid_eval_top10(similarity, q_pid, q_cam, g_pid, g_cam)
    gains = {}
    for weight in cfg["weights"]:
        if weight == 0:
            continue
        after, _ = reid_eval_top10(reorder(similarity, picked, scores, weight), q_pid, q_cam, g_pid, g_cam)
        gains[weight] = after["mAP10"] - before["mAP10"]
    best = max(gains, key=gains.get)
    return gains[best], best, {str(k): round(v, 6) for k, v in gains.items()}


def train(model, sources, validation, cfg, device, seed, log):
    """sources: list of (banks, window groups); the first is ours, the second - when present - external."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * cfg["steps_per_epoch"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], total_steps=total_steps,
                                                    pct_start=0.1)
    rng = np.random.default_rng(seed)
    best = (-1e9, None, 0, cfg["weights"][-1])
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        for _ in range(cfg["steps_per_epoch"]):
            external = len(sources) > 1 and rng.random() < cfg["external_share"]
            banks, groups = sources[1] if external else sources[0]
            chosen = [groups[i] for i in rng.integers(len(groups), size=cfg["groups_per_batch"])]
            candidates = np.array([g[1] for g in chosen])
            positive = torch.as_tensor(np.array([g[2] for g in chosen]), device=device)
            junk = torch.as_tensor(np.array([g[3] for g in chosen]), device=device)
            anchors = np.repeat([g[0] for g in chosen], candidates.shape[1])
            a = take_tokens(banks, anchors, rng, cfg["token_keep"], device)
            b = take_tokens(banks, candidates.ravel(), rng, cfg["token_keep"], device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(a, b)
            logits = logits.float().view(len(chosen), -1)
            loss = window_loss(logits, positive, junk)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
            ranked_first = logits.masked_fill(junk, float("-inf")).argmax(1)
            correct += int(positive.gather(1, ranked_first[:, None]).sum())
            seen += len(chosen)
        gain, weight, all_gains = validation_gain(model, *validation, cfg, device)
        if gain > best[0]:
            best = (gain, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch, weight)
        record = {"epoch": epoch, "loss": total / cfg["steps_per_epoch"], "positive_first": correct / max(seen, 1),
                  "held_out_map10_gain": gain, "held_out_by_weight": all_gains,
                  "elapsed_min": (time.time() - start) / 60}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)
    model.load_state_dict(best[1])
    return {"best_epoch": best[2], "best_held_out_gain": best[0], "weight_chosen_on_held_out": best[3]}


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
    top10, ap = reid_eval_top10(ranked, q_pid, q_cam, g_pid, g_cam)
    full, _ = reid_eval(ranked, q_pid, q_cam, g_pid, g_cam)
    best = pick_threshold(refusal_curve(reduced, q_pid, q_cam, g_pid[keep], g_cam[keep]))
    return {"mAP10": float(top10["mAP10"]), "R1": float(top10["R1"]), "R5": float(top10["R5"]),
            "mAP": float(full["mAP"]), "mINP": float(full["mINP"]), "refusal": float(best["score"]),
            "F1": float(best["F1"]), "TNR": float(best["TNR"]), "threshold": float(best["threshold"])}, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", action="store_true", help="score a saved reranker.pt instead of training one")
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
    tcfg = {**cfg["train"], "window": cfg["window"], "weights": cfg["weights"]}  # the loop needs both
    query, gallery, no_match = load_fold(Path(val_cfg["out_dir"]), cfg["eval_fold"])
    hw, crops = cfg["model"]["input_hw"], data_cfg["crops_dir"]
    cache = Path(tcfg.get("token_cache", "/dev/shm/rerank_tokens")) / cfg["checkpoint"].replace(".pt", "")

    def banks_of(name, frame, mirrored=True):
        views = [False, True] if mirrored else [False]
        name = f"{name}_{len(frame)}"  # the row count is part of the key: a different subset is a different bank
        return [cached(cache / f"{name}{'_flip' if flip else ''}.npy",
                       lambda flip=flip: token_bank(tower, plain_loader(frame, crops, hw, mean, std, 64, flip=flip),
                                                    device))
                for flip in views]

    def embeddings_of(name, frame):
        return cached(cache / f"{name}_{len(frame)}_emb.npy",
                      lambda: torch.from_numpy(crop_embeddings(encoder, plain_loader(frame, crops, hw, mean, std,
                                                                                     64), device))).numpy()

    def source_of(name, frame, mirrored=True):
        embeddings = embeddings_of(name, frame)
        vehicle = pd.factorize(frame.vehicle_id.to_numpy())[0]
        groups = window_groups(embeddings, vehicle, frame.camera_id.to_numpy(), cfg["window"], device)
        positives = float(np.mean([g[2].sum() for g in groups])) if groups else 0.0
        if not groups:
            raise ValueError(f"source {name} has no cross-camera pairs: {len(frame)} rows")
        print(json.dumps({"source": name, "rows": len(frame), "vehicles": int(frame.vehicle_id.nunique()),
                          "groups": len(groups), "positives_per_window": round(positives, 3)}), flush=True)
        return banks_of(name, frame, mirrored), groups

    train_df = folds[folds.fold.isin(tcfg["folds"])].reset_index(drop=True)
    sources = [source_of("train", train_df)]
    if tcfg.get("external"):
        external = external_table(data_cfg, tcfg["external"])
        keep_n = tcfg["external_images_per_vehicle"]
        external = (external.sample(frac=1.0, random_state=0).groupby("vehicle_id", group_keys=False)
                    .head(keep_n).reset_index(drop=True))
        sources.append(source_of("external", external))
    # the held-out fold is scored through its own protocol split, the same shape the dev fold is scored in
    validation_query, validation_gallery, _ = load_fold(Path(val_cfg["out_dir"]), tcfg["validation_fold"])
    validation = ((banks_of("val_query", validation_query, mirrored=False)[0],
                   banks_of("val_gallery", validation_gallery, mirrored=False)[0]),
                  (validation_query, validation_gallery),
                  (normalise(embeddings_of("val_query", validation_query)),
                   normalise(embeddings_of("val_gallery", validation_gallery))))

    query_tokens = banks_of("query", query, mirrored=False)[0]
    gallery_tokens = banks_of("gallery", gallery, mirrored=False)[0]

    model = PairReranker(query_tokens.shape[-1], cfg["width"], cfg["blocks"], cfg["heads"],
                         cfg.get("dropout", 0.0)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if args.load:
        model.load_state_dict(torch.load(out / "reranker.pt", map_location=device)["state_dict"])
        selection = {"loaded": str(out / "reranker.pt")}
    else:
        with (out / "train_log.jsonl").open("w") as log:
            selection = train(model, sources, validation, tcfg, device, args.seed, log)
    print(json.dumps({"stage": "selected", **selection}), flush=True)

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
        results[f"w{weight}"], ap = evaluate(ranked, reduced, query, gallery, keep)
        query.assign(AP=ap).to_csv(out / f"per_query_ap_w{weight}.csv", index=False)
        print(json.dumps({"weight": weight, **{k: round(v, 6) for k, v in results[f"w{weight}"].items()}}), flush=True)

    summary = {"exp_id": cfg["exp_id"], "protocol": val_cfg["version"], "fold": cfg["eval_fold"], "seed": args.seed,
               "window": cfg["window"], **selection, "reranker_parameters": parameters,
               "reranker_mb": round(parameters * 4 / 1e6, 2), "variants": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out / "reranker.pt")
    print(json.dumps({"stage": "done", **{k: summary[k] for k in ("reranker_parameters", "reranker_mb")}}), flush=True)


if __name__ == "__main__":
    main()
