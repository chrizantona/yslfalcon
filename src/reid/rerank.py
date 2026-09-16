"""k-reciprocal re-ranking (Zhong et al., CVPR 2017) of saved fold embeddings, scored with the frozen protocol.

Distances are recomputed from the neighbourhoods of query and gallery items together, so re-ranking is a batch step
over all queries at once and its time is reported separately from the per-vehicle latency.
Reads artifacts/<source>/seed<k>/{query,gallery}_emb.npy and writes artifacts/<exp_id>/seed<k>/ like reid.train.

Usage: PYTHONPATH=src python -m reid.rerank --val-config configs/val_v1.yaml --config configs/exp006_rerank_exp003.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.metrics import reid_eval


def k_reciprocal_vectors(dist, k1, k2):
    """Row-normalised weights over each item's expanded k-reciprocal neighbours, averaged over its k2 nearest items."""
    n = len(dist)
    rank = np.argsort(dist, axis=1, kind="stable")  # rank[i, 0] is i itself

    def reciprocal(i, k):
        forward = rank[i, :k + 1]
        return forward[(rank[forward, :k + 1] == i).any(1)]

    V = np.zeros((n, n), np.float32)
    for i in range(n):
        neighbours = reciprocal(i, k1)
        expanded = [neighbours]
        for c in neighbours:
            candidates = reciprocal(c, int(round(k1 / 2)))
            if len(np.intersect1d(candidates, neighbours)) > 2 / 3 * len(candidates):
                expanded.append(candidates)
        index = np.unique(np.concatenate(expanded))
        weight = np.exp(-dist[i, index])
        V[i, index] = weight / weight.sum()
    return V[rank[:, :k2]].mean(1) if k2 > 1 else V


def jaccard_distance(V, n_rows):
    """1 - sum(min) / sum(max) between the first n_rows vectors and all vectors; rows of V sum to 1."""
    columns = [np.flatnonzero(V[:, j]) for j in range(V.shape[1])]
    dist = np.zeros((n_rows, len(V)), np.float32)
    for i in range(n_rows):
        overlap = np.zeros(len(V), np.float32)
        for j in np.flatnonzero(V[i]):
            overlap[columns[j]] += np.minimum(V[i, j], V[columns[j], j])
        dist[i] = 1 - overlap / (2 - overlap)
    return dist


def k_reciprocal_rerank(query_emb, gallery_emb, k1=20, k2=6, lam=0.3):
    """(n_query, n_gallery) re-ranked distance for L2-normalised embeddings; lower is closer."""
    x = np.vstack([query_emb, gallery_emb]).astype(np.float32)
    n_query = len(query_emb)
    dist = np.maximum(2 - 2 * x @ x.T, 0)  # squared Euclidean distance
    dist /= dist.max(axis=1, keepdims=True)
    jaccard = jaccard_distance(k_reciprocal_vectors(dist, k1, k2), n_query)
    return ((1 - lam) * jaccard + lam * dist[:n_query])[:, n_query:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    val_cfg, cfg = load_yaml(args.val_config), load_yaml(args.config)
    query, gallery, no_match = load_fold(val_cfg["out_dir"], cfg["eval_fold"])
    rcfg = cfg["rerank"]

    def score(query_emb, gallery_emb):  # similarity for the protocol: higher is closer
        return -k_reciprocal_rerank(query_emb, gallery_emb, rcfg["k1"], rcfg["k2"], rcfg["lambda"])

    for seed in cfg["seeds"]:
        source = Path("artifacts") / cfg["source_exp"] / f"seed{seed}"
        embeddings = {"query": np.load(source / "query_emb.npy").astype(np.float32),
                      "gallery": np.load(source / "gallery_emb.npy").astype(np.float32)}
        start = time.perf_counter()
        score(embeddings["query"], embeddings["gallery"])
        rerank_seconds = time.perf_counter() - start

        result = evaluate_fold(lambda df: embeddings["query"] if df is query else embeddings["gallery"],
                               query, gallery, no_match, cfg["refusal"], score_fn=score)
        # the saved float16 embeddings must reproduce the source run, otherwise the comparison is not paired
        source_map, _ = reid_eval(embeddings["query"] @ embeddings["gallery"].T, query.vehicle_id.to_numpy(),
                                  query.camera_id.to_numpy(), gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy())
        header = {"exp_id": cfg["exp_id"], "seed": seed, "source": f"{cfg['source_exp']}/seed{seed}",
                  "protocol": val_cfg["version"], "fold": cfg["eval_fold"], "n_query": len(query),
                  "n_gallery": len(gallery), "rerank": rcfg, "rerank_seconds": rerank_seconds,
                  "source_mAP": json.loads((source / "metrics.json").read_text())["retrieval"]["mAP"],
                  "source_mAP_saved_emb": source_map["mAP"]}
        metrics = save_fold_results(Path("artifacts") / cfg["exp_id"] / f"seed{seed}", result, query, header,
                                    emb_dtype=np.float16)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
