"""k-reciprocal re-ranking (Zhong et al., CVPR 2017) of saved fold embeddings, scored with the frozen protocol.

Two variants:
- batch (k_reciprocal_rerank): neighbourhoods of query and gallery items together. Not allowed for submissions: the
  organisers' streaming protocol forbids using other queries. Kept to reproduce EXP-006, which is invalid for that reason;
- streaming (StreamingRerank, rerank.streaming: true): the gallery is processed once, every query on its own.
Reads artifacts/<source>/seed<k>/ or val_fold<k>/ embeddings and writes artifacts/<exp_id>/<same run name>/.

Usage: PYTHONPATH=src python -m reid.rerank --val-config configs/val_v2.yaml --config configs/exp019_streaming_rerank_exp016.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.metrics import reid_eval
from reid.recompute import run_dirs


def reciprocal(rank, i, k):
    """Items among the k nearest of i (with i itself) that also have i among their k nearest."""
    forward = rank[i, :k + 1]
    return forward[(rank[forward, :k + 1] == i).any(1)]


def expanded_weights(dist_row, neighbours, candidate_sets):
    """Row-normalised exp(-dist) over the neighbours plus each candidate set overlapping them by more than 2/3."""
    expanded = [neighbours]
    for candidates in candidate_sets:
        if len(np.intersect1d(candidates, neighbours)) > 2 / 3 * len(candidates):
            expanded.append(candidates)
    index = np.unique(np.concatenate(expanded))
    weight = np.exp(-dist_row[index])
    row = np.zeros(len(dist_row), np.float32)
    row[index] = weight / weight.sum()
    return row


def k_reciprocal_weights(dist, k1):
    """(rank matrix, expanded k-reciprocal weights of every item before local query expansion)."""
    rank = np.argsort(dist, axis=1, kind="stable")  # rank[i, 0] is i itself
    half = int(round(k1 / 2))
    rows = []
    for i in range(len(dist)):
        neighbours = reciprocal(rank, i, k1)
        rows.append(expanded_weights(dist[i], neighbours, [reciprocal(rank, c, half) for c in neighbours]))
    return rank, np.stack(rows)


def k_reciprocal_vectors(dist, k1, k2):
    """Row-normalised weights over each item's expanded k-reciprocal neighbours, averaged over its k2 nearest items."""
    rank, V = k_reciprocal_weights(dist, k1)
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


class StreamingRerank:
    """k-reciprocal re-ranking under the streaming protocol: the gallery is processed once, each query on its own.

    Gallery neighbourhoods, weights and their local expansion use the gallery alone. For a query, a gallery item is a
    reciprocal neighbour when it is among the query's k1 nearest and the query would be among its k1 nearest; the
    expansion uses the gallery's own half-size reciprocal sets. The query vector is averaged with its k2 - 1 nearest
    gallery items; distance = (1 - lam) Jaccard + lam normalised squared distance. Gallery items that share no
    neighbours with the query keep their original order."""

    def __init__(self, gallery_emb, k1=20, k2=6, lam=0.3):
        self.gallery, self.k1, self.k2, self.lam = gallery_emb.astype(np.float32), k1, k2, lam
        dist = np.maximum(2 - 2 * self.gallery @ self.gallery.T, 0)
        self.kth = np.sort(dist, axis=1)[:, min(k1, len(dist) - 1)]  # distance to the k1-th nearest other item
        rank, self.weights = k_reciprocal_weights(dist / dist.max(axis=1, keepdims=True), k1)
        self.half = [reciprocal(rank, i, int(round(k1 / 2))) for i in range(len(dist))]
        self.vectors = self.weights[rank[:, :k2]].mean(1) if k2 > 1 else self.weights
        self.columns = [np.flatnonzero(self.vectors[:, j]) for j in range(len(dist))]

    def distance(self, query_emb):
        """(n_gallery,) re-ranked distance for one L2-normalised query; lower is closer."""
        raw = np.maximum(2 - 2 * self.gallery @ query_emb.astype(np.float32), 0)
        dist = raw / raw.max()
        order = np.argsort(dist, kind="stable")
        forward = order[:self.k1]
        neighbours = forward[raw[forward] <= self.kth[forward]]
        if len(neighbours) == 0:  # nobody would take the query as a neighbour: plain distance order
            return (1 - self.lam) + self.lam * dist
        v = expanded_weights(dist, neighbours, [self.half[c] for c in neighbours])
        if self.k2 > 1:
            v = (v + self.weights[order[:self.k2 - 1]].sum(0)) / self.k2
        overlap = np.zeros(len(self.gallery), np.float32)
        for j in np.flatnonzero(v):
            overlap[self.columns[j]] += np.minimum(v[j], self.vectors[self.columns[j], j])
        return (1 - self.lam) * (1 - overlap / (2 - overlap)) + self.lam * dist


def streaming_similarity(query_emb, gallery_emb, k1, k2, lam):
    """Similarity for the protocol (higher is closer); every row depends on its query and the gallery only."""
    rerank = StreamingRerank(gallery_emb, k1, k2, lam)
    return -np.stack([rerank.distance(q) for q in query_emb])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    val_cfg, cfg = load_yaml(args.val_config), load_yaml(args.config)
    query, gallery, no_match = load_fold(val_cfg["out_dir"], cfg["eval_fold"])
    rcfg = cfg["rerank"]

    def score(query_emb, gallery_emb):  # similarity for the protocol: higher is closer
        if rcfg.get("streaming"):
            return streaming_similarity(query_emb, gallery_emb, rcfg["k1"], rcfg["k2"], rcfg["lambda"])
        return -k_reciprocal_rerank(query_emb, gallery_emb, rcfg["k1"], rcfg["k2"], rcfg["lambda"])

    for source in run_dirs(Path("artifacts") / cfg["source_exp"], cfg["eval_fold"]):
        embeddings = {"query": np.load(source / "query_emb.npy").astype(np.float32),
                      "gallery": np.load(source / "gallery_emb.npy").astype(np.float32)}
        start = time.perf_counter()
        score(embeddings["query"], embeddings["gallery"])
        timing = {"rerank_seconds": time.perf_counter() - start}
        if rcfg.get("streaming"):
            start = time.perf_counter()
            rerank = StreamingRerank(embeddings["gallery"], rcfg["k1"], rcfg["k2"], rcfg["lambda"])
            timing["gallery_build_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            for q in embeddings["query"]:
                rerank.distance(q)
            timing["per_query_ms"] = (time.perf_counter() - start) * 1000 / len(embeddings["query"])

        result = evaluate_fold(lambda df: embeddings["query"] if df is query else embeddings["gallery"],
                               query, gallery, no_match, score_fn=score)
        # the saved float16 embeddings must reproduce the source run, otherwise the comparison is not paired
        source_map, _ = reid_eval(embeddings["query"] @ embeddings["gallery"].T, query.vehicle_id.to_numpy(),
                                  query.camera_id.to_numpy(), gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy())
        header = {"exp_id": cfg["exp_id"], "source": str(source), "protocol": val_cfg["version"], "fold": cfg["eval_fold"],
                  "n_query": len(query), "n_gallery": len(gallery), "rerank": rcfg, **timing,
                  "source_mAP": json.loads((source / "metrics.json").read_text())["retrieval"]["mAP"],
                  "source_mAP_saved_emb": source_map["mAP"]}
        if source.name.startswith("seed"):
            header["seed"] = int(source.name[4:])
        metrics = save_fold_results(Path("artifacts") / cfg["exp_id"] / source.name, result, query, header,
                                    emb_dtype=np.float16)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
