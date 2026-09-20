"""Pseudo-labels for unlabelled images from an ensemble of trained models (EXP-162).

Images are linked when they are within each other's k nearest neighbours and the similarity clears a z-score
threshold; the connected components of that graph become pseudo vehicles. Components larger than max_size are
dropped - our real vehicles have 4 to 8 frames - and so are singletons.

Two modes:
- calibrate: cluster saved fold embeddings, where the true vehicle ids are known, and score the clusters. The
  threshold is chosen here, on the dev fold, and only then applied to unlabelled images;
- apply: embed the competition test CSVs with the ensemble members and write image_id, vehicle_id, camera_id for
  training. The test images carry no camera labels, so every image counts as its own camera, as for VehicleID.

Usage: PYTHONPATH=src python -m reid.pseudo_labels calibrate --val-config configs/val_v2.yaml \
           --config configs/exp162_pseudo_labels.yaml --out artifacts/EXP-162/calibration.json
       PYTHONPATH=src python -m reid.pseudo_labels apply --data-config configs/data.yaml \
           --config configs/exp162_pseudo_labels.yaml --out protocol/pseudo/test_pseudo.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid.data import load_csv, load_yaml
from reid.embed import embed_boxes, pick_device
from reid.ensemble import mean_embedding
from reid.evaluate import load_fold


def mutual_edges(embeddings, k, z_threshold):
    """Pairs that list each other among their k nearest and whose similarity clears the z-score threshold."""
    similarity = embeddings @ embeddings.T
    n = len(similarity)
    # the row statistics are taken without the diagonal: a self-similarity of 1 skews them, and -inf destroys them
    others = similarity[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    z = (similarity - others.mean(1, keepdims=True)) / np.maximum(others.std(1, keepdims=True), 1e-12)
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argsort(-similarity, axis=1)[:, :k]
    neighbours = [set(row) for row in nearest]
    return [(i, int(j)) for i, row in enumerate(nearest) for j in row
            if i < j and i in neighbours[j] and z[i, j] >= z_threshold and z[j, i] >= z_threshold]


def components(n, edges, max_size):
    """Connected components of the graph as a label per image; -1 for singletons and for components over max_size."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    labels = np.full(n, -1)
    for cluster, members in enumerate(g for g in groups.values() if 1 < len(g) <= max_size):
        labels[members] = cluster
    return labels


def score_clusters(labels, truth):
    """How good the clusters are when the true vehicle of every image is known."""
    used = labels >= 0
    if not used.any():
        return {"images": 0, "clusters": 0, "coverage": 0.0, "pure_clusters": 0.0, "image_purity": 0.0,
                "vehicles_split": 0.0}
    frame = pd.DataFrame({"cluster": labels[used], "vehicle": np.asarray(truth)[used]})
    by_cluster = frame.groupby("cluster").vehicle
    sizes = by_cluster.size()
    majority = by_cluster.agg(lambda v: v.value_counts().iloc[0])
    vehicles_per_cluster = by_cluster.nunique()
    return {"images": int(used.sum()), "clusters": int(len(sizes)), "coverage": float(used.mean()),
            "pure_clusters": float((vehicles_per_cluster == 1).mean()),  # clusters holding a single vehicle
            "image_purity": float(majority.sum() / sizes.sum()),  # images sitting with their own majority vehicle
            "vehicles_split": float(frame.groupby("vehicle").cluster.nunique().gt(1).mean())}


def ensemble_embeddings(members, images_dir, rows, data_cfg, device):
    """Mean of the L2-normalised embeddings of the ensemble members over the given annotated boxes."""
    from reid.inference import load_trained_encoder  # local: reid.inference imports the model builders
    parts = []
    for spec in members:
        model_cfg = load_yaml(spec["config"])["model"]
        encoder, _, mean, std = load_trained_encoder(model_cfg, Path(data_cfg["checkpoint_dir"]) / spec["checkpoint"],
                                                     data_cfg.get("external_weights_dir"), spec.get("sha256"))
        encoder = encoder.eval().to(device)
        parts.append(embed_boxes(encoder, rows, images_dir, model_cfg["input_hw"], mean, std,
                                 data_cfg["crop_max_side"], device=device, batch_size=128, num_workers=8))
        del encoder
    return mean_embedding(parts)


def calibrate(cfg, val_cfg):
    """Cluster the dev fold, where the answer is known, over the configured grid of k and z thresholds."""
    fold = cfg["eval_fold"]
    query, gallery, _ = load_fold(Path(val_cfg["out_dir"]), fold)
    rows = pd.concat([query, gallery], ignore_index=True)
    embeddings = mean_embedding([np.concatenate([np.load(Path(d) / "query_emb.npy"),
                                                 np.load(Path(d) / "gallery_emb.npy")]).astype(np.float32)
                                 for d in cfg["calibration_dirs"]])
    results = []
    for k in cfg["grid"]["k"]:
        for threshold in cfg["grid"]["z_threshold"]:
            labels = components(len(rows), mutual_edges(embeddings, k, threshold), cfg["max_size"])
            results.append({"k": k, "z_threshold": threshold,
                            **score_clusters(labels, rows.vehicle_id.to_numpy())})
            print(json.dumps({key: round(v, 4) if isinstance(v, float) else v for key, v in results[-1].items()}),
                  flush=True)
    return results


def apply(cfg, data_cfg):
    """Pseudo vehicle ids for the competition test images, with the calibrated k and threshold."""
    dataset_dir = Path(data_cfg["dataset_dir"])
    rows = pd.concat([load_csv(dataset_dir, "query"), load_csv(dataset_dir, "gallery")], ignore_index=True)
    embeddings = ensemble_embeddings(cfg["members"], dataset_dir / "images", rows, data_cfg, pick_device())
    labels = components(len(rows), mutual_edges(embeddings, cfg["k"], cfg["z_threshold"]), cfg["max_size"])
    kept = labels >= 0
    table = pd.DataFrame({"image_id": rows.image_id[kept].to_numpy(),
                          "vehicle_id": [f"pl_{c}" for c in labels[kept]],
                          "camera_id": np.arange(int(kept.sum())),  # no camera labels on the test: one per image
                          "x": rows.x[kept].to_numpy(), "y": rows.y[kept].to_numpy(),
                          "w": rows.w[kept].to_numpy(), "h": rows.h[kept].to_numpy()})
    print(json.dumps({"images": int(kept.sum()), "of": len(rows), "vehicles": int(table.vehicle_id.nunique()),
                      "sizes": table.vehicle_id.value_counts().value_counts().sort_index().to_dict()}), flush=True)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["calibrate", "apply"])
    parser.add_argument("--data-config")
    parser.add_argument("--val-config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "calibrate":
        results = calibrate(cfg, load_yaml(args.val_config))
        out.write_text(json.dumps({"exp_id": cfg["exp_id"], "max_size": cfg["max_size"], "grid": results}, indent=2))
    else:
        apply(cfg, load_yaml(args.data_config)).to_csv(out, index=False)


if __name__ == "__main__":
    main()
