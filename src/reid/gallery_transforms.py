"""Gallery-side transforms of saved fold embeddings, scored with the frozen protocol (EXP-104).

Nothing here trains and nothing looks at another query: the streaming protocol allows the whole gallery, so a transform
fitted on the gallery is legal, query expansion over gallery items is legal, and query-to-query information is not used.
- base: the saved embeddings, unchanged; it must reproduce the recorded mAP@10 and is the comparison base;
- centre: subtract the gallery mean from both sides and renormalise;
- whiten: PCA-whitening fitted on the gallery, its eigenvalue scaling raised to power (0 = none, 1 = full), renormalised;
- alpha_qe: every query becomes the weighted mean of itself and its k nearest gallery items, weights cos ** alpha.
The refusal threshold is recalibrated for every variant, exactly as reid.evaluate does for the untransformed model.

Usage: PYTHONPATH=src python -m reid.gallery_transforms --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp104_gallery_transforms.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.metrics import paired_bootstrap


def normalise(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def centre(query, gallery):
    """Both sides moved by the gallery mean: the common direction of a camera set carries no identity."""
    mean = gallery.mean(0)
    return normalise(query - mean), normalise(gallery - mean)


def whiten(query, gallery, power):
    """PCA-whitening fitted on the gallery; power 0 leaves the embeddings alone, 1 is full whitening."""
    mean = gallery.mean(0)
    values, vectors = np.linalg.eigh(np.cov((gallery - mean).T))
    scale = np.maximum(values, 1e-12) ** (-power / 2)
    transform = (vectors * scale) @ vectors.T
    return normalise((query - mean) @ transform), normalise((gallery - mean) @ transform)


def alpha_qe(query, gallery, k, alpha):
    """Alpha-query expansion over gallery items only (Radenovic et al., 2019): the query keeps weight 1."""
    similarity = query @ gallery.T
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    weights = np.take_along_axis(similarity, nearest, axis=1).clip(min=0) ** alpha
    expanded = query + (weights[:, :, None] * gallery[nearest]).sum(1)
    return normalise(expanded), gallery


def apply_variant(query, gallery, spec):
    kind = spec["type"]
    if kind == "base":
        return query, gallery
    if kind == "centre":
        return centre(query, gallery)
    if kind == "whiten":
        return whiten(query, gallery, spec["power"])
    if kind == "alpha_qe":
        return alpha_qe(query, gallery, spec["k"], spec["alpha"])
    raise ValueError(f"unknown transform {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    val_cfg, cfg = load_yaml(args.val_config), load_yaml(args.config)
    fold = cfg["eval_fold"]
    query, gallery, no_match = load_fold(Path(val_cfg["out_dir"]), fold)
    source = Path(cfg["source_dir"])
    q0 = np.load(source / "query_emb.npy").astype(np.float32)
    g0 = np.load(source / "gallery_emb.npy").astype(np.float32)
    if len(q0) != len(query) or len(g0) != len(gallery):
        raise ValueError(f"{source}: embeddings do not match fold {fold}")
    out_root = Path("artifacts") / cfg["exp_id"]

    results, base_ap = {}, None
    for name, spec in cfg["variants"].items():
        q, g = apply_variant(normalise(q0), normalise(g0), spec)
        embeddings = {id(query): q, id(gallery): g}
        result = evaluate_fold(lambda df: embeddings[id(df)], query, gallery, no_match)
        header = {"exp_id": cfg["exp_id"], "variant": name, "transform": spec, "protocol": val_cfg["version"],
                  "fold": fold, "source_dir": str(source), "n_query": len(query), "n_gallery": len(gallery)}
        metrics = save_fold_results(out_root / name, result, query, header, emb_dtype=np.float16,
                                    save_embeddings=spec["type"] != "base")
        entry = {"transform": spec, "mAP10": metrics["retrieval"]["mAP10"], "R1": metrics["retrieval"]["R1"],
                 "R5": metrics["retrieval"]["R5"], "refusal": metrics["refusal"]["score"],
                 "threshold": metrics["refusal"]["threshold"]}
        if name == cfg["base_variant"]:
            base_ap = result["ap"]
        else:
            gain, low, high = paired_bootstrap(base_ap, result["ap"], query.vehicle_id.to_numpy())
            entry |= {"gain_vs_base": gain, "ci95": [low, high],
                      "decision": "accepted" if low > 0 else ("rejected" if high < 0 else "inconclusive")}
        results[name] = entry
        print(json.dumps({name: {k: round(v, 6) if isinstance(v, float) else v for k, v in entry.items()}}), flush=True)

    summary = {"exp_id": cfg["exp_id"], "protocol": val_cfg["version"], "fold": fold, "source_dir": str(source),
               "base_variant": cfg["base_variant"], "variants": results}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
