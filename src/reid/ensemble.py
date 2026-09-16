"""Ensemble the seeds of one experiment on its saved fold embeddings: mean of L2-normalised embeddings, renormalised.

Costs one forward pass per member at inference. Writes artifacts/<exp_id>/val_fold<k>/ like reid.evaluate.

Usage: PYTHONPATH=src python -m reid.ensemble --data-config configs/data.yaml --val-config configs/val_v1.yaml \
           --config configs/exp015_ensemble_exp010_seeds.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.evaluate import evaluate_fold, load_fold, save_fold_results


def mean_embedding(embeddings):
    """Rows of each (n, d) array are L2-normalised, averaged over the arrays and normalised again."""
    mean = np.mean([e / np.linalg.norm(e, axis=1, keepdims=True) for e in embeddings], axis=0)
    return (mean / np.linalg.norm(mean, axis=1, keepdims=True)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config")  # unused; keeps the call identical to reid.evaluate in pod jobs
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    val_cfg, cfg = load_yaml(args.val_config), load_yaml(args.config)
    fold = cfg["eval_fold"]
    query, gallery, no_match = load_fold(val_cfg["out_dir"], fold)

    source = Path("artifacts") / cfg["source_exp"]
    emb = {part: mean_embedding([np.load(source / f"seed{s}" / f"{part}_emb.npy").astype(np.float32)
                                 for s in cfg["seeds"]])
           for part in ("query", "gallery")}
    result = evaluate_fold(lambda df: emb["query"] if df is query else emb["gallery"],
                           query, gallery, no_match)
    header = {"exp_id": cfg["exp_id"], "protocol": val_cfg["version"], "fold": fold, "n_query": len(query),
              "n_gallery": len(gallery), "members": [f"{cfg['source_exp']}/seed{s}" for s in cfg["seeds"]]}
    metrics = save_fold_results(Path("artifacts") / cfg["exp_id"] / f"val_fold{fold}", result, query, header,
                                emb_dtype=np.float16)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
