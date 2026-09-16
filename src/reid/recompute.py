"""Re-score finished experiments under another protocol version from their saved dev-fold embeddings, without re-embedding.

Query and gallery of the new version must be the same rows in the same order as when the embeddings were saved
(val_v2 copies them from val_v1). For each experiment the runs are artifacts/<exp>/seed<k>/ or artifacts/<exp>/val_fold<k>/;
results go to artifacts/<exp>/<version>/<run>/ plus summary.json and per_query_ap_mean.csv in artifacts/<exp>/<version>/.

Usage: PYTHONPATH=src python -m reid.recompute --val-config configs/val_v2.yaml --exps EXP-000 EXP-001
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.aggregate import summarise
from reid.data import load_yaml
from reid.evaluate import evaluate_fold, load_fold, save_fold_results


def run_dirs(exp_dir, fold):
    seeds = sorted(exp_dir.glob("seed*/query_emb.npy"))
    return [p.parent for p in seeds] or [exp_dir / f"val_fold{fold}"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--exps", nargs="+", required=True)
    args = parser.parse_args()
    val_cfg = load_yaml(args.val_config)
    fold = val_cfg["dev_fold"]
    query, gallery, no_match = load_fold(val_cfg["out_dir"], fold)

    for exp in args.exps:
        exp_dir, out_root = Path("artifacts") / exp, Path("artifacts") / exp / val_cfg["version"]
        outs = []
        for source in run_dirs(exp_dir, fold):
            emb = {part: np.load(source / f"{part}_emb.npy").astype(np.float32) for part in ("query", "gallery")}
            assert len(emb["query"]) == len(query) and len(emb["gallery"]) == len(gallery), f"{source}: rows differ"
            result = evaluate_fold(lambda df: emb["query"] if df is query else emb["gallery"], query, gallery, no_match)
            header = {"exp_id": exp, "protocol": val_cfg["version"], "fold": fold, "n_query": len(query),
                      "n_gallery": len(gallery), "source": str(source)}
            outs.append(out_root / source.name)
            save_fold_results(outs[-1], result, query, header, save_embeddings=False)
        seeds = [int(d.name[4:]) for d in outs if d.name.startswith("seed")]
        summary = summarise(out_root, outs, exp, seeds)
        print(exp, json.dumps({k: round(v["mean"], 4) for k, v in {**summary["retrieval"], **summary["refusal"]}.items()
                               if k in ("mAP10", "R1", "mAP", "F1", "TNR", "score", "F1_strict")}), flush=True)


if __name__ == "__main__":
    main()
