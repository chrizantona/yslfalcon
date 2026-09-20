"""Ensemble saved fold embeddings: mean of L2-normalised embeddings, renormalised.

One forward pass per member at inference, so an ensemble is not a submission candidate under the 2 GB weight limit;
it is a reference point and a possible distillation teacher.

Two config shapes:
- source_exp + seeds: the seeds of one experiment, artifacts/<source_exp>/seed<k>/ (EXP-015);
- variants: {name: [embedding dirs]} with base_variant: several ensembles in one run, each scored and bootstrapped
  against the base, as in reid.gallery_transforms (EXP-150).

Usage: PYTHONPATH=src python -m reid.ensemble --data-config configs/data.yaml --val-config configs/val_v1.yaml \
           --config configs/exp015_ensemble_exp010_seeds.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.metrics import paired_bootstrap


def mean_embedding(embeddings):
    """Rows of each (n, d) array are L2-normalised, averaged over the arrays and normalised again."""
    mean = np.mean([e / np.linalg.norm(e, axis=1, keepdims=True) for e in embeddings], axis=0)
    return (mean / np.linalg.norm(mean, axis=1, keepdims=True)).astype(np.float32)


def ensemble_of(dirs):
    """(query, gallery) embeddings of the ensemble over the given directories."""
    return {part: mean_embedding([np.load(Path(d) / f"{part}_emb.npy").astype(np.float32) for d in dirs])
            for part in ("query", "gallery")}


def score(emb, query, gallery, no_match):
    return evaluate_fold(lambda df: emb["query"] if df is query else emb["gallery"], query, gallery, no_match)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config")  # unused; keeps the call identical to reid.evaluate in pod jobs
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    val_cfg, cfg = load_yaml(args.val_config), load_yaml(args.config)
    fold = cfg["eval_fold"]
    query, gallery, no_match = load_fold(val_cfg["out_dir"], fold)
    out_root = Path("artifacts") / cfg["exp_id"]

    if not cfg.get("variants"):  # one ensemble of the seeds of one experiment
        source = Path("artifacts") / cfg["source_exp"]
        result = score(ensemble_of([source / f"seed{s}" for s in cfg["seeds"]]), query, gallery, no_match)
        header = {"exp_id": cfg["exp_id"], "protocol": val_cfg["version"], "fold": fold, "n_query": len(query),
                  "n_gallery": len(gallery), "members": [f"{cfg['source_exp']}/seed{s}" for s in cfg["seeds"]]}
        print(json.dumps(save_fold_results(out_root / f"val_fold{fold}", result, query, header,
                                           emb_dtype=np.float16), indent=2))
        return

    results, base_ap = {}, None
    for name, members in cfg["variants"].items():
        result = score(ensemble_of(members), query, gallery, no_match)
        header = {"exp_id": cfg["exp_id"], "variant": name, "members": members, "protocol": val_cfg["version"],
                  "fold": fold, "n_query": len(query), "n_gallery": len(gallery)}
        metrics = save_fold_results(out_root / name, result, query, header, emb_dtype=np.float16)
        entry = {"members": members, "mAP10": metrics["retrieval"]["mAP10"], "R1": metrics["retrieval"]["R1"],
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

    (out_root / "summary.json").write_text(json.dumps(
        {"exp_id": cfg["exp_id"], "protocol": val_cfg["version"], "fold": fold,
         "base_variant": cfg["base_variant"], "variants": results}, indent=2))


if __name__ == "__main__":
    main()
