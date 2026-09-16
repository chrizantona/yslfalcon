"""Aggregate the seeds of one experiment: mean and std of dev-fold metrics, seed-averaged per-query AP.

Writes artifacts/<exp_id>/summary.json and artifacts/<exp_id>/per_query_ap_mean.csv.
Usage: PYTHONPATH=src python -m reid.aggregate --config configs/exp001_clip_vitb16_bot.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid.data import load_yaml


def mean_std(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_yaml(parser.parse_args().config)
    root, seeds = Path("artifacts") / cfg["exp_id"], cfg["seeds"]

    runs = [json.loads((root / f"seed{s}" / "metrics.json").read_text()) for s in seeds]
    summary = {"exp_id": cfg["exp_id"], "protocol": runs[0]["protocol"], "fold": runs[0]["fold"], "seeds": seeds}
    for key in ("train_minutes", "rerank_seconds", "checkpoint_mb"):  # training runs vs re-ranking runs
        if key in runs[0]:
            summary[key] = mean_std([r[key] for r in runs])
    for section in ("retrieval", "refusal"):
        summary[section] = {key: mean_std([r[section][key] for r in runs]) for key in runs[0][section]}
    summary["checkpoints"] = [run.get("checkpoint") for run in runs]  # file names on the pod, weights stay there
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    per_seed = [pd.read_csv(root / f"seed{s}" / "per_query_ap.csv") for s in seeds]
    mean_ap = per_seed[0][["image_id", "vehicle_id", "camera_id"]].copy()
    mean_ap["AP"] = np.mean([df.set_index("image_id").AP.loc[mean_ap.image_id].to_numpy() for df in per_seed], axis=0)
    mean_ap.to_csv(root / "per_query_ap_mean.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
