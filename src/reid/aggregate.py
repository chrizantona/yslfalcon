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


def summarise(root, run_dirs, exp_id, seeds):
    """summary.json and per_query_ap_mean.csv in root from the metrics.json / per_query_ap.csv of each run dir."""
    root = Path(root)
    runs = [json.loads((d / "metrics.json").read_text()) for d in run_dirs]
    summary = {"exp_id": exp_id, "protocol": runs[0]["protocol"], "fold": runs[0]["fold"], "seeds": seeds}
    for key in ("train_minutes", "rerank_seconds", "checkpoint_mb"):  # training runs vs re-ranking runs
        if key in runs[0]:
            summary[key] = mean_std([r[key] for r in runs])
    for section in ("retrieval", "refusal"):
        summary[section] = {key: mean_std([r[section][key] for r in runs]) for key in runs[0][section]}
    summary["checkpoints"] = [run.get("checkpoint") for run in runs]  # file names on the pod, weights stay there
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    per_run = [pd.read_csv(d / "per_query_ap.csv").set_index("image_id") for d in run_dirs]
    mean_ap = per_run[0].reset_index()[["image_id", "vehicle_id", "camera_id"]]
    for column in [c for c in ("AP", "AP_full") if c in per_run[0]]:
        mean_ap[column] = np.mean([df[column].loc[mean_ap.image_id].to_numpy() for df in per_run], axis=0)
    mean_ap.to_csv(root / "per_query_ap_mean.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_yaml(parser.parse_args().config)
    root, seeds = Path("artifacts") / cfg["exp_id"], cfg["seeds"]
    summary = summarise(root, [root / f"seed{s}" for s in seeds], cfg["exp_id"], seeds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
