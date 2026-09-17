"""Paired vehicle bootstrap and seed-noise decision on an unchanged validation protocol."""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.compare import load_ap
from reid.metrics import paired_bootstrap


def describe(path):
    summary = Path(path) / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        metric = data["retrieval"]["mAP10"]
        return data, metric["mean"], metric["std"]
    data = json.loads(next(Path(path).glob("val_fold*/metrics.json")).read_text())
    return data, data["retrieval"]["mAP10"], 0.0


def compare(base, new):
    b, base_mean, base_std = describe(base)
    n, new_mean, new_std = describe(new)
    if (b["protocol"], b["fold"]) != (n["protocol"], n["fold"]):
        raise ValueError("cannot compare different validation protocols or folds")
    old, updated = load_ap(base), load_ap(new)
    if set(old.image_id) != set(updated.image_id):
        raise ValueError("query sets differ")
    merged = old.merge(updated[["image_id", "AP"]], on="image_id", suffixes=("_base", "_new"), validate="one_to_one")
    if not np.isclose(merged.AP_base.mean(), base_mean, atol=1e-7) or not np.isclose(merged.AP_new.mean(), new_mean, atol=1e-7):
        raise ValueError("AP files do not reproduce recorded mAP@10")
    gain, low, high = paired_bootstrap(merged.AP_base.to_numpy(), merged.AP_new.to_numpy(), merged.vehicle_id.to_numpy())
    noise = max(base_std, new_std)
    # Single checkpoint evaluations have no measured seed spread; zero is not evidence of zero noise.
    clears_noise = noise > 0 and gain > noise
    decision = "accepted" if gain > 0 and (clears_noise or low > 0) else ("rejected" if high < 0 else "inconclusive")
    return {"base": b["exp_id"], "new": n["exp_id"], "protocol": n["protocol"], "fold": n["fold"],
            "metric": "mAP@10", "base_mean": base_mean, "new_mean": new_mean, "gain": gain,
            "ci95": [low, high], "seed_noise": noise, "decision": decision,
            "decision_scope": "quality vs parent; H200 does not establish A5000 performance",
            "bootstrap_unit": "vehicle_id", "n_boot": 2000, "bootstrap_seed": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(args.base, args.new)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
