"""Recorded gate for the conditional overnight pretraining chain; thresholds live in the experiment config."""
import argparse
import json
from pathlib import Path

from reid.data import load_yaml


def decide(summary, gate):
    metric = summary["retrieval"]["mAP10"]
    spread = max(metric["std"], gate["parent_std"])
    return {"continue": metric["mean"] - gate["parent_mean"] > spread,
            "parent": gate["parent"], "parent_mean": gate["parent_mean"], "parent_std": gate["parent_std"],
            "mean": metric["mean"], "std": metric["std"], "required_gain": spread,
            "delta_vs_exp016": metric["mean"] - gate["exp016_mean"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    summary = json.loads((Path("artifacts") / cfg["exp_id"] / "summary.json").read_text())
    result = decide(summary, cfg["night_gate"])
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
