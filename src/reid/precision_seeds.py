"""FP16 validation and saved query/gallery/AP for every seed of an AFEM/OSNet training config."""
import argparse
import gc
import json
from pathlib import Path

import torch

from reid.aggregate import summarise
from reid.data import load_yaml
from reid.precision_eval import score_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data, val, cfg = load_yaml(args.data_config), load_yaml(args.val_config), load_yaml(args.config)
    if cfg["benchmark"]["precision"] != "fp16":
        raise ValueError("AFEM/OSNet seed precision check expects the frozen fp16 recipe")
    root = Path("artifacts") / cfg["exp_id"]
    run_dirs = []
    for seed in cfg["seeds"]:
        saved = json.loads((root / f"seed{seed}" / "metrics.json").read_text())
        child_id = f'{cfg["exp_id"]}/fp16_check/seed{seed}'
        metrics = score_checkpoint(data, val, cfg, saved["checkpoint"], child_id)
        print(json.dumps(metrics), flush=True)
        run_dirs.append(Path("artifacts") / child_id / "val_fold0")
        gc.collect()
        torch.cuda.empty_cache()
    summary = summarise(root / "fp16_check", run_dirs, cfg["exp_id"] + "/fp16_check", cfg["seeds"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
