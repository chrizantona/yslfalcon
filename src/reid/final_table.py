"""Every fixed candidate on the dev fold and the holdout in the single-gallery setting of the test, nothing fitted.

EXP-171 compared the candidates on fold 4 in the old protocol (retrieval on the whole gallery, refusal after alpha-QE
over it, which leaked). Here each one runs as at test time (reid.official_check.realistic), is scored with the
organisers' definitions at the plateau-centre threshold EXP-178 found on fold 0, and optionally takes the viewpoint
penalty of EXP-190 with its fold-0 delta. Quality points = 45 mAP@10 + 10 (0.7 F1 + 0.3 TNR).

Usage: PYTHONPATH=src python -m reid.final_table --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp191_final_table.yaml --fold 4
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from reid.data import load_yaml
from reid.embed import pick_device
from reid.evaluate import load_fold
from reid.official_check import realistic
from reid.view_penalty import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    candidates = load_yaml(cfg["candidates_from"])["candidates"]
    thresholds = json.loads(Path(cfg["thresholds_from"]).read_text())
    out = Path("artifacts") / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)
    query, gallery_full, no_match = load_fold(Path(val_cfg["out_dir"]), args.fold)
    gallery = gallery_full[~gallery_full.vehicle_id.isin(no_match)].reset_index(drop=True)  # one gallery, as at test
    views = pd.read_csv("protocol/attributes.csv").set_index("image_id").view
    same = (views.reindex(query.image_id).to_numpy()[:, None] == views.reindex(gallery.image_id).to_numpy()[None])
    device, rows = pick_device(), []
    for name, spec in candidates.items():
        threshold = thresholds[name]["plateau"]["centre"]
        similarity, confidence, _, _ = realistic(spec, query, gallery, data_cfg, device)
        for penalty in ([0.0, cfg["view_penalty"]] if name in cfg["view_penalty_on"] else [0.0]):
            result, _ = score(similarity - penalty * same, confidence, threshold, query, gallery)
            rows.append({"fold": args.fold, "pipeline": name + (" + ракурс" if penalty else ""),
                         "query_expansion": bool(spec.get("alpha_qe")), "threshold": threshold, **result,
                         "points": 45 * result["mAP10"] + 10 * result["refusal"]})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(out / f"fold{args.fold}.csv", index=False)


if __name__ == "__main__":
    main()
