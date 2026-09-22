"""The viewpoint of EXP-190 read from our own patch tokens instead of CLIP.

The softmax head of EXP-190 on the final embedding matched the CLIP views on only 72% of frames: the re-ID embedding
is trained to ignore the side of the car. The patch tokens the tower pools into that embedding still carry it, and
they cost nothing at inference - every forward computes them. A softmax head on the mean token, or on the mean of each
quarter of the 16 x 16 token grid (where the headlights or the tail lights are), is fitted on training folds 1-3
against the CLIP labels; its views then replace CLIP's in the penalty sim - delta * same(q, g), with delta and the
refusal threshold carried from EXP-190 unchanged. Agreement with CLIP needs no identity labels, so it is reported on
every fold; mAP@10 on the dev fold decides, the holdout is a check.

Usage: PYTHONPATH=src python -m reid.view_tokens --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp192_view_tokens.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reid.data import load_yaml
from reid.embed import pick_device
from reid.evaluate import load_fold
from reid.inference import load_trained_encoder
from reid.metrics import paired_bootstrap
from reid.official_check import realistic
from reid.rerank_pairs import plain_loader
from reid.view_penalty import VIEWS, fit_view_head, score, view_probs


@torch.no_grad()
def token_features(tower, loader, device):
    """(mean token, mean token of each 2 x 2 quarter of the grid) per row, float32."""
    means, quarters = [], []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            tokens = tower.trunk.forward_features(x.to(device)).float()
        n, t, d = tokens.shape
        side = int(t ** 0.5)
        grid = tokens.reshape(n, 2, side // 2, 2, side // 2, d).mean((2, 4)).reshape(n, 4 * d)
        means.append(tokens.mean(1).cpu().numpy())
        quarters.append(grid.cpu().numpy())
    return {"mean": np.concatenate(means), "quarters": np.concatenate(quarters)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    spec, device = cfg["candidate"], pick_device()
    protocol, out = Path(val_cfg["out_dir"]), Path("artifacts") / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)
    views = pd.read_csv("protocol/attributes.csv").set_index("image_id").view
    code = {v: k for k, v in enumerate(VIEWS)}

    model_cfg = load_yaml(spec["config"])["model"]
    encoder, _, mean, std = load_trained_encoder(model_cfg, Path(data_cfg["checkpoint_dir"]) / spec["checkpoint"],
                                                 data_cfg.get("external_weights_dir"))
    tower = encoder.eval().to(device)[0]

    def features(rows):
        return token_features(tower, plain_loader(rows, data_cfg["crops_dir"], model_cfg["input_hw"], mean, std, 64),
                              device)

    folds = pd.read_csv(protocol / "folds.csv")
    train = folds[folds.fold.isin(cfg["head"]["folds"])].reset_index(drop=True)
    train_x, train_y = features(train), views.reindex(train.image_id).map(code).to_numpy()
    h = cfg["head"]
    heads = {k: fit_view_head(train_x[k], train_y, h["epochs"], h["lr"], h["weight_decay"]) for k in train_x}
    report = {"train_agreement": {k: float((view_probs(train_x[k], heads[k]).argmax(1) == train_y).mean())
                                  for k in heads}}

    for fold in cfg["folds"]:
        query, gallery_full, no_match = load_fold(protocol, fold)
        gallery = gallery_full[~gallery_full.vehicle_id.isin(no_match)].reset_index(drop=True)  # one gallery, as at test
        cache = Path(cfg["cache_dir"]) / f"fold{fold}.npz"                    # EXP-190 saved these for the EXP-164 soup
        if not cache.exists():                                                # another encoder: embed the fold once
            _, confidence, q_emb, g_emb = realistic({k: v for k, v in spec.items() if k != "threshold"},
                                                    query, gallery, data_cfg, device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, q_emb=q_emb, g_emb=g_emb, confidence=confidence)
        cached = np.load(cache)
        similarity, confidence = cached["q_emb"] @ cached["g_emb"].T, cached["confidence"]
        q_x, g_x = features(query), features(gallery)
        clip_q, clip_g = views.reindex(query.image_id).to_numpy(), views.reindex(gallery.image_id).to_numpy()
        base, base_ap = score(similarity, confidence, spec["threshold"], query, gallery)
        entry = {"base": base}
        variants = {"clip": (clip_q, clip_g)}
        for k in heads:
            q_view = np.asarray(VIEWS)[view_probs(q_x[k], heads[k]).argmax(1)]
            g_view = np.asarray(VIEWS)[view_probs(g_x[k], heads[k]).argmax(1)]
            entry[f"agreement_{k}"] = float(np.mean(np.r_[q_view == clip_q, g_view == clip_g]))
            variants[f"tokens_{k}"] = (q_view, g_view)
        keep = ~np.isnan(base_ap)
        for name, (q_view, g_view) in variants.items():
            same = (q_view[:, None] == g_view[None]).astype(np.float32)
            result, ap = score(similarity - cfg["delta"] * same, confidence, spec["threshold"], query, gallery)
            gain, low, high = paired_bootstrap(base_ap[keep], ap[keep], query.vehicle_id.to_numpy()[keep])
            entry[name] = {**result, "gain": gain, "ci": [low, high],
                           "points": 45 * result["mAP10"] + 10 * result["refusal"]}
        report[f"fold{fold}"] = entry
        print(json.dumps({f"fold{fold}": entry}), flush=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
