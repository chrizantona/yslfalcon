"""A penalty on candidates shot from the query's own viewpoint, fitted on the dev fold and checked once on the holdout.

The error analysis of the final configuration (EXP-164 + re-ranker) found that 77% of the lost AP@10 sits on true
matches shot from another side than the query, and that a wrong first answer shares the query's viewpoint 80% of the
time against 37% for a random gallery frame: look-alikes seen from the same side outrank the right car seen from
another. The score becomes

    sim - delta * same(q, g),   same = 1 when the views agree (hard) or sum_v p_q(v) p_g(v) (soft).

Views come from pixels, never from the camera: the CLIP zero-shot labels of protocol/attributes.csv, or a softmax head
on our own embedding fitted on the training folds against those labels, which costs nothing at inference. Everything
runs in the single-gallery setting of the test (reid.official_check.realistic); delta is chosen on cfg.fit_fold and
scored once on cfg.check_fold, with the refusal threshold carried as it is.

Usage: PYTHONPATH=src python -m reid.view_penalty --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp190_view_penalty.yaml
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
from reid.gallery_transforms import normalise
from reid.holdout import embeddings
from reid.inference import load_trained_encoder
from reid.metrics import paired_bootstrap, refusal_curve, reid_eval_top10
from reid.official_check import realistic

VIEWS = ["front", "back", "side"]


def fit_view_head(x, labels, epochs, lr, weight_decay, seed=0):
    """Softmax regression from embeddings to CLIP view labels; returns (weight, bias) as numpy."""
    torch.manual_seed(seed)
    x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(labels)
    head = torch.nn.Linear(x.shape[1], len(VIEWS))
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimiser.zero_grad()
        torch.nn.functional.cross_entropy(head(x), y).backward()
        optimiser.step()
    return head.weight.detach().numpy(), head.bias.detach().numpy()


def view_probs(x, head):
    logits = x @ head[0].T + head[1]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def same_view(variant, q_views, g_views, q_probs, g_probs):
    """nq x ng agreement of views: 0/1 for the hard variants, a probability for the soft one."""
    if variant == "clip_hard":
        return (q_views[:, None] == g_views[None]).astype(np.float32)
    if variant == "head_hard":
        return (q_probs.argmax(1)[:, None] == g_probs.argmax(1)[None]).astype(np.float32)
    if variant == "head_soft":
        return (q_probs @ g_probs.T).astype(np.float32)
    raise ValueError(f"unknown variant {variant}")


def score(similarity, confidence, threshold, query, gallery):
    """mAP@10 with per-query AP, and the organisers' refusal score at the carried threshold."""
    args = (query.vehicle_id.to_numpy(), query.camera_id.to_numpy(), gallery.vehicle_id.to_numpy(),
            gallery.camera_id.to_numpy())
    retrieval, ap = reid_eval_top10(similarity, *args)
    refusal = refusal_curve(similarity, *args, confidence=confidence, grid=threshold).iloc[0]
    return {"mAP10": retrieval["mAP10"], "R1": retrieval["R1"], "refusal": float(refusal["score_official"]),
            "F1": float(refusal["F1_official"]), "TNR": float(refusal["TNR"])}, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    spec, device = cfg["candidate"], pick_device()
    protocol = Path(val_cfg["out_dir"])
    out = Path("artifacts") / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)
    attr = pd.read_csv("protocol/attributes.csv").set_index("image_id")
    code = {v: k for k, v in enumerate(VIEWS)}

    # the view head, on embeddings of the training folds from the same encoder
    folds = pd.read_csv(protocol / "folds.csv")
    train_rows = folds[folds.fold.isin(cfg["head"]["folds"])].reset_index(drop=True)
    model_cfg = load_yaml(spec["config"])["model"]
    encoder, _, mean, std = load_trained_encoder(model_cfg, Path(data_cfg["checkpoint_dir"]) / spec["checkpoint"],
                                                 data_cfg.get("external_weights_dir"))
    encoder = encoder.eval().to(device)
    train_emb = normalise(embeddings(encoder, train_rows, Path(data_cfg["dataset_dir"]) / "images", model_cfg,
                                     data_cfg, mean, std, device))
    del encoder
    torch.cuda.empty_cache()
    labels = attr.view.reindex(train_rows.image_id).map(code).to_numpy()
    h = cfg["head"]
    head = fit_view_head(train_emb, labels, h["epochs"], h["lr"], h["weight_decay"])
    report = {"head": {"train_accuracy": float((view_probs(train_emb, head).argmax(1) == labels).mean())}}

    folds_out = {}
    for fold in (cfg["fit_fold"], cfg["check_fold"]):
        query, gallery_full, no_match = load_fold(protocol, fold)
        gallery = gallery_full[~gallery_full.vehicle_id.isin(no_match)].reset_index(drop=True)  # one gallery, as at test
        cache = out / f"fold{fold}.npz"
        if cache.exists():
            saved = np.load(cache)
            similarity, confidence, q_emb, g_emb = (saved[k] for k in ("similarity", "confidence", "q_emb", "g_emb"))
        else:
            similarity, confidence, q_emb, g_emb = realistic(spec, query, gallery, data_cfg, device)
            np.savez(cache, similarity=similarity, confidence=confidence, q_emb=q_emb, g_emb=g_emb)
        q_views, g_views = attr.view.reindex(query.image_id).to_numpy(), attr.view.reindex(gallery.image_id).to_numpy()
        q_probs, g_probs = view_probs(q_emb, head), view_probs(g_emb, head)
        report["head"][f"fold{fold}_accuracy"] = float(np.mean(np.r_[
            q_probs.argmax(1) == pd.Series(q_views).map(code).to_numpy(),
            g_probs.argmax(1) == pd.Series(g_views).map(code).to_numpy()]))
        base, base_ap = score(similarity, confidence, spec["threshold"], query, gallery)
        rows, aps = [], {}
        for variant in cfg["variants"]:
            same = same_view(variant, q_views, g_views, q_probs, g_probs)
            for delta in cfg["deltas"]:
                result, ap = score(similarity - delta * same, confidence, spec["threshold"], query, gallery)
                rows.append({"fold": fold, "variant": variant, "delta": delta, **result})
                aps[(variant, delta)] = ap
        folds_out[fold] = (pd.DataFrame(rows), aps, base, base_ap, query)
        print(json.dumps({"fold": fold, "queries": len(query), "gallery": len(gallery), "base": base}), flush=True)

    curves = pd.concat([f[0] for f in folds_out.values()], ignore_index=True)
    curves.to_csv(out / "curves.csv", index=False)
    fit, check = folds_out[cfg["fit_fold"]], folds_out[cfg["check_fold"]]
    report["base"] = {"fit": fit[2], "check": check[2]}
    report["variants"] = {}
    for variant in cfg["variants"]:
        on_fit = fit[0][fit[0].variant == variant]
        delta = float(on_fit.loc[on_fit.mAP10.idxmax(), "delta"])  # chosen on the dev fold only
        entry = {"delta": delta}
        for name, (table, aps, base, base_ap, query) in (("fit", fit), ("check", check)):
            row = table[(table.variant == variant) & (table.delta == delta)].iloc[0]
            keep = ~np.isnan(base_ap)
            gain, low, high = paired_bootstrap(base_ap[keep], aps[(variant, delta)][keep],
                                               query.vehicle_id.to_numpy()[keep])
            entry[name] = {"mAP10": float(row.mAP10), "gain": gain, "ci": [low, high],
                           "refusal": float(row.refusal), "refusal_gain": float(row.refusal - base["refusal"]),
                           "points": 45 * gain + 10 * float(row.refusal - base["refusal"])}
        report["variants"][variant] = entry
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
