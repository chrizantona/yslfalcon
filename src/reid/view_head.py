"""The viewpoint head of EXP-192/204 as a saved artefact, and the penalty it feeds at submission time.

The error analysis found that a wrong first answer shares the query's viewpoint 80% of the time against 37% for a
random gallery frame, so the score becomes sim - delta * [view(q) = view(g)]. The view comes from a softmax head on
the mean patch token, fitted against the CLIP zero-shot labels of protocol/attributes.csv; EXP-204 confirmed the
penalty on five leave-one-fold-out models (mean +0.0031 mAP@10, positive on all five). At inference the head costs one
1152x3 matrix product on a tensor the tower computes anyway: the tokens are captured where the attention pool receives
them, which is exactly what reid.view_tokens fitted on.

Refusal is decided on the encoder's own similarity, before the penalty, because that is how the threshold of EXP-202
was calibrated; the penalty only reorders the candidates.

Usage: PYTHONPATH=src python -m reid.view_head --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp208_view_head.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np

VIEWS = ["front", "back", "side"]


def mean_patch_tokens(encoder, run):
    """(result of run(), mean patch token per row) - the tokens are captured on the way into the attention pool.

    CUDA graphs replay captured kernels without firing hooks, and reid.benchmark.InferencePrecision uses one for
    batches of a single image, so the row count is checked against the result instead of trusted."""
    tower = encoder.encoder[0] if hasattr(encoder, "encoder") else encoder[0]
    store = []
    handle = tower.trunk.attn_pool.register_forward_pre_hook(
        lambda module, args: store.append(args[0].detach().float().mean(1).cpu().numpy()))
    try:
        result = run()
    finally:
        handle.remove()
    tokens = np.concatenate(store) if store else np.empty((0, 0), np.float32)
    if len(tokens) != len(result):
        raise RuntimeError(f"captured {len(tokens)} tokens for {len(result)} rows: hooks did not fire on every forward")
    return result, tokens


def views(tokens, head):
    """Hard view label per row."""
    return np.asarray(VIEWS)[(tokens @ head["weight"].T + head["bias"]).argmax(1)]


def penalty(q_views, g_views, delta):
    """nq x ng matrix to subtract from the similarity."""
    return delta * (q_views[:, None] == g_views[None]).astype(np.float32)


def save(path, weight, bias, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weight=weight, bias=bias, views=np.asarray(VIEWS), meta=np.asarray(json.dumps(meta)))


def load(path):
    saved = np.load(path, allow_pickle=False)
    if list(saved["views"]) != VIEWS:
        raise ValueError(f"the head was fitted for views {list(saved['views'])}")
    return {"weight": saved["weight"].astype(np.float32), "bias": saved["bias"].astype(np.float32),
            "meta": json.loads(str(saved["meta"]))}


def main():
    import pandas as pd
    import torch

    from reid.benchmark import InferencePrecision
    from reid.data import load_yaml
    from reid.embed import embed_boxes, pick_device
    from reid.inference import load_trained_encoder
    from reid.view_penalty import fit_view_head

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    device, spec = pick_device(), cfg["candidate"]
    model_cfg = load_yaml(spec["config"])["model"]
    encoder, _, mean, std = load_trained_encoder(model_cfg, Path(data_cfg["checkpoint_dir"]) / spec["checkpoint"],
                                                 data_cfg.get("external_weights_dir"))
    encoder = InferencePrecision(encoder.eval().to(device), cfg.get("precision", "fp16"), device)

    folds = pd.read_csv(Path(val_cfg["out_dir"]) / "folds.csv")
    rows = folds[folds.fold.isin(cfg["folds"])].reset_index(drop=True)
    images = Path(data_cfg["dataset_dir"]) / "images"
    _, tokens = mean_patch_tokens(encoder, lambda: embed_boxes(
        encoder, rows, images, model_cfg["input_hw"], mean, std, data_cfg["crop_max_side"], device=device,
        **cfg["embed"]))

    labels = pd.read_csv("protocol/attributes.csv").set_index("image_id").view
    code = {v: k for k, v in enumerate(VIEWS)}
    y = labels.reindex(rows.image_id).map(code).to_numpy()
    if np.isnan(y).any():
        raise ValueError("protocol/attributes.csv has no view for some training rows")
    h = cfg["head"]
    weight, bias = fit_view_head(tokens, y.astype(np.int64), h["epochs"], h["lr"], h["weight_decay"])
    agreement = float((views(tokens, {"weight": weight, "bias": bias}) == labels.reindex(rows.image_id).to_numpy()).mean())
    meta = {"exp_id": cfg["exp_id"], "folds": cfg["folds"], "rows": int(len(rows)), "source": spec["checkpoint"],
            "precision": cfg.get("precision", "fp16"), "train_agreement_with_clip": agreement, **h}
    out = Path(data_cfg["checkpoint_dir"]) / cfg["head_name"]
    if out.exists():
        raise FileExistsError(out)
    save(out, weight, bias, meta)
    report = {**meta, "file": str(out), "size_kb": round(out.stat().st_size / 1024, 1)}
    Path("artifacts", cfg["exp_id"]).mkdir(parents=True, exist_ok=True)
    Path("artifacts", cfg["exp_id"], "head.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
