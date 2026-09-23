"""The viewpoint head of EXP-192/204 as a saved artefact, and the penalty it feeds at submission time.

The error analysis found that a wrong first answer shares the query's viewpoint 80% of the time against 37% for a
random gallery frame, so the score becomes sim - delta * [view(q) = view(g)]. The view comes from a softmax head on
the mean patch token, fitted against the CLIP zero-shot labels of protocol/attributes.csv; EXP-204 confirmed the
penalty on five leave-one-fold-out models (mean +0.0031 mAP@10, positive on all five). At inference the head costs one
1152x3 matrix product on a tensor the tower computes anyway: the tokens are captured where the attention pool receives
them, which is exactly what reid.view_tokens fitted on.

Refusal is decided on the encoder's own similarity, before the penalty, because that is how the threshold of EXP-202
was calibrated; the penalty only reorders the candidates.

The head is fitted once, offline, and shipped as a 15 KB file next to the checkpoint; the fitting CLI and the
labels it needs live on the `experiments` branch (reid.view_head there, configs/exp208_view_head.yaml).
"""
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
