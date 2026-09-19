"""Train the local patch branch of reid.local_patches on the frozen EXP-054 soup and score it for several alphas.

The SigLIP 2 trunk, its MAP pooling and the global BNNeck (with its statistics) stay frozen in eval mode; only the local
head and its ArcFace classifier learn (EXP-053 recipe: imprinting, SupCon, PK sampler, augmentations, warm-up + cosine).
Type mean has nothing to learn and is only scored (EXP-083). Evaluation is fp32: at alpha 0 it reproduces EXP-054.

One seed per call. Every alpha of model.local.alphas is written to artifacts/<exp_id>/alpha_<a>/seed<k>/ (the files of
reid.evaluate); model.local.alpha, fixed in the config beforehand, is also written to artifacts/<exp_id>/seed<k>/ for
reid.aggregate and the pod job. The head (a few MB) goes to checkpoint_dir under the configured name.

Usage: PYTHONPATH=src python -m reid.train_local --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp084_exp054_local_attention.yaml --seed 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from reid.aggregate import summarise
from reid.data import BoxDataset, load_yaml
from reid.embed import _worker_init as _embed_worker_init
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.local_patches import LocalPatchHead, combine, load_base, tokens_and_global
from reid.models import checkpoint_name
from reid.train import CropDataset, PKSampler, _worker_init, supervised_contrastive, training_table, warmup_cosine


@torch.no_grad()
def embed_parts(encoder, head, df, images_dir, cfg, data_cfg, mean, std, device):
    """Unit-length (g, l) for the rows of df, fp32."""
    head.eval()
    loader = DataLoader(BoxDataset(df, images_dir, cfg["model"]["input_hw"], mean, std, data_cfg["crop_max_side"]),
                        batch_size=cfg["embed"]["batch_size"], num_workers=cfg["embed"]["num_workers"],
                        worker_init_fn=_embed_worker_init)
    g_parts, l_parts = [], []
    for x in loader:
        tokens, g = tokens_and_global(encoder, x.to(device))
        g_parts.append(torch.nn.functional.normalize(g.float(), dim=1).cpu())
        l_parts.append(head.embedding(tokens).cpu())
    return torch.cat(g_parts), torch.cat(l_parts)


@torch.no_grad()
def imprint(encoder, head, loader, device):
    """ArcFace directions = normalised mean local feature per class after the initial BNNeck (train images only)."""
    head.eval()
    sums = torch.zeros_like(head.classifier.weight, device=device)
    counts = torch.zeros(len(sums), device=device)
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, _ = tokens_and_global(encoder, x.to(device))
            emb = torch.nn.functional.normalize(head.neck(head.features(tokens)).float(), dim=1)
        sums.index_add_(0, y.to(device), emb)
        counts += torch.bincount(y.to(device), minlength=len(counts))
    head.classifier.weight.copy_(torch.nn.functional.normalize(sums / counts[:, None], dim=1))
    return {"classes": len(counts), "images": int(counts.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = load_yaml(args.data_config), load_yaml(args.val_config), load_yaml(args.config)
    tcfg, local, seed, device = cfg["train"], cfg["model"]["local"], args.seed, "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    encoder, dim, mean, std = load_base(cfg["model"], data_cfg["checkpoint_dir"], data_cfg.get("external_weights_dir"))
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    train_df, crops_dir, classes = training_table(cfg, data_cfg, val_cfg)
    query, gallery, no_match = load_fold(Path(val_cfg["out_dir"]), cfg["eval_fold"])
    head = LocalPatchHead(dim, local, len(classes), tcfg["id_head"]).to(device)
    out_root = Path("artifacts") / cfg["exp_id"]
    header = {"exp_id": cfg["exp_id"], "seed": seed, "protocol": val_cfg["version"], "fold": cfg["eval_fold"],
              "n_query": len(query), "n_gallery": len(gallery)}
    start = time.time()

    if local["type"] == "attention":
        params = [p for p in head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=tcfg["lr_head"], weight_decay=tcfg["weight_decay"])
        P, K = tcfg["vehicles_per_batch"], tcfg["images_per_vehicle"]
        sampler = PKSampler(train_df, P, K, seed)
        loader = DataLoader(CropDataset(train_df, crops_dir, cfg["model"]["input_hw"], mean, std, tcfg),
                            batch_sampler=sampler, num_workers=tcfg["num_workers"], pin_memory=True,
                            persistent_workers=True, worker_init_fn=_worker_init,
                            generator=torch.Generator().manual_seed(seed))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine(tcfg["epochs"] * len(sampler), tcfg["warmup_epochs"] * len(sampler)))
        cross_entropy = torch.nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
        plain = DataLoader(CropDataset(train_df, crops_dir, cfg["model"]["input_hw"], mean, std,
                                       {**tcfg, "flip_p": 0, "pad": 0, "erasing_p": 0}),
                           batch_size=tcfg["id_head"]["imprint_batch_size"], num_workers=tcfg["num_workers"],
                           worker_init_fn=_worker_init)
        header["imprinting"] = imprint(encoder, head, plain, device)
        out_root.joinpath(f"seed{seed}").mkdir(parents=True, exist_ok=True)
        log = (out_root / f"seed{seed}" / "train_log.jsonl").open("w")
        print(f"{len(classes)} vehicles, {len(train_df)} images, {len(sampler)} batches/epoch, "
              f"{sum(p.numel() for p in params) / 1e6:.2f}M trainable parameters", flush=True)
        for epoch in range(1, tcfg["epochs"] + 1):
            sampler.set_epoch(epoch)
            head.train()
            totals, steps, epoch_start = np.zeros(3), 0, time.time()
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        tokens, _ = tokens_and_global(encoder, x)
                    feat = head.features(tokens)
                    logits = head.classifier(head.neck(feat), y)
                loss_id = cross_entropy(logits.float(), y)
                loss_supcon = supervised_contrastive(feat.float(), y, tcfg["metric_loss"]["temperature"])
                loss = loss_id + loss_supcon
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"loss {loss.item()} at epoch {epoch}, step {steps}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                totals += [loss_id.item(), loss_supcon.item(), (logits.argmax(1) == y).float().mean().item()]
                steps += 1
            record = {"epoch": epoch, "loss_id": totals[0] / steps, "loss_supcon": totals[1] / steps,
                      "train_acc": totals[2] / steps, "lr": optimizer.param_groups[0]["lr"],
                      "img_per_s": steps * P * K / (time.time() - epoch_start), "elapsed_min": (time.time() - start) / 60}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    header["train_minutes"] = (time.time() - start) / 60

    # the head alone; the base soup is referenced by name and checksum in the config
    checkpoints = Path(data_cfg["checkpoint_dir"])
    name = checkpoint_name(cfg, seed)
    if (checkpoints / name).exists():
        raise FileExistsError(checkpoints / name)
    torch.save({"exp_id": cfg["exp_id"], "seed": seed, "config": cfg, "state_dict": head.state_dict()},
               checkpoints / name)
    header["checkpoint"] = name
    header["checkpoint_mb"] = round((checkpoints / name).stat().st_size / 1e6, 1)

    images_dir = Path(data_cfg["dataset_dir"]) / "images"
    parts = {id(df): embed_parts(encoder, head, df, images_dir, cfg, data_cfg, mean, std, device)
             for df in (query, gallery)}
    for alpha in local["alphas"]:
        embeddings = {key: combine(g, l, alpha).numpy() for key, (g, l) in parts.items()}
        result = evaluate_fold(lambda df: embeddings[id(df)], query, gallery, no_match)
        metrics = save_fold_results(out_root / f"alpha_{alpha:.2f}" / f"seed{seed}", result, query,
                                    {**header, "alpha": alpha}, emb_dtype=np.float16)
        print(json.dumps({"alpha": alpha, **{k: metrics["retrieval"][k] for k in ("mAP10", "R1", "R5")},
                          "refusal": metrics["refusal"]["score"]}), flush=True)
        if alpha == local["alpha"]:
            save_fold_results(out_root / f"seed{seed}", result, query, {**header, "alpha": alpha},
                              emb_dtype=np.float16)
    # per-alpha seed summaries once every seed of the config is written (each seed process tries; the last one wins)
    for alpha in local["alphas"]:
        root = out_root / f"alpha_{alpha:.2f}"
        seeds = [s for s in cfg["seeds"] if (root / f"seed{s}" / "metrics.json").exists()]
        if len(seeds) == len(cfg["seeds"]):
            summarise(root, [root / f"seed{s}" for s in seeds], cfg["exp_id"], seeds)


if __name__ == "__main__":
    main()
