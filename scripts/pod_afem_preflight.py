"""Actual PK batch, finite gradients, inference precision and VRAM checks before AFEM/OSNet runs.

An optional structural check without EXP-054 weights never authorises scientific training.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reid.afem_models import build_extended_model, OSNetAppearance
from reid.benchmark import InferencePrecision
from reid.data import load_yaml
from reid.optim import optimizer_groups
from reid.train import CropDataset, PKSampler, training_table, supervised_contrastive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--val-config", default="configs/val_v2.yaml")
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg, data, val = [load_yaml(p) for p in (args.config, args.data_config, args.val_config)]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    table, crops, classes = training_table(cfg, data, val)
    tcfg, mcfg = cfg["train"], cfg["model"]
    model, dim, mean, std = build_extended_model(mcfg, len(classes), data["external_weights_dir"],
                                               load_pretrained=not args.structural_only, head=tcfg["id_head"])
    # Even a structural fusion check uses verified OSNet weights, never person-specialised weights.
    if args.structural_only and mcfg["backbone"] == "siglip_osnet":
        model.backbone.appearance = OSNetAppearance(mcfg["appearance"], data["external_weights_dir"], True)
    model = model.cuda().train()
    groups = optimizer_groups(model, tcfg)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])
    dataset = CropDataset(table, crops, mcfg["input_hw"], mean, std, tcfg)
    sampler = PKSampler(table, tcfg["vehicles_per_batch"], tcfg["images_per_vehicle"], 0)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    x, y = next(iter(loader))
    x, y = x.cuda(), y.cuda()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(x, y)
        feat, logits = output[:2]
        loss = F.cross_entropy(logits.float(), y, label_smoothing=tcfg["label_smoothing"])
        loss = loss + supervised_contrastive(feat.float(), y, tcfg["metric_loss"]["temperature"])
        if len(output) == 4:
            aux = F.cross_entropy(output[3].float(), y, label_smoothing=tcfg["label_smoothing"])
            aux = aux + supervised_contrastive(output[2].float(), y, tcfg["metric_loss"]["temperature"])
            loss = loss + tcfg["auxiliary_weight"] * aux
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not missing, missing
        assert not nonfinite, nonfinite
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    train_peak = torch.cuda.max_memory_reserved()
    del optimizer, groups, output, feat, logits, loss
    for p in model.parameters():
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fp32 = model(x[:8]).float()
        assert torch.isfinite(fp32).all()
        # Training checkpoint is float32; these are conservative full-state sizes including unused auxiliary heads.
        state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        encoder = InferencePrecision(model, "fp16", "cuda").cuda().eval()
        batched = F.normalize(encoder(x[:8]).float(), dim=1)
        single = F.normalize(encoder(x[2:3]).float(), dim=1)
        repeated = F.normalize(encoder(x[2:3]).float(), dim=1)
        assert torch.isfinite(batched).all() and torch.isfinite(single).all()
        assert torch.equal(single, repeated), "inference is not deterministic"
        delta = (batched[2] - single[0]).abs().max().item()
        assert delta < 0.005, delta  # expected low-precision batch-kernel rounding, no query interaction
        large = encoder(x.repeat(2, 1, 1, 1))
        assert torch.isfinite(large).all() and len(large) == 256
    torch.cuda.synchronize()
    assert state_bytes <= 2_000_000_000, state_bytes
    receipt = {"exp_id": cfg["exp_id"], "config": args.config,
               "scientific_initialisation_verified": not args.structural_only,
               "batch": len(x), "train_losses": losses, "all_trainable_gradients_finite": True,
               "train_peak_reserved_bytes": train_peak,
               "infer_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "full_state_bytes": state_bytes, "embedding_dim": dim,
               "source_dim": getattr(model.backbone, "source_dim", None),
               "mean": mean, "std": std, "fp16_query_batch_rounding_max_abs": delta,
               "fp16_repeated_query_bit_identical": True, "fp16_batch256_finite": True,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
