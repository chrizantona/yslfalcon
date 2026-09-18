"""AdamW groups: the original recipe, or layer decay for an OpenCLIP ViT."""
import re


def optimizer_groups(model, cfg):
    head = [p for p in [*model.bnneck.parameters(), *model.classifier.parameters()] if p.requires_grad]
    for name in ("auxiliary_neck", "auxiliary_classifier"):
        module = getattr(model, name, None)
        if module is not None:
            head.extend(p for p in module.parameters() if p.requires_grad)
    decay = float(cfg.get("layer_decay", 1.0))
    if not 0 < decay <= 1:
        raise ValueError("layer_decay must be in (0, 1]")
    if hasattr(model.backbone, "optimizer_groups"):
        if decay != 1:
            raise ValueError("Layer decay is not configured for AFEM/OSNet backbones")
        return model.backbone.optimizer_groups(cfg) + [{"params": head, "lr": cfg["lr_head"], "branch": "reid_heads"}]
    if decay == 1:
        return [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg["lr_backbone"]},
            {"params": head, "lr": cfg["lr_head"]},
        ]
    blocks = model.backbone.transformer.resblocks
    depth = len(blocks)
    groups = {}
    for name, parameter in model.backbone.named_parameters():
        if not parameter.requires_grad:
            continue
        match = re.match(r"transformer\.resblocks\.(\d+)\.", name)
        if match:
            layer = int(match.group(1)) + 1
            if not 1 <= layer <= depth:
                raise ValueError(f"unknown transformer block: {name}")
        elif name.startswith(("conv1.", "ln_pre.")) or name in ("class_embedding", "positional_embedding"):
            layer = 0
        elif name.startswith("ln_post.") or name == "proj":
            layer = depth
        else:
            raise ValueError(f"unmapped ViT parameter: {name}")
        group = groups.setdefault(layer, {"params": [], "lr": cfg["lr_backbone"] * decay ** (depth - layer),
                                          "layer": layer})
        group["params"].append(parameter)
    return [groups[layer] for layer in sorted(groups)] + [{"params": head, "lr": cfg["lr_head"]}]
