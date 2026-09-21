"""AdamW groups: the original recipe, or layer decay for native and timm-backed OpenCLIP ViTs."""
import re


def _module_parameters(members):
    """Flatten OpenCLIP layer_groups members, which can be Parameters or Modules."""
    parameters = []
    for member in members:
        if hasattr(member, "requires_grad"):  # torch.nn.Parameter, without importing torch just for this type check
            parameters.append(member)
        else:
            parameters.extend(member.parameters())
    return [parameter for parameter in parameters if parameter.requires_grad]


def _timm_layer_groups(backbone):
    """Layer groups for OpenCLIP 3.3's TimmModel, before TimmModel.layer_groups was added upstream."""
    try:
        from timm.models import group_parameters
    except ImportError:  # compatibility with older timm
        from timm.models.helpers import group_parameters
    named = group_parameters(backbone.trunk, backbone.trunk.group_matcher())
    layers = [(f"layer.{layer}", [backbone.trunk.get_parameter(name) for name in named[layer]])
              for layer in sorted(named)]
    if any(parameter.requires_grad for parameter in backbone.head.parameters()):
        layers.append(("proj", [backbone.head]))
    return layers


def optimizer_groups(model, cfg):
    head = [p for p in [*model.bnneck.parameters(), *model.classifier.parameters()] if p.requires_grad]
    for name in ("auxiliary_neck", "auxiliary_classifier"):
        module = getattr(model, name, None)
        if module is not None:
            head.extend(p for p in module.parameters() if p.requires_grad)
    # the query offsets of a multi-query pool start from scratch: at the backbone rate they would move ~3% in a run
    offsets = [p for name, p in model.backbone.named_parameters() if name.endswith("query_offsets")]
    head.extend(offsets)
    decay = float(cfg.get("layer_decay", 1.0))
    if not 0 < decay <= 1:
        raise ValueError("layer_decay must be in (0, 1]")
    if hasattr(model.backbone, "optimizer_groups"):
        if decay != 1:
            raise ValueError("Layer decay is not configured for AFEM/OSNet backbones")
        return model.backbone.optimizer_groups(cfg) + [{"params": head, "lr": cfg["lr_head"], "branch": "reid_heads"}]
    if decay == 1:
        return [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad and all(p is not o for o in offsets)],
             "lr": cfg["lr_backbone"]},
            {"params": head, "lr": cfg["lr_head"]},
        ]
    if hasattr(model.backbone, "layer_groups") or (hasattr(model.backbone, "trunk")
                                                    and hasattr(model.backbone.trunk, "group_matcher")):
        # New OpenCLIP supplies layer_groups; pinned 3.3 exposes the same native timm grouping through its trunk.
        layers = (model.backbone.layer_groups() if hasattr(model.backbone, "layer_groups")
                  else _timm_layer_groups(model.backbone))
        groups = []
        top = len(layers) - 1
        for layer, (name, members) in enumerate(layers):
            parameters = _module_parameters(members)
            if parameters:
                groups.append({"params": parameters, "lr": cfg["lr_backbone"] * decay ** (top - layer),
                               "layer": layer, "branch": name})
        covered = {id(parameter) for group in groups for parameter in group["params"]}
        expected = {id(parameter) for parameter in model.backbone.parameters() if parameter.requires_grad}
        if covered != expected or sum(len(group["params"]) for group in groups) != len(covered):
            raise ValueError("OpenCLIP layer_groups must cover every trainable backbone parameter exactly once")
        return groups + [{"params": head, "lr": cfg["lr_head"], "branch": "reid_heads"}]
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
