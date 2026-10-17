"""Backbone construction for the ReID model configuration."""
import hashlib
from pathlib import Path

import open_clip
import timm
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from timm.data import resolve_data_config

from reid.teachers import load_fastreid

from ._model_pooling import MultiQueryPool, apply_drop_path


def build_backbone(cfg, external_weights_dir=None, load_pretrained=True):
    """(module mapping images to a global feature, feature dim, normalisation mean, std) for a model config."""
    if cfg["backbone"] == "fastreid_r50ibn":  # public FastReID checkpoint as the start; its BN head is dropped
        model, feat_dim, mean, std = load_fastreid(Path(external_weights_dir) / cfg["weights"])
        model.bottleneck = None  # ReIDModel adds a fresh BNNeck, as for the other backbones
        return model, feat_dim, mean, std
    if cfg["backbone"] == "clip":
        pretrained = cfg["pretrained"]
        kwargs = {}
        if cfg.get("weights") or not load_pretrained:
            metadata = open_clip.get_pretrained_cfg(cfg["name"], pretrained)
            kwargs = {f"image_{key}": metadata[key] for key in ("mean", "std", "interpolation", "resize_mode")
                      if key in metadata}
        if cfg.get("weights") and load_pretrained:
            path = Path(external_weights_dir) / cfg["weights"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != cfg["sha256"]:
                raise ValueError(f"checksum mismatch: {path}")
            # A local checkpoint does not carry the original tag's preprocessing metadata.
            pretrained = str(path)
        if not load_pretrained:
            pretrained = None  # a complete trained state follows; no extra public weights at inference
        if cfg.get("interpolate_positional_embedding"):
            kwargs["force_image_size"] = tuple(cfg["input_hw"])
        model, _, _ = open_clip.create_model_and_transforms(cfg["name"], pretrained=pretrained, **kwargs)
        preprocess = getattr(model.visual, "preprocess_cfg", None) or {}
        mean, std = preprocess.get("mean", OPENAI_DATASET_MEAN), preprocess.get("std", OPENAI_DATASET_STD)
        # timm-based towers (EVA02, SigLIP 2) have no output_dim; their projection ends at the model's embed_dim
        dim = getattr(model.visual, "output_dim", None) or open_clip.get_model_config(cfg["name"])["embed_dim"]
        if cfg.get("drop_path_rate"):
            apply_drop_path(model.visual.trunk, cfg["drop_path_rate"])
        if cfg.get("pool_queries", 1) > 1:
            trunk = model.visual.trunk
            trunk.attn_pool = MultiQueryPool(trunk.attn_pool, cfg["pool_queries"], cfg["pool_offset_norm"])
        return model.visual, dim, mean, std
    if cfg["backbone"] == "timm":
        # load_pretrained is False when a trained state follows: no point fetching public weights to overwrite
        model = timm.create_model(cfg["name"], pretrained=load_pretrained, num_classes=0)
        if cfg.get("last_stride_1"):  # keep a 2x larger final feature map, as in the ReID bag of tricks
            block = model.layer4[0]
            block.conv2.stride = (1, 1)
            block.downsample[0].stride = (1, 1)
        data_cfg = resolve_data_config({}, model=model)
        return model, model.num_features, data_cfg["mean"], data_cfg["std"]
    raise ValueError(f"unknown backbone {cfg['backbone']}")
