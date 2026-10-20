"""Offline model adapters. The preview descriptor never makes identity claims."""

import hashlib
import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .core import CONFIDENCE_KIND, SCORE_DEFINITION, FeatureBundle, normalise
from .images import crop_image, validate_image


class PreviewEncoder:
    mode = "preview"
    model_id = "preview-rgb-layout-v1-NOT-REID"
    dimension = 192
    rerank_window = 0
    calibration_policy = None
    real_model_connected = False
    device = "cpu"
    precision = "float32"
    weights_bytes = 0

    def encode(self, image: bytes, bbox=None):
        with Image.open(BytesIO(crop_image(image, bbox))) as frame:
            values = np.asarray(frame.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32)
        # Positive floor gives even a fully black image a defined unit descriptor.
        return FeatureBundle(normalise((values / 255.0 + 0.01).ravel()), None, self.model_id)

    def rerank(self, query, candidates, similarities):
        return np.asarray(similarities, dtype=np.float32).copy()


def view_rerank(similarities, query_view, candidate_views, delta):
    """Order a window by sim - delta * [same viewpoint], then hand the incoming values back in that order.

    The error analysis behind the model found that a wrong first answer shares the query's viewpoint 80% of
    the time against 37% for a random frame, so a candidate shot from the query's own side is pushed down by
    delta. The service accepts only a permutation of the scores it sent, which keeps the calibrated confidence
    scale intact; the penalty therefore decides the order and nothing else. Candidates without a view (-1)
    are never penalised."""
    similarities = np.asarray(similarities, dtype=np.float32)
    views = np.asarray(candidate_views)
    if similarities.ndim != 1 or views.shape != similarities.shape:
        raise RuntimeError("Invalid viewpoint rerank input")
    same = (views == query_view) & (views >= 0) & (query_view >= 0)
    order = np.argsort(-(similarities - np.float32(delta) * same), kind="stable")
    result = np.empty_like(similarities)
    result[order] = np.sort(similarities)[::-1]
    return result


def _view_index(feature):
    tokens = feature.tokens
    if tokens is None or np.asarray(tokens).size == 0:
        return -1
    return int(np.asarray(tokens).reshape(-1).argmax())


class RepositoryEncoder:
    """The trained model behind the inference port.

    SigLIP 2 So400m/14 at 224, fine-tuned for vehicle re-identification, run in fp16, plus a 1152x3 softmax
    head that reads the viewpoint from the mean patch token the tower computes anyway. Everything comes
    from the bundle manifest and the model configuration shipped with the runtime image: nothing is
    downloaded, and a missing or altered file, or a GPU that is not there, stops the startup instead of
    degrading to the preview. The model code lives in the runtime image only (service/runtime), so this
    module imports it lazily and the API image stays free of PyTorch."""

    mode = "production"
    real_model_connected = True
    rerank_window = 50  # the service caps the window at 50; the penalty moves a candidate by at most delta
    # Galleries the refusal threshold was validated on: single-gallery folds of 636-793 frames, test 750.
    gallery_size_range = (500, 1000)

    def __init__(self, manifest_path: str, device="cuda"):
        manifest, root, weights_bytes = self.verified_bundle(manifest_path)

        import torch
        import yaml
        from reid import view_head
        from reid.benchmark import InferencePrecision
        from reid.inference import load_trained_encoder

        config_path = Path(os.environ.get("FALCON_MODEL_CONFIG", "/app/runtime/final_model.yaml"))
        cfg = yaml.safe_load(config_path.read_text())
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("FALCON_MODEL_DEVICE asks for CUDA, but no GPU is visible to the runtime")
        torch_device = "cuda" if device.startswith("cuda") else "cpu"
        precision = cfg.get("inference", {}).get("precision", "fp16") if torch_device == "cuda" else "fp32"
        encoder, dimension, mean, std = load_trained_encoder(cfg["model"], root / cfg["checkpoint_name"])
        self._encoder = InferencePrecision(encoder.eval().to(torch_device), precision, torch_device)
        self._head = view_head.load(root / cfg["view"]["head"])
        self._delta = float(cfg["view"]["delta"])
        self._mean, self._std, self._input_hw = mean, std, cfg["model"]["input_hw"]
        self._max_side, self._torch, self._device_name = 512, torch, torch_device
        self._lock = threading.Lock()  # the token hook is module-wide: one forward at a time
        self._view_head = view_head

        self.model_id = manifest["model_id"]
        self.dimension = int(dimension)
        self.device = "cuda:0" if torch_device == "cuda" else "cpu"
        self.precision = {"fp16": "float16", "bf16": "bfloat16"}.get(precision, "float32")
        self.weights_bytes = int(weights_bytes)
        lo, hi = self.gallery_size_range
        self.calibration_policy = {
            "model_id": self.model_id,
            "confidence_kind": CONFIDENCE_KIND,
            "score_definition": SCORE_DEFINITION,
            "threshold": float(cfg["refusal"]["threshold"]),
            "gallery_size_min": lo,
            "gallery_size_max": hi,
            "validation_reference": "leave-one-fold-out calibration on five folds (EXP-209): plateau centre of "
            "the top-1 z-score on a gallery without the query's own-camera frames, as in the closed test, plus the "
            "seed-to-soup offset",
        }
        warmup = BytesIO()
        Image.new("RGB", (64, 48), (90, 110, 130)).save(warmup, "PNG")
        self.encode(warmup.getvalue())  # a startup that cannot embed must fail here, not on the first request

    @staticmethod
    def verified_bundle(manifest_path):
        """(manifest, directory, weight bytes) after checking every listed file, before any model code loads.

        Runs without PyTorch, so a missing, truncated or altered bundle is rejected the same way everywhere."""
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise RuntimeError(f"Runtime bundle manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1 or not manifest.get("model_id"):
            raise RuntimeError("runtime-bundle.json must use schema_version 1 and name the model")
        root, weights_bytes = manifest_path.parent, 0
        for item in manifest["files"]:
            name = item["path"]
            if Path(name).name != name:
                raise RuntimeError(f"Bundle paths must be plain file names: {name}")
            path = root / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"]:
                raise RuntimeError(f"Bundle file is missing or has the wrong size: {name}")
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    checksum.update(chunk)
            if checksum.hexdigest() != item["sha256"]:
                raise RuntimeError(f"Bundle file failed its sha256 check: {name}")
            if item.get("role") == "weight":
                weights_bytes += item["bytes"]
        if weights_bytes != manifest.get("total_weights_bytes"):
            raise RuntimeError("Bundle weight total does not match its files")
        return manifest, root, weights_bytes

    def _forward(self, batch):
        """Eager forward: the CUDA graph used for single images would replay without firing the token hook."""
        with self._torch.no_grad():
            output = self._encoder.eager(batch).float()
        return self._torch.nn.functional.normalize(output, dim=-1).cpu().numpy()

    def encode(self, image: bytes, bbox=None):
        import cv2
        from reid.data import crop_box, to_model_input

        original, box, _, _ = validate_image(image, bbox)
        frame = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Image could not be decoded")
        batch = to_model_input(crop_box(frame, box, self._max_side), self._input_hw, self._mean, self._std)
        batch = batch[None].to(self._device_name)
        with self._lock:
            if self._device_name == "cuda":
                self._torch.cuda.synchronize()  # the clock starts on an idle device with the batch in place
            started = time.perf_counter()
            embedding, tokens = self._view_head.mean_patch_tokens(self._encoder, lambda: self._forward(batch))
            # _forward returns host memory, so the clock stops once the device has finished
            forward_ms = (time.perf_counter() - started) * 1000
        logits = tokens @ self._head["weight"].T + self._head["bias"]
        probabilities = np.exp(logits - logits.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        return FeatureBundle(
            normalise(embedding[0]), probabilities.astype(np.float32), self.model_id, forward_ms
        )

    def rerank(self, query, candidates, similarities):
        views = np.array([_view_index(item) for item in candidates])
        return view_rerank(similarities, _view_index(query), views, self._delta)


def redistribute_scores(similarities, logits, weight):
    """Same rank blending and value permutation as reid.rerank_pairs.reorder."""
    similarities, logits = np.asarray(similarities), np.asarray(logits)
    if similarities.shape != logits.shape or similarities.ndim != 1 or not np.isfinite(logits).all():
        raise RuntimeError("Invalid pairwise logits")
    encoder_rank = np.argsort(np.argsort(-similarities))
    pair_rank = np.argsort(np.argsort(-logits))
    mixed = (1 - weight) * encoder_rank + weight * pair_rank
    places = np.argsort(np.argsort(mixed, kind="stable"))
    return np.sort(similarities)[::-1][places].astype(np.float32)
