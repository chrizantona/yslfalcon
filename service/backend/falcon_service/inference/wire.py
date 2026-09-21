"""Bounded NPZ transport: numeric arrays only and never pickle deserialization."""

import zipfile
from io import BytesIO

import numpy as np

from .core import FeatureBundle

MAX_WIRE_BYTES = 64 * 1024 * 1024


def pack_features(features, **extra):
    if not features or len({feature.model_id for feature in features}) != 1:
        raise ValueError("Wire features must share one model version")
    payload = {
        "embeddings": np.stack([item.embedding for item in features]).astype(np.float32),
        "model_id": np.asarray(features[0].model_id),
    }
    have_tokens = [item.tokens is not None for item in features]
    if any(have_tokens) and not all(have_tokens):
        raise ValueError("Mixed token availability")
    if all(have_tokens):
        payload["tokens"] = np.stack([item.tokens for item in features])
    payload.update(extra)
    stream = BytesIO()
    np.savez(stream, **payload)
    result = stream.getvalue()
    if len(result) > MAX_WIRE_BYTES:
        raise ValueError("Feature message exceeds 64 MiB")
    return result


def unpack_features(data):
    if not data or len(data) > MAX_WIRE_BYTES:
        raise ValueError("Invalid feature message size")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if (
                len(archive.infolist()) > 8
                or sum(item.file_size for item in archive.infolist()) > MAX_WIRE_BYTES
            ):
                raise ValueError("Feature archive expands beyond allowed limits")
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            if not {"embeddings", "model_id"} <= set(archive.files):
                raise ValueError("Incomplete feature message")
            vectors = archive["embeddings"]
            identity = archive["model_id"]
            if identity.shape != () or identity.dtype.kind != "U" or len(str(identity)) > 256:
                raise ValueError("Invalid model identity")
            if (
                vectors.dtype != np.float32
                or vectors.ndim != 2
                or not 1 <= len(vectors) <= 51
                or not 1 <= vectors.shape[1] <= 16384
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("Invalid embedding matrix")
            tokens = archive["tokens"] if "tokens" in archive.files else None
            if tokens is not None and (
                tokens.dtype.kind != "f"
                or tokens.ndim != 3
                or len(tokens) != len(vectors)
                or not np.isfinite(tokens).all()
            ):
                raise ValueError("Invalid token bank")
            extras = {}
            for key in set(archive.files) - {"embeddings", "model_id", "tokens"}:
                value = archive[key]
                if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                    raise ValueError("Invalid extra array")
                extras[key] = value.copy()
            features = [
                FeatureBundle(vector.copy(), None if tokens is None else tokens[i].copy(), str(identity))
                for i, vector in enumerate(vectors)
            ]
    except (OSError, zipfile.BadZipFile, KeyError, TypeError) as exc:
        raise ValueError("Invalid feature archive") from exc
    return features, extras
