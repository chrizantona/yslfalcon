"""Local immutable artifact store; user filenames are never filesystem paths."""

import io
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from .errors import ServiceError

if TYPE_CHECKING:
    from .inference import FeatureBundle

_KEY = re.compile(r"^[a-z]+/[a-f0-9]{32}\.[a-z0-9]+$")


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise ServiceError("invalid_artifact", "Некорректный идентификатор файла", 404)
        target = (self.root / key).resolve()
        if not target.is_relative_to(self.root):
            raise ServiceError("invalid_artifact", "Некорректный идентификатор файла", 404)
        return target

    def put(self, data: bytes, category: str, extension: str) -> str:
        key = f"{category}/{uuid4().hex}.{extension}"
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique files are never overwritten; publishing a DB reference happens later.
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return key

    def read(self, key: str) -> bytes:
        try:
            return self.path(key).read_bytes()
        except FileNotFoundError as exc:
            raise ServiceError("artifact_unavailable", "Сохранённый файл недоступен", 410) from exc

    def put_feature(self, feature: "FeatureBundle") -> str:
        data = io.BytesIO()
        values = {"embedding": feature.embedding.astype(np.float32), "model_id": np.array(feature.model_id)}
        if feature.tokens is not None:
            values["tokens"] = feature.tokens
        np.savez_compressed(data, **values)
        return self.put(data.getvalue(), "features", "npz")

    def read_feature(self, key: str) -> "FeatureBundle":
        from .inference import FeatureBundle

        with np.load(io.BytesIO(self.read(key)), allow_pickle=False) as stored:
            return FeatureBundle(
                embedding=stored["embedding"],
                tokens=stored["tokens"] if "tokens" in stored.files else None,
                model_id=str(stored["model_id"].item()),
            )
