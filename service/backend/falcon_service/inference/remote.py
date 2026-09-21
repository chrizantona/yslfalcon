"""HTTP adapter for the internal inference port; credentials never enter image URLs."""

import json
from urllib.parse import urlparse

import httpx
import numpy as np

from .images import validate_image
from .wire import MAX_WIRE_BYTES, pack_features, unpack_features


class RemoteEncoder:
    def __init__(self, url: str, token: str, timeout=60):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise RuntimeError("Runtime URL must be an HTTP(S) origin without credentials")
        if not token:
            raise RuntimeError("Runtime bearer token is required")
        self._client = httpx.Client(
            base_url=url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"},
            follow_redirects=False,
        )
        try:
            response = self._client.get("/internal/v1/capabilities")
            response.raise_for_status()
            capabilities = response.json()
            self.model_id, self.mode = capabilities["model_id"], capabilities["mode"]
            self.dimension = int(capabilities["dimension"])
            self.rerank_window = int(capabilities.get("rerank_window", 0))
            self.calibration_policy = capabilities.get("calibration_policy")
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self._client.close()
            raise RuntimeError("Model runtime is unavailable or returned invalid capabilities") from exc

    def _request(self, path, **kwargs):
        try:
            with self._client.stream("POST", path, **kwargs) as response:
                response.raise_for_status()
                chunks, count = [], 0
                for chunk in response.iter_bytes():
                    count += len(chunk)
                    if count > MAX_WIRE_BYTES:
                        raise RuntimeError("Runtime response exceeds the message limit")
                    chunks.append(chunk)
                return b"".join(chunks)
        except httpx.HTTPError as exc:
            raise RuntimeError("Model runtime request failed") from exc

    def encode(self, image: bytes, bbox=None):
        validate_image(image, bbox)
        data = self._request(
            "/internal/v1/encode", files={"image": ("upload", image)}, data={"bbox": json.dumps(bbox)}
        )
        features, _ = unpack_features(data)
        if (
            len(features) != 1
            or features[0].model_id != self.model_id
            or len(features[0].embedding) != self.dimension
        ):
            raise RuntimeError("Runtime changed model identity or feature shape")
        return features[0]

    def rerank(self, query, candidates, similarities):
        if not self.rerank_window:
            return np.asarray(similarities, dtype=np.float32).copy()
        body = pack_features([query, *candidates], similarities=np.asarray(similarities, dtype=np.float32))
        data = self._request(
            "/internal/v1/rerank", content=body, headers={"Content-Type": "application/octet-stream"}
        )
        returned, extras = unpack_features(data)
        if returned[0].model_id != self.model_id or "scores" not in extras:
            raise RuntimeError("Runtime returned invalid reranking response")
        return extras["scores"]

    def close(self):
        self._client.close()
