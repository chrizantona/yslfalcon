"""HTTP adapter for the internal inference port; credentials never enter image URLs."""

import json
from dataclasses import replace
from urllib.parse import urlparse

import httpx
import numpy as np

from ..logging_setup import request_id_context
from ..runtime_schemas import RuntimeCapabilities
from .images import validate_image
from .wire import MAX_WIRE_BYTES, pack_features, unpack_features


class RemoteEncoder:
    def __init__(self, url: str, token: str, timeout=60, *, handshake_timeout: float | None = None):
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
            response = self._client.get(
                "/internal/v1/capabilities",
                timeout=timeout if handshake_timeout is None else handshake_timeout,
            )
            response.raise_for_status()
            capabilities = RuntimeCapabilities.model_validate(response.json())
            self._capabilities = capabilities
            self.model_id, self.mode = capabilities.model_id, capabilities.mode
            self.dimension = capabilities.dimension
            self.rerank_window = capabilities.rerank_window
            self.calibration_policy = capabilities.calibration_policy
            self.real_model_connected = capabilities.real_model_connected
            self.device = capabilities.device
            self.precision = capabilities.precision
            self.weights_bytes = capabilities.weights_bytes
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self._client.close()
            raise RuntimeError("Model runtime is unavailable or returned invalid capabilities") from exc

    def check_ready(self) -> None:
        """Probe the current runtime without replacing the active model contract."""
        request_id = request_id_context.get()
        headers = {"X-Request-ID": request_id} if request_id else {}
        try:
            response = self._client.get("/internal/v1/capabilities", timeout=3, headers=headers)
            response.raise_for_status()
            capabilities = RuntimeCapabilities.model_validate(response.json())
            if capabilities != self._capabilities:
                raise ValueError("Runtime capabilities changed after initialization")
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise RuntimeError("Model runtime is unavailable or its capabilities changed") from exc

    def _request(self, path, **kwargs):
        request_id = request_id_context.get()
        if request_id:
            kwargs["headers"] = {**kwargs.get("headers", {}), "X-Request-ID": request_id}
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
        features, extras = unpack_features(data)
        if (
            len(features) != 1
            or features[0].model_id != self.model_id
            or len(features[0].embedding) != self.dimension
        ):
            raise RuntimeError("Runtime changed model identity or feature shape")
        forward = extras.get("forward_ms")
        if forward is None:
            return features[0]
        if forward.shape != () or not 0 <= float(forward) < 600_000:
            raise RuntimeError("Runtime returned an invalid forward time")
        return replace(features[0], forward_ms=float(forward))

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
