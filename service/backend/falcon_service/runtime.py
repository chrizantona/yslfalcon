"""One encoder factory shared by API and durable workers."""

import json
import threading

from .config import Settings
from .inference import Encoder


class EncoderRuntime:
    def __init__(self, settings: Settings, encoder: Encoder | None = None):
        self.settings = settings
        self.encoder = encoder
        self.error: str | None = None
        self.lock = threading.RLock()
        self.policy = None

    def initialize(self) -> None:
        from .inference import PreviewEncoder, RemoteEncoder, RepositoryEncoder

        try:
            if self.encoder is None:
                if self.settings.engine_mode == "preview":
                    self.encoder = PreviewEncoder()
                elif self.settings.engine_mode == "remote":
                    self.encoder = RemoteEncoder(self.settings.runtime_url, self.settings.runtime_token)
                elif self.settings.model_manifest:
                    self.encoder = RepositoryEncoder(self.settings.model_manifest, self.settings.model_device)
                else:
                    raise RuntimeError("FALCON_MODEL_MANIFEST is required in repository mode")
            if self.settings.expected_model_id and self.encoder.model_id != self.settings.expected_model_id:
                raise RuntimeError("Runtime model identity does not match FALCON_EXPECTED_MODEL_ID")
            if (
                self.settings.expected_weights_bytes is not None
                and self.encoder.weights_bytes != self.settings.expected_weights_bytes
            ):
                raise RuntimeError("Runtime weight inventory does not match FALCON_EXPECTED_WEIGHTS_BYTES")
            self.policy = getattr(self.encoder, "calibration_policy", None)
            if self.settings.calibration_path:
                self.policy = json.loads(self.settings.calibration_path.read_text())
            self.error = None
        except Exception as exc:
            self.error = str(exc)
            self.encoder = None

    def require(self) -> Encoder:
        from .errors import ServiceError

        if self.encoder is None and self.settings.engine_mode == "remote":
            with self.lock:
                if self.encoder is None:
                    self.initialize()
        if self.encoder is None:
            raise ServiceError(
                "model_unavailable", "Модель недоступна: проверьте конфигурацию и локальные веса", 503
            )
        return self.encoder

    def capabilities(self) -> dict:
        encoder = self.encoder
        return {
            "mode": getattr(encoder, "mode", self.settings.engine_mode),
            "model_id": getattr(encoder, "model_id", None),
            "dimension": getattr(encoder, "dimension", None),
            "ready": encoder is not None,
            "real_model_connected": bool(getattr(encoder, "real_model_connected", False)),
            "device": getattr(encoder, "device", None),
            "precision": getattr(encoder, "precision", None),
            "weights_bytes": getattr(encoder, "weights_bytes", None),
        }
