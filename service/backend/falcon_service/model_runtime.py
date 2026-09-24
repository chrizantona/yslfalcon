"""Separate internal runtime: the preview descriptor, or the trained model with FALCON_ENCODER=production."""

import json
import os
import secrets
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from .inference import PreviewEncoder, RepositoryEncoder
from .inference.images import MAX_IMAGE_BYTES
from .inference.wire import MAX_WIRE_BYTES, pack_features, unpack_features
from .runtime_schemas import (
    RERANK_REQUEST,
    RUNTIME_ERRORS,
    RuntimeCapabilities,
    RuntimeLiveResponse,
    feature_response,
)


def create_app(encoder=None, token=None):
    state = {"encoder": encoder, "token": token}
    try:
        max_concurrency = int(os.environ.get("FALCON_RUNTIME_MAX_CONCURRENCY", "2"))
    except ValueError as exc:
        raise RuntimeError("FALCON_RUNTIME_MAX_CONCURRENCY must be an integer") from exc
    if not 1 <= max_concurrency <= 128:
        raise RuntimeError("FALCON_RUNTIME_MAX_CONCURRENCY must be between 1 and 128")
    capacity = threading.BoundedSemaphore(max_concurrency)

    @asynccontextmanager
    async def lifespan(app):
        state["token"] = state["token"] or os.environ.get("FALCON_RUNTIME_TOKEN")
        if not state["token"] or len(state["token"]) < 24:
            raise RuntimeError("FALCON_RUNTIME_TOKEN must contain at least 24 characters")
        if state["encoder"] is None:
            mode = os.environ.get("FALCON_ENCODER", "preview")
            if mode == "preview":
                state["encoder"] = PreviewEncoder()
            elif mode == "production":
                manifest = os.environ.get("FALCON_MODEL_MANIFEST")
                if not manifest:
                    raise RuntimeError("FALCON_MODEL_MANIFEST is required by the production encoder")
                state["encoder"] = RepositoryEncoder(manifest, os.environ.get("FALCON_MODEL_DEVICE", "cuda"))
            else:
                raise RuntimeError(f"Unknown FALCON_ENCODER {mode!r}: expected preview or production")
        yield

    app = FastAPI(
        title="Falcon internal preview runtime",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        responses=RUNTIME_ERRORS,
    )
    bearer = HTTPBearer(auto_error=False, scheme_name="RuntimeBearer")

    def authenticate(authorization: HTTPAuthorizationCredentials | None = Security(bearer)):
        supplied = authorization.credentials if authorization and authorization.scheme == "Bearer" else ""
        if not state["token"] or not secrets.compare_digest(supplied.encode(), state["token"].encode()):
            raise HTTPException(401, "Runtime authentication required")

    def execute(callable_, *args):
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Runtime capacity exhausted", headers={"Retry-After": "1"})
        try:
            return callable_(*args)
        finally:
            capacity.release()

    @app.get("/health/live", response_model=RuntimeLiveResponse)
    def live():
        return {"status": "ok"}

    @app.get(
        "/internal/v1/capabilities", dependencies=[Depends(authenticate)], response_model=RuntimeCapabilities
    )
    def capabilities():
        current = state["encoder"]
        return {
            "mode": current.mode,
            "model_id": current.model_id,
            "dimension": current.dimension,
            "rerank_window": current.rerank_window,
            "calibration_policy": current.calibration_policy,
            "real_model_connected": current.real_model_connected,
            "device": current.device,
            "precision": current.precision,
            "weights_bytes": current.weights_bytes,
        }

    @app.post(
        "/internal/v1/encode",
        dependencies=[Depends(authenticate)],
        response_class=Response,
        responses=feature_response("One normalized embedding (N=1) for the validated image crop."),
    )
    async def encode(image: UploadFile = File(...), bbox: str | None = Form(default=None)):
        data = await image.read(MAX_IMAGE_BYTES + 1)
        await image.close()
        try:
            box = json.loads(bbox) if bbox is not None else None
            feature = await run_in_threadpool(execute, state["encoder"].encode, data, box)
            return Response(pack_features([feature]), media_type="application/octet-stream")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post(
        "/internal/v1/rerank",
        dependencies=[Depends(authenticate)],
        response_class=Response,
        responses=feature_response("The query feature (N=1) plus scores float32[candidate_count]."),
        openapi_extra=RERANK_REQUEST,
    )
    async def rerank(request: Request):
        chunks, length = [], 0
        async for chunk in request.stream():
            length += len(chunk)
            if length > MAX_WIRE_BYTES:
                raise HTTPException(413, "Feature message too large")
            chunks.append(chunk)
        try:
            features, extras = unpack_features(b"".join(chunks))
            if len(features) < 2 or features[0].model_id != state["encoder"].model_id:
                raise ValueError("Incompatible feature message")
            similarities = extras.get("similarities")
            if similarities is None or similarities.shape != (len(features) - 1,):
                raise ValueError("Invalid similarities vector")
            scores = await run_in_threadpool(
                execute, state["encoder"].rerank, features[0], features[1:], similarities
            )
            return Response(
                pack_features([features[0]], scores=scores), media_type="application/octet-stream"
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
