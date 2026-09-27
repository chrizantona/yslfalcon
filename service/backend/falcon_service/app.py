"""Application composition root; no credentials or models are created on import."""

import html
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from . import __version__
from .auth import router as auth_router
from .config import Settings
from .db import Database
from .errors import ServiceError
from .indexes import GalleryIndexStore
from .logging_setup import configure_logging
from .response_schemas import ERROR_RESPONSES
from .routes import artifacts, galleries, searches, system
from .runtime import EncoderRuntime
from .storage import ArtifactStore

logger = logging.getLogger(__name__)
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")


class RequestLimits:
    """Enforce total streamed body size before multipart buffering can grow unbounded."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        limit = (
            min(self.limit, 4 * 1024 * 1024)
            if headers.get(b"content-type", b"").startswith(b"application/json")
            else self.limit
        )
        length = headers.get(b"content-length")
        try:
            too_large = length is not None and (int(length) > limit or int(length) < 0)
        except ValueError:
            too_large = True
        if too_large:
            response = JSONResponse(
                {
                    "error": {"code": "request_too_large", "message": "Превышен размер запроса"},
                    "request_id": scope.get("state", {}).get("request_id", str(uuid4())),
                },
                status_code=413,
            )
            return await response(scope, receive, send)
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    scope["falcon_body_too_large"] = True
                    raise ServiceError("request_too_large", "Превышен размер запроса", 413)
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None, encoder=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application):
        configure_logging()
        application.state.settings = settings
        application.state.db = Database(settings.database_url)
        application.state.db.initialize()
        application.state.store = ArtifactStore(settings.storage_root)
        application.state.indexes = GalleryIndexStore(
            application.state.db, application.state.store, settings.index_cache_size
        )
        application.state.runtime = EncoderRuntime(settings, encoder)
        await run_in_threadpool(application.state.runtime.initialize)
        try:
            yield
        finally:
            application.state.db.close()

    application = FastAPI(
        title="Falcon visual search service",
        version=__version__,
        lifespan=lifespan,
        description="Workspace-scoped service. Preview mode has no connected vehicle ReID model.",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
        responses=ERROR_RESPONSES,
    )
    application.add_middleware(RequestLimits, limit=settings.max_request_bytes)

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        supplied_id = request.headers.get("x-request-id", "")
        request.state.request_id = supplied_id if REQUEST_ID.fullmatch(supplied_id) else str(uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        route = request.scope.get("route")
        logger.info(
            "http_request",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "route": getattr(route, "path", "unmatched"),
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )
        return response

    def error_response(request: Request, code: str, message: str, status: int):
        request_id = getattr(request.state, "request_id", str(uuid4()))
        return JSONResponse(
            {
                "error": {"code": code, "message": message},
                "request_id": request_id,
            },
            status_code=status,
            headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
        )

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return error_response(request, exc.code, exc.message, exc.status)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _exc: RequestValidationError):
        # Pydantic's default payload echoes input, including login credentials.
        return error_response(
            request, "invalid_request", "Некорректные поля запроса; сверьтесь с OpenAPI", 422
        )

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.scope.get("falcon_body_too_large"):
            return error_response(request, "request_too_large", "Превышен размер запроса", 413)
        return error_response(request, "http_error", str(exc.detail), exc.status_code)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.error(
            "http_unexpected_error",
            extra={"request_id": request.state.request_id, "error_type": type(exc).__name__},
        )
        return error_response(
            request, "internal_error", "Ошибка сервиса; используйте request_id для диагностики", 500
        )

    for router in [auth_router, galleries.router, searches.router, artifacts.router, system.router]:
        application.include_router(router, prefix="/api/v1")

    @application.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    def offline_docs():
        schema = application.openapi()

        def code(value):
            return "<pre>" + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"

        operations_html = []
        for path, operations in schema["paths"].items():
            for method, operation in operations.items():
                security = "Bearer token or session cookie" if operation.get("security") else "Public"
                responses = "".join(
                    f"<details><summary>{status} — {html.escape(response.get('description', ''))}</summary>"
                    + code(response)
                    + "</details>"
                    for status, response in operation["responses"].items()
                )
                operations_html.append(
                    f"<details class='operation'><summary><b>{html.escape(method.upper())}</b> "
                    f"<code>{html.escape(path)}</code></summary>"
                    f"<h3>{html.escape(operation.get('summary', ''))}</h3><p>{security}</p>"
                    + (
                        "<h4>Parameters</h4>" + code(operation["parameters"])
                        if operation.get("parameters")
                        else ""
                    )
                    + (
                        "<h4>Request body</h4>" + code(operation["requestBody"])
                        if operation.get("requestBody")
                        else ""
                    )
                    + "<h4>Responses</h4>"
                    + responses
                    + "</details>"
                )
        models_html = "".join(
            f"<details id='{html.escape(name, quote=True)}'><summary>{html.escape(name)}</summary>"
            + code(model)
            + "</details>"
            for name, model in sorted(schema["components"]["schemas"].items())
        )
        return (
            """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Falcon API reference</title>
<style>
:root{font-family:system-ui,sans-serif;color:#243247;background:#f5f6f8;color-scheme:light}
body{max-width:1040px;margin:0 auto;padding:40px 24px;line-height:1.6}h1{letter-spacing:-.04em}
a{color:#365cc0}pre{background:#f1f3f7;padding:16px;overflow:auto;border-radius:8px;font-size:12px}
code{font-family:ui-monospace,monospace;overflow-wrap:anywhere}details{margin:8px 0}
summary{cursor:pointer;padding:8px 0}summary:focus-visible{outline:2px solid #365cc0;outline-offset:4px}
.operation{background:white;border:1px solid #dfe4ec;border-radius:12px;padding:12px 20px}
.operation summary b{display:inline-block;min-width:68px;color:#365cc0}
</style></head><body><p>FALCON / INTEGRATION</p><h1>API reference</h1>
<p>This reference and every schema are served locally. No external scripts, fonts or network requests.</p>
<p><a href="/api/v1/openapi.json" download="falcon-openapi.json">Download OpenAPI JSON</a>
 · <a href="#schemas">Response and request schemas</a></p>
<p>Integrations send <code>Authorization: Bearer &lt;token&gt;</code>.
The browser can instead use its HttpOnly session cookie. Cookie mutations require an allowed
<code>Origin</code> and <code>X-CSRF-Token</code>. All objects belong to the authenticated workspace.</p>
<p>Errors use <code>{"error":{"code":"…","message":"…"},"request_id":"…"}</code>.
The response includes the same <code>X-Request-ID</code>; a supplied safe ID is preserved.
Search retries must reuse <code>Idempotency-Key</code> and the identical input.</p>
<p>Preview is an explicit service stub. Scores are not vehicle identity probabilities and
<code>accepted_candidates</code> remains empty without validated calibration.</p>
<h2>Operations</h2>"""
            + "".join(operations_html)
            + "<h2 id='schemas'>Schemas</h2>"
            + models_html
            + "</body></html>"
        )

    return application


app = create_app()
