"""Application composition root; no credentials or models are created on import."""

import html
import logging
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
from .logging_setup import configure_logging
from .routes import artifacts, galleries, searches, system
from .runtime import EncoderRuntime
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


class RequestLimits:
    """Enforce total streamed body size before multipart buffering can grow unbounded."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        limit = (
            min(self.limit, 1024 * 1024)
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
                    "request_id": str(uuid4()),
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
    )
    application.add_middleware(RequestLimits, limit=settings.max_request_bytes)

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = str(uuid4())
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
        return JSONResponse(
            {
                "error": {"code": code, "message": message},
                "request_id": getattr(request.state, "request_id", str(uuid4())),
            },
            status_code=status,
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
        logger.error("Unhandled error request_id=%s type=%s", request.state.request_id, type(exc).__name__)
        return error_response(
            request, "internal_error", "Ошибка сервиса; используйте request_id для диагностики", 500
        )

    for router in [auth_router, galleries.router, searches.router, artifacts.router, system.router]:
        application.include_router(router, prefix="/api/v1")

    @application.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    def offline_docs():
        paths = application.openapi()["paths"]
        lines = []
        for path, operations in paths.items():
            for method, operation in operations.items():
                lines.append(
                    f"<tr><td>{html.escape(method.upper())}</td><td><code>{html.escape(path)}</code></td><td>{html.escape(operation.get('summary', ''))}</td></tr>"
                )
        return (
            "<!doctype html><html lang='en'><meta charset='utf-8'><title>Falcon API</title><body><h1>Falcon API</h1><p>Offline API reference. Authenticate business requests using Authorization: Bearer &lt;token&gt;.</p><p><a href='/api/v1/openapi.json'>Download OpenAPI JSON</a></p><table><thead><tr><th>Method</th><th>Path</th><th>Operation</th></tr></thead><tbody>"
            + "".join(lines)
            + "</tbody></table></body></html>"
        )

    return application


app = create_app()
