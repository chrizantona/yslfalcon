"""Structured operational logging without request bodies, headers or query strings."""

import json
import logging
import re
import time
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_context: ContextVar[str | None] = ContextVar("falcon_request_id", default=None)
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
logger = logging.getLogger("falcon_service.http")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "service": "falcon_service",
            # Uvicorn can pass a preformatted startup traceback or arbitrary exception
            # values as its message, in addition to exc_info for HTTP failures.
            "message": (
                "server_error"
                if record.name.startswith("uvicorn") and record.levelno >= logging.ERROR
                else record.getMessage()
            ),
        }
        for key in (
            "request_id",
            "method",
            "route",
            "status",
            "duration_ms",
            "job_id",
            "job_kind",
            "error_type",
            "component",
            "attempt",
            "retry_in_seconds",
            "engine_ready",
        ):
            if hasattr(record, key):
                event[key] = getattr(record, key)
        if "request_id" not in event and request_id_context.get() is not None:
            event["request_id"] = request_id_context.get()
        if record.exc_info:
            event["exception_type"] = record.exc_info[0].__name__
            # Exception messages and source lines can contain credentials or user input.
            event["traceback"] = [
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(record.exc_info[2])
            ]
        return json.dumps(event, ensure_ascii=False)


def configure_logging() -> None:
    logger = logging.getLogger("falcon_service")
    if not any(getattr(handler, "falcon_json", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler.falcon_json = True
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    server_logger = logging.getLogger("uvicorn.error")
    if not server_logger.handlers:
        server_logger.addHandler(
            next(handler for handler in logger.handlers if getattr(handler, "falcon_json", False))
        )
    for handler in server_logger.handlers:
        handler.setFormatter(JsonFormatter())
    server_logger.propagate = False


class RequestLoggingMiddleware:
    """Log every HTTP outcome, including failures, without buffering a request or response."""

    def __init__(self, app: ASGIApp, component: str):
        self.app = app
        self.component = component

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied_id = Headers(scope=scope).get("x-request-id", "")
        request_id = supplied_id if REQUEST_ID.fullmatch(supplied_id) else str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        context_token = request_id_context.set(request_id)
        started = time.perf_counter()
        status = 500
        error_info = None

        async def send_response(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Cache-Control"] = "no-store"
            await send(message)

        try:
            await self.app(scope, receive, send_response)
        except Exception as exc:
            status = 500
            error_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            route = scope.get("route")
            level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
            try:
                logger.log(
                    level,
                    "http_request",
                    extra={
                        "component": self.component,
                        "request_id": request_id,
                        "method": scope["method"],
                        "route": getattr(route, "path", "unmatched"),
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                    exc_info=error_info,
                )
            finally:
                request_id_context.reset(context_token)
