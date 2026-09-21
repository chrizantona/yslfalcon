"""Workspace authorization, revocable sessions and CSRF checks."""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta

from fastapi import APIRouter, Depends, Request, Response, Security
from fastapi.security import APIKeyCookie, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .db import now
from .errors import ServiceError
from .models import BrowserSession

router = APIRouter(prefix="/auth", tags=["Authentication"])
ROLE_LEVEL = {"viewer": 0, "operator": 1, "admin": 2}
bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")
session_scheme = APIKeyCookie(
    name="falcon_session",
    auto_error=False,
    scheme_name="SessionCookie",
    description="HttpOnly session from /api/v1/auth/session. Mutating cookie requests also require an allowed Origin and X-CSRF-Token.",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Principal:
    workspace: str
    role: str
    actor: str


def identity_for_token(request: Request, token: str) -> Principal | None:
    # Constant-time comparison also avoids exposing configured credential prefixes.
    for configured, identity in request.app.state.settings.auth_tokens.items():
        if secrets.compare_digest(configured.encode("utf-8"), token.encode("utf-8")):
            return Principal(identity["workspace"], identity["role"], digest(token)[:16])
    return None


def check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin not in request.app.state.settings.allowed_origins:
        raise ServiceError("origin_forbidden", "Источник запроса не разрешён", 403)


def principal(
    request: Request,
    _bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    _session: str | None = Security(session_scheme),
) -> Principal:
    authorization = request.headers.get("authorization", "")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        identity = identity_for_token(request, token) if scheme.lower() == "bearer" else None
        if identity is None:
            raise ServiceError("unauthorized", "Неверный токен доступа", 401)
        return identity
    session_token = request.cookies.get("falcon_session")
    if not session_token:
        raise ServiceError("unauthorized", "Требуется авторизация", 401)
    with request.app.state.db.transaction() as session:
        browser_session = session.get(BrowserSession, digest(session_token))
        if browser_session is None or browser_session.expires_at <= now():
            raise ServiceError("unauthorized", "Сессия завершена", 401)
        identity = next(
            (
                Principal(item["workspace"], item["role"], digest(token)[:16])
                for token, item in request.app.state.settings.auth_tokens.items()
                if secrets.compare_digest(digest(token), browser_session.credential_hash)
            ),
            None,
        )
        if identity is None:
            raise ServiceError("unauthorized", "Доступ отозван", 401)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            check_origin(request)
            csrf = request.headers.get("x-csrf-token", "")
            if not csrf or not secrets.compare_digest(digest(csrf), browser_session.csrf_hash):
                raise ServiceError("csrf_failed", "Требуется действующий CSRF-токен", 403)
        return identity


def require_role(role: str):
    def authorize(identity: Principal = Depends(principal)) -> Principal:
        if ROLE_LEVEL[identity.role] < ROLE_LEVEL[role]:
            raise ServiceError("forbidden", "Недостаточно прав для этого действия", 403)
        return identity

    return authorize


class Login(BaseModel):
    token: str = Field(min_length=1, max_length=4096)


@router.post("/session")
def login(body: Login, request: Request, response: Response):
    check_origin(request)
    identity = identity_for_token(request, body.token)
    if identity is None:
        raise ServiceError("unauthorized", "Неверный токен доступа", 401)
    session_token, csrf_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    settings = request.app.state.settings
    with request.app.state.db.transaction() as session:
        old_token = request.cookies.get("falcon_session")
        if old_token:
            previous = session.get(BrowserSession, digest(old_token))
            if previous:
                session.delete(previous)
        session.add(
            BrowserSession(
                token_hash=digest(session_token),
                credential_hash=digest(body.token),
                csrf_hash=digest(csrf_token),
                expires_at=now() + timedelta(hours=settings.session_hours),
            )
        )
    common = {
        "secure": settings.cookie_secure,
        "samesite": "strict",
        "path": "/",
        "max_age": settings.session_hours * 3600,
    }
    response.set_cookie("falcon_session", session_token, httponly=True, **common)
    response.set_cookie("falcon_csrf", csrf_token, httponly=False, **common)
    response.headers["Cache-Control"] = "no-store"
    return {
        "role": identity.role,
        "workspace": identity.workspace,
        "workspace_id": identity.workspace,
        "csrf_token": csrf_token,
    }


@router.delete("/session", status_code=204)
def logout(request: Request, response: Response, _: Principal = Depends(principal)):
    token = request.cookies.get("falcon_session")
    if token:
        with request.app.state.db.transaction() as session:
            existing = session.get(BrowserSession, digest(token))
            if existing:
                session.delete(existing)
    response.delete_cookie("falcon_session", path="/")
    response.delete_cookie("falcon_csrf", path="/")


@router.get("/me")
def me(request: Request, identity: Principal = Depends(principal)):
    return {
        "role": identity.role,
        "workspace": identity.workspace,
        "workspace_id": identity.workspace,
        "csrf_token": request.cookies.get("falcon_csrf"),
    }
