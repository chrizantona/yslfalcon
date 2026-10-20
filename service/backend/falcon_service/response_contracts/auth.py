"""Authenticated workspace identity contracts."""

from .common import ResponseModel, Role


class IdentityResponse(ResponseModel):
    role: Role
    workspace: str
    workspace_id: str
    csrf_token: str | None
