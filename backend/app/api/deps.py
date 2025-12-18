from __future__ import annotations

from typing import Optional
from fastapi import Header

from backend.app.services.auth_service import resolve_token


def _bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        return ""
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return ""


def require_user_id(authorization: Optional[str] = Header(default=None)) -> int:
    """
    FastAPI dependency: returns user_id if session token valid, else raises RuntimeError.
    """
    token = _bearer_token(authorization)
    uid = resolve_token(token)
    if not uid:
        raise RuntimeError("Unauthorized")
    return int(uid)
