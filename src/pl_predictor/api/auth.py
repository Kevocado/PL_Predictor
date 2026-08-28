"""auth.py — guest-password gate for the public, read-only deployment.

Only active when `config.PUBLIC_MODE` is true (set for the public Render
deployment; unset for local/private use, where this is a complete no-op).
Applied once as ASGI middleware in `main.py` so it covers the built
frontend's static files too, not just `/api/*` — a browser's native HTTP
Basic Auth prompt on first load is the entire "login," no frontend login
page or token handling needed.
"""

from __future__ import annotations

import base64
import binascii
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..config import GUEST_PASSWORD, PUBLIC_MODE


def _unauthorized() -> Response:
    return Response(
        status_code=401,
        content="Authentication required.",
        headers={"WWW-Authenticate": 'Basic realm="PL Predictor"'},
    )


def _password_matches(authorization_header: str) -> bool:
    if not GUEST_PASSWORD or not authorization_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(authorization_header.removeprefix("Basic ")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    _, _, password = decoded.partition(":")
    # Any username is accepted — this gates access to the app, not identity.
    return secrets.compare_digest(password, GUEST_PASSWORD)


class GuestAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not PUBLIC_MODE:
            return await call_next(request)
        if _password_matches(request.headers.get("authorization", "")):
            return await call_next(request)
        return _unauthorized()
