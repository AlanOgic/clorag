"""Bearer token auth wrapper for MCP HTTP transport."""

import secrets

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BearerAuthMiddleware:
    """ASGI middleware that requires a valid Bearer token on HTTP requests.

    Only checks HTTP requests. Non-HTTP scopes (lifespan, etc.) pass through.
    Uses secrets.compare_digest for timing-safe comparison.
    """

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self._key = api_key.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            request = Request(scope)
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer "):
                resp = JSONResponse(
                    status_code=401,
                    content={"error": "Missing Bearer token"},
                )
                await resp(scope, receive, send)
                return
            token = auth.removeprefix("Bearer ").strip().encode()
            if not secrets.compare_digest(token, self._key):
                resp = JSONResponse(
                    status_code=401,
                    content={"error": "Invalid API key"},
                )
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def apply_bearer_auth(app: ASGIApp, api_key: str) -> ASGIApp:
    """Wrap an ASGI app with Bearer token authentication."""
    return BearerAuthMiddleware(app, api_key)
