"""Bearer-token auth for /v1/* and /mcp.

PREDICTALOT_AUTH_TOKENS is a comma-separated list of acceptable bearer tokens.
Empty list = open (refused at startup unless PREDICTALOT_ALLOW_NO_AUTH=1).

Token may also arrive as the ``apiToken`` query parameter for clients that
can't set headers (Cursor / Claude Desktop fallback) — checked at ASGI scope
level by MCPWithAuth.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Query

from . import config


def check_open_auth_allowed() -> None:
    """Refuse to start if no tokens AND no explicit allow-no-auth opt-in.

    Called from server startup, not as a request dep.
    """
    if config.AUTH_TOKENS or config.ALLOW_NO_AUTH:
        return
    raise RuntimeError(
        "PREDICTALOT_AUTH_TOKENS is empty and PREDICTALOT_ALLOW_NO_AUTH != 1. "
        "Either set tokens or explicitly opt-in to open auth."
    )


def check_bearer(
    authorization: str | None = Header(default=None),
    apiToken: str | None = Query(default=None),  # noqa: N803
) -> None:
    """FastAPI dependency: enforce bearer token if any are configured."""
    tokens = config.AUTH_TOKENS
    if not tokens:
        return

    presented: str | None = None
    if authorization and authorization.startswith("Bearer "):
        presented = authorization[len("Bearer ") :].strip()
    elif apiToken:
        presented = apiToken.strip()

    if not presented:
        raise HTTPException(status_code=401, detail="missing bearer token")
    if presented not in tokens:
        raise HTTPException(status_code=401, detail="invalid bearer token")
