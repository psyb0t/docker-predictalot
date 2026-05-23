"""Bearer-token auth for /v1/* and /mcp.

PREDICTALOT_AUTH_TOKENS is a comma-separated list of acceptable bearer tokens.
Empty list = open (refused at startup unless PREDICTALOT_ALLOW_NO_AUTH=1).

Token may also arrive as the ``apiToken`` query parameter for clients that
can't set headers (Cursor / Claude Desktop fallback) — checked at ASGI scope
level by MCPWithAuth.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Query

from . import config


def _token_matches(presented: str, tokens: list[str]) -> bool:
    """Constant-time membership check. `tokens` is small (typ. 1-3 entries).

    Compares as bytes so non-ASCII input doesn't raise — `hmac.compare_digest`
    on `str` is ASCII-only and a single non-ASCII byte in the presented token
    would otherwise crash the auth path with a 500 instead of returning 401.
    """
    presented_b = presented.encode("utf-8", errors="replace")
    ok = False
    for t in tokens:
        # `or` is short-circuit but compare_digest itself doesn't leak the
        # match position because each call has the same time cost; we
        # additionally avoid using `in` so the loop runs to completion.
        if hmac.compare_digest(presented_b, t.encode("utf-8", errors="replace")):
            ok = True
    return ok


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
    if not _token_matches(presented, tokens):
        raise HTTPException(status_code=401, detail="invalid bearer token")
