"""GET /healthz and GET /v1/models.

Both routes are unauthenticated by design — operators need them to monitor
the service. /v1/models doesn't leak anything sensitive (slugs are public).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .. import config, models

router = APIRouter(tags=["meta"])


@router.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@router.get("/v1/models")
def list_models() -> dict[str, Any]:
    out = []
    for slug in config.MODEL_SLUGS:
        backend = models.get(slug)
        out.append(
            {
                "slug": slug,
                "loaded": backend.loaded(),
                "lastUsedSecsAgo": backend.last_used_secs_ago(),
                "idleTimeoutSecs": config.idle_timeout_for(slug),
            }
        )
    return {"models": out}
