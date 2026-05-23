"""GET /healthz — unauthenticated liveness probe.

Per-type model listings now live under /v1/<type>/models (one per type
router); the unified /v1/models that existed in v0.1 is gone.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["meta"])


@router.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}
