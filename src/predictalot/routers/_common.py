"""Shared helpers for per-type routers."""

from __future__ import annotations

from typing import Any

from .. import config, types


def build_type_models_response(type_slug: str, models: Any) -> dict[str, Any]:
    """Build the {type, models[]} payload for GET /v1/<type>/models."""
    out = []
    for slug in types.members(type_slug):
        backend = models.get(slug)
        out.append(
            {
                "slug": slug,
                "loaded": backend.loaded(),
                "lastUsedSecsAgo": backend.last_used_secs_ago(),
                "idleTimeoutSecs": config.idle_timeout_for(slug),
            }
        )
    return {"type": type_slug, "models": out}
