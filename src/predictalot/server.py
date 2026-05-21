"""FastAPI app + uvicorn entrypoint.

Lifespan:
  - startup:  validate auth config, optionally prefetch + preload models,
              spawn the idle-unload sweeper background task.
  - shutdown: cancel the sweeper.

Endpoints:
  GET  /healthz
  GET  /v1/models
  POST /v1/forecast
  POST /mcp/* (streamable-http, via MCPWithAuth)
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import config, fetch as fetch_module, models
from .auth import check_open_auth_allowed
from .logging import configure as configure_logging
from .routers.forecast import router as forecast_router
from .routers.meta import router as meta_router

log = logging.getLogger("predictalot.server")

SWEEPER_INTERVAL_SECONDS = 60.0


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds PREDICTALOT_MAX_BODY_SIZE.

    This is a quick rejection at the boundary — we don't read the body to
    measure (streaming would be more accurate but most clients send
    Content-Length). Returns 413.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > self._max:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"request body too large; max {self._max} bytes"
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


_mcp_lifespan_cm: Any = None
_sweeper_task: asyncio.Task[None] | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    check_open_auth_allowed()

    if config.PREFETCH:
        log.info("prefetch: %s", config.PREFETCH)
        rc = await asyncio.to_thread(fetch_module.fetch, list(config.PREFETCH))
        if rc != 0:
            log.warning("prefetch reported failures (continuing)")

    for slug in config.PRELOAD:
        log.info("preload: %s", slug)
        try:
            await models.get(slug).get_model()
        except Exception:  # noqa: BLE001
            log.exception("preload %s failed", slug)

    global _sweeper_task
    _sweeper_task = asyncio.create_task(_idle_sweeper(), name="predictalot-idle-sweeper")

    try:
        if _mcp_lifespan_cm is not None:
            async with _mcp_lifespan_cm:
                yield
        else:
            yield
    finally:
        if _sweeper_task is not None:
            _sweeper_task.cancel()
            try:
                await _sweeper_task
            except (asyncio.CancelledError, Exception):
                pass


async def _idle_sweeper() -> None:
    """Periodically unload models that have been idle too long."""
    while True:
        try:
            await asyncio.sleep(SWEEPER_INTERVAL_SECONDS)
            now = time.monotonic()
            for slug in config.MODEL_SLUGS:
                backend = models.get(slug)
                if not backend.loaded():
                    continue
                timeout = config.idle_timeout_for(slug)
                if timeout <= 0:
                    continue
                last = backend.last_used_secs_ago()
                if last is None:
                    continue
                if last < timeout:
                    continue
                log.info("idle sweeper: unloading %s (idle %.1fs >= %.1fs)", slug, last, timeout)
                try:
                    await backend.unload()
                except Exception:  # noqa: BLE001
                    log.exception("idle sweeper: unload %s failed", slug)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("idle sweeper iteration failed")


app = FastAPI(
    title="predictalot",
    description=(
        "Unified forecast API over Chronos-2, TimesFM 2.5, Moirai-2. "
        "POST /v1/forecast — model selected by the `model` field in the body."
    ),
    lifespan=_lifespan,
)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=config.MAX_BODY_SIZE)
app.include_router(meta_router)
app.include_router(forecast_router)


def _mount_mcp() -> None:
    global _mcp_lifespan_cm
    try:
        from .mcp_server import MCPWithAuth, build_mcp_app

        mcp_app = build_mcp_app()
    except Exception:  # noqa: BLE001
        log.exception("mcp: failed to build MCP app — /mcp not mounted")
        return
    app.mount("/mcp", MCPWithAuth(mcp_app))
    _mcp_lifespan_cm = mcp_app.router.lifespan_context(mcp_app)
    log.info("mcp: mounted /mcp")


_mount_mcp()


def main() -> int:
    configure_logging()
    import uvicorn

    log.info("predictalot: starting on %s:%d", config.HOST, config.PORT)
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
