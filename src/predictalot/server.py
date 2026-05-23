"""FastAPI app + uvicorn entrypoint.

Lifespan:
  - startup:  validate auth config, optionally prefetch + preload models,
              spawn the idle-unload sweeper background task.
  - shutdown: cancel the sweeper.

Endpoints:
  GET  /healthz
  POST /v1/univariate/{forecast,forecast/ensemble} + GET /v1/univariate/models
  POST /v1/multivariate/{forecast,forecast/ensemble} + GET /v1/multivariate/models
  POST /v1/covariates/past/{forecast,forecast/ensemble} + GET /v1/covariates/past/models
  POST /v1/covariates/future/{forecast,forecast/ensemble} + GET /v1/covariates/future/models
  POST /v1/covariates/{forecast,forecast/ensemble} + GET /v1/covariates/models
  POST /v1/samples/{forecast,forecast/ensemble} + GET /v1/samples/models
  POST /mcp/* (streamable-http, via MCPWithAuth)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, MutableMapping

from fastapi import FastAPI

from . import config, fetch as fetch_module, models
from .auth import check_open_auth_allowed
from .logging import configure as configure_logging
from .routers.covariates import router as covariates_router
from .routers.covariates_future import router as covariates_future_router
from .routers.covariates_past import router as covariates_past_router
from .routers.meta import router as meta_router
from .routers.multivariate import router as multivariate_router
from .routers.samples import router as samples_router
from .routers.univariate import router as univariate_router

log = logging.getLogger("predictalot.server")

SWEEPER_INTERVAL_SECONDS = 60.0


class BodySizeLimitMiddleware:
    """Raw ASGI middleware enforcing PREDICTALOT_MAX_BODY_SIZE on every request.

    Two-layer defense:
      1. Cheap reject on Content-Length when present (no body buffered).
      2. Wrap the ``receive`` callable so even chunked / no-Content-Length
         requests are rejected once accumulated body bytes exceed the limit.

    Without (2), a client omitting Content-Length (legal under HTTP/1.1
    chunked encoding) bypasses the size limit entirely and the downstream
    server buffers an unbounded body before Pydantic validation runs.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self._app = app
        self._max = max_bytes

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        # Cheap rejection on declared Content-Length.
        for name, value in scope.get("headers", []):
            if name != b"content-length":
                continue
            try:
                declared = int(value)
            except ValueError:
                break
            if declared > self._max:
                await self._send_413(send)
                return
            break

        # Streaming rejection — wraps receive to enforce the cap on actual
        # bytes pushed by the client.
        max_bytes = self._max
        seen = 0
        rejected = False

        async def _capped_receive() -> MutableMapping[str, Any]:
            nonlocal seen, rejected
            message = await receive()
            if message["type"] != "http.request":
                return message
            if rejected:
                # Drain remaining chunks silently so the ASGI server doesn't
                # hang waiting for more_body=False.
                return {"type": "http.request", "body": b"", "more_body": False}
            body = message.get("body", b"")
            seen += len(body)
            if seen > max_bytes:
                rejected = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return message

        send_started = False

        async def _capped_send(message: MutableMapping[str, Any]) -> None:
            nonlocal send_started
            # If the request was rejected mid-receive, intercept any
            # downstream response and replace it with 413. The
            # downstream app sees a truncated body and may itself error,
            # but we own the wire so this guarantees a clean 413.
            if rejected and not send_started:
                send_started = True
                await self._send_413(send)
                return
            if rejected:
                return  # swallow further chunks
            send_started = True
            await send(message)

        await self._app(scope, _capped_receive, _capped_send)

    async def _send_413(self, send: Any) -> None:
        body = (
            f'{{"detail":"request body too large; max {self._max} bytes"}}'
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


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
        "Unified forecast API over Chronos-2, TimesFM 2.5, Moirai-2, Toto-1, "
        "Sundial-base-128m. Forecasts are routed by type — POST to "
        "/v1/<type>/forecast or /v1/<type>/forecast/ensemble with `model` "
        "selected in the body. Types: univariate, multivariate, "
        "covariates/past, covariates/future, covariates (past+future), samples."
    ),
    lifespan=_lifespan,
)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=config.MAX_BODY_SIZE)
app.include_router(meta_router)
app.include_router(univariate_router)
app.include_router(multivariate_router)
app.include_router(covariates_past_router)
app.include_router(covariates_future_router)
app.include_router(covariates_router)
app.include_router(samples_router)


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
