"""MCP server exposed via streamable-http at /mcp.

Three named forecasting tools (one per slug) — LLM agents discover named
capabilities more reliably than they pick a polymorphic ``forecast(model=...)``
with an enum. Each is a one-line wrapper over the dispatch.forecast() call.

Auth: PREDICTALOT_AUTH_TOKENS checked at ASGI scope level (mirror aicodebox).
Token may arrive as ``Authorization: Bearer ...`` or ``?apiToken=...``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from . import config, dispatch

log = logging.getLogger("predictalot.mcp")


def build_mcp_app() -> Any:
    """Construct the FastMCP ASGI app. Lazy import so the api module doesn't
    pay the import cost when MCP is unused."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("predictalot", streamable_http_path="/")

    @mcp.tool()
    async def forecast_chronos_2(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int = 2048,
        unload: bool = False,
    ) -> str:
        """Forecast with chronos-2.

        Args:
            context: One inner list per time series (`List[List[float]]`).
            horizon: How many future steps to predict.
            quantile_levels: Subset of {0.1, ..., 0.9}. Default [0.1, 0.5, 0.9].
            context_length: Max history points fed to the model (default 2048).
            unload: Tear down the model after this response.
        """
        return await _run("chronos-2", context, horizon, quantile_levels, context_length, unload)

    @mcp.tool()
    async def forecast_timesfm_2_5(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int = 2048,
        unload: bool = False,
    ) -> str:
        """Forecast with timesfm-2.5.

        Args: same as forecast_chronos_2. Horizon must be <= compile-time
        max_horizon (default 512, multiple of 128).
        """
        return await _run("timesfm-2.5", context, horizon, quantile_levels, context_length, unload)

    @mcp.tool()
    async def forecast_sundial_base_128m(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int = 2880,
        unload: bool = False,
    ) -> str:
        """Forecast with sundial-base-128m (thuml/sundial-base-128m).

        ICML 2025 Oral, GIFT-Eval #1 MASE (May 2025), 1T training points.
        Runs in its own sidecar process due to a transformers 4.40 pin —
        from the API surface it looks identical to the other backends.
        Default context_length 2880. Args otherwise as forecast_chronos_2.
        """
        return await _run(
            "sundial-base-128m", context, horizon, quantile_levels, context_length, unload
        )

    @mcp.tool()
    async def forecast_toto_1(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int = 4096,
        unload: bool = False,
    ) -> str:
        """Forecast with toto-1 (Datadog Toto-Open-Base-1.0).

        Decoder-only transformer, multivariate-native (we use it univariate),
        probabilistic via Student-T mixture (256 samples → empirical quantiles).
        Default context_length 4096. Args otherwise as forecast_chronos_2.
        """
        return await _run("toto-1", context, horizon, quantile_levels, context_length, unload)

    @mcp.tool()
    async def forecast_moirai_2(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int = 4000,
        unload: bool = False,
    ) -> str:
        """Forecast with moirai-2.

        Args: same as forecast_chronos_2. Default context_length 4000.
        """
        return await _run("moirai-2", context, horizon, quantile_levels, context_length, unload)

    @mcp.tool()
    async def forecast_ensemble(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        weights: dict[str, float] | None = None,
        unload: bool = False,
    ) -> str:
        """Run the forecasters in parallel and return the weighted-mean
        forecast. Response also includes each contributing model's individual
        forecast + applied weight.

        Args:
            context: List[List[float]] — one inner list per series.
            horizon: future steps to predict.
            quantile_levels: subset of {0.1..0.9}. Default [0.1, 0.5, 0.9].
            context_length: per-model history cap. None = each model's default.
            weights: per-model weight map {slug: float >= 0}. None = uniform.
                Weight 0 skips the model entirely (use this to disable it).
            unload: tear down each contributing model after the response.
        """
        try:
            result = await dispatch.forecast_ensemble(
                context=context,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                weights=weights,
                unload_after=unload,
            )
            return json.dumps(result)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc), "model": "ensemble"})

    @mcp.tool()
    async def list_models() -> str:
        """List available model slugs with their loaded state and last-used info."""
        from . import models

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
        return json.dumps({"models": out})

    return mcp.streamable_http_app()


async def _run(
    model: str,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int,
    unload: bool,
) -> str:
    try:
        result = await dispatch.forecast(
            model=model,
            context=context,
            horizon=horizon,
            quantile_levels=quantile_levels,
            context_length=context_length,
            unload_after=unload,
        )
        return json.dumps(result)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc), "model": model})


def _scope_auth_ok(scope: dict[str, Any]) -> bool:
    tokens = config.AUTH_TOKENS
    if not tokens:
        return True
    headers = {k: v for k, v in scope.get("headers", [])}
    auth = headers.get(b"authorization", b"").decode()
    if auth.startswith("Bearer "):
        token = auth[len("Bearer ") :].strip()
        if token in tokens:
            return True
    qs = scope.get("query_string", b"").decode()
    for part in qs.split("&"):
        if part.startswith("apiToken="):
            presented = part[len("apiToken=") :]
            if presented in tokens:
                return True
    return False


class MCPWithAuth:
    """ASGI wrapper enforcing bearer-token auth before passing to MCP."""

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return
        if not _scope_auth_ok(scope):
            log.warning("mcp: auth failed")
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [[b"content-type", b"application/json"]],
                }
            )
            await send({"type": "http.response.body", "body": b'{"detail":"unauthorized"}'})
            return
        await self._app(scope, receive, send)
