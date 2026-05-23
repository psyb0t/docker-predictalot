"""MCP server exposed via streamable-http at /mcp.

Tool surface mirrors the v0.2 type-routed HTTP API. For every forecast type
there are three classes of tool:

  * ``forecast_<type>_<model>`` — single-model forecast for one (type, model)
    pair. One tool per cell in the type-to-members matrix.
  * ``forecast_<type>_ensemble`` — weighted ensemble over every model that
    supports the type. Per-type, not per-model.
  * ``list_<type>_models`` — show which models implement the type and their
    runtime state (loaded / last-used / idle-timeout).

LLM agents pick a named capability instead of routing through a polymorphic
``forecast(model=..., type=...)`` — concrete names yield more reliable tool
selection. The matrix is the same one declared in :mod:`predictalot.types`.

Auth: PREDICTALOT_AUTH_TOKENS is checked at ASGI scope level (mirrors
aicodebox). Token may arrive as ``Authorization: Bearer ...`` or as
``?apiToken=...`` query string.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any, Awaitable, MutableMapping

from . import config, dispatch, types
from .auth import _token_matches

log = logging.getLogger("predictalot.mcp")


def _norm(slug: str) -> str:
    """Convert a model / type slug to a Python-identifier-safe token."""
    return slug.replace("-", "_").replace(".", "_")


_SAFE_USER_EXCEPTIONS = (
    ValueError,
    dispatch.UnknownModelError,
    dispatch.BadQuantileLevelsError,
    types.ModelDoesNotSupportTypeError,
    types.UnknownTypeError,
)


async def _call_json(coro: Awaitable[dict[str, Any]], context: str) -> str:
    try:
        result = await coro
        return json.dumps(result)
    except _SAFE_USER_EXCEPTIONS as exc:
        # User-input validation errors: message is intentionally surfaced so
        # the caller (LLM agent) can correct its request.
        return json.dumps({"error": str(exc), "context": context})
    except Exception:  # noqa: BLE001
        # Internal errors: do not leak file paths, module names, stack frame
        # fragments. Full trace is logged server-side; client gets a generic
        # message that includes only the routing context.
        log.exception("mcp tool %s failed", context)
        return json.dumps(
            {
                "error": "internal error; see server logs",
                "context": context,
            }
        )


def build_mcp_app() -> Any:
    """Construct the FastMCP ASGI app. Lazy import so the api module doesn't
    pay the import cost when MCP is unused."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("predictalot", streamable_http_path="/")

    _register_univariate(mcp)
    _register_multivariate(mcp)
    _register_covariates_past(mcp)
    _register_covariates_future(mcp)
    _register_covariates_both(mcp)
    _register_samples(mcp)

    return mcp.streamable_http_app()


# ─── univariate ──────────────────────────────────────────────────────────────


def _register_univariate(mcp: Any) -> None:
    for model_slug in types.members(types.TYPE_UNIVARIATE):
        _make_univariate_tool(mcp, model_slug)
    _make_univariate_ensemble(mcp)
    _make_list_models_tool(mcp, types.TYPE_UNIVARIATE)


_UNIVARIATE_DESCRIPTION = (
    "Univariate quantile forecast with {model}.\n\n"
    "Each inner list of context is one independent time series; the backend "
    "runs them as a batch. Returns per-series median and quantiles (one path "
    "per requested quantile level).\n\n"
    "Args:\n"
    "  context: list[list[float]] — one series per inner list.\n"
    "  horizon: future steps to predict.\n"
    "  quantile_levels: subset of {{0.1..0.9}}. Default [0.1, 0.5, 0.9].\n"
    "  context_length: history points fed to the model. None = backend default.\n"
    "  unload: tear the model down after the response."
)


def _make_univariate_tool(mcp: Any, model_slug: str) -> None:
    norm = _norm(model_slug)

    @mcp.tool(
        name=f"forecast_univariate_{norm}",
        description=_UNIVARIATE_DESCRIPTION.format(model=model_slug),
    )
    async def _tool(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.dispatch_univariate(
                model=model_slug,
                context=context,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                unload_after=unload,
            ),
            context=f"univariate/{model_slug}",
        )


def _make_univariate_ensemble(mcp: Any) -> None:
    @mcp.tool(
        name="forecast_univariate_ensemble",
        description=(
            "Univariate ensemble — weighted mean across every model that "
            "supports univariate forecasting.\n\n"
            "Response carries the aggregated median and quantiles plus each "
            "contributing model's individual forecast and applied weight.\n\n"
            "Args:\n"
            "  context, horizon, quantile_levels, context_length, unload: "
            "as in forecast_univariate_<model>.\n"
            "  weights: {slug: float >= 0}. Missing slugs default to 1.0; "
            "weight 0 disables that model. None = uniform."
        ),
    )
    async def _tool(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        weights: dict[str, float] | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.ensemble_univariate(
                context=context,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                weights=weights,
                unload_after=unload,
            ),
            context="univariate/ensemble",
        )


# ─── multivariate ────────────────────────────────────────────────────────────


def _register_multivariate(mcp: Any) -> None:
    for model_slug in types.members(types.TYPE_MULTIVARIATE):
        _make_multivariate_tool(mcp, model_slug)
    _make_multivariate_ensemble(mcp)
    _make_list_models_tool(mcp, types.TYPE_MULTIVARIATE)


_MULTIVARIATE_DESCRIPTION = (
    "Multivariate quantile forecast with {model}.\n\n"
    "context is list[list[list[float]]] — outer dim is independent series, "
    "middle is channels (variates) within a series, inner is time. Channels "
    "are forecast jointly per series. median and quantiles are shaped "
    "[series][channel][time].\n\n"
    "Args:\n"
    "  context: list[list[list[float]]].\n"
    "  horizon: future steps to predict.\n"
    "  quantile_levels: subset of {{0.1..0.9}}. Default [0.1, 0.5, 0.9].\n"
    "  context_length: history points per channel. None = backend default.\n"
    "  unload: tear the model down after the response."
)


def _make_multivariate_tool(mcp: Any, model_slug: str) -> None:
    norm = _norm(model_slug)

    @mcp.tool(
        name=f"forecast_multivariate_{norm}",
        description=_MULTIVARIATE_DESCRIPTION.format(model=model_slug),
    )
    async def _tool(
        context: list[list[list[float]]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.dispatch_multivariate(
                model=model_slug,
                context=context,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                unload_after=unload,
            ),
            context=f"multivariate/{model_slug}",
        )


def _make_multivariate_ensemble(mcp: Any) -> None:
    @mcp.tool(
        name="forecast_multivariate_ensemble",
        description=(
            "Multivariate ensemble — weighted mean across every model that "
            "supports multivariate forecasting. Args otherwise as "
            "forecast_multivariate_<model> + per-slug weights."
        ),
    )
    async def _tool(
        context: list[list[list[float]]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        weights: dict[str, float] | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.ensemble_multivariate(
                context=context,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                weights=weights,
                unload_after=unload,
            ),
            context="multivariate/ensemble",
        )


# ─── covariates: past ────────────────────────────────────────────────────────


def _register_covariates_past(mcp: Any) -> None:
    for model_slug in types.members(types.TYPE_COVARIATES_PAST):
        _make_covariates_past_tool(mcp, model_slug)
    _make_covariates_past_ensemble(mcp)
    _make_list_models_tool(mcp, types.TYPE_COVARIATES_PAST)


_COVARIATES_PAST_DESCRIPTION = (
    "Past-only covariates forecast with {model}.\n\n"
    "Forecasts each context series conditioned on covariates whose values are "
    "known up to t but not into the future. Each entry of past_covariates is "
    "a {{name: values}} mapping for the matching series; every value array "
    "must be the same length as that series' context.\n\n"
    "Args:\n"
    "  context: list[list[float]] — target series.\n"
    "  past_covariates: list[dict[str, list[float]]] — one mapping per series.\n"
    "  horizon, quantile_levels, context_length, unload: as elsewhere."
)


def _make_covariates_past_tool(mcp: Any, model_slug: str) -> None:
    norm = _norm(model_slug)

    @mcp.tool(
        name=f"forecast_covariates_past_{norm}",
        description=_COVARIATES_PAST_DESCRIPTION.format(model=model_slug),
    )
    async def _tool(
        context: list[list[float]],
        past_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.dispatch_covariates_past(
                model=model_slug,
                context=context,
                past_covariates=past_covariates,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                unload_after=unload,
            ),
            context=f"covariates-past/{model_slug}",
        )


def _make_covariates_past_ensemble(mcp: Any) -> None:
    @mcp.tool(
        name="forecast_covariates_past_ensemble",
        description=(
            "Past-only covariates ensemble. Args as "
            "forecast_covariates_past_<model> + per-slug weights."
        ),
    )
    async def _tool(
        context: list[list[float]],
        past_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        weights: dict[str, float] | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.ensemble_covariates_past(
                context=context,
                past_covariates=past_covariates,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                weights=weights,
                unload_after=unload,
            ),
            context="covariates-past/ensemble",
        )


# ─── covariates: future ──────────────────────────────────────────────────────


def _register_covariates_future(mcp: Any) -> None:
    for model_slug in types.members(types.TYPE_COVARIATES_FUTURE):
        _make_covariates_future_tool(mcp, model_slug)
    _make_covariates_future_ensemble(mcp)
    _make_list_models_tool(mcp, types.TYPE_COVARIATES_FUTURE)


_COVARIATES_FUTURE_DESCRIPTION = (
    "Future-only covariates forecast with {model}.\n\n"
    "Forecasts each context series conditioned on covariates known only over "
    "the future window (length == horizon). Useful for ahead-of-time "
    "scheduling signals (price, weather forecast, planned promotion) where "
    "you have no observed history of the covariate.\n\n"
    "Args:\n"
    "  context: list[list[float]] — target series.\n"
    "  future_covariates: list[dict[str, list[float]]] — one mapping per "
    "series, each value array of length horizon.\n"
    "  horizon, quantile_levels, context_length, unload: as elsewhere."
)


def _make_covariates_future_tool(mcp: Any, model_slug: str) -> None:
    norm = _norm(model_slug)

    @mcp.tool(
        name=f"forecast_covariates_future_{norm}",
        description=_COVARIATES_FUTURE_DESCRIPTION.format(model=model_slug),
    )
    async def _tool(
        context: list[list[float]],
        future_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.dispatch_covariates_future(
                model=model_slug,
                context=context,
                future_covariates=future_covariates,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                unload_after=unload,
            ),
            context=f"covariates-future/{model_slug}",
        )


def _make_covariates_future_ensemble(mcp: Any) -> None:
    @mcp.tool(
        name="forecast_covariates_future_ensemble",
        description=(
            "Future-only covariates ensemble. Args as "
            "forecast_covariates_future_<model> + per-slug weights. Only "
            "useful once more than one backend supports the type."
        ),
    )
    async def _tool(
        context: list[list[float]],
        future_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        weights: dict[str, float] | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.ensemble_covariates_future(
                context=context,
                future_covariates=future_covariates,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                weights=weights,
                unload_after=unload,
            ),
            context="covariates-future/ensemble",
        )


# ─── covariates: past + future ───────────────────────────────────────────────


def _register_covariates_both(mcp: Any) -> None:
    for model_slug in types.members(types.TYPE_COVARIATES_BOTH):
        _make_covariates_both_tool(mcp, model_slug)
    _make_covariates_both_ensemble(mcp)
    _make_list_models_tool(mcp, types.TYPE_COVARIATES_BOTH)


_COVARIATES_BOTH_DESCRIPTION = (
    "Past + future covariates forecast with {model}.\n\n"
    "Each series can carry both past covariates (length == series length) "
    "and future covariates (length == horizon). Future-only covariate names "
    "that don't also appear in past_covariates for that series will be "
    "rejected by the backend (chronos-2 requires every future-cov key to "
    "also be present in past-cov).\n\n"
    "Args:\n"
    "  context: list[list[float]] — target series.\n"
    "  past_covariates: list[dict[str, list[float]]].\n"
    "  future_covariates: list[dict[str, list[float]]].\n"
    "  horizon, quantile_levels, context_length, unload: as elsewhere."
)


def _make_covariates_both_tool(mcp: Any, model_slug: str) -> None:
    norm = _norm(model_slug)

    @mcp.tool(
        name=f"forecast_covariates_both_{norm}",
        description=_COVARIATES_BOTH_DESCRIPTION.format(model=model_slug),
    )
    async def _tool(
        context: list[list[float]],
        past_covariates: list[dict[str, list[float]]],
        future_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.dispatch_covariates(
                model=model_slug,
                context=context,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                unload_after=unload,
            ),
            context=f"covariates/{model_slug}",
        )


def _make_covariates_both_ensemble(mcp: Any) -> None:
    @mcp.tool(
        name="forecast_covariates_both_ensemble",
        description=(
            "Past + future covariates ensemble. Args as "
            "forecast_covariates_both_<model> + per-slug weights. Currently "
            "only chronos-2 supports this type."
        ),
    )
    async def _tool(
        context: list[list[float]],
        past_covariates: list[dict[str, list[float]]],
        future_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float] | None = None,
        context_length: int | None = None,
        weights: dict[str, float] | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.ensemble_covariates(
                context=context,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                horizon=horizon,
                quantile_levels=quantile_levels,
                context_length=context_length,
                weights=weights,
                unload_after=unload,
            ),
            context="covariates/ensemble",
        )


# ─── samples ─────────────────────────────────────────────────────────────────


def _register_samples(mcp: Any) -> None:
    for model_slug in types.members(types.TYPE_SAMPLES):
        _make_samples_tool(mcp, model_slug)
    _make_samples_ensemble(mcp)
    _make_list_models_tool(mcp, types.TYPE_SAMPLES)


_SAMPLES_DESCRIPTION = (
    "Raw sample-path forecast with {model}.\n\n"
    "Returns num_samples Monte Carlo paths per series instead of quantiles — "
    "use this when you want to compute custom risk metrics, joint "
    "distributions across timesteps, or scenario analysis over the raw "
    "draws. Response carries samples shaped [series][sample][time] plus "
    "per-series median.\n\n"
    "Args:\n"
    "  context: list[list[float]] — target series.\n"
    "  horizon: future steps to predict.\n"
    "  num_samples: how many sample paths to draw per series. None = backend "
    "default (64).\n"
    "  context_length: history per series. None = backend default.\n"
    "  unload: tear the model down after the response."
)


def _make_samples_tool(mcp: Any, model_slug: str) -> None:
    norm = _norm(model_slug)

    @mcp.tool(
        name=f"forecast_samples_{norm}",
        description=_SAMPLES_DESCRIPTION.format(model=model_slug),
    )
    async def _tool(
        context: list[list[float]],
        horizon: int,
        num_samples: int | None = None,
        context_length: int | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.dispatch_samples(
                model=model_slug,
                context=context,
                horizon=horizon,
                num_samples=num_samples,
                context_length=context_length,
                unload_after=unload,
            ),
            context=f"samples/{model_slug}",
        )


def _make_samples_ensemble(mcp: Any) -> None:
    @mcp.tool(
        name="forecast_samples_ensemble",
        description=(
            "Samples ensemble — concatenates sample paths from every "
            "supporting model along the sample axis.\n\n"
            "Per-model sample count is max(1, round(weight * num_samples)). "
            "median is recomputed over the pooled paths.\n\n"
            "Args:\n"
            "  context, horizon, num_samples, context_length, unload: same as "
            "forecast_samples_<model>.\n"
            "  weights: {slug: float >= 0}. None = uniform; weight 0 disables "
            "a model."
        ),
    )
    async def _tool(
        context: list[list[float]],
        horizon: int,
        num_samples: int | None = None,
        context_length: int | None = None,
        weights: dict[str, float] | None = None,
        unload: bool = False,
    ) -> str:
        return await _call_json(
            dispatch.ensemble_samples(
                context=context,
                horizon=horizon,
                num_samples=num_samples,
                context_length=context_length,
                weights=weights,
                unload_after=unload,
            ),
            context="samples/ensemble",
        )


# ─── list_<type>_models ──────────────────────────────────────────────────────


def _make_list_models_tool(mcp: Any, type_slug: str) -> None:
    type_norm = _norm(type_slug)
    members = types.members(type_slug)

    @mcp.tool(
        name=f"list_{type_norm}_models",
        description=(
            f"List models that implement the {type_slug!r} forecast type. "
            "Returns {type, models: [{slug, loaded, lastUsedSecsAgo, "
            "idleTimeoutSecs}]}."
        ),
    )
    async def _tool() -> str:
        from . import models as models_pkg

        out = []
        for slug in members:
            backend = models_pkg.get(slug)
            out.append(
                {
                    "slug": slug,
                    "loaded": backend.loaded(),
                    "lastUsedSecsAgo": backend.last_used_secs_ago(),
                    "idleTimeoutSecs": config.idle_timeout_for(slug),
                }
            )
        return json.dumps({"type": type_slug, "models": out})


# ─── ASGI auth wrapper ───────────────────────────────────────────────────────


def _scope_auth_ok(scope: MutableMapping[str, Any]) -> bool:
    tokens = config.AUTH_TOKENS
    if not tokens:
        return True
    headers = {k: v for k, v in scope.get("headers", [])}
    auth = headers.get(b"authorization", b"").decode()
    if auth.startswith("Bearer "):
        token = auth[len("Bearer ") :].strip()
        if _token_matches(token, tokens):
            return True
    qs_bytes = scope.get("query_string", b"")
    # parse_qs handles percent-decoding (incl. '+' → space mapping that
    # matches form-encoded clients) and tolerates repeated keys.
    parsed = urllib.parse.parse_qs(qs_bytes.decode(errors="replace"))
    for presented in parsed.get("apiToken", []):
        if _token_matches(presented, tokens):
            return True
    return False


class MCPWithAuth:
    """ASGI wrapper enforcing bearer-token auth before passing to MCP."""

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        # MCP streamable-http only uses HTTP. Lifespan and any other ASGI
        # scope types pass straight through (lifespan must not be gated by
        # auth); a websocket would not reach this branch under streamable-http
        # but if upstream ever adds one, we explicitly refuse to send HTTP
        # ASGI messages on a non-HTTP scope.
        if scope["type"] != "http":
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
