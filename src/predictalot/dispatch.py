"""Forecast dispatcher.

`forecast(...)` is the single entry point used by both the HTTP router and the
MCP tools — keeps wire shape, validation, default-filling, and per-model
unload-hint handling in one place.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import config, models

log = logging.getLogger("predictalot.dispatch")


class UnknownModelError(ValueError):
    pass


class BadQuantileLevelsError(ValueError):
    pass


def _validate_quantile_levels(levels: list[float]) -> list[float]:
    if not levels:
        raise BadQuantileLevelsError("quantileLevels must not be empty")
    seen: set[float] = set()
    out: list[float] = []
    for q in levels:
        # Only accept values exactly representable in tenths between 0.1 and 0.9.
        # Use *10 + round to dodge IEEE-754 noise (e.g. 0.7 storing as 0.69999...).
        scaled = float(q) * 10
        as_int = round(scaled)
        if abs(scaled - as_int) > 1e-9 or not 1 <= as_int <= 9:
            raise BadQuantileLevelsError(
                f"quantile level {q} not supported; valid: "
                f"{list(config.ALLOWED_QUANTILE_LEVELS)}"
            )
        rounded = as_int / 10.0
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(rounded)
    return out


def _validate_context(context: list[list[float]]) -> None:
    if not context:
        raise ValueError("context must not be empty")
    for i, series in enumerate(context):
        if not series:
            raise ValueError(f"context[{i}] is empty")


async def forecast(
    model: str,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float] | None = None,
    context_length: int | None = None,
    unload_after: bool = False,
) -> dict[str, Any]:
    """Dispatch to the named model backend; return the unified result dict."""
    if model not in config.MODEL_SLUGS:
        raise UnknownModelError(
            f"unknown model {model!r}; valid: {list(config.MODEL_SLUGS)}"
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    _validate_context(context)

    q_levels = _validate_quantile_levels(
        quantile_levels if quantile_levels is not None else config.DEFAULT_QUANTILE_LEVELS
    )

    ctx_len = (
        context_length
        if context_length is not None
        else config.DEFAULT_CONTEXT_LENGTH[model]
    )
    if ctx_len <= 0:
        raise ValueError(f"contextLength must be > 0, got {ctx_len}")

    backend = models.get(model)
    result = await backend.predict(context, horizon, q_levels, ctx_len)

    if unload_after:
        try:
            await backend.unload()
        except Exception:  # noqa: BLE001
            log.exception("unload after request failed for %s", model)

    return result


async def forecast_ensemble(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float] | None = None,
    context_length: int | None = None,
    weights: dict[str, float] | None = None,
    unload_after: bool = False,
) -> dict[str, Any]:
    """Run multiple models in parallel and return the weighted-mean forecast.

    Args:
        weights: per-model weight map (model slug → non-negative float).
            None → all models weighted 1.0 (uniform).
            Absent key → weight 1.0 (default).
            Weight 0 → skipped (not called, not in result). Use this to
                disable a model from the ensemble.
            Weights are normalized internally; the response echoes the
            normalized weights used.

    Returns the weighted median + quantiles plus an `individual` map with each
    contributing model's full forecast (and its applied weight) so callers can
    inspect dissent. Failure of any included model fails the whole call.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    _validate_context(context)

    q_levels = _validate_quantile_levels(
        quantile_levels if quantile_levels is not None else config.DEFAULT_QUANTILE_LEVELS
    )

    # ─── resolve weights ──────────────────────────────────────────────────
    raw_weights: dict[str, float] = {slug: 1.0 for slug in config.MODEL_SLUGS}
    if weights is not None:
        for slug, w in weights.items():
            if slug not in config.MODEL_SLUGS:
                raise ValueError(
                    f"weights contains unknown model {slug!r}; valid: {list(config.MODEL_SLUGS)}"
                )
            if w < 0:
                raise ValueError(f"weight for {slug} must be >= 0, got {w}")
            raw_weights[slug] = float(w)

    active = [(slug, w) for slug, w in raw_weights.items() if w > 0]
    if not active:
        raise ValueError("every model weight is 0 — at least one must be > 0")

    total = sum(w for _, w in active)
    norm_weights: dict[str, float] = {slug: w / total for slug, w in active}

    # ─── run included models in parallel ──────────────────────────────────
    async def _one(slug: str) -> dict[str, Any]:
        ctx_len = (
            context_length
            if context_length is not None
            else config.DEFAULT_CONTEXT_LENGTH[slug]
        )
        backend = models.get(slug)
        out = await backend.predict(context, horizon, q_levels, ctx_len)
        if unload_after:
            try:
                await backend.unload()
            except Exception:  # noqa: BLE001
                log.exception("unload after request failed for %s", slug)
        return out

    active_slugs = [slug for slug, _ in active]
    results_list = await asyncio.gather(
        *[_one(slug) for slug in active_slugs],
        return_exceptions=False,
    )
    # Stamp the (normalized) weight into each individual result so downstream
    # callers can see, per-model, exactly what contribution it made. If the
    # caller passed no weights, every active model gets 1/N — same weight,
    # explicitly returned.
    individual: dict[str, dict[str, Any]] = {}
    for slug, result in zip(active_slugs, results_list):
        individual[slug] = {**result, "weight": norm_weights[slug]}

    # ─── weighted mean across included models ─────────────────────────────
    n_series = len(individual[active_slugs[0]]["median"])

    median_avg: list[list[float]] = [
        [
            sum(
                individual[slug]["median"][s][t] * norm_weights[slug]
                for slug in active_slugs
            )
            for t in range(horizon)
        ]
        for s in range(n_series)
    ]

    quantiles_avg: dict[str, list[list[float]]] = {}
    for q_key in individual[active_slugs[0]]["quantiles"]:
        quantiles_avg[q_key] = [
            [
                sum(
                    individual[slug]["quantiles"][q_key][s][t] * norm_weights[slug]
                    for slug in active_slugs
                )
                for t in range(horizon)
            ]
            for s in range(n_series)
        ]

    return {
        "model": "ensemble",
        "horizon": horizon,
        "quantile_levels": list(q_levels),
        "median": median_avg,
        "quantiles": quantiles_avg,
        "ensemble_members": active_slugs,
        "weights": norm_weights,
        "individual": individual,
    }
