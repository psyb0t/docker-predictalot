"""Per-type dispatch + per-type ensemble.

For each forecast type there's:
  * ``dispatch_<type>(model, ...)`` — call a single named backend's
    ``predict_<type>`` after validating that the model supports the type.
  * ``ensemble_<type>(...)`` — fan out to every supporting model in parallel
    (filtered by `weights`), aggregate, return the unified shape.

Validation (quantile levels, context shape, horizon, model membership) lives
here so the routers stay thin.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from . import config, models, types

log = logging.getLogger("predictalot.dispatch")


# ─── shared validators ───────────────────────────────────────────────────────


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


def _validate_context_1d(context: list[list[float]]) -> None:
    if not context:
        raise ValueError("context must not be empty")
    for i, series in enumerate(context):
        if not series:
            raise ValueError(f"context[{i}] is empty")


def _validate_context_2d(context: list[list[list[float]]]) -> None:
    if not context:
        raise ValueError("context must not be empty")
    first_n_channels: int | None = None
    for i, series in enumerate(context):
        if not series:
            raise ValueError(f"context[{i}] has no channels")
        if first_n_channels is None:
            first_n_channels = len(series)
        elif len(series) != first_n_channels:
            raise ValueError(
                f"context[{i}] has {len(series)} channels but context[0] has "
                f"{first_n_channels}; all series must have the same channel count"
            )
        for c, ch in enumerate(series):
            if not ch:
                raise ValueError(f"context[{i}][{c}] is empty")


def _validate_covariates_shape(
    label: str,
    context: list[list[float]],
    covariates: list[dict[str, list[float]]],
) -> None:
    """Cross-validate covariate entry count against context series count, and
    require uniform covariate-name sets across series.
    """
    if len(covariates) != len(context):
        raise ValueError(
            f"{label} length ({len(covariates)}) must equal context length "
            f"({len(context)})"
        )
    if not covariates:
        return
    expected_keys = frozenset(covariates[0].keys())
    for i, entry in enumerate(covariates):
        if frozenset(entry.keys()) != expected_keys:
            raise ValueError(
                f"{label}[{i}] keys {sorted(entry.keys())} differ from "
                f"{label}[0] keys {sorted(expected_keys)}; every series must "
                f"share the same covariate names"
            )


def _resolve_quantiles(quantile_levels: list[float] | None) -> list[float]:
    return _validate_quantile_levels(
        quantile_levels if quantile_levels is not None else config.DEFAULT_QUANTILE_LEVELS
    )


def _resolve_ctx_len(model: str, context_length: int | None) -> int:
    ctx = context_length if context_length is not None else config.DEFAULT_CONTEXT_LENGTH[model]
    if ctx <= 0:
        raise ValueError(f"contextLength must be > 0, got {ctx}")
    return ctx


def _check_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")


def _check_model_for_type(model: str, type_slug: str) -> None:
    if model not in config.MODEL_SLUGS:
        raise UnknownModelError(
            f"unknown model {model!r}; valid: {list(config.MODEL_SLUGS)}"
        )
    types.assert_supported(type_slug, model)


def _resolve_weights(type_slug: str, weights: dict[str, float] | None) -> dict[str, float]:
    """Normalize ensemble weights against a type's member set.

    Returns {slug: normalized_weight} for the *active* (weight > 0) members only.
    """
    members = types.members(type_slug)
    raw: dict[str, float] = {slug: 1.0 for slug in members}
    if weights is not None:
        for slug, w in weights.items():
            if slug not in members:
                raise ValueError(
                    f"weights contains {slug!r} which is not a member of type "
                    f"{type_slug!r}; valid: {list(members)}"
                )
            wf = float(w)
            if not math.isfinite(wf):
                raise ValueError(
                    f"weight for {slug} must be a finite non-negative number, "
                    f"got {w}"
                )
            if wf < 0:
                raise ValueError(f"weight for {slug} must be >= 0, got {w}")
            raw[slug] = wf

    active = [(s, w) for s, w in raw.items() if w > 0]
    if not active:
        raise ValueError(
            f"every member weight is 0 for type {type_slug!r} — at least one must be > 0"
        )
    total = sum(w for _, w in active)
    return {s: w / total for s, w in active}


async def _maybe_unload(slug: str, unload_after: bool) -> None:
    if not unload_after:
        return
    try:
        await models.get(slug).unload()
    except Exception:  # noqa: BLE001
        log.exception("unload after request failed for %s", slug)


# ─── extras + per-member override helpers ───────────────────────────────────


def _resolve_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce a possibly-None extra dict into a plain dict for backends."""
    return dict(extra) if extra else {}


def _member_override(
    slug: str,
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return the per-member override dict for `slug` (empty if none)."""
    if not overrides:
        return {}
    if slug in overrides and isinstance(overrides[slug], dict):
        return dict(overrides[slug])
    return {}


def _merge_member_config(
    slug: str,
    *,
    global_context_length: int | None,
    global_quantile_levels: list[float] | None,
    global_extra: dict[str, Any] | None,
    overrides: dict[str, dict[str, Any]] | None,
) -> tuple[int, list[float] | None, dict[str, Any]]:
    """Compute the EFFECTIVE (context_length, quantile_levels, extra)
    for a single ensemble member, given the global config + an optional
    per-member override map.

    Override map shape: ``{slug: {contextLength?: int, quantileLevels?,
    extra?: {...}}}``. Each present key replaces the global value FOR
    THAT MEMBER ONLY. Other members continue using globals.

    Quantile levels are pinned to the global value if not overridden so
    every member returns the same quantile set (the ensemble averages
    quantile-by-quantile).
    """
    ov = _member_override(slug, overrides)
    # camelCase + snake_case both accepted in the override dict so
    # callers don't have to remember which alias the wire uses.
    ctx_override = ov.get("context_length", ov.get("contextLength"))
    q_override = ov.get("quantile_levels", ov.get("quantileLevels"))
    extra_override = ov.get("extra")

    ctx = _resolve_ctx_len(
        slug,
        ctx_override if ctx_override is not None else global_context_length,
    )
    q = (
        _resolve_quantiles(q_override)
        if q_override is not None
        else _resolve_quantiles(global_quantile_levels)
    )

    # Extras merge: per-member values override globals key-by-key.
    eff_extra = _resolve_extra(global_extra)
    if extra_override:
        eff_extra.update(extra_override)
    return ctx, q, eff_extra


# ─── univariate ──────────────────────────────────────────────────────────────


async def dispatch_univariate(
    model: str,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _check_model_for_type(model, types.TYPE_UNIVARIATE)
    _check_horizon(horizon)
    _validate_context_1d(context)
    q = _resolve_quantiles(quantile_levels)
    ctx = _resolve_ctx_len(model, context_length)

    backend = models.get(model)
    result = await backend.predict_univariate(
        context, horizon, q, ctx, extra=_resolve_extra(extra),
    )
    await _maybe_unload(model, unload_after)
    return result


async def ensemble_univariate(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    weights: dict[str, float] | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
    member_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _check_horizon(horizon)
    _validate_context_1d(context)
    q = _resolve_quantiles(quantile_levels)
    norm = _resolve_weights(types.TYPE_UNIVARIATE, weights)

    async def _one(slug: str) -> dict[str, Any]:
        ctx, q_eff, eff_extra = _merge_member_config(
            slug,
            global_context_length=context_length,
            global_quantile_levels=quantile_levels,
            global_extra=extra,
            overrides=member_overrides,
        )
        out = await models.get(slug).predict_univariate(
            context, horizon, q_eff, ctx, extra=eff_extra,
        )
        await _maybe_unload(slug, unload_after)
        return out

    active = list(norm.keys())
    results = await asyncio.gather(*[_one(s) for s in active], return_exceptions=False)
    individual = {slug: {**res, "weight": norm[slug]} for slug, res in zip(active, results)}

    n_series = len(individual[active[0]]["median"])
    median_avg = [
        [
            sum(individual[s]["median"][i][t] * norm[s] for s in active)
            for t in range(horizon)
        ]
        for i in range(n_series)
    ]
    quantiles_avg: dict[str, list[list[float]]] = {}
    for q_key in individual[active[0]]["quantiles"]:
        quantiles_avg[q_key] = [
            [
                sum(individual[s]["quantiles"][q_key][i][t] * norm[s] for s in active)
                for t in range(horizon)
            ]
            for i in range(n_series)
        ]

    return {
        "model": "ensemble",
        "horizon": horizon,
        "quantile_levels": list(q),
        "median": median_avg,
        "quantiles": quantiles_avg,
        "ensemble_members": active,
        "weights": norm,
        "individual": individual,
    }


# ─── multivariate ────────────────────────────────────────────────────────────


async def dispatch_multivariate(
    model: str,
    context: list[list[list[float]]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _check_model_for_type(model, types.TYPE_MULTIVARIATE)
    _check_horizon(horizon)
    _validate_context_2d(context)
    q = _resolve_quantiles(quantile_levels)
    ctx = _resolve_ctx_len(model, context_length)

    backend = models.get(model)
    result = await backend.predict_multivariate(context, horizon, q, ctx, extra=_resolve_extra(extra))
    await _maybe_unload(model, unload_after)
    return result


async def ensemble_multivariate(
    context: list[list[list[float]]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    weights: dict[str, float] | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
    member_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _check_horizon(horizon)
    _validate_context_2d(context)
    q = _resolve_quantiles(quantile_levels)
    norm = _resolve_weights(types.TYPE_MULTIVARIATE, weights)

    async def _one(slug: str) -> dict[str, Any]:
        ctx, q_eff, eff_extra = _merge_member_config(
            slug,
            global_context_length=context_length,
            global_quantile_levels=quantile_levels,
            global_extra=extra,
            overrides=member_overrides,
        )
        out = await models.get(slug).predict_multivariate(context, horizon, q_eff, ctx, extra=eff_extra)
        await _maybe_unload(slug, unload_after)
        return out

    active = list(norm.keys())
    results = await asyncio.gather(*[_one(s) for s in active], return_exceptions=False)
    individual = {slug: {**res, "weight": norm[slug]} for slug, res in zip(active, results)}

    n_series = len(individual[active[0]]["median"])
    n_channels = len(individual[active[0]]["median"][0])
    median_avg = [
        [
            [
                sum(individual[s]["median"][i][c][t] * norm[s] for s in active)
                for t in range(horizon)
            ]
            for c in range(n_channels)
        ]
        for i in range(n_series)
    ]
    quantiles_avg: dict[str, list[list[list[float]]]] = {}
    for q_key in individual[active[0]]["quantiles"]:
        quantiles_avg[q_key] = [
            [
                [
                    sum(individual[s]["quantiles"][q_key][i][c][t] * norm[s] for s in active)
                    for t in range(horizon)
                ]
                for c in range(n_channels)
            ]
            for i in range(n_series)
        ]

    return {
        "model": "ensemble",
        "horizon": horizon,
        "quantile_levels": list(q),
        "median": median_avg,
        "quantiles": quantiles_avg,
        "ensemble_members": active,
        "weights": norm,
        "individual": individual,
    }


# ─── covariates: past only ───────────────────────────────────────────────────


async def dispatch_covariates_past(
    model: str,
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _check_model_for_type(model, types.TYPE_COVARIATES_PAST)
    _check_horizon(horizon)
    _validate_context_1d(context)
    _validate_covariates_shape("pastCovariates", context, past_covariates)
    q = _resolve_quantiles(quantile_levels)
    ctx = _resolve_ctx_len(model, context_length)

    backend = models.get(model)
    result = await backend.predict_covariates_past(context, past_covariates, horizon, q, ctx, extra=_resolve_extra(extra))
    await _maybe_unload(model, unload_after)
    return result


async def ensemble_covariates_past(
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    weights: dict[str, float] | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
    member_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _check_horizon(horizon)
    _validate_context_1d(context)
    _validate_covariates_shape("pastCovariates", context, past_covariates)
    q = _resolve_quantiles(quantile_levels)
    norm = _resolve_weights(types.TYPE_COVARIATES_PAST, weights)

    async def _one(slug: str) -> dict[str, Any]:
        ctx, q_eff, eff_extra = _merge_member_config(
            slug,
            global_context_length=context_length,
            global_quantile_levels=quantile_levels,
            global_extra=extra,
            overrides=member_overrides,
        )
        out = await models.get(slug).predict_covariates_past(context, past_covariates, horizon, q_eff, ctx, extra=eff_extra)
        await _maybe_unload(slug, unload_after)
        return out

    return await _aggregate_quantile_ensemble(types.TYPE_COVARIATES_PAST, norm, _one, horizon, q)


# ─── covariates: future only ─────────────────────────────────────────────────


async def dispatch_covariates_future(
    model: str,
    context: list[list[float]],
    future_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _check_model_for_type(model, types.TYPE_COVARIATES_FUTURE)
    _check_horizon(horizon)
    _validate_context_1d(context)
    _validate_covariates_shape("futureCovariates", context, future_covariates)
    q = _resolve_quantiles(quantile_levels)
    ctx = _resolve_ctx_len(model, context_length)

    backend = models.get(model)
    result = await backend.predict_covariates_future(context, future_covariates, horizon, q, ctx, extra=_resolve_extra(extra))
    await _maybe_unload(model, unload_after)
    return result


async def ensemble_covariates_future(
    context: list[list[float]],
    future_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    weights: dict[str, float] | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
    member_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _check_horizon(horizon)
    _validate_context_1d(context)
    _validate_covariates_shape("futureCovariates", context, future_covariates)
    q = _resolve_quantiles(quantile_levels)
    norm = _resolve_weights(types.TYPE_COVARIATES_FUTURE, weights)

    async def _one(slug: str) -> dict[str, Any]:
        ctx, q_eff, eff_extra = _merge_member_config(
            slug,
            global_context_length=context_length,
            global_quantile_levels=quantile_levels,
            global_extra=extra,
            overrides=member_overrides,
        )
        out = await models.get(slug).predict_covariates_future(context, future_covariates, horizon, q_eff, ctx, extra=eff_extra)
        await _maybe_unload(slug, unload_after)
        return out

    return await _aggregate_quantile_ensemble(types.TYPE_COVARIATES_FUTURE, norm, _one, horizon, q)


# ─── covariates: past + future ───────────────────────────────────────────────


async def dispatch_covariates(
    model: str,
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]],
    future_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _check_model_for_type(model, types.TYPE_COVARIATES_BOTH)
    _check_horizon(horizon)
    _validate_context_1d(context)
    _validate_covariates_shape("pastCovariates", context, past_covariates)
    _validate_covariates_shape("futureCovariates", context, future_covariates)
    q = _resolve_quantiles(quantile_levels)
    ctx = _resolve_ctx_len(model, context_length)

    backend = models.get(model)
    result = await backend.predict_covariates_both(context, past_covariates, future_covariates, horizon, q, ctx, extra=_resolve_extra(extra))
    await _maybe_unload(model, unload_after)
    return result


async def ensemble_covariates(
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]],
    future_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    weights: dict[str, float] | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
    member_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _check_horizon(horizon)
    _validate_context_1d(context)
    _validate_covariates_shape("pastCovariates", context, past_covariates)
    _validate_covariates_shape("futureCovariates", context, future_covariates)
    q = _resolve_quantiles(quantile_levels)
    norm = _resolve_weights(types.TYPE_COVARIATES_BOTH, weights)

    async def _one(slug: str) -> dict[str, Any]:
        ctx, q_eff, eff_extra = _merge_member_config(
            slug,
            global_context_length=context_length,
            global_quantile_levels=quantile_levels,
            global_extra=extra,
            overrides=member_overrides,
        )
        out = await models.get(slug).predict_covariates_both(context, past_covariates, future_covariates, horizon, q_eff, ctx, extra=eff_extra)
        await _maybe_unload(slug, unload_after)
        return out

    return await _aggregate_quantile_ensemble(types.TYPE_COVARIATES_BOTH, norm, _one, horizon, q)


# ─── samples ─────────────────────────────────────────────────────────────────

# Default total samples for a samples request when caller omits num_samples.
DEFAULT_SAMPLES = 64


async def dispatch_samples(
    model: str,
    context: list[list[float]],
    horizon: int,
    num_samples: int | None,
    context_length: int | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _check_model_for_type(model, types.TYPE_SAMPLES)
    _check_horizon(horizon)
    _validate_context_1d(context)
    n = num_samples if num_samples is not None else DEFAULT_SAMPLES
    if n <= 0:
        raise ValueError(f"numSamples must be > 0, got {n}")
    ctx = _resolve_ctx_len(model, context_length)

    backend = models.get(model)
    result = await backend.predict_samples(
        context, horizon, n, ctx, extra=_resolve_extra(extra),
    )
    await _maybe_unload(model, unload_after)
    return result


async def ensemble_samples(
    context: list[list[float]],
    horizon: int,
    num_samples: int | None,
    context_length: int | None,
    weights: dict[str, float] | None,
    unload_after: bool,
    extra: dict[str, Any] | None = None,
    member_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _check_horizon(horizon)
    _validate_context_1d(context)
    n = num_samples if num_samples is not None else DEFAULT_SAMPLES
    if n <= 0:
        raise ValueError(f"numSamples must be > 0, got {n}")
    norm = _resolve_weights(types.TYPE_SAMPLES, weights)

    # Per-member sample count = round(weight * total), minimum 1.
    per_member: dict[str, int] = {}
    for slug, w in norm.items():
        share = max(1, round(w * n))
        per_member[slug] = share

    async def _one(slug: str) -> dict[str, Any]:
        ctx, _, eff_extra = _merge_member_config(
            slug,
            global_context_length=context_length,
            global_quantile_levels=None,
            global_extra=extra,
            overrides=member_overrides,
        )
        out = await models.get(slug).predict_samples(
            context, horizon, per_member[slug], ctx, extra=eff_extra,
        )
        await _maybe_unload(slug, unload_after)
        return out

    active = list(norm.keys())
    results = await asyncio.gather(*[_one(s) for s in active], return_exceptions=False)
    individual = {slug: {**res, "weight": norm[slug]} for slug, res in zip(active, results)}

    # Concat samples across members along the sample axis (per series).
    import numpy as np

    n_series = len(context)
    combined_samples: list[list[list[float]]] = []
    medians: list[list[float]] = []
    for i in range(n_series):
        per_series_chunks = [np.asarray(individual[s]["samples"][i]) for s in active]
        combined = np.concatenate(per_series_chunks, axis=0)  # [total_samples, horizon]
        combined_samples.append(combined.tolist())
        medians.append(np.median(combined, axis=0).tolist())

    total_samples = sum(per_member[s] for s in active)
    return {
        "model": "ensemble",
        "horizon": horizon,
        "num_samples": total_samples,
        "samples": combined_samples,
        "median": medians,
        "ensemble_members": active,
        "weights": norm,
        "individual": individual,
    }


# ─── shared aggregator for univariate-target quantile ensembles ──────────────


async def _aggregate_quantile_ensemble(
    type_slug: str,
    norm: dict[str, float],
    one_call,  # async fn slug -> result dict
    horizon: int,
    q_levels: list[float],
) -> dict[str, Any]:
    active = list(norm.keys())
    results = await asyncio.gather(*[one_call(s) for s in active], return_exceptions=False)
    individual = {slug: {**res, "weight": norm[slug]} for slug, res in zip(active, results)}

    n_series = len(individual[active[0]]["median"])
    median_avg = [
        [
            sum(individual[s]["median"][i][t] * norm[s] for s in active)
            for t in range(horizon)
        ]
        for i in range(n_series)
    ]
    quantiles_avg: dict[str, list[list[float]]] = {}
    for q_key in individual[active[0]]["quantiles"]:
        quantiles_avg[q_key] = [
            [
                sum(individual[s]["quantiles"][q_key][i][t] * norm[s] for s in active)
                for t in range(horizon)
            ]
            for i in range(n_series)
        ]

    log.debug("aggregated %s ensemble over %d members", type_slug, len(active))
    return {
        "model": "ensemble",
        "horizon": horizon,
        "quantile_levels": list(q_levels),
        "median": median_avg,
        "quantiles": quantiles_avg,
        "ensemble_members": active,
        "weights": norm,
        "individual": individual,
    }
