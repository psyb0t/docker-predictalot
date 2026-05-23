"""Moirai-2 backend.

Native output is 9 fixed quantiles (0.1..0.9). The Moirai2Module (weights)
is cached once; per-mode we build a Moirai2Forecast wrapper with the right
target_dim / past_feat_dynamic_real_dim. Wrappers are cached by their
dimension tuple — rewrapping is cheap (just re-binds the module).

Supported types: univariate, multivariate (UPSTREAM-UNTESTED, see footgun
in `.research_files/moirai2-modes.md`), covariates-past.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any

from .. import config, storage, types
from ..device import resolve_device

SLUG = "moirai-2"

SUPPORTED_TYPES: frozenset[str] = frozenset(
    {
        types.TYPE_UNIVARIATE,
        types.TYPE_MULTIVARIATE,
        types.TYPE_COVARIATES_PAST,
    }
)

log = logging.getLogger(f"predictalot.models.{SLUG}")

_lock = asyncio.Lock()
_module: Any = None  # cached Moirai2Module (the weights)
# Wrapper cache: (target_dim, past_feat_dim) -> Moirai2Forecast
_wrappers: dict[tuple[int, int], Any] = {}
_last_used: float | None = None

_NATIVE_QUANTILES: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 10))


class HorizonTooLargeError(ValueError):
    pass


def loaded() -> bool:
    return _module is not None


def last_used_secs_ago() -> float | None:
    if _last_used is None:
        return None
    return time.monotonic() - _last_used


def _bump_last_used() -> None:
    global _last_used
    _last_used = time.monotonic()


async def get_model() -> Any:
    """Load the Moirai2Module weights + build the default (univariate) wrapper."""
    global _module
    if _module is not None:
        return _module
    async with _lock:
        if _module is not None:
            return _module
        path = await asyncio.to_thread(storage.ensure_snapshot, SLUG)
        log.info("loading moirai-2 from %s", path)
        _module = await asyncio.to_thread(_load_module_sync, str(path))
        # Pre-build the univariate wrapper (most common path).
        _wrappers[(1, 0)] = _build_wrapper_sync(_module, target_dim=1, past_feat_dim=0)
        log.info(
            "moirai-2 loaded (wrapper context_length=%d, prediction_length=%d)",
            config.MOIRAI_MAX_CONTEXT,
            config.MOIRAI_MAX_HORIZON,
        )
        return _module


def _load_module_sync(path: str) -> Any:
    from uni2ts.model.moirai2 import Moirai2Module

    return Moirai2Module.from_pretrained(path)


def _build_wrapper_sync(module: Any, target_dim: int, past_feat_dim: int) -> Any:
    from uni2ts.model.moirai2 import Moirai2Forecast

    return Moirai2Forecast(
        module=module,
        prediction_length=config.MOIRAI_MAX_HORIZON,
        context_length=config.MOIRAI_MAX_CONTEXT,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=past_feat_dim,
    ).to(resolve_device())


def _get_or_build_wrapper(target_dim: int, past_feat_dim: int) -> Any:
    key = (target_dim, past_feat_dim)
    if key in _wrappers:
        return _wrappers[key]
    log.info(
        "moirai-2: building new wrapper target_dim=%d past_feat_dim=%d", target_dim, past_feat_dim
    )
    _wrappers[key] = _build_wrapper_sync(_module, target_dim, past_feat_dim)
    return _wrappers[key]


async def unload() -> None:
    global _module, _last_used
    async with _lock:
        if _module is None:
            return
        log.info("unloading moirai-2")
        _module = None
        _wrappers.clear()
        _last_used = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _check_horizon(horizon: int) -> None:
    if horizon > config.MOIRAI_MAX_HORIZON:
        raise HorizonTooLargeError(
            f"moirai-2: horizon {horizon} > max {config.MOIRAI_MAX_HORIZON} "
            f"(compile-time cap). Increase PREDICTALOT_MOIRAI_MAX_HORIZON and "
            f"restart to support larger horizons."
        )


# ─── univariate ───────────────────────────────────────────────────────────────


async def predict_univariate(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    _check_horizon(horizon)
    await get_model()
    async with _lock:
        forecast = _get_or_build_wrapper(target_dim=1, past_feat_dim=0)
        result = await asyncio.to_thread(
            _predict_univariate_sync, forecast, context, horizon, quantile_levels, context_length
        )
        _bump_last_used()
        return result


def _predict_univariate_sync(
    forecast: Any,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    device = resolve_device()
    wrapper_ctx = config.MOIRAI_MAX_CONTEXT

    all_quantiles = []
    for series in context:
        effective_ctx = min(context_length if context_length > 0 else wrapper_ctx, wrapper_ctx)
        sliced = series[-effective_ctx:]
        actual_len = len(sliced)

        padded = np.zeros(wrapper_ctx, dtype=np.float32)
        padded[-actual_len:] = sliced
        past_target = torch.from_numpy(padded.reshape(1, wrapper_ctx, 1)).to(device)

        past_observed = torch.ones((1, wrapper_ctx, 1), dtype=torch.bool, device=device)
        past_observed[:, : wrapper_ctx - actual_len, :] = False
        past_is_pad = torch.zeros((1, wrapper_ctx), dtype=torch.bool, device=device)
        past_is_pad[:, : wrapper_ctx - actual_len] = True

        with torch.no_grad():
            out = forecast(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
            )
        out_arr = out.detach().cpu().numpy()
        if out_arr.ndim == 4:
            out_arr = out_arr[..., 0]
        out_arr = out_arr[:, :, :horizon]
        all_quantiles.append(out_arr[0].T)  # [horizon, 9]

    return _pack_quantiles_1d(all_quantiles, horizon, quantile_levels)


# ─── multivariate ─────────────────────────────────────────────────────────────


async def predict_multivariate(
    context: list[list[list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    _check_horizon(horizon)
    await get_model()

    if not context or not context[0]:
        raise ValueError("multivariate context is empty")
    n_channels = len(context[0])
    if n_channels < 1:
        raise ValueError("multivariate series must have at least 1 channel")
    for i, series in enumerate(context):
        if len(series) != n_channels:
            raise ValueError(
                f"context[{i}] has {len(series)} channels; series 0 has {n_channels}. "
                f"moirai-2 multivariate requires uniform channel counts across series"
            )

    log.warning(
        "moirai-2 multivariate is upstream-untested (see Moirai2Forecast._format_preds "
        "docstring). Verify channel order matches input before relying on output."
    )

    async with _lock:
        forecast = _get_or_build_wrapper(target_dim=n_channels, past_feat_dim=0)
        result = await asyncio.to_thread(
            _predict_multivariate_sync,
            forecast,
            context,
            horizon,
            quantile_levels,
            context_length,
            n_channels,
        )
        _bump_last_used()
        return result


def _predict_multivariate_sync(
    forecast: Any,
    context: list[list[list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
    n_channels: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    device = resolve_device()
    wrapper_ctx = config.MOIRAI_MAX_CONTEXT

    per_series_outputs = []
    for series in context:  # series: list[channel][time]
        # All channels in this series must share length.
        lens = {len(ch) for ch in series}
        if len(lens) != 1:
            raise ValueError(
                f"multivariate series has channels of differing lengths {lens}; "
                f"all channels must align in time"
            )
        # Slice each channel to context length.
        sliced_channels = [
            ch[-context_length:] if context_length > 0 else ch for ch in series
        ]
        actual_len = len(sliced_channels[0])
        effective_actual = min(actual_len, wrapper_ctx)
        sliced_channels = [ch[-effective_actual:] for ch in sliced_channels]

        # past_target shape: [B=1, C=wrapper_ctx, N=n_channels]
        data = np.zeros((1, wrapper_ctx, n_channels), dtype=np.float32)
        for ch_idx, ch in enumerate(sliced_channels):
            data[0, -effective_actual:, ch_idx] = ch
        past_target = torch.from_numpy(data).to(device)

        past_observed = torch.zeros((1, wrapper_ctx, n_channels), dtype=torch.bool, device=device)
        past_observed[:, -effective_actual:, :] = True
        past_is_pad = torch.ones((1, wrapper_ctx), dtype=torch.bool, device=device)
        past_is_pad[:, -effective_actual:] = False

        with torch.no_grad():
            out = forecast(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
            )
        # Multivariate output shape: [B=1, 9, H_max, N]
        out_arr = out.detach().cpu().numpy()
        if out_arr.ndim == 3:
            # Defensive: rewrapped univariate -> add channel axis
            out_arr = out_arr[..., np.newaxis]
        out_arr = out_arr[:, :, :horizon, :]
        per_series_outputs.append(out_arr[0])  # [9, H, N]

    # Build response shape: median [series][channel][time], quantiles[key][series][channel][time]
    out_quantiles: dict[str, list[list[list[float]]]] = {}
    median_idx = _NATIVE_QUANTILES.index(0.5)
    for q_level in quantile_levels:
        try:
            q_idx = _NATIVE_QUANTILES.index(round(q_level, 1))
        except ValueError as exc:
            raise ValueError(
                f"moirai-2: quantile level {q_level} not in supported set {_NATIVE_QUANTILES}"
            ) from exc
        out_quantiles[_qkey(q_level)] = [
            # series_q: [9, H, N] -> swap to [N, H] for one quantile
            series_q[q_idx].T.tolist() for series_q in per_series_outputs
        ]
    median = [series_q[median_idx].T.tolist() for series_q in per_series_outputs]

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": median,
        "quantiles": out_quantiles,
    }


# ─── covariates: past only ────────────────────────────────────────────────────


async def predict_covariates_past(
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    _check_horizon(horizon)
    await get_model()

    n_series = len(context)
    if len(past_covariates) != n_series:
        raise ValueError(
            f"past_covariates length {len(past_covariates)} != context series count {n_series}"
        )
    # Every series must have the same covariate names (positional flat layout).
    ref_names = sorted(past_covariates[0].keys()) if past_covariates else []
    for i, block in enumerate(past_covariates):
        if sorted(block.keys()) != ref_names:
            raise ValueError(
                f"past_covariates[{i}] names {sorted(block.keys())} differ from "
                f"series 0 names {ref_names}; all series must share covariate names"
            )
    n_past_feat = len(ref_names)
    if n_past_feat == 0:
        raise ValueError(
            "covariates-past requires at least one past covariate per series; got 0"
        )

    async with _lock:
        forecast = _get_or_build_wrapper(target_dim=1, past_feat_dim=n_past_feat)
        result = await asyncio.to_thread(
            _predict_covariates_past_sync,
            forecast,
            context,
            past_covariates,
            ref_names,
            horizon,
            quantile_levels,
            context_length,
        )
        _bump_last_used()
        return result


def _predict_covariates_past_sync(
    forecast: Any,
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]],
    cov_names: list[str],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    device = resolve_device()
    wrapper_ctx = config.MOIRAI_MAX_CONTEXT
    K = len(cov_names)

    all_quantiles = []  # [series][horizon, 9]
    for i, target in enumerate(context):
        # Validate covariate alignment against target length.
        block = past_covariates[i]
        target_len = len(target)
        for name in cov_names:
            if len(block[name]) != target_len:
                raise ValueError(
                    f"past_covariates[{i}][{name!r}] length {len(block[name])} != "
                    f"target series length {target_len}"
                )

        effective_ctx = min(context_length if context_length > 0 else wrapper_ctx, wrapper_ctx)
        sliced_target = target[-effective_ctx:]
        actual_len = len(sliced_target)
        sliced_block = {name: block[name][-actual_len:] for name in cov_names}

        # past_target: [1, wrapper_ctx, 1]
        padded = np.zeros(wrapper_ctx, dtype=np.float32)
        padded[-actual_len:] = sliced_target
        past_target = torch.from_numpy(padded.reshape(1, wrapper_ctx, 1)).to(device)

        past_observed = torch.ones((1, wrapper_ctx, 1), dtype=torch.bool, device=device)
        past_observed[:, : wrapper_ctx - actual_len, :] = False
        past_is_pad = torch.zeros((1, wrapper_ctx), dtype=torch.bool, device=device)
        past_is_pad[:, : wrapper_ctx - actual_len] = True

        # past_feat_dynamic_real: [1, wrapper_ctx, K]
        cov_arr = np.zeros((wrapper_ctx, K), dtype=np.float32)
        for k_idx, name in enumerate(cov_names):
            cov_arr[-actual_len:, k_idx] = sliced_block[name]
        past_feat = torch.from_numpy(cov_arr.reshape(1, wrapper_ctx, K)).to(device)

        past_feat_observed = torch.zeros(
            (1, wrapper_ctx, K), dtype=torch.bool, device=device
        )
        past_feat_observed[:, -actual_len:, :] = True

        with torch.no_grad():
            out = forecast(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
                past_feat_dynamic_real=past_feat,
                past_observed_feat_dynamic_real=past_feat_observed,
            )
        out_arr = out.detach().cpu().numpy()
        if out_arr.ndim == 4:
            out_arr = out_arr[..., 0]
        out_arr = out_arr[:, :, :horizon]
        all_quantiles.append(out_arr[0].T)  # [horizon, 9]

    return _pack_quantiles_1d(all_quantiles, horizon, quantile_levels)


# ─── shared helpers ───────────────────────────────────────────────────────────


def _pack_quantiles_1d(
    all_quantiles: list[Any],  # list of [horizon, 9] numpy arrays
    horizon: int,
    quantile_levels: list[float],
) -> dict[str, Any]:
    out_quantiles: dict[str, list[list[float]]] = {}
    median_idx = _NATIVE_QUANTILES.index(0.5)

    for q_level in quantile_levels:
        try:
            q_idx = _NATIVE_QUANTILES.index(round(q_level, 1))
        except ValueError as exc:
            raise ValueError(
                f"moirai-2: quantile level {q_level} not in supported set {_NATIVE_QUANTILES}"
            ) from exc
        out_quantiles[_qkey(q_level)] = [
            series_q[:, q_idx].tolist() for series_q in all_quantiles
        ]

    median = [series_q[:, median_idx].tolist() for series_q in all_quantiles]

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": median,
        "quantiles": out_quantiles,
    }


def _qkey(q: float) -> str:
    return f"{q:.1f}"
