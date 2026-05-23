"""Chronos-2 backend.

Native API accepts arbitrary quantile_levels in {0.1, ..., 0.9}. Median comes
from the `mean` return value (q=0.5 by Chronos's convention).

Supported types: univariate, multivariate, covariates-past, covariates-future,
covariates-both.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any

from .. import storage, types
from ..device import resolve_device

SLUG = "chronos-2"

SUPPORTED_TYPES: frozenset[str] = frozenset(
    {
        types.TYPE_UNIVARIATE,
        types.TYPE_MULTIVARIATE,
        types.TYPE_COVARIATES_PAST,
        types.TYPE_COVARIATES_FUTURE,
        types.TYPE_COVARIATES_BOTH,
    }
)

log = logging.getLogger(f"predictalot.models.{SLUG}")

_lock = asyncio.Lock()
_model: Any = None
_last_used: float | None = None


def loaded() -> bool:
    return _model is not None


def last_used_secs_ago() -> float | None:
    if _last_used is None:
        return None
    return time.monotonic() - _last_used


def _bump_last_used() -> None:
    global _last_used
    _last_used = time.monotonic()


async def get_model() -> Any:
    global _model
    if _model is not None:
        return _model
    async with _lock:
        if _model is not None:
            return _model
        path = await asyncio.to_thread(storage.ensure_snapshot, SLUG)
        log.info("loading chronos-2 from %s", path)
        _model = await asyncio.to_thread(_load_model_sync, str(path))
        log.info("chronos-2 loaded")
        return _model


def _load_model_sync(path: str) -> Any:
    import torch
    from chronos import BaseChronosPipeline

    device = resolve_device()
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    pipeline = BaseChronosPipeline.from_pretrained(
        path,
        device_map=device,
        torch_dtype=dtype,
    )
    return pipeline


async def unload() -> None:
    global _model, _last_used
    async with _lock:
        if _model is None:
            return
        log.info("unloading chronos-2")
        _model = None
        _last_used = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ─── univariate ───────────────────────────────────────────────────────────────


async def predict_univariate(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    model = await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_univariate_sync, model, context, horizon, quantile_levels, context_length
        )
        _bump_last_used()
        return result


def _predict_univariate_sync(
    pipeline: Any,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import torch

    sliced = [series[-context_length:] if context_length > 0 else series for series in context]
    inputs = [torch.tensor(s, dtype=torch.float32) for s in sliced]

    quantiles, mean = pipeline.predict_quantiles(
        inputs=inputs,
        prediction_length=horizon,
        quantile_levels=quantile_levels,
    )
    # quantiles[i]: [n_variates=1, H, Q]; mean[i]: [n_variates=1, H].
    out_quantiles: dict[str, list[list[float]]] = {}
    for q_idx, q_level in enumerate(quantile_levels):
        key = _qkey(q_level)
        out_quantiles[key] = [
            q_tensor[0, :, q_idx].detach().cpu().tolist() for q_tensor in quantiles
        ]

    median = [m_tensor[0].detach().cpu().tolist() for m_tensor in mean]

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": median,
        "quantiles": out_quantiles,
    }


# ─── multivariate ─────────────────────────────────────────────────────────────


async def predict_multivariate(
    context: list[list[list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    model = await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_multivariate_sync, model, context, horizon, quantile_levels, context_length
        )
        _bump_last_used()
        return result


def _predict_multivariate_sync(
    pipeline: Any,
    context: list[list[list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import torch

    # context: [series][channel][time]. Chronos accepts a list of 2D tensors
    # of shape (n_variates, history_length); variable per series.
    inputs: list[torch.Tensor] = []
    for series in context:
        n_var = len(series)
        if n_var < 1:
            raise ValueError("multivariate series must have at least 1 channel")
        # Slice each channel to context_length (trailing window).
        sliced = [ch[-context_length:] if context_length > 0 else ch for ch in series]
        # All channels in one series must share length; enforce.
        lens = {len(ch) for ch in sliced}
        if len(lens) != 1:
            raise ValueError(
                f"multivariate series has channels of differing lengths {lens}; "
                f"all channels of a series must align in time"
            )
        arr = torch.tensor(sliced, dtype=torch.float32)  # [n_variates, history]
        inputs.append(arr)

    quantiles, mean = pipeline.predict_quantiles(
        inputs=inputs,
        prediction_length=horizon,
        quantile_levels=quantile_levels,
    )
    # quantiles[i]: [n_variates, H, Q]; mean[i]: [n_variates, H].
    out_quantiles: dict[str, list[list[list[float]]]] = {}
    for q_idx, q_level in enumerate(quantile_levels):
        key = _qkey(q_level)
        # per series → per channel → per time
        out_quantiles[key] = [
            q_tensor[:, :, q_idx].detach().cpu().tolist() for q_tensor in quantiles
        ]

    median = [m_tensor.detach().cpu().tolist() for m_tensor in mean]

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": median,
        "quantiles": out_quantiles,
    }


# ─── covariates (past / future / both) ────────────────────────────────────────


async def predict_covariates_past(
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    return await _run_covariates(
        context=context,
        past_covariates=past_covariates,
        future_covariates=None,
        horizon=horizon,
        quantile_levels=quantile_levels,
        context_length=context_length,
    )


async def predict_covariates_future(
    context: list[list[float]],
    future_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    # Chronos requires every future_covariate name to also appear in
    # past_covariates. For the "future-only" public type, synthesize a
    # past-covariate block per series by tiling the FIRST future value
    # back across the target history (any sane placeholder works — the
    # model gets a nan-free past block and the future block carries the
    # actual known-future information).
    past_covariates: list[dict[str, list[float]]] = []
    for s_idx, target in enumerate(context):
        future_block = future_covariates[s_idx] if s_idx < len(future_covariates) else {}
        past_block: dict[str, list[float]] = {}
        for name, vals in future_block.items():
            # Use the first known future value as a constant past stand-in;
            # consistent length = len(target).
            stand_in = vals[0] if vals else 0.0
            past_block[name] = [float(stand_in)] * len(target)
        past_covariates.append(past_block)

    return await _run_covariates(
        context=context,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        horizon=horizon,
        quantile_levels=quantile_levels,
        context_length=context_length,
    )


async def predict_covariates_both(
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]],
    future_covariates: list[dict[str, list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    return await _run_covariates(
        context=context,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        horizon=horizon,
        quantile_levels=quantile_levels,
        context_length=context_length,
    )


async def _run_covariates(
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]] | None,
    future_covariates: list[dict[str, list[float]]] | None,
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    model = await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_covariates_sync,
            model,
            context,
            past_covariates,
            future_covariates,
            horizon,
            quantile_levels,
            context_length,
        )
        _bump_last_used()
        return result


def _predict_covariates_sync(
    pipeline: Any,
    context: list[list[float]],
    past_covariates: list[dict[str, list[float]]] | None,
    future_covariates: list[dict[str, list[float]]] | None,
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import numpy as np

    n_series = len(context)
    if past_covariates is not None and len(past_covariates) != n_series:
        raise ValueError(
            f"past_covariates length {len(past_covariates)} != context series count {n_series}"
        )
    if future_covariates is not None and len(future_covariates) != n_series:
        raise ValueError(
            f"future_covariates length {len(future_covariates)} != context series count {n_series}"
        )

    inputs: list[dict[str, Any]] = []
    for i, target in enumerate(context):
        sliced = target[-context_length:] if context_length > 0 else target
        actual_ctx_len = len(sliced)
        entry: dict[str, Any] = {"target": np.asarray(sliced, dtype=np.float32)}

        if past_covariates is not None:
            past_block = past_covariates[i]
            past_out: dict[str, np.ndarray] = {}
            for name, vals in past_block.items():
                arr = np.asarray(vals, dtype=np.float32)
                if arr.shape != (len(target),):
                    raise ValueError(
                        f"past_covariates[{i}][{name!r}] length {arr.shape[0]} != "
                        f"target series length {len(target)}"
                    )
                # Match the context_length slice we did on the target.
                past_out[name] = arr[-actual_ctx_len:]
            if past_out:
                entry["past_covariates"] = past_out

        if future_covariates is not None:
            fut_block = future_covariates[i]
            fut_out: dict[str, np.ndarray] = {}
            for name, vals in fut_block.items():
                arr = np.asarray(vals, dtype=np.float32)
                if arr.shape != (horizon,):
                    raise ValueError(
                        f"future_covariates[{i}][{name!r}] length "
                        f"{arr.shape[0]} != horizon {horizon}"
                    )
                if past_covariates is not None and name not in past_covariates[i]:
                    raise ValueError(
                        f"future_covariates[{i}] has key {name!r} not present in "
                        f"past_covariates[{i}] — chronos-2 requires every future "
                        f"covariate to also have a past series"
                    )
                fut_out[name] = arr
            if fut_out:
                entry["future_covariates"] = fut_out

        inputs.append(entry)

    quantiles, mean = pipeline.predict_quantiles(
        inputs=inputs,
        prediction_length=horizon,
        quantile_levels=quantile_levels,
    )
    # univariate target → quantiles[i]: [1, H, Q]; mean[i]: [1, H].
    out_quantiles: dict[str, list[list[float]]] = {}
    for q_idx, q_level in enumerate(quantile_levels):
        out_quantiles[_qkey(q_level)] = [
            q_tensor[0, :, q_idx].detach().cpu().tolist() for q_tensor in quantiles
        ]
    median = [m_tensor[0].detach().cpu().tolist() for m_tensor in mean]

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": median,
        "quantiles": out_quantiles,
    }


def _qkey(q: float) -> str:
    return f"{q:.1f}"
