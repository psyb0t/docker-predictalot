"""Chronos-2 backend.

Native API accepts arbitrary quantile_levels in {0.1, ..., 0.9}. Median comes
from the `mean` return value (q=0.5 by Chronos's convention).
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any

from .. import storage
from ..device import resolve_device

SLUG = "chronos-2"

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
    # Lazy imports — heavy ML deps only touched when actually loading.
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


async def predict(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    model = await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_sync, model, context, horizon, quantile_levels, context_length
        )
        _bump_last_used()
        return result


def _predict_sync(
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
    # Chronos-2 returns:
    #   quantiles: list[Tensor], one tensor per input series, shape [n_variates, H, Q]
    #   mean:      list[Tensor], one tensor per input series, shape [n_variates, H]
    # For univariate input (our case), n_variates == 1. Squeeze it.

    out_quantiles: dict[str, list[list[float]]] = {}
    for q_idx, q_level in enumerate(quantile_levels):
        key = _quantile_key(q_level)
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


def _quantile_key(q: float) -> str:
    return f"{q:.1f}"
