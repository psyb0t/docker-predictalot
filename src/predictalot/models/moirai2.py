"""Moirai-2 backend.

Native output is 9 fixed quantiles (0.1..0.9). The Moirai2Module is cached
once; per-request we re-wrap it in a Moirai2Forecast(prediction_length=h,
context_length=c). The wrapper move-to-device is cheap.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any

from .. import config, storage
from ..device import resolve_device

SLUG = "moirai-2"

log = logging.getLogger(f"predictalot.models.{SLUG}")

_lock = asyncio.Lock()
_module: Any = None  # cached Moirai2Module (the weights)
_forecast: Any = None  # cached Moirai2Forecast wrapper at WRAPPER_CONTEXT_LENGTH
_last_used: float | None = None

_NATIVE_QUANTILES: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 10))

# Wrapper dimensions come from env-configurable settings in config.py:
#   PREDICTALOT_MOIRAI_MAX_CONTEXT (default 4000)
#   PREDICTALOT_MOIRAI_MAX_HORIZON (default 512)
# These are baked into Moirai2Forecast at model-load time. Per-request inputs
# shorter than the max-context are zero-padded with past_is_pad=True on the
# padded positions; horizons must be <= max-horizon, and the wrapper output
# is sliced to the actual requested horizon.


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
    """Load module + build forecast wrapper at WRAPPER_CONTEXT_LENGTH (once)."""
    global _module, _forecast
    if _module is not None:
        return _module
    async with _lock:
        if _module is not None:
            return _module
        path = await asyncio.to_thread(storage.ensure_snapshot, SLUG)
        log.info("loading moirai-2 from %s", path)
        _module, _forecast = await asyncio.to_thread(_load_sync, str(path))
        log.info(
            "moirai-2 loaded (wrapper context_length=%d, prediction_length=%d)",
            config.MOIRAI_MAX_CONTEXT,
            config.MOIRAI_MAX_HORIZON,
        )
        return _module


def _load_sync(path: str) -> tuple[Any, Any]:
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    module = Moirai2Module.from_pretrained(path)
    # Build the forecast wrapper once with our fixed context length. The
    # prediction_length is also baked in at this point; we use the max
    # horizon any reasonable caller would ask for. (Horizons shorter than
    # this just get sliced at the wrapper output side.)
    forecast = Moirai2Forecast(
        module=module,
        prediction_length=config.MOIRAI_MAX_HORIZON,
        context_length=config.MOIRAI_MAX_CONTEXT,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(resolve_device())
    return module, forecast


async def unload() -> None:
    global _module, _forecast, _last_used
    async with _lock:
        if _module is None:
            return
        log.info("unloading moirai-2")
        _module = None
        _forecast = None
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
    if horizon > config.MOIRAI_MAX_HORIZON:
        raise HorizonTooLargeError(
            f"moirai-2: horizon {horizon} > max {config.MOIRAI_MAX_HORIZON} "
            f"(compile-time cap). Increase PREDICTALOT_MOIRAI_MAX_HORIZON and "
            f"restart to support larger horizons."
        )
    await get_model()  # ensures _module + _forecast are populated
    async with _lock:
        result = await asyncio.to_thread(
            _predict_sync, _forecast, context, horizon, quantile_levels, context_length
        )
        _bump_last_used()
        return result


def _predict_sync(
    forecast: Any,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    device = resolve_device()
    wrapper_ctx = config.MOIRAI_MAX_CONTEXT  # the dimension the wrapper expects

    # The wrapper's forward expects past_target of exactly shape
    # [B, wrapper_ctx, target_dim]. Build that by:
    #   - slicing the user series to min(actual_len, context_length, wrapper_ctx)
    #   - left-padding with zeros to wrapper_ctx
    #   - flagging padded positions in past_is_pad
    all_quantiles = []  # list of [horizon, 9] per series
    for series in context:
        # Step 1: enforce user's context_length cap; clamp to wrapper_ctx.
        effective_ctx = min(context_length if context_length > 0 else wrapper_ctx, wrapper_ctx)
        sliced = series[-effective_ctx:]
        actual_len = len(sliced)

        # Step 2: left-pad to wrapper_ctx.
        padded = np.zeros(wrapper_ctx, dtype=np.float32)
        padded[-actual_len:] = sliced
        past_target = torch.from_numpy(padded.reshape(1, wrapper_ctx, 1)).to(device)

        # Step 3: observed/pad masks — first (wrapper_ctx - actual_len) are pad.
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
        # forward returns [batch, num_quantiles, future_time, *tgt].
        # future_time == WRAPPER_PREDICTION_LENGTH; we slice to user's horizon.
        out_arr = out.detach().cpu().numpy()
        if out_arr.ndim == 4:
            out_arr = out_arr[..., 0]
        # out_arr shape [batch, 9, future_time] → slice future_time, transpose
        out_arr = out_arr[:, :, :horizon]
        all_quantiles.append(out_arr[0].T)  # [horizon, 9]

    out_quantiles: dict[str, list[list[float]]] = {}
    median_list: list[list[float]] = []
    median_idx = _NATIVE_QUANTILES.index(0.5)

    for q_level in quantile_levels:
        try:
            q_idx = _NATIVE_QUANTILES.index(round(q_level, 1))
        except ValueError as exc:
            raise ValueError(
                f"moirai-2: quantile level {q_level} not in supported set {_NATIVE_QUANTILES}"
            ) from exc
        out_quantiles[_quantile_key(q_level)] = [
            series_q[:, q_idx].tolist() for series_q in all_quantiles
        ]

    median_list = [series_q[:, median_idx].tolist() for series_q in all_quantiles]

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": median_list,
        "quantiles": out_quantiles,
    }


def _quantile_key(q: float) -> str:
    return f"{q:.1f}"
