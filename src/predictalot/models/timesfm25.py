"""TimesFM 2.5 backend.

Compile-time max_horizon (multiple of 128) + max_context (multiple of 32) are
baked in; per-request horizon over max_horizon → 400. Native output is 9
fixed quantiles (0.1..0.9) — we filter to the requested subset. Channel 0 of
the quantile_forecast is the point/median forecast.

Supported types: univariate only — TimesFM 2.5 has no native multivariate,
covariate, or sample-paths interface.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any

from .. import config, storage, types

SLUG = "timesfm-2.5"

SUPPORTED_TYPES: frozenset[str] = frozenset({types.TYPE_UNIVARIATE})

log = logging.getLogger(f"predictalot.models.{SLUG}")

_lock = asyncio.Lock()
_model: Any = None
_last_used: float | None = None

_NATIVE_QUANTILES: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 10))


class HorizonTooLargeError(ValueError):
    pass


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
        log.info("loading timesfm-2.5 from %s", path)
        _model = await asyncio.to_thread(_load_model_sync, str(path))
        log.info("timesfm-2.5 loaded")
        return _model


def _load_model_sync(path: str) -> Any:
    import json
    from pathlib import Path

    import timesfm
    from timesfm import ForecastConfig

    snapshot = Path(path)
    with open(snapshot / "config.json") as f:
        model_config = json.load(f)

    model = timesfm.TimesFM_2p5_200M_torch(config=model_config, torch_compile=False)
    model.model.load_checkpoint(
        str(snapshot / "model.safetensors"), torch_compile=False
    )
    model.compile(
        ForecastConfig(
            max_context=config.TIMESFM_MAX_CONTEXT,
            max_horizon=config.TIMESFM_MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


async def unload() -> None:
    global _model, _last_used
    async with _lock:
        if _model is None:
            return
        log.info("unloading timesfm-2.5")
        _model = None
        _last_used = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


async def predict_univariate(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if horizon > config.TIMESFM_MAX_HORIZON:
        raise HorizonTooLargeError(
            f"timesfm-2.5: horizon {horizon} > max_horizon {config.TIMESFM_MAX_HORIZON} "
            f"(compile-time cap). Increase PREDICTALOT_TIMESFM_MAX_HORIZON and restart."
        )
    model = await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_sync, model, context, horizon, quantile_levels, context_length
        )
        _bump_last_used()
        return result


def _predict_sync(
    model: Any,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import numpy as np

    wrapper_ctx = model.forecast_config.max_context

    values, masks = [], []
    for series in context:
        sliced = series[-context_length:] if context_length > 0 else series
        arr = np.asarray(sliced, dtype=np.float32)
        if arr.size >= wrapper_ctx:
            arr = arr[-wrapper_ctx:]
        else:
            arr = np.pad(arr, (wrapper_ctx - arr.size, 0), mode="edge")
        values.append(arr)
        masks.append(np.zeros(wrapper_ctx, dtype=bool))

    point_forecast, quantile_forecast = model.compiled_decode(horizon, values, masks)
    q_arr = np.asarray(quantile_forecast)
    median = np.asarray(point_forecast).tolist()

    out_quantiles: dict[str, list[list[float]]] = {}
    for q_level in quantile_levels:
        try:
            channel = _NATIVE_QUANTILES.index(round(q_level, 1)) + 1
        except ValueError as exc:
            raise ValueError(
                f"timesfm-2.5: quantile level {q_level} not in supported set {_NATIVE_QUANTILES}"
            ) from exc
        out_quantiles[_qkey(q_level)] = [
            q_arr[b, :, channel].tolist() for b in range(q_arr.shape[0])
        ]

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": median,
        "quantiles": out_quantiles,
    }


def _qkey(q: float) -> str:
    return f"{q:.1f}"
