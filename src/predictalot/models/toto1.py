"""Toto-1 backend (Datadog/Toto-Open-Base-1.0).

Decoder-only transformer with Proportional Factorized Space-Time Attention,
multivariate-native, Student-T mixture probabilistic output. We sample N
times and take empirical percentiles to populate the quantiles dict.

Supported types: univariate, multivariate, samples.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any

from .. import storage, types
from ..device import resolve_device

SLUG = "toto-1"

SUPPORTED_TYPES: frozenset[str] = frozenset(
    {
        types.TYPE_UNIVARIATE,
        types.TYPE_MULTIVARIATE,
        types.TYPE_SAMPLES,
    }
)

log = logging.getLogger(f"predictalot.models.{SLUG}")

_lock = asyncio.Lock()
_model: Any = None
_forecaster: Any = None
_last_used: float | None = None

# Default sample count for quantile estimation. 256 mirrors the upstream notebook.
NUM_SAMPLES = 256


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
    global _model, _forecaster
    if _model is not None:
        return _model
    async with _lock:
        if _model is not None:
            return _model
        path = await asyncio.to_thread(storage.ensure_snapshot, SLUG)
        log.info("loading toto-1 from %s", path)
        _model, _forecaster = await asyncio.to_thread(_load_sync, str(path))
        log.info("toto-1 loaded")
        return _model


def _load_sync(path: str) -> tuple[Any, Any]:
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto

    model = Toto.from_pretrained(path).to(resolve_device()).eval()
    forecaster = TotoForecaster(model.model)
    return model, forecaster


async def unload() -> None:
    global _model, _forecaster, _last_used
    async with _lock:
        if _model is None:
            return
        log.info("unloading toto-1")
        _model = None
        _forecaster = None
        _last_used = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _samples_per_batch_for(num_samples: int) -> int:
    """Pick the largest factor of num_samples <= 64 (CPU-friendly chunk size).

    Toto's forecaster asserts num_samples % samples_per_batch == 0. On CPU,
    keeping the inner batch small avoids memory pressure; 64 is a sane cap.
    """
    for s in (64, 32, 16, 8, 4, 2, 1):
        if num_samples % s == 0 and s <= num_samples:
            return s
    return 1


def _make_inputs_univariate(series: list[float], context_length: int, device: str) -> Any:
    """Build a MaskedTimeseries for one univariate series. V=1."""
    import torch
    from toto.data.util.dataset import MaskedTimeseries

    sliced = series[-context_length:] if context_length > 0 else series
    arr = torch.tensor(sliced, dtype=torch.float32, device=device).reshape(1, -1)
    return MaskedTimeseries(
        series=arr,
        padding_mask=torch.ones_like(arr, dtype=torch.bool),
        id_mask=torch.zeros_like(arr, dtype=torch.long),
        timestamp_seconds=torch.zeros_like(arr, dtype=torch.long),
        time_interval_seconds=torch.full((1,), 60, device=device, dtype=torch.long),
    )


def _make_inputs_multivariate(
    series_channels: list[list[float]], context_length: int, device: str
) -> Any:
    """Build a MaskedTimeseries for one multivariate series with V channels.

    Channels are stacked along the variate axis with a shared id_mask so the
    model cross-attends across channels (per toto1-modes.md §1).
    """
    import torch
    from toto.data.util.dataset import MaskedTimeseries

    sliced = [ch[-context_length:] if context_length > 0 else ch for ch in series_channels]
    lens = {len(ch) for ch in sliced}
    if len(lens) != 1:
        raise ValueError(
            f"multivariate series has channels of differing lengths {lens}; "
            f"all channels must align in time"
        )
    n_var = len(sliced)
    arr = torch.tensor(sliced, dtype=torch.float32, device=device)  # [V, T]
    return MaskedTimeseries(
        series=arr,
        padding_mask=torch.ones_like(arr, dtype=torch.bool),
        id_mask=torch.zeros_like(arr, dtype=torch.long),
        timestamp_seconds=torch.zeros_like(arr, dtype=torch.long),
        time_interval_seconds=torch.full((n_var,), 60, device=device, dtype=torch.long),
    )


# ─── univariate ───────────────────────────────────────────────────────────────


async def predict_univariate(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_univariate_sync, _forecaster, context, horizon, quantile_levels, context_length
        )
        _bump_last_used()
        return result


def _predict_univariate_sync(
    forecaster: Any,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import torch

    device = resolve_device()
    all_medians: list[list[float]] = []
    out_quantiles: dict[str, list[list[float]]] = {_qkey(q): [] for q in quantile_levels}

    spb = _samples_per_batch_for(NUM_SAMPLES)
    for series in context:
        inputs = _make_inputs_univariate(series, context_length, device)
        with torch.no_grad():
            result = forecaster.forecast(
                inputs,
                prediction_length=horizon,
                num_samples=NUM_SAMPLES,
                samples_per_batch=spb,
            )
        # result.median / quantile(q): shape [1, 1, H]
        all_medians.append(result.median[0, 0].detach().cpu().tolist())
        for q in quantile_levels:
            out_quantiles[_qkey(q)].append(result.quantile(q)[0, 0].detach().cpu().tolist())

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": all_medians,
        "quantiles": out_quantiles,
    }


# ─── multivariate ─────────────────────────────────────────────────────────────


async def predict_multivariate(
    context: list[list[list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_multivariate_sync,
            _forecaster,
            context,
            horizon,
            quantile_levels,
            context_length,
        )
        _bump_last_used()
        return result


def _predict_multivariate_sync(
    forecaster: Any,
    context: list[list[list[float]]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import torch

    device = resolve_device()
    spb = _samples_per_batch_for(NUM_SAMPLES)

    out_quantiles: dict[str, list[list[list[float]]]] = {_qkey(q): [] for q in quantile_levels}
    all_medians: list[list[list[float]]] = []  # [series][channel][time]

    for series_channels in context:
        inputs = _make_inputs_multivariate(series_channels, context_length, device)
        with torch.no_grad():
            result = forecaster.forecast(
                inputs,
                prediction_length=horizon,
                num_samples=NUM_SAMPLES,
                samples_per_batch=spb,
            )
        # result.median: [1, V, H], result.quantile(q): [1, V, H]
        all_medians.append(result.median[0].detach().cpu().tolist())  # [V, H]
        for q in quantile_levels:
            out_quantiles[_qkey(q)].append(
                result.quantile(q)[0].detach().cpu().tolist()
            )

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": all_medians,
        "quantiles": out_quantiles,
    }


# ─── samples ──────────────────────────────────────────────────────────────────


async def predict_samples(
    context: list[list[float]],
    horizon: int,
    num_samples: int,
    context_length: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_samples_sync, _forecaster, context, horizon, num_samples, context_length
        )
        _bump_last_used()
        return result


def _predict_samples_sync(
    forecaster: Any,
    context: list[list[float]],
    horizon: int,
    num_samples: int,
    context_length: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    device = resolve_device()
    spb = _samples_per_batch_for(num_samples)

    all_samples: list[list[list[float]]] = []  # [series][sample][time]
    all_medians: list[list[float]] = []

    for series in context:
        inputs = _make_inputs_univariate(series, context_length, device)
        with torch.no_grad():
            result = forecaster.forecast(
                inputs,
                prediction_length=horizon,
                num_samples=num_samples,
                samples_per_batch=spb,
            )
        # result.samples: [1, V=1, H, N]
        samples = result.samples[0, 0].detach().cpu().numpy()  # [H, N]
        # Reshape to [N, H] (sample-paths first).
        samples_n_h = samples.T  # [N, H]
        all_samples.append(samples_n_h.tolist())
        median = np.median(samples_n_h, axis=0).tolist()
        all_medians.append(median)

    return {
        "model": SLUG,
        "horizon": horizon,
        "num_samples": num_samples,
        "samples": all_samples,
        "median": all_medians,
    }


def _qkey(q: float) -> str:
    return f"{q:.1f}"
