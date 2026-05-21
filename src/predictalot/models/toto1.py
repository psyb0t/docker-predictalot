"""Toto-1 backend (Datadog/Toto-Open-Base-1.0).

Decoder-only transformer with Proportional Factorized Space-Time Attention,
multivariate-native, Student-T mixture probabilistic output. We sample N
times and take empirical percentiles to populate the quantiles dict.

Installed in the Dockerfile via `toto-ts==0.1.4 --no-deps` because the
package pins torch==2.7.0 / transformers==4.52.1 / gluonts==0.15.1 which all
conflict with the chronos/uni2ts stack. Verified to work against our actual
torch 2.4.1 + transformers 4.57.6 + gluonts 0.14.4 in a manual smoke test
before integration.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any

from .. import storage
from ..device import resolve_device

SLUG = "toto-1"

log = logging.getLogger(f"predictalot.models.{SLUG}")

_lock = asyncio.Lock()
_model: Any = None
_forecaster: Any = None
_last_used: float | None = None

# Number of Monte-Carlo samples for quantile estimation. Higher = smoother
# quantile estimates but linearly more expensive. 256 is the value used in
# Toto's own quick-start notebook.
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


async def predict(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    await get_model()
    async with _lock:
        result = await asyncio.to_thread(
            _predict_sync, _forecaster, context, horizon, quantile_levels, context_length
        )
        _bump_last_used()
        return result


def _predict_sync(
    forecaster: Any,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    import torch
    from toto.data.util.dataset import MaskedTimeseries

    device = resolve_device()

    # Process each series independently (variable-length safe). Toto natively
    # supports multi-channel batching but our wire shape allows per-series
    # different lengths, so we loop.
    all_medians: list[list[float]] = []
    out_quantiles: dict[str, list[list[float]]] = {
        _quantile_key(q): [] for q in quantile_levels
    }

    for series in context:
        sliced = series[-context_length:] if context_length > 0 else series
        arr = torch.tensor(sliced, dtype=torch.float32, device=device).reshape(1, -1)
        inputs = MaskedTimeseries(
            series=arr,
            padding_mask=torch.ones_like(arr, dtype=torch.bool),
            id_mask=torch.zeros_like(arr, dtype=torch.long),
            # Toto's API expects these fields but the current model release
            # does not use them — any sane placeholder works.
            timestamp_seconds=torch.zeros_like(arr),
            time_interval_seconds=torch.full((1,), 60, device=device),
        )
        with torch.no_grad():
            result = forecaster.forecast(
                inputs,
                prediction_length=horizon,
                num_samples=NUM_SAMPLES,
                samples_per_batch=NUM_SAMPLES,
            )
        # result.median / quantile(q): shape [batch=1, channels=1, horizon].
        # Squeeze both leading dims to get a 1D list of length `horizon`.
        all_medians.append(result.median[0, 0].detach().cpu().tolist())
        for q in quantile_levels:
            q_vals = result.quantile(q)[0, 0].detach().cpu().tolist()
            out_quantiles[_quantile_key(q)].append(q_vals)

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": all_medians,
        "quantiles": out_quantiles,
    }


def _quantile_key(q: float) -> str:
    return f"{q:.1f}"
