"""Sundial worker — sidecar FastAPI service running in /opt/sundial-venv.

Lives in its own venv with transformers==4.40.1 because Sundial's published
model code uses several DynamicCache internals that were removed in
transformers 4.42+. The main predictalot service talks to this worker over
a unix socket (default `/tmp/predictalot/sundial.sock`) using plain HTTP.

Endpoints:
    GET  /healthz
    POST /forecast   — quantile output (univariate type)
    POST /samples    — raw sample paths (samples type)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SLUG = "sundial-base-128m"
HF_REPO_ID = "thuml/sundial-base-128m"

log = logging.getLogger("sundial_worker")

_lock = asyncio.Lock()
_model: Any = None


def _device() -> str:
    d = os.environ.get("PREDICTALOT_DEVICE", "auto")
    if d != "auto":
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_dir() -> str:
    return os.environ.get(
        "PREDICTALOT_SUNDIAL_MODEL_DIR", "/models/sundial-base-128m"
    )


def _default_num_samples() -> int:
    try:
        return int(os.environ.get("PREDICTALOT_SUNDIAL_NUM_SAMPLES", "64"))
    except ValueError:
        return 64


def _load_sync() -> Any:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    path = _model_dir()
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "config.json")):
        log.info("downloading %s → %s", HF_REPO_ID, path)
        os.makedirs(path, exist_ok=True)
        snapshot_download(repo_id=HF_REPO_ID, local_dir=path)

    log.info("loading sundial from %s", path)
    m = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.float32
    )
    m = m.to(_device()).eval()
    return m


async def _get_model() -> Any:
    global _model
    if _model is not None:
        return _model
    async with _lock:
        if _model is not None:
            return _model
        _model = await asyncio.to_thread(_load_sync)
        return _model


class ForecastRequest(BaseModel):
    context: list[list[float]] = Field(..., description="One inner list per series.")
    horizon: int = Field(..., gt=0)
    quantile_levels: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    context_length: int = Field(default=2880, gt=0)


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    quantile_levels: list[float]
    median: list[list[float]]
    quantiles: dict[str, list[list[float]]]


class SamplesRequest(BaseModel):
    context: list[list[float]] = Field(..., description="One inner list per series.")
    horizon: int = Field(..., gt=0)
    num_samples: int | None = Field(default=None, gt=0)
    context_length: int = Field(default=2880, gt=0)


class SamplesResponse(BaseModel):
    model: str
    horizon: int
    num_samples: int
    samples: list[list[list[float]]]
    median: list[list[float]]


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield


app = FastAPI(title="sundial_worker", lifespan=_lifespan)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "model": SLUG, "loaded": _model is not None}


def _validate_context(context: list[list[float]]) -> None:
    if not context:
        raise HTTPException(status_code=400, detail="context must not be empty")
    for i, s in enumerate(context):
        if not s:
            raise HTTPException(status_code=400, detail=f"context[{i}] is empty")


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> dict[str, Any]:
    _validate_context(req.context)
    try:
        model = await _get_model()
    except Exception as exc:  # noqa: BLE001
        log.exception("failed to load sundial")
        raise HTTPException(status_code=503, detail=f"load failed: {exc}") from exc

    async with _lock:
        try:
            return await asyncio.to_thread(
                _forecast_sync,
                model,
                req.context,
                req.horizon,
                req.quantile_levels,
                req.context_length,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("sundial inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/samples", response_model=SamplesResponse)
async def samples(req: SamplesRequest) -> dict[str, Any]:
    _validate_context(req.context)
    try:
        model = await _get_model()
    except Exception as exc:  # noqa: BLE001
        log.exception("failed to load sundial")
        raise HTTPException(status_code=503, detail=f"load failed: {exc}") from exc

    n_samples = req.num_samples if req.num_samples is not None else _default_num_samples()

    async with _lock:
        try:
            return await asyncio.to_thread(
                _samples_sync,
                model,
                req.context,
                req.horizon,
                n_samples,
                req.context_length,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("sundial sample-gen failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


def _forecast_sync(
    model: Any,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    n_samples = _default_num_samples()

    all_medians: list[list[float]] = []
    out_quantiles: dict[str, list[list[float]]] = {
        f"{q:.1f}": [] for q in quantile_levels
    }

    for series in context:
        samples_arr = _generate_samples(model, series, horizon, n_samples, context_length)
        # samples_arr: (1, n_samples, horizon)
        median = _percentile_along_samples(samples_arr, 0.5)
        all_medians.append(median[0].tolist())
        for q in quantile_levels:
            q_arr = _percentile_along_samples(samples_arr, q)
            out_quantiles[f"{q:.1f}"].append(q_arr[0].tolist())

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": all_medians,
        "quantiles": out_quantiles,
    }


def _samples_sync(
    model: Any,
    context: list[list[float]],
    horizon: int,
    num_samples: int,
    context_length: int,
) -> dict[str, Any]:
    import numpy as np

    all_samples: list[list[list[float]]] = []
    all_medians: list[list[float]] = []

    for series in context:
        samples_arr = _generate_samples(model, series, horizon, num_samples, context_length)
        # samples_arr: (1, n_samples, horizon) → strip batch dim → (n_samples, horizon)
        s = samples_arr[0]
        all_samples.append(s.tolist())
        all_medians.append(np.median(s, axis=0).tolist())

    return {
        "model": SLUG,
        "horizon": horizon,
        "num_samples": num_samples,
        "samples": all_samples,
        "median": all_medians,
    }


def _generate_samples(
    model: Any,
    series: list[float],
    horizon: int,
    num_samples: int,
    context_length: int,
):
    """Run model.generate on a single series; return numpy (1, num_samples, horizon)."""
    device = _device()
    sliced = series[-context_length:] if context_length > 0 else series
    x = torch.tensor(sliced, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        out = model.generate(
            x, max_new_tokens=horizon, num_samples=num_samples, revin=True
        )
    return out.detach().cpu().numpy()


def _percentile_along_samples(samples, q: float):
    """samples shape: [batch, n_samples, horizon]. Returns [batch, horizon]."""
    import numpy as np

    return np.percentile(samples, q * 100, axis=1)
