"""Sundial worker — sidecar FastAPI service running in /opt/sundial-venv.

Lives in its own venv with transformers==4.40.1 because Sundial's published
model code uses several DynamicCache internals that were removed in
transformers 4.42+ (seen_tokens, get_max_length, get_usable_length, plus a
4D-mask shape change deep in modeling_attn_mask_utils). Patching all of
them from the outside was too brittle — easier to just pin the version.

The main predictalot service talks to this worker over a unix socket
(default `/tmp/predictalot/sundial.sock`) using plain HTTP. From the main
service's perspective sundial is just another forecast backend.

Run via uvicorn:

    /opt/sundial-venv/bin/uvicorn sundial_worker.server:app \\
        --uds /tmp/predictalot/sundial.sock

Env vars (all read at startup):
    PREDICTALOT_SUNDIAL_MODEL_DIR   — local snapshot dir (default /models/sundial-base-128m)
    PREDICTALOT_DEVICE              — auto/cpu/cuda (default auto)
    PREDICTALOT_SUNDIAL_NUM_SAMPLES — samples for quantile estimation (default 64)
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


def _num_samples() -> int:
    try:
        return int(os.environ.get("PREDICTALOT_SUNDIAL_NUM_SAMPLES", "64"))
    except ValueError:
        return 64


def _load_sync() -> Any:
    """Download+load. Called in a thread because it does blocking I/O."""
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


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield


app = FastAPI(title="sundial_worker", lifespan=_lifespan)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "model": SLUG, "loaded": _model is not None}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> dict[str, Any]:
    if not req.context:
        raise HTTPException(status_code=400, detail="context must not be empty")
    for i, s in enumerate(req.context):
        if not s:
            raise HTTPException(status_code=400, detail=f"context[{i}] is empty")

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


def _forecast_sync(
    model: Any,
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    device = _device()
    n_samples = _num_samples()

    all_medians: list[list[float]] = []
    out_quantiles: dict[str, list[list[float]]] = {
        f"{q:.1f}": [] for q in quantile_levels
    }

    for series in context:
        sliced = series[-context_length:] if context_length > 0 else series
        x = torch.tensor(sliced, dtype=torch.float32, device=device).reshape(1, -1)

        with torch.no_grad():
            # Returns [batch, num_samples, horizon] — sundial generates
            # multiple sample paths and we take percentiles across them.
            out = model.generate(
                x, max_new_tokens=horizon, num_samples=n_samples, revin=True
            )
        # out: [1, n_samples, horizon]
        samples = out.detach().cpu().numpy()  # shape: (1, n_samples, horizon)

        # Median is the 0.5 quantile of the samples.
        median = _percentile_along_samples(samples, 0.5)
        all_medians.append(median[0].tolist())  # series 0 of [1, horizon]

        for q in quantile_levels:
            q_arr = _percentile_along_samples(samples, q)
            out_quantiles[f"{q:.1f}"].append(q_arr[0].tolist())

    return {
        "model": SLUG,
        "horizon": horizon,
        "quantile_levels": list(quantile_levels),
        "median": all_medians,
        "quantiles": out_quantiles,
    }


def _percentile_along_samples(samples, q: float):
    """samples shape: [batch, n_samples, horizon]. Returns [batch, horizon]."""
    import numpy as np

    return np.percentile(samples, q * 100, axis=1)
