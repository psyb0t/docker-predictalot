"""Sundial backend (thuml/sundial-base-128m) — talks to a sidecar worker.

Sundial's published model code uses transformers 4.40 internals that were
removed in 4.42+, so it lives in its own venv (`/opt/sundial-venv`) and runs
as a tiny FastAPI worker on a unix socket. From the predictalot API's
perspective sundial looks identical to chronos/timesfm/moirai/toto.

Supported types: univariate, samples. Sundial is univariate-only at the
model level (see `.research_files/sundial-modes.md` §5).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from .. import types

SLUG = "sundial-base-128m"

SUPPORTED_TYPES: frozenset[str] = frozenset(
    {
        types.TYPE_UNIVARIATE,
        types.TYPE_SAMPLES,
    }
)

log = logging.getLogger(f"predictalot.models.{SLUG}")

SUNDIAL_SOCK = os.environ.get(
    "PREDICTALOT_SUNDIAL_SOCK", "/tmp/predictalot/sundial.sock"
)
WORKER_READY_TIMEOUT = float(
    os.environ.get("PREDICTALOT_SUNDIAL_READY_TIMEOUT", "60.0")
)

_lock = asyncio.Lock()
_client: httpx.AsyncClient | None = None
_loaded: bool = False
_last_used: float | None = None


def loaded() -> bool:
    return _loaded


def last_used_secs_ago() -> float | None:
    if _last_used is None:
        return None
    return time.monotonic() - _last_used


def _bump_last_used() -> None:
    global _last_used
    _last_used = time.monotonic()


def _get_or_create_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        transport = httpx.AsyncHTTPTransport(uds=SUNDIAL_SOCK)
        _client = httpx.AsyncClient(
            transport=transport, base_url="http://sundial", timeout=600.0
        )
    return _client


async def _wait_for_worker(timeout: float = WORKER_READY_TIMEOUT) -> None:
    client = _get_or_create_client()
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = await client.get("/healthz", timeout=2.0)
            if r.status_code == 200:
                return
            last_err = RuntimeError(f"worker /healthz returned {r.status_code}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        await asyncio.sleep(1.0)
    raise RuntimeError(
        f"sundial worker not reachable within {timeout}s on {SUNDIAL_SOCK} "
        f"(last error: {last_err})"
    )


async def get_model() -> Any:
    global _loaded
    if _loaded:
        return _get_or_create_client()
    async with _lock:
        if _loaded:
            return _get_or_create_client()
        log.info("waiting for sundial worker on %s", SUNDIAL_SOCK)
        await _wait_for_worker()
        _loaded = True
        log.info("sundial worker reachable")
        return _get_or_create_client()


async def unload() -> None:
    global _loaded, _last_used
    async with _lock:
        if not _loaded:
            return
        log.info("marking sundial unloaded (worker process untouched)")
        _loaded = False
        _last_used = None


# ─── univariate ───────────────────────────────────────────────────────────────


async def predict_univariate(
    context: list[list[float]],
    horizon: int,
    quantile_levels: list[float],
    context_length: int,
) -> dict[str, Any]:
    await get_model()
    async with _lock:
        client = _get_or_create_client()
        body = {
            "context": context,
            "horizon": horizon,
            "quantile_levels": list(quantile_levels),
            "context_length": context_length,
        }
        return await _post_json(client, "/forecast", body)


# ─── samples ──────────────────────────────────────────────────────────────────


async def predict_samples(
    context: list[list[float]],
    horizon: int,
    num_samples: int,
    context_length: int,
) -> dict[str, Any]:
    await get_model()
    async with _lock:
        client = _get_or_create_client()
        body = {
            "context": context,
            "horizon": horizon,
            "num_samples": num_samples,
            "context_length": context_length,
        }
        return await _post_json(client, "/samples", body)


async def _post_json(client: httpx.AsyncClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        r = await client.post(path, json=body)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"sundial worker request to {path} failed: {exc}") from exc
    if r.status_code != 200:
        raise RuntimeError(
            f"sundial worker {path} returned {r.status_code}: {r.text[:200]}"
        )
    _bump_last_used()
    return r.json()
