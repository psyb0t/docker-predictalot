"""Sundial backend (thuml/sundial-base-128m) — talks to a sidecar worker.

Sundial's published model code uses transformers 4.40 internals that were
removed in 4.42+ (DynamicCache.seen_tokens / get_max_length /
get_usable_length, plus a 4D-mask shape change). Shimming them from the
outside breaks deeper inside transformers itself; sundial really does need
transformers==4.40.

So sundial lives in its own venv (`/opt/sundial-venv`), runs as a tiny
background FastAPI worker on a unix socket, and we talk to it like any
other forecast backend. From the predictalot API's perspective, sundial
looks identical to chronos/timesfm/moirai/toto — same wire shape, same
predict() interface, same per-model lock + lazy-load semantics.

If the worker isn't reachable (still booting after container start, or
crashed and being restarted by the entrypoint's restart loop), this
backend returns 503 — same as a snapshot download failure would.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

SLUG = "sundial-base-128m"

log = logging.getLogger(f"predictalot.models.{SLUG}")

# Sundial worker connection — unix socket path is fixed by the entrypoint
# (with env override for tests).
SUNDIAL_SOCK = os.environ.get(
    "PREDICTALOT_SUNDIAL_SOCK", "/tmp/predictalot/sundial.sock"
)
# How long to wait for the worker to come up on first request.
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
    """Build the httpx client lazily so import time stays cheap."""
    global _client
    if _client is None:
        transport = httpx.AsyncHTTPTransport(uds=SUNDIAL_SOCK)
        _client = httpx.AsyncClient(
            transport=transport, base_url="http://sundial", timeout=600.0
        )
    return _client


async def _wait_for_worker(timeout: float = WORKER_READY_TIMEOUT) -> None:
    """Block until /healthz on the worker returns 200."""
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
    """For sundial, 'loading' means the sidecar worker is reachable + healthy.
    The actual weights load lives inside the worker process."""
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
    """Mark unloaded on our side; we don't tear down the worker process
    itself (it's managed by the entrypoint restart loop). The worker keeps
    its own model loaded indefinitely — its memory is its own concern."""
    global _loaded, _last_used
    async with _lock:
        if not _loaded:
            return
        log.info("marking sundial unloaded (worker process untouched)")
        _loaded = False
        _last_used = None


async def predict(
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
        try:
            r = await client.post("/forecast", json=body)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"sundial worker request failed: {exc}") from exc
        if r.status_code != 200:
            raise RuntimeError(
                f"sundial worker returned {r.status_code}: {r.text[:200]}"
            )
        _bump_last_used()
        return r.json()
