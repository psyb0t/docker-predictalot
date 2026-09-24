"""Real container proof for full foundation-model teardown."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration


_SUNDIAL_SLUG = "sundial-base-128m"
_UNIVARIATE_FORECAST_PATH = "/v1/timeseries/univariate/forecast"
_UNLOAD_PATH = "/v1/models/unload"
_WORKER_SOCKET = "/tmp/predictalot/sundial.sock"
_WORKER_HEALTH_PATH = "/healthz"
_WORKER_PYTHON = "/opt/venv/bin/python"
_WORKER_HEALTH_SCRIPT = f"""\
import asyncio
import json
import httpx


async def main() -> None:
    transport = httpx.AsyncHTTPTransport(uds={_WORKER_SOCKET!r})
    async with httpx.AsyncClient(transport=transport, base_url='http://sundial') as client:
        response = await client.get({_WORKER_HEALTH_PATH!r})
        response.raise_for_status()
        print(json.dumps(response.json()))


asyncio.run(main())
"""


def _worker_health(container_name: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "docker",
            "exec",
            container_name,
            _WORKER_PYTHON,
            "-c",
            _WORKER_HEALTH_SCRIPT,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_public_unload_releases_the_real_sundial_worker(
    http_client: httpx.Client,
    predictalot_container: dict[str, object],
) -> None:
    forecast = http_client.post(
        _UNIVARIATE_FORECAST_PATH,
        json={
            "model": _SUNDIAL_SLUG,
            "context": [[float(index) for index in range(1, 50)]],
            "config": {"horizon": 2},
        },
    )
    assert forecast.status_code == 200, forecast.text
    assert _worker_health(str(predictalot_container["name"])) == {
        "ok": True,
        "model": _SUNDIAL_SLUG,
        "loaded": True,
    }

    unload = http_client.post(_UNLOAD_PATH)

    assert unload.status_code == 200, unload.text
    sundial_result = next(
        model for model in unload.json()["models"] if model["slug"] == _SUNDIAL_SLUG
    )
    assert sundial_result == {"slug": _SUNDIAL_SLUG, "wasLoaded": True}
    assert _worker_health(str(predictalot_container["name"])) == {
        "ok": True,
        "model": _SUNDIAL_SLUG,
        "loaded": False,
    }
