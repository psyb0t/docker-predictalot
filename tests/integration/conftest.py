"""Integration test harness.

Builds the right Docker image for the host (CUDA if nvidia-container-toolkit
is set up, otherwise CPU), starts it as `predictalot-int-<timestamp>` on a
free port with a host-mounted model cache, waits for /healthz, yields a
session-scoped fixture with the base URL + auth token, then docker rm -f's
the container on session teardown.

Image tags are throw-away (`psyb0t/predictalot-test:{cpu,cuda}`). Docker
layer caching makes subsequent builds fast.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Model cache lives in-tree (gitignored). Repo-relative so it follows the
# checkout; survives `make clean`; easy to nuke (`rm -rf tests/integration/.fixtures`).
# Override via PREDICTALOT_TEST_MODEL_CACHE if you want it elsewhere.
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "tests" / "integration" / ".fixtures" / "models"
MODEL_CACHE = Path(
    os.environ.get("PREDICTALOT_TEST_MODEL_CACHE", str(DEFAULT_MODEL_CACHE))
)


def _has_cuda() -> bool:
    """True if the host docker daemon has the nvidia runtime configured.

    We test only via `docker info` because integration tests may run inside a
    DIND dev container where the local `nvidia-smi` binary isn't installed —
    but the host docker daemon (reached via the mounted socket) still knows
    about its nvidia runtime.
    """
    try:
        out = subprocess.check_output(
            ["docker", "info"], stderr=subprocess.DEVNULL, timeout=10
        ).decode()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False
    # `docker info` lists `Runtimes: io.containerd... nvidia runc` when the
    # nvidia-container-toolkit is installed.
    return "nvidia" in out.lower()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_healthz(base_url: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/healthz", timeout=2.0)
            if r.status_code == 200:
                return
            last_err = RuntimeError(f"healthz returned {r.status_code}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.0)
    raise TimeoutError(
        f"healthz did not respond within {timeout}s on {base_url} (last error: {last_err})"
    )


CUDA_AVAILABLE = _has_cuda()


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return CUDA_AVAILABLE


@pytest.fixture(scope="session")
def predictalot_image() -> str:
    """Build the appropriate image for this host; return its tag."""
    if CUDA_AVAILABLE:
        tag = "psyb0t/predictalot-test:cuda"
        dockerfile = "Dockerfile.cuda"
    else:
        tag = "psyb0t/predictalot-test:cpu"
        dockerfile = "Dockerfile"

    print(f"\n[integration] building {tag} (from {dockerfile}) — first run may take ~10 min")
    subprocess.check_call(
        ["docker", "build", "-f", dockerfile, "-t", tag, str(PROJECT_ROOT)],
    )
    return tag


@pytest.fixture(scope="session")
def predictalot_container(predictalot_image: str):
    """Run the container, yield connection info, tear down on teardown."""
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    name = f"predictalot-int-{int(time.time())}"
    token = "integration-test-token"

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", name,
        "-v", f"{MODEL_CACHE}:/models",
        "-e", f"PREDICTALOT_AUTH_TOKENS={token}",
        "-p", f"{port}:8080",
    ]
    if CUDA_AVAILABLE:
        cmd += ["--gpus", "all"]
    cmd.append(predictalot_image)

    print(f"[integration] starting {name} on :{port} (cuda={CUDA_AVAILABLE})")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)

    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_healthz(base_url, timeout=180.0)
        yield {
            "base_url": base_url,
            "token": token,
            "name": name,
            "cuda": CUDA_AVAILABLE,
        }
    finally:
        print(f"[integration] removing {name}")
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


@pytest.fixture
def http_client(predictalot_container) -> Iterator[httpx.Client]:
    info = predictalot_container
    client = httpx.Client(
        base_url=info["base_url"],
        headers={"Authorization": f"Bearer {info['token']}"},
        timeout=600.0,  # first request per model downloads weights — can be slow
    )
    try:
        yield client
    finally:
        client.close()
