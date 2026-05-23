"""BodySizeLimitMiddleware — declared-content-length rejection + streaming
byte-count rejection for clients that omit Content-Length (chunked transfer)."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from predictalot.server import BodySizeLimitMiddleware


def _app_with_limit(max_bytes: int) -> TestClient:
    app = FastAPI()

    @app.post("/echo")
    async def echo(payload: dict) -> dict:
        return {"len": len(payload)}

    # Add as raw ASGI middleware — Starlette applies these around the routed
    # app, matching the production wiring in server.py.
    wrapped_app = BodySizeLimitMiddleware(app, max_bytes=max_bytes)
    return TestClient(wrapped_app)


def test_declared_content_length_over_limit_rejected() -> None:
    """Cheap rejection: oversize Content-Length header → 413 without the
    middleware needing to buffer the body."""
    client = _app_with_limit(64)
    huge_body = b'{"junk":"' + b"x" * 256 + b'"}'
    resp = client.post(
        "/echo",
        headers={"Content-Type": "application/json"},
        content=huge_body,
    )
    assert resp.status_code == 413
    assert "too large" in resp.text.lower()


def test_under_limit_passes() -> None:
    client = _app_with_limit(1024)
    resp = client.post("/echo", json={"a": 1})
    assert resp.status_code == 200
    assert resp.json() == {"len": 1}


def test_streaming_over_limit_rejected() -> None:
    """Streaming rejection: client sends NO Content-Length and pushes body
    chunks. The cheap declared-length check can't fire — only the
    ``_capped_receive`` byte counter can catch this. Exercises the entire
    point of the S1 rewrite.

    We drive the ASGI app directly because TestClient/httpx normalize
    Content-Length on byte payloads. Hand-crafting the ASGI receive
    sequence is the cleanest way to model a chunked transfer.
    """

    app = FastAPI()

    @app.post("/echo")
    async def echo(payload: dict) -> dict:
        return {"len": len(payload)}

    wrapped: Any = BodySizeLimitMiddleware(app, max_bytes=64)

    chunks = [b"x" * 40, b"x" * 40, b"x" * 40]  # 120 bytes, no CL header
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/echo",
        "raw_path": b"/echo",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"transfer-encoding", b"chunked"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8080),
    }

    recv_queue: list[dict[str, Any]] = [
        {"type": "http.request", "body": c, "more_body": True} for c in chunks
    ]
    recv_queue.append({"type": "http.request", "body": b"", "more_body": False})

    async def receive() -> dict[str, Any]:
        if recv_queue:
            return recv_queue.pop(0)
        return {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(wrapped(scope, receive, send))

    starts = [m for m in sent if m["type"] == "http.response.start"]
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert len(starts) == 1, sent  # exactly one response, no double-start
    assert starts[0]["status"] == 413, sent
    assert any(b"too large" in m.get("body", b"").lower() for m in bodies), sent
