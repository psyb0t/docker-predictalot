"""MCP lifecycle tool contract through the mounted streamable HTTP server."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

_MCP_PATH = "/mcp/"
_MCP_SESSION_ID_HEADER = "Mcp-Session-Id"
_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Authorization": "Bearer testtoken",
    "Content-Type": "application/json",
    "Host": "localhost:8080",
}
_PROTOCOL_VERSION = "2025-11-25"
_INITIALIZE_METHOD = "initialize"
_INITIALIZED_METHOD = "notifications/initialized"
_TOOLS_CALL_METHOD = "tools/call"
_TOOLS_LIST_METHOD = "tools/list"
_UNLOAD_TOOL_NAME = "unload_models"
_SSE_DATA_PREFIX = "data: "
_MISSING_SSE_DATA_MESSAGE = "MCP response did not contain an SSE data frame"


@pytest.fixture
def mcp_client(client: TestClient) -> Iterator[TestClient]:
    with client:
        yield client


def test_mcp_unload_models_uses_the_shared_lifecycle(mcp_client: TestClient) -> None:
    initialize = mcp_client.post(
        _MCP_PATH,
        headers=_MCP_HEADERS,
        json=_message(
            _INITIALIZE_METHOD,
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "predictalot-test", "version": "1"},
            },
            request_id=1,
        ),
    )
    assert initialize.status_code == 200, initialize.text
    session_headers = {
        **_MCP_HEADERS,
        _MCP_SESSION_ID_HEADER: initialize.headers[_MCP_SESSION_ID_HEADER],
    }

    initialized = mcp_client.post(
        _MCP_PATH,
        headers=session_headers,
        json={"jsonrpc": "2.0", "method": _INITIALIZED_METHOD},
    )
    assert initialized.status_code in {200, 202}, initialized.text

    tools = mcp_client.post(
        _MCP_PATH,
        headers=session_headers,
        json=_message(_TOOLS_LIST_METHOD, {}, request_id=2),
    )
    assert tools.status_code == 200, tools.text
    tool_names = {tool["name"] for tool in _sse_response(tools)["result"]["tools"]}
    assert _UNLOAD_TOOL_NAME in tool_names

    unload = mcp_client.post(
        _MCP_PATH,
        headers=session_headers,
        json=_message(
            _TOOLS_CALL_METHOD,
            {"name": _UNLOAD_TOOL_NAME, "arguments": {}},
            request_id=3,
        ),
    )
    assert unload.status_code == 200, unload.text
    result = _sse_response(unload)["result"]
    assert result["isError"] is False
    body = json.loads(result["content"][0]["text"])
    assert body == {
        "status": "unloaded",
        "models": [
            {"slug": "chronos-2", "wasLoaded": False},
            {"slug": "timesfm-2.5", "wasLoaded": False},
            {"slug": "moirai-2", "wasLoaded": False},
            {"slug": "toto-1", "wasLoaded": False},
            {"slug": "sundial-base-128m", "wasLoaded": False},
        ],
    }


def _message(method: str, params: dict[str, object], request_id: int) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


def _sse_response(response: httpx.Response) -> dict[str, Any]:
    for line in response.text.splitlines():
        if line.startswith(_SSE_DATA_PREFIX):
            return cast(dict[str, Any], json.loads(line.removeprefix(_SSE_DATA_PREFIX)))

    raise AssertionError(f"{_MISSING_SSE_DATA_MESSAGE}: {response.text!r}")
