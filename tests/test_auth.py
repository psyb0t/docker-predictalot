from __future__ import annotations

from fastapi.testclient import TestClient


class TestBearerAuth:
    def test_no_token_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            json={"model": "chronos-2", "context": [[1.0, 2.0]], "config": {"horizon": 1}},
        )
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "chronos-2", "context": [[1.0, 2.0]], "config": {"horizon": 1}},
        )
        assert resp.status_code == 401

    def test_correct_token_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers={"Authorization": "Bearer testtoken"},
            json={"model": "chronos-2", "context": [[1.0, 2.0]], "config": {"horizon": 1}},
        )
        assert resp.status_code == 200

    def test_query_param_token_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast?apiToken=testtoken",
            json={"model": "chronos-2", "context": [[1.0, 2.0]], "config": {"horizon": 1}},
        )
        assert resp.status_code == 200

    def test_open_mode_allows_unauthed(self, open_client: TestClient) -> None:
        resp = open_client.post(
            "/v1/forecast",
            json={"model": "chronos-2", "context": [[1.0, 2.0]], "config": {"horizon": 1}},
        )
        assert resp.status_code == 200
