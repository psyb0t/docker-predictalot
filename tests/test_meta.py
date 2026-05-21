from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealth:
    def test_healthz_unauthenticated(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


class TestListModels:
    def test_list_models_unauthenticated(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        slugs = [m["slug"] for m in body["models"]]
        assert sorted(slugs) == sorted(
            ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"]
        )
        for m in body["models"]:
            assert "loaded" in m
            assert "lastUsedSecsAgo" in m
            assert "idleTimeoutSecs" in m
