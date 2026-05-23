from __future__ import annotations

from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}


class TestHealth:
    def test_healthz_unauthenticated(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


class TestPerTypeModelsEndpoint:
    """Each forecast type has its own GET /v1/<type>/models endpoint."""

    def _expect_members(
        self, client: TestClient, url: str, type_slug: str, expected: list[str]
    ) -> None:
        resp = client.get(url)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["type"] == type_slug
        slugs = [m["slug"] for m in body["models"]]
        assert sorted(slugs) == sorted(expected)
        for m in body["models"]:
            assert "loaded" in m
            assert "lastUsedSecsAgo" in m
            assert "idleTimeoutSecs" in m

    def test_univariate_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/univariate/models",
            "univariate",
            ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"],
        )

    def test_multivariate_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/multivariate/models",
            "multivariate",
            ["chronos-2", "moirai-2", "toto-1"],
        )

    def test_covariates_past_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/covariates/past/models",
            "covariates-past",
            ["chronos-2", "moirai-2"],
        )

    def test_covariates_future_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/covariates/future/models",
            "covariates-future",
            ["chronos-2"],
        )

    def test_covariates_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/covariates/models",
            "covariates-both",
            ["chronos-2"],
        )

    def test_samples_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/samples/models",
            "samples",
            ["toto-1", "sundial-base-128m"],
        )

    def test_models_endpoints_unauthenticated(self, client: TestClient) -> None:
        """All /v1/<type>/models endpoints are publicly readable (no bearer)."""
        for url in (
            "/v1/univariate/models",
            "/v1/multivariate/models",
            "/v1/covariates/past/models",
            "/v1/covariates/future/models",
            "/v1/covariates/models",
            "/v1/samples/models",
        ):
            resp = client.get(url)
            assert resp.status_code == 200, f"{url}: {resp.text}"
