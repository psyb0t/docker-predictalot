from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

HEADERS = {"Authorization": "Bearer testtoken"}


class TestHealth:
    def test_healthz_unauthenticated(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


class TestFoundationModelLifecycle:
    def test_unload_requires_bearer(self, client: TestClient) -> None:
        response = client.post("/v1/models/unload")

        assert response.status_code == 401

    def test_unload_reports_each_foundation_model(self, client: TestClient) -> None:
        response = client.post("/v1/models/unload", headers=HEADERS)

        assert response.status_code == 200, response.text
        assert response.json() == {
            "status": "unloaded",
            "models": [
                {"slug": "chronos-2", "wasLoaded": False},
                {"slug": "timesfm-2.5", "wasLoaded": False},
                {"slug": "moirai-2", "wasLoaded": False},
                {"slug": "toto-1", "wasLoaded": False},
                {"slug": "sundial-base-128m", "wasLoaded": False},
            ],
        }

    def test_unload_returns_conflict_while_a_forecast_is_active(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from predictalot.errors import ModelBusyError
        from predictalot.routers import models as models_router

        async def busy_unload() -> dict[str, object]:
            raise ModelBusyError("a foundation model is processing a forecast")

        monkeypatch.setattr(models_router.model_lifecycle, "unload_all", busy_unload)

        response = client.post("/v1/models/unload", headers=HEADERS)

        assert response.status_code == 409
        assert response.json() == {"detail": "a foundation model is processing a forecast"}

    def test_unload_hides_backend_failure_details(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from predictalot.errors import ModelUnloadError
        from predictalot.routers import models as models_router

        async def failed_unload() -> dict[str, object]:
            raise ModelUnloadError("worker socket at /private/path failed")

        monkeypatch.setattr(models_router.model_lifecycle, "unload_all", failed_unload)

        response = client.post("/v1/models/unload", headers=HEADERS)

        assert response.status_code == 503
        assert response.json() == {"detail": "foundation-model unload failed"}


class TestPerTypeModelsEndpoint:
    """Each forecast type has its own GET /v1/<type>/models endpoint."""

    def _expect_members(
        self, client: TestClient, url: str, type_slug: str, expected: list[str]
    ) -> None:
        resp = client.get(url, headers=HEADERS)
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
            "/v1/timeseries/univariate/models",
            "univariate",
            ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"],
        )

    def test_multivariate_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/timeseries/multivariate/models",
            "multivariate",
            ["chronos-2", "moirai-2", "toto-1"],
        )

    def test_covariates_past_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/timeseries/covariates/past/models",
            "covariates-past",
            ["chronos-2", "moirai-2"],
        )

    def test_covariates_future_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/timeseries/covariates/future/models",
            "covariates-future",
            ["chronos-2"],
        )

    def test_covariates_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/timeseries/covariates/models",
            "covariates-both",
            ["chronos-2"],
        )

    def test_samples_models(self, client: TestClient) -> None:
        self._expect_members(
            client,
            "/v1/timeseries/samples/models",
            "samples",
            ["toto-1", "sundial-base-128m"],
        )

    def test_models_endpoints_require_bearer(self, client: TestClient) -> None:
        """All /v1/<type>/models endpoints reject unauthenticated reads.

        Listing reveals which models are installed, their loaded state and
        last-used timestamps — usage-pattern information that must not leak
        to unauthenticated callers.
        """
        for url in (
            "/v1/timeseries/univariate/models",
            "/v1/timeseries/multivariate/models",
            "/v1/timeseries/covariates/past/models",
            "/v1/timeseries/covariates/future/models",
            "/v1/timeseries/covariates/models",
            "/v1/timeseries/samples/models",
        ):
            resp = client.get(url)
            assert resp.status_code == 401, f"{url}: {resp.status_code} {resp.text}"
            resp_bad = client.get(url, headers={"Authorization": "Bearer WRONG"})
            assert resp_bad.status_code == 401, f"{url}: {resp_bad.text}"

    def test_models_endpoints_open_when_auth_disabled(self, open_client: TestClient) -> None:
        """With auth disabled (ALLOW_NO_AUTH + empty token list) /models is open."""
        for url in (
            "/v1/timeseries/univariate/models",
            "/v1/timeseries/multivariate/models",
            "/v1/timeseries/covariates/past/models",
            "/v1/timeseries/covariates/future/models",
            "/v1/timeseries/covariates/models",
            "/v1/timeseries/samples/models",
        ):
            resp = open_client.get(url)
            assert resp.status_code == 200, f"{url}: {resp.text}"
