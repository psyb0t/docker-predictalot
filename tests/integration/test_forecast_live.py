"""End-to-end forecast tests against a real container with real ML libs.

First run per model downloads HuggingFace weights into the bind-mounted cache
(~120 MB chronos-2 + ~200 MB timesfm-2.5 + ~50 MB moirai-2). Cached on host
so subsequent runs are fast.

Covers the v0.2 type-routed surface — one univariate happy path per backend,
plus a smoke test for each per-type ``/forecast`` and the per-type
``/forecast/ensemble`` endpoints where >1 model supports the type.
"""

from __future__ import annotations

import math

import httpx
import pytest


pytestmark = pytest.mark.integration


UNIVARIATE_URL = "/v1/univariate/forecast"
UNIVARIATE_ENSEMBLE_URL = "/v1/univariate/forecast/ensemble"
MULTIVARIATE_URL = "/v1/multivariate/forecast"
COVARIATES_PAST_URL = "/v1/covariates/past/forecast"
COVARIATES_FUTURE_URL = "/v1/covariates/future/forecast"
COVARIATES_BOTH_URL = "/v1/covariates/forecast"
SAMPLES_URL = "/v1/samples/forecast"


def _sane_floats(values: list[float]) -> None:
    for v in values:
        assert isinstance(v, (int, float))
        assert not math.isnan(v), f"NaN in prediction: {values}"
        assert math.isfinite(v), f"non-finite in prediction: {values}"
        assert -1e9 < v < 1e9, f"absurd magnitude in prediction: {v}"


class TestUnivariateLive:
    @pytest.mark.parametrize(
        "model",
        ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"],
    )
    def test_single_series(self, http_client: httpx.Client, model: str) -> None:
        context = [[float(i) for i in range(1, 50)]]
        resp = http_client.post(
            UNIVARIATE_URL,
            json={"model": model, "context": context, "config": {"horizon": 5}},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == model
        assert body["horizon"] == 5
        assert body["quantileLevels"] == [0.1, 0.5, 0.9]

        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 5
        _sane_floats(body["median"][0])

        for q_key in ("0.1", "0.5", "0.9"):
            assert q_key in body["quantiles"]
            assert len(body["quantiles"][q_key]) == 1
            assert len(body["quantiles"][q_key][0]) == 5
            _sane_floats(body["quantiles"][q_key][0])

        # Quantile ordering with small tolerance for numerical noise.
        for step in range(5):
            q1 = body["quantiles"]["0.1"][0][step]
            q5 = body["quantiles"]["0.5"][0][step]
            q9 = body["quantiles"]["0.9"][0][step]
            assert q1 <= q5 + 1e-3, f"q0.1 > q0.5 at step {step}: {q1} > {q5}"
            assert q5 <= q9 + 1e-3, f"q0.5 > q0.9 at step {step}: {q5} > {q9}"

    @pytest.mark.parametrize(
        "model",
        ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"],
    )
    def test_batched_series(self, http_client: httpx.Client, model: str) -> None:
        context = [
            [float(i) for i in range(1, 30)],
            [float(i * 10) for i in range(1, 30)],
            [float(i * 100) for i in range(1, 30)],
        ]
        resp = http_client.post(
            UNIVARIATE_URL,
            json={"model": model, "context": context, "config": {"horizon": 3}},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["median"]) == 3
        for series in body["median"]:
            assert len(series) == 3
            _sane_floats(series)

    def test_custom_quantile_levels(self, http_client: httpx.Client) -> None:
        resp = http_client.post(
            UNIVARIATE_URL,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "config": {"horizon": 2, "quantileLevels": [0.2, 0.5, 0.8]},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["quantileLevels"] == [0.2, 0.5, 0.8]
        assert set(body["quantiles"].keys()) == {"0.2", "0.5", "0.8"}

    def test_unload_flag(self, http_client: httpx.Client) -> None:
        r1 = http_client.post(
            UNIVARIATE_URL,
            json={
                "model": "moirai-2",
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 1},
            },
        )
        assert r1.status_code == 200

        info_loaded = http_client.get("/v1/univariate/models").json()
        moirai_info = next(m for m in info_loaded["models"] if m["slug"] == "moirai-2")
        assert moirai_info["loaded"] is True

        r2 = http_client.post(
            UNIVARIATE_URL,
            json={
                "model": "moirai-2",
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 1},
                "unload": True,
            },
        )
        assert r2.status_code == 200

        info_unloaded = http_client.get("/v1/univariate/models").json()
        moirai_info = next(m for m in info_unloaded["models"] if m["slug"] == "moirai-2")
        assert moirai_info["loaded"] is False

    def test_ensemble(self, http_client: httpx.Client) -> None:
        context = [[float(i) for i in range(1, 50)]]
        resp = http_client.post(
            UNIVARIATE_ENSEMBLE_URL,
            json={"context": context, "config": {"horizon": 5}},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "ensemble"
        assert body["horizon"] == 5
        assert sorted(body["ensembleMembers"]) == sorted(
            ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"]
        )
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 5
        _sane_floats(body["median"][0])
        for q in ("0.1", "0.5", "0.9"):
            _sane_floats(body["quantiles"][q][0])


class TestMultivariateLive:
    @pytest.mark.parametrize("model", ["chronos-2", "moirai-2", "toto-1"])
    def test_two_channel_series(self, http_client: httpx.Client, model: str) -> None:
        # One series, two correlated channels, 30 obs.
        context = [
            [
                [float(i) for i in range(1, 31)],
                [float(i * 10) for i in range(1, 31)],
            ]
        ]
        resp = http_client.post(
            MULTIVARIATE_URL,
            json={"model": model, "context": context, "config": {"horizon": 3}},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == model
        assert body["horizon"] == 3
        # [series][channel][time]
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 2
        assert len(body["median"][0][0]) == 3
        for ch in body["median"][0]:
            _sane_floats(ch)


class TestCovariatesPastLive:
    @pytest.mark.parametrize("model", ["chronos-2", "moirai-2"])
    def test_with_two_past_covariates(
        self, http_client: httpx.Client, model: str
    ) -> None:
        ctx_len = 40
        context = [[float(i) for i in range(1, ctx_len + 1)]]
        past = [
            {
                "temp": [float(20 + (i % 5)) for i in range(ctx_len)],
                "promo": [float(i % 7 == 0) for i in range(ctx_len)],
            }
        ]
        resp = http_client.post(
            COVARIATES_PAST_URL,
            json={
                "model": model,
                "context": context,
                "pastCovariates": past,
                "config": {"horizon": 3},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == model
        assert body["horizon"] == 3
        assert len(body["median"]) == 1
        _sane_floats(body["median"][0])


class TestCovariatesFutureLive:
    def test_chronos2_with_future_only(self, http_client: httpx.Client) -> None:
        ctx_len = 40
        horizon = 3
        context = [[float(i) for i in range(1, ctx_len + 1)]]
        future = [{"price": [9.5, 9.6, 9.7][:horizon]}]
        resp = http_client.post(
            COVARIATES_FUTURE_URL,
            json={
                "model": "chronos-2",
                "context": context,
                "futureCovariates": future,
                "config": {"horizon": horizon},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["horizon"] == horizon
        _sane_floats(body["median"][0])


class TestCovariatesBothLive:
    def test_chronos2_with_past_and_future(self, http_client: httpx.Client) -> None:
        ctx_len = 40
        horizon = 3
        context = [[float(i) for i in range(1, ctx_len + 1)]]
        past = [{"price": [float(9 + (i % 3) * 0.1) for i in range(ctx_len)]}]
        future = [{"price": [9.5, 9.6, 9.7][:horizon]}]
        resp = http_client.post(
            COVARIATES_BOTH_URL,
            json={
                "model": "chronos-2",
                "context": context,
                "pastCovariates": past,
                "futureCovariates": future,
                "config": {"horizon": horizon},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["horizon"] == horizon
        _sane_floats(body["median"][0])


class TestSamplesLive:
    @pytest.mark.parametrize("model", ["toto-1", "sundial-base-128m"])
    def test_samples_shape_and_count(
        self, http_client: httpx.Client, model: str
    ) -> None:
        context = [[float(i) for i in range(1, 40)]]
        resp = http_client.post(
            SAMPLES_URL,
            json={
                "model": model,
                "context": context,
                "config": {"horizon": 3, "numSamples": 8},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == model
        assert body["numSamples"] == 8
        assert len(body["samples"]) == 1
        assert len(body["samples"][0]) == 8
        assert len(body["samples"][0][0]) == 3
        for path in body["samples"][0]:
            _sane_floats(path)
        _sane_floats(body["median"][0])


class TestCudaSmoke:
    def test_cuda_when_available(
        self, http_client: httpx.Client, cuda_available: bool
    ) -> None:
        """If we're running on a CUDA host, the container should be on GPU."""
        if not cuda_available:
            pytest.skip("no CUDA on host — running CPU image")

        # No direct API to ask "are you on GPU?". Trust --gpus all + a working
        # forecast as a sufficient smoke; a deeper check would docker-exec
        # nvidia-smi against the container.
        resp = http_client.post(
            UNIVARIATE_URL,
            json={
                "model": "moirai-2",
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 200
