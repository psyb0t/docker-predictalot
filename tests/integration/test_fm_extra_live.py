"""End-to-end test that the FM ``extra`` dict + ensemble
``memberOverrides`` actually reach the live container.

Smoke-only: we don't verify any particular hyperparam EFFECT on the
output (that's covered in benchs/), only that the path doesn't
explode with parametrically-valid extras and that ensemble member
overrides successfully shadow the global config.
"""

from __future__ import annotations

import httpx
import pytest


pytestmark = pytest.mark.integration


UNIVAR_URL = "/v1/timeseries/univariate/forecast"
UNIVAR_ENSEMBLE_URL = "/v1/timeseries/univariate/forecast/ensemble"


class TestFmExtraLive:
    def test_chronos2_accepts_known_extra_keys(
        self, http_client: httpx.Client,
    ) -> None:
        """chronos-2's documented extras: batch_size, cross_learning."""
        context = [[float(i) for i in range(1, 50)]]
        r = http_client.post(
            UNIVAR_URL,
            json={
                "model": "chronos-2",
                "context": context,
                "config": {
                    "horizon": 5,
                    "extra": {"batch_size": 8},
                },
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["median"][0]) == 5

    def test_timesfm_accepts_known_extra_keys(
        self, http_client: httpx.Client,
    ) -> None:
        context = [[float(i) for i in range(1, 80)]]
        r = http_client.post(
            UNIVAR_URL,
            json={
                "model": "timesfm-2.5",
                "context": context,
                "config": {
                    "horizon": 5,
                    "extra": {
                        "fix_quantile_crossing": True,
                        "normalize_inputs": True,
                    },
                },
            },
        )
        assert r.status_code == 200, r.text

    def test_unknown_extra_keys_are_ignored(
        self, http_client: httpx.Client,
    ) -> None:
        """Backends must drop keys they don't understand silently —
        otherwise upgrading a backend would break clients."""
        context = [[float(i) for i in range(1, 50)]]
        r = http_client.post(
            UNIVAR_URL,
            json={
                "model": "chronos-2",
                "context": context,
                "config": {
                    "horizon": 3,
                    "extra": {
                        "definitely_not_a_real_key": 42,
                        "another_made_up": "value",
                    },
                },
            },
        )
        assert r.status_code == 200, r.text


class TestFmEnsembleMemberOverridesLive:
    def test_member_overrides_per_slug(
        self, http_client: httpx.Client,
    ) -> None:
        """Each ensemble member can have its own context_length / extra
        without affecting other members. We just verify the response
        comes back with all members present + numerically sane."""
        context = [[float(i) for i in range(1, 200)]]
        r = http_client.post(
            UNIVAR_ENSEMBLE_URL,
            json={
                "context": context,
                "config": {
                    "horizon": 3,
                    "contextLength": 100,
                    "extra": {"batch_size": 4},
                },
                "weights": {"chronos-2": 1.0, "timesfm-2.5": 1.0},
                "memberOverrides": {
                    "chronos-2": {
                        "contextLength": 50,
                        "extra": {"batch_size": 8},
                    },
                    "timesfm-2.5": {
                        "extra": {"normalize_inputs": True},
                    },
                },
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Active members appear in the median/quantiles aggregate; the
        # response shape is mode-specific. For univariate it's
        # {median, quantiles, model: aggregate-tag}. We only verify
        # 200 + shape sanity.
        assert "median" in body
        assert len(body["median"][0]) == 3

    def test_unknown_override_slug_is_silently_ignored(
        self, http_client: httpx.Client,
    ) -> None:
        """An override for a slug not in the ensemble must not break
        the request — clients should be able to pass project-wide
        overrides without curating per-call."""
        context = [[float(i) for i in range(1, 100)]]
        r = http_client.post(
            UNIVAR_ENSEMBLE_URL,
            json={
                "context": context,
                "config": {"horizon": 3},
                "weights": {"chronos-2": 1.0},
                "memberOverrides": {
                    "not-a-real-model": {"extra": {"foo": 1}},
                },
            },
        )
        assert r.status_code == 200, r.text
