"""Verify the FM routers are mounted under /v1/timeseries/* after the
hard cutover rename. Old /v1/<type>/* paths should 404.
"""

from __future__ import annotations

import pytest


AUTH = {"Authorization": "Bearer testtoken"}


def test_new_univariate_models_path_works(client) -> None:
    r = client.get("/v1/timeseries/univariate/models", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["type"] == "univariate"


def test_new_multivariate_models_path_works(client) -> None:
    r = client.get("/v1/timeseries/multivariate/models", headers=AUTH)
    assert r.status_code == 200


def test_new_covariates_past_models_path_works(client) -> None:
    r = client.get("/v1/timeseries/covariates/past/models", headers=AUTH)
    assert r.status_code == 200


def test_new_covariates_future_models_path_works(client) -> None:
    r = client.get("/v1/timeseries/covariates/future/models", headers=AUTH)
    assert r.status_code == 200


def test_new_covariates_both_models_path_works(client) -> None:
    r = client.get("/v1/timeseries/covariates/models", headers=AUTH)
    assert r.status_code == 200


def test_new_samples_models_path_works(client) -> None:
    r = client.get("/v1/timeseries/samples/models", headers=AUTH)
    assert r.status_code == 200


def test_old_univariate_path_404s(client) -> None:
    r = client.get("/v1/univariate/models", headers=AUTH)
    assert r.status_code == 404


def test_old_multivariate_path_404s(client) -> None:
    r = client.get("/v1/multivariate/models", headers=AUTH)
    assert r.status_code == 404


def test_old_covariates_past_path_404s(client) -> None:
    r = client.get("/v1/covariates/past/models", headers=AUTH)
    assert r.status_code == 404


def test_old_samples_path_404s(client) -> None:
    r = client.get("/v1/samples/models", headers=AUTH)
    assert r.status_code == 404


def test_tabular_keeps_its_own_namespace(client) -> None:
    """/v1/tabular/ is intentionally NOT under /v1/timeseries/. The
    backends listing endpoint lazy-imports lightgbm/xgboost/sklearn, so
    skip when the heavy ML stack isn't installed in the test image."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    r = client.get("/v1/tabular/backends", headers=AUTH)
    assert r.status_code == 200
