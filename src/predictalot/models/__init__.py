"""Model backends — one module per supported slug.

Two registries live here:

* ``BACKENDS`` — the foundation-model stack (chronos-2, timesfm-2.5,
  moirai-2, toto-1, sundial-base-128m). Each module exposes:
      SLUG: str
      SUPPORTED_TYPES: frozenset[str]  — see predictalot.types
      async get_model() -> object       — lazy loader, idempotent
      async unload() -> None
      def  loaded() -> bool
      def  last_used_secs_ago() -> float | None
      plus a ``predict_<type>`` async function for each
      SUPPORTED_TYPES member.

* ``TABULAR_BACKENDS`` — supervised tabular learners (lightgbm,
  xgboost, logistic; tabpfn deferred). Each module exposes:
      SLUG: str
      DISPLAY_NAME: str
      SUPPORTED_MODES: frozenset[str]  — direction / value / quantile
      def train(X, y, feature_names, config) -> TrainOutput
      def predict(blob, X, mode, quantile_levels) -> PredictionDict

The two stacks are intentionally separate registries because their
lifecycles differ: FM backends are stateless one-shot predictors
backed by vendor-shipped weights; tabular backends are libraries
that train per-user-request models stored on disk via
``tabular_storage``.

``get()`` and ``get_tabular_backend()`` are the public lookup
entrypoints; consumers should not iterate the registry dicts
directly.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import chronos2, moirai2, sundial, timesfm25, toto1

BACKENDS: dict[str, Any] = {
    "chronos-2": chronos2,
    "timesfm-2.5": timesfm25,
    "moirai-2": moirai2,
    "toto-1": toto1,
    "sundial-base-128m": sundial,
}


# Tabular backends are lazy-loaded so importing `predictalot.models`
# doesn't pull in lightgbm / xgboost / sklearn at module-import time.
# That matters because:
#   1. The dev image ships without the heavy ML stack — only the prod
#      image installs lightgbm/xgboost from the hashed lockfiles. We
#      want every test that doesn't TOUCH a tabular backend to still
#      collect + run under the dev image.
#   2. FastAPI imports this module on startup; eager-loading optional
#      backends would slow boot.
#
# TabPFN deferred entirely: caps scikit-learn<1.7 while the project
# pins scikit-learn==1.8. Re-add once upstream bumps the cap.
_TABULAR_MODULE_NAMES: dict[str, str] = {
    "lightgbm":      "predictalot.models.lightgbm_be",
    "xgboost":       "predictalot.models.xgboost_be",
    "hist-gbt":      "predictalot.models.hist_gbt_be",
    "random-forest": "predictalot.models.random_forest_be",
    "logistic":      "predictalot.models.logistic_be",
    "mlp":           "predictalot.models.mlp_be",
    "svm-rbf":       "predictalot.models.svm_rbf_be",
    "knn":           "predictalot.models.knn_be",
    "naive-bayes":   "predictalot.models.naive_bayes_be",
}

_tabular_cache: dict[str, Any] = {}


def get(slug: str) -> Any:
    """Return the FM backend module for a model slug. KeyError if unknown."""
    return BACKENDS[slug]


def get_tabular_backend(slug: str) -> Any:
    """Lazy-load + return the tabular backend module for a slug.

    Raises KeyError on unknown slug. The actual import (lightgbm etc.)
    happens on first lookup, so the dev image — which doesn't ship
    the heavy ML stack — can still import predictalot.models for
    everything else.
    """
    if slug not in _TABULAR_MODULE_NAMES:
        raise KeyError(
            f"unknown tabular backend {slug!r}; valid: "
            f"{sorted(_TABULAR_MODULE_NAMES)}"
        )
    if slug not in _tabular_cache:
        _tabular_cache[slug] = importlib.import_module(
            _TABULAR_MODULE_NAMES[slug]
        )
    return _tabular_cache[slug]


def tabular_backend_slugs() -> list[str]:
    """All registered tabular backend slugs (whether or not loaded)."""
    return sorted(_TABULAR_MODULE_NAMES)
