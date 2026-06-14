"""sklearn HistGradientBoosting tabular backend.

Pure-sklearn alternative to lightgbm; same algorithm family
(histogram-based gradient boosting). Useful when lightgbm isn't
available, or as a third GBT voice in an ensemble for diversity.

Native quantile mode via loss="quantile" + quantile=q.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    feature_indices_from_names,
    monotone_vector,
)

SLUG = "hist-gbt"
DISPLAY_NAME = "HistGradientBoosting (sklearn)"
CATEGORY = "boosting"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   max_iter, l2_regularization, max_bins, min_samples_leaf, max_leaf_nodes


def _common_params(
    config: TrainConfigLike, feature_names: list[str],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_iter": config.n_estimators or 200,
        "learning_rate": config.learning_rate or 0.05,
        "max_depth": config.max_depth,  # None → unlimited
        "max_leaf_nodes": (
            config.num_leaves
            if config.num_leaves is not None else 31
        ),
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    }
    if config.random_state is not None:
        params["random_state"] = config.random_state

    cat_idx = feature_indices_from_names(
        feature_names, config.categorical_features,
    )
    if cat_idx is not None:
        params["categorical_features"] = cat_idx

    mono = monotone_vector(feature_names, config.monotonic_constraints)
    if mono is not None:
        params["monotonic_cst"] = mono

    if config.extra:
        for k in (
            "max_iter", "l2_regularization", "max_bins",
            "min_samples_leaf", "max_leaf_nodes",
        ):
            if k in config.extra:
                params[k] = config.extra[k]

    if (
        config.early_stopping_rounds is not None
        and config.validation_fraction
    ):
        params["early_stopping"] = True
        params["n_iter_no_change"] = config.early_stopping_rounds
        params["validation_fraction"] = config.validation_fraction
    return params


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: TrainConfigLike,
    sample_weight: np.ndarray | None = None,
) -> TrainOutput:
    mode = config.mode
    fit_kw: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight

    importance: dict[str, float] = {}

    if mode == "direction":
        params = _common_params(config, feature_names)
        if config.class_weight is not None:
            params["class_weight"] = config.class_weight
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X, y, **fit_kw)
        importance = _importance_fallback(clf, X, y, feature_names)
        payload = {"variant": "direction", "model": clf}
    elif mode == "value":
        reg = HistGradientBoostingRegressor(
            **_common_params(config, feature_names),
            loss="squared_error",
        )
        reg.fit(X, y, **fit_kw)
        importance = _importance_fallback(reg, X, y, feature_names)
        payload = {"variant": "value", "model": reg}
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        models: dict[str, Any] = {}
        for q in config.quantile_levels:
            reg = HistGradientBoostingRegressor(
                **_common_params(config, feature_names),
                loss="quantile",
                quantile=float(q),
            )
            reg.fit(X, y, **fit_kw)
            models[f"{q:.1f}"] = reg
            if abs(q - 0.5) < 1e-9:
                importance = _importance_fallback(reg, X, y, feature_names)
        if not importance:
            importance = _importance_fallback(
                next(iter(models.values())), X, y, feature_names,
            )
        payload = {"variant": "quantile", "models": models}
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    buf = io.BytesIO()
    pickle.dump(payload, buf)
    return {"blob": buf.getvalue(), "importance": importance}


def _importance_fallback(
    model: Any, X: np.ndarray, y: np.ndarray, feature_names: list[str],
) -> dict[str, float]:
    """HistGradientBoosting doesn't expose `feature_importances_` like
    other tree models. Approximate via permutation importance on the
    training set — slow but informative. Skip if too few rows."""
    if X.shape[0] < 200:
        return {fn: 1.0 / max(len(feature_names), 1) for fn in feature_names}
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(model, X, y, n_repeats=3, random_state=0, n_jobs=-1)
    arr = np.asarray(pi.importances_mean, dtype=np.float64)
    arr = np.clip(arr, 0, None)
    total = float(arr.sum()) or 1.0
    return {fn: float(v) / total for fn, v in zip(feature_names, arr)}


def predict(
    blob: bytes,
    X: np.ndarray,
    mode: str,
    quantile_levels: list[float] | None = None,
) -> PredictionDict:
    payload = pickle.loads(blob)
    variant = payload["variant"]
    if variant != mode:
        raise ValueError(
            f"model was trained for mode={variant!r}, forecast requested mode={mode!r}"
        )
    if mode == "direction":
        proba = payload["model"].predict_proba(X)
        prob_up = np.asarray(proba[:, 1], dtype=np.float64)
        return {"prob_up": prob_up}
    if mode == "value":
        pred = np.asarray(payload["model"].predict(X), dtype=np.float64)
        return {"predicted": pred}
    if mode == "quantile":
        models: dict[str, Any] = payload["models"]
        out: dict[str, np.ndarray] = {}
        median: np.ndarray | None = None
        for q_key, m in models.items():
            arr = np.asarray(m.predict(X), dtype=np.float64)
            out[q_key] = arr
            if abs(float(q_key) - 0.5) < 1e-9:
                median = arr
        if median is None:
            stacked = np.stack(list(out.values()), axis=0)
            median = stacked.mean(axis=0)
        return {"median": median, "quantiles": out}
    raise ValueError(f"unsupported mode {mode!r}")
