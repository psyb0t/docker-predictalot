"""LightGBM tabular backend.

Direction → LGBMClassifier(objective='binary')
Value     → LGBMRegressor(objective='regression')
Quantile  → one LGBMRegressor(objective='quantile', alpha=q) per quantile,
            stored as a dict[str, bytes] inside the blob.

Blob format: pickle of a small dict {"variant": ..., "models": ...} so we
don't need a custom on-disk format.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import lightgbm as lgb
import numpy as np

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    feature_indices_from_names,
    monotone_vector,
)

SLUG = "lightgbm"
DISPLAY_NAME = "LightGBM"
CATEGORY = "boosting"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   subsample          row-subsample per tree (default 1.0)
#   colsample_bytree   feature-subsample per tree (default 1.0)
#   reg_alpha          L1 regularization
#   reg_lambda         L2 regularization
#   boosting_type      "gbdt" | "dart" | "goss" (default "gbdt")


def _build_params(
    config: TrainConfigLike,
    feature_names: list[str],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": config.n_estimators or 400,
        "learning_rate": config.learning_rate or 0.05,
        "num_leaves": config.num_leaves or 31,
        "max_depth": config.max_depth or -1,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    if config.random_state is not None:
        params["random_state"] = config.random_state
    if config.class_weight is not None and overrides.get("objective") == "binary":
        params["class_weight"] = config.class_weight
    mono = monotone_vector(feature_names, config.monotonic_constraints)
    if mono is not None:
        params["monotone_constraints"] = mono
    if config.extra:
        for k in ("subsample", "colsample_bytree", "reg_alpha",
                  "reg_lambda", "boosting_type"):
            if k in config.extra:
                params[k] = config.extra[k]
    params.update(overrides)
    return params


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: TrainConfigLike,
    sample_weight: np.ndarray | None = None,
) -> TrainOutput:
    mode = config.mode
    cat_idx = feature_indices_from_names(
        feature_names, config.categorical_features,
    )
    fit_kw: dict[str, Any] = {"feature_name": feature_names}
    if cat_idx is not None:
        fit_kw["categorical_feature"] = cat_idx
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight

    importance: dict[str, float] = {}

    if mode == "direction":
        cls = lgb.LGBMClassifier(
            **_build_params(config, feature_names, {"objective": "binary"})
        )
        cls.fit(X, y, **fit_kw)
        importance = _importance(cls, feature_names)
        payload = {"variant": "direction", "model": cls}
    elif mode == "value":
        reg = lgb.LGBMRegressor(
            **_build_params(config, feature_names, {"objective": "regression"})
        )
        reg.fit(X, y, **fit_kw)
        importance = _importance(reg, feature_names)
        payload = {"variant": "value", "model": reg}
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        models: dict[str, Any] = {}
        for q in config.quantile_levels:
            reg = lgb.LGBMRegressor(
                **_build_params(
                    config, feature_names,
                    {"objective": "quantile", "alpha": float(q)},
                )
            )
            reg.fit(X, y, **fit_kw)
            models[f"{q:.1f}"] = reg
            if abs(q - 0.5) < 1e-9:
                importance = _importance(reg, feature_names)
        if not importance:
            importance = _importance(next(iter(models.values())), feature_names)
        payload = {"variant": "quantile", "models": models}
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    buf = io.BytesIO()
    pickle.dump(payload, buf)
    return {"blob": buf.getvalue(), "importance": importance}


def _importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Gain-based importance normalized to sum=1."""
    arr = model.booster_.feature_importance(importance_type="gain")
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
        # Classifier emits [[p0, p1], ...] — class 1 is "up".
        prob_up = np.asarray(proba[:, 1], dtype=np.float64)
        return {"prob_up": prob_up}

    if mode == "value":
        pred = np.asarray(payload["model"].predict(X), dtype=np.float64)
        return {"predicted": pred}

    if mode == "quantile":
        models: dict[str, Any] = payload["models"]
        quantiles_out: dict[str, np.ndarray] = {}
        median: np.ndarray | None = None
        for q_key, model in models.items():
            arr = np.asarray(model.predict(X), dtype=np.float64)
            quantiles_out[q_key] = arr
            if abs(float(q_key) - 0.5) < 1e-9:
                median = arr
        if median is None:
            # Synthesize median as mean across available quantiles.
            stacked = np.stack(list(quantiles_out.values()), axis=0)
            median = stacked.mean(axis=0)
        return {"median": median, "quantiles": quantiles_out}

    raise ValueError(f"unsupported mode {mode!r}")
