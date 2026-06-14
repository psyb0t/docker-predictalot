"""Linear baseline — sklearn LogisticRegression for direction, Ridge for value.

Purpose: cheap sanity check. If LogisticRegression's dirHit ≈ GBT's dirHit
on the same features, the features have ~linear signal at best. If GBT
beats logistic by a lot, the nonlinear interactions matter.

Quantile mode is implemented via sklearn's QuantileRegressor (slow but works).
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import (
    LogisticRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.preprocessing import StandardScaler

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    extra_get,
)

SLUG = "logistic"
DISPLAY_NAME = "Logistic / Ridge / QuantileRegressor (linear baselines)"
CATEGORY = "linear"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   C                 inverse regularization strength for LogisticRegression
#                     (default 1.0)
#   penalty           "l1" | "l2" | "elasticnet" | None (default "l2")
#   l1_ratio          float for elasticnet penalty
#   solver            sklearn solver name (default "lbfgs")
#   alpha             Ridge alpha (default 1.0)
#   quantile_alpha    QuantileRegressor alpha L1-reg (default 0.1)
#   max_iter          iteration cap (default 2000)


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: TrainConfigLike,
    sample_weight: np.ndarray | None = None,
) -> TrainOutput:
    mode = config.mode
    importance: dict[str, float] = {}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    seed = config.random_state if config.random_state is not None else 0
    max_iter = int(extra_get(config, "max_iter", 2000))
    fit_kw: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight

    if mode == "direction":
        clf = LogisticRegression(
            max_iter=max_iter,
            C=float(extra_get(config, "C", 1.0)),
            penalty=extra_get(config, "penalty", "l2"),
            l1_ratio=extra_get(config, "l1_ratio", None),
            solver=extra_get(config, "solver", "lbfgs"),
            class_weight=config.class_weight,
            random_state=seed,
        )
        clf.fit(Xs, y, **fit_kw)
        importance = _linear_importance(clf.coef_[0], feature_names)
        payload = {"variant": "direction", "model": clf, "scaler": scaler}
    elif mode == "value":
        reg = Ridge(
            alpha=float(extra_get(config, "alpha", 1.0)),
            random_state=seed,
        )
        reg.fit(Xs, y, **fit_kw)
        importance = _linear_importance(reg.coef_, feature_names)
        payload = {"variant": "value", "model": reg, "scaler": scaler}
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        models: dict[str, Any] = {}
        for q in config.quantile_levels:
            qr = QuantileRegressor(
                quantile=float(q),
                alpha=float(extra_get(config, "quantile_alpha", 0.1)),
                solver="highs",
            )
            qr.fit(Xs, y, **fit_kw)
            models[f"{q:.1f}"] = qr
            if abs(q - 0.5) < 1e-9:
                importance = _linear_importance(qr.coef_, feature_names)
        if not importance:
            importance = _linear_importance(
                next(iter(models.values())).coef_, feature_names
            )
        payload = {
            "variant": "quantile",
            "models": models,
            "scaler": scaler,
        }
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    buf = io.BytesIO()
    pickle.dump(payload, buf)
    return {"blob": buf.getvalue(), "importance": importance}


def _linear_importance(
    coef: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    abs_coef = np.abs(np.asarray(coef, dtype=np.float64))
    total = float(abs_coef.sum()) or 1.0
    return {fn: float(v) / total for fn, v in zip(feature_names, abs_coef)}


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

    scaler: StandardScaler = payload["scaler"]
    Xs = scaler.transform(X)

    if mode == "direction":
        proba = payload["model"].predict_proba(Xs)
        prob_up = np.asarray(proba[:, 1], dtype=np.float64)
        return {"prob_up": prob_up}

    if mode == "value":
        pred = np.asarray(payload["model"].predict(Xs), dtype=np.float64)
        return {"predicted": pred}

    if mode == "quantile":
        models: dict[str, Any] = payload["models"]
        quantiles_out: dict[str, np.ndarray] = {}
        median: np.ndarray | None = None
        for q_key, model in models.items():
            arr = np.asarray(model.predict(Xs), dtype=np.float64)
            quantiles_out[q_key] = arr
            if abs(float(q_key) - 0.5) < 1e-9:
                median = arr
        if median is None:
            stacked = np.stack(list(quantiles_out.values()), axis=0)
            median = stacked.mean(axis=0)
        return {"median": median, "quantiles": quantiles_out}

    raise ValueError(f"unsupported mode {mode!r}")
