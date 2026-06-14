"""sklearn MLP tabular backend.

Multi-layer perceptron — neural baseline. Useful diagnostic: if MLP
beats logistic but matches GBT, the GBT's nonlinearities are doing
the work. If MLP beats GBT, there's nonlinear interaction GBT misses.

Quantile mode falls back to a 3-MLP fit at q={low, 0.5, high} using
the pinball loss — sklearn doesn't have a quantile MLP, so we
approximate by fitting on (residuals shifted by the quantile target).
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    extra_get,
)

SLUG = "mlp"
DISPLAY_NAME = "Multi-Layer Perceptron (sklearn)"
CATEGORY = "neural"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   hidden_layer_sizes  tuple of int (default (64, 32))
#   activation          "relu" | "tanh" | "logistic" | "identity"
#   alpha               L2 reg (default 1e-4)
#   learning_rate_init  Adam initial LR (default 1e-3)
#   max_iter            (default 500)
#   early_stopping      (default True)
#   batch_size          (default "auto")


def _common_params(config: TrainConfigLike) -> dict[str, Any]:
    return {
        "hidden_layer_sizes": tuple(extra_get(
            config, "hidden_layer_sizes", (64, 32),
        )),
        "activation": extra_get(config, "activation", "relu"),
        "alpha": float(extra_get(config, "alpha", 1e-4)),
        "learning_rate_init": float(
            extra_get(config, "learning_rate_init", 1e-3),
        ),
        "max_iter": int(extra_get(config, "max_iter", 500)),
        "early_stopping": bool(extra_get(config, "early_stopping", True)),
        "batch_size": extra_get(config, "batch_size", "auto"),
        "random_state": (
            config.random_state if config.random_state is not None else 0
        ),
    }


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: TrainConfigLike,
    sample_weight: np.ndarray | None = None,
) -> TrainOutput:
    mode = config.mode
    # MLP needs feature standardization for stable training.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # sklearn MLP doesn't accept sample_weight; emulate by repeating
    # high-weighted rows. Skip when weights are uniform/None.
    if sample_weight is not None and not np.allclose(sample_weight, sample_weight[0]):
        # Naive: scale weights to integer repeats summing to 2*N max.
        scaled = np.asarray(sample_weight, dtype=np.float64)
        scaled = scaled / scaled.max() * 2.0
        reps = np.clip(np.round(scaled).astype(np.int64), 1, 8)
        Xs = np.repeat(Xs, reps, axis=0)
        y = np.repeat(y, reps, axis=0)

    importance: dict[str, float] = {}

    if mode == "direction":
        clf = MLPClassifier(**_common_params(config))
        clf.fit(Xs, y)
        importance = _importance(clf, X, y, feature_names)
        payload = {"variant": "direction", "model": clf, "scaler": scaler}
    elif mode == "value":
        reg = MLPRegressor(**_common_params(config))
        reg.fit(Xs, y)
        importance = _importance(reg, X, y, feature_names)
        payload = {"variant": "value", "model": reg, "scaler": scaler}
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        # Approximate per-quantile via a regression MLP plus per-quantile
        # shift derived from the residual distribution. Train one base
        # regressor + compute residual quantiles at predict time.
        base = MLPRegressor(**_common_params(config))
        base.fit(Xs, y)
        pred_train = base.predict(Xs)
        residuals = np.asarray(y, dtype=np.float64) - pred_train
        importance = _importance(base, X, y, feature_names)
        payload = {
            "variant": "quantile",
            "model": base,
            "scaler": scaler,
            "residuals": residuals,
            "quantile_levels": list(config.quantile_levels),
        }
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    buf = io.BytesIO()
    pickle.dump(payload, buf)
    return {"blob": buf.getvalue(), "importance": importance}


def _importance(
    model: Any, X: np.ndarray, y: np.ndarray, feature_names: list[str],
) -> dict[str, float]:
    """Permutation importance on the training set (sklearn MLP has no
    native importance score)."""
    if X.shape[0] < 200:
        return {fn: 1.0 / max(len(feature_names), 1) for fn in feature_names}
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(model, X, y, n_repeats=3, random_state=0, n_jobs=-1)
    arr = np.clip(np.asarray(pi.importances_mean, dtype=np.float64), 0, None)
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
        base = payload["model"]
        residuals: np.ndarray = payload["residuals"]
        base_pred = np.asarray(base.predict(Xs), dtype=np.float64)
        levels = (
            quantile_levels
            if quantile_levels is not None
            else payload.get("quantile_levels") or [0.1, 0.5, 0.9]
        )
        out: dict[str, np.ndarray] = {}
        median: np.ndarray | None = None
        for q in levels:
            shift = float(np.quantile(residuals, float(q)))
            arr = base_pred + shift
            out[f"{q:.1f}"] = arr
            if abs(q - 0.5) < 1e-9:
                median = arr
        if median is None:
            median = base_pred
        return {"median": median, "quantiles": out}
    raise ValueError(f"unsupported mode {mode!r}")
