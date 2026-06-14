"""XGBoost tabular backend.

Sister to lightgbm. Different regularization; sometimes wins where lgbm
loses. Same interface, same modes.

Blob format: pickle of {"variant": ..., "model": ...}. xgboost has native
JSON save/load but we use pickle here for code consistency across backends.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
import xgboost as xgb

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    monotone_vector,
)

SLUG = "xgboost"
DISPLAY_NAME = "XGBoost"
CATEGORY = "boosting"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   subsample / colsample_bytree / colsample_bylevel / reg_alpha /
#   reg_lambda / scale_pos_weight / grow_policy / gamma


def _build_params(
    config: TrainConfigLike,
    feature_names: list[str],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": config.n_estimators or 400,
        "learning_rate": config.learning_rate or 0.05,
        "max_depth": config.max_depth or 6,
        "min_child_weight": 5,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if config.random_state is not None:
        params["random_state"] = config.random_state
    mono = monotone_vector(feature_names, config.monotonic_constraints)
    if mono is not None:
        params["monotone_constraints"] = "(" + ",".join(str(m) for m in mono) + ")"
    if config.categorical_features:
        # XGBoost 3.x supports native categorical via enable_categorical
        # when the input is a DataFrame with dtype='category'. We can't
        # set that on a raw ndarray here; document it as a known
        # limitation. Pass enable_categorical so callers using
        # DataFrame inputs (future improvement) get the right behavior.
        params["enable_categorical"] = True
    if config.extra:
        for k in (
            "subsample", "colsample_bytree", "colsample_bylevel",
            "reg_alpha", "reg_lambda", "scale_pos_weight",
            "grow_policy", "gamma",
        ):
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

    fit_kw: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight

    # scale_pos_weight for direction-mode imbalance, derived from
    # class_weight="balanced" if requested + no explicit value in extras.
    direction_overrides: dict[str, Any] = {
        "objective": "binary:logistic", "eval_metric": "logloss",
    }
    if (
        config.class_weight == "balanced"
        and (not config.extra or "scale_pos_weight" not in config.extra)
    ):
        n_pos = float(np.sum(y == 1)) or 1.0
        n_neg = float(np.sum(y == 0)) or 1.0
        direction_overrides["scale_pos_weight"] = n_neg / n_pos

    importance: dict[str, float] = {}

    if mode == "direction":
        cls = xgb.XGBClassifier(
            **_build_params(config, feature_names, direction_overrides)
        )
        cls.fit(X, y, **fit_kw)
        importance = _importance(cls, feature_names)
        payload = {"variant": "direction", "model": cls}
    elif mode == "value":
        reg = xgb.XGBRegressor(
            **_build_params(
                config, feature_names, {"objective": "reg:squarederror"}
            )
        )
        reg.fit(X, y, **fit_kw)
        importance = _importance(reg, feature_names)
        payload = {"variant": "value", "model": reg}
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        models: dict[str, Any] = {}
        for q in config.quantile_levels:
            reg = xgb.XGBRegressor(
                **_build_params(
                    config, feature_names,
                    {
                        "objective": "reg:quantileerror",
                        "quantile_alpha": float(q),
                    },
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
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    if not raw:
        return {fn: 0.0 for fn in feature_names}
    total = sum(raw.values()) or 1.0
    out: dict[str, float] = {}
    for i, fn in enumerate(feature_names):
        key = f"f{i}"
        out[fn] = float(raw.get(key, 0.0)) / total
    return out


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
        quantiles_out: dict[str, np.ndarray] = {}
        median: np.ndarray | None = None
        for q_key, model in models.items():
            arr = np.asarray(model.predict(X), dtype=np.float64)
            quantiles_out[q_key] = arr
            if abs(float(q_key) - 0.5) < 1e-9:
                median = arr
        if median is None:
            stacked = np.stack(list(quantiles_out.values()), axis=0)
            median = stacked.mean(axis=0)
        return {"median": median, "quantiles": quantiles_out}

    raise ValueError(f"unsupported mode {mode!r}")
