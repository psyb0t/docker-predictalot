"""POST /v1/tabular/{train,forecast}, GET/DELETE /v1/tabular/models[/{id}].

Per-call shape:
  train    — fits one model PER SERIES, saves under model_id.
  forecast — loads model_id, applies to latest snapshot per series.

Multi-series support: each entry in `target` / `features` is one
independent training set. Most callers will send a single series; the
container supports batches because that's how the FM routers work.

NOTE on multi-series for train: with the current single-blob storage,
this endpoint trains on the CONCATENATION of series (treats them as
independent rows in a larger pool). For per-series persistent models,
call /train multiple times with distinct model_ids.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..auth import check_bearer
from .. import models, tabular_features as features, tabular_storage as storage
from .tabular_schemas import (
    EnsembleForecastRequest,
    EnsembleForecastResponse,
    ForecastRequest,
    ForecastResponse,
    TabularModelInfo,
    TabularModelsResponse,
    TrainRequest,
    TrainResponse,
)

log = logging.getLogger("predictalot.tabular.router")

router = APIRouter(prefix="/v1/tabular", tags=["tabular"])


@router.get("/backends", dependencies=[Depends(check_bearer)])
def list_backends() -> dict[str, Any]:
    """List all available tabular backends + their supported modes."""
    return {
        "backends": [
            {
                "slug": b.SLUG,
                "displayName": b.DISPLAY_NAME,
                "category": getattr(b, "CATEGORY", "other"),
                "supportedModes": sorted(b.SUPPORTED_MODES),
            }
            for b in (models.get_tabular_backend(s) for s in models.tabular_backend_slugs())
        ],
    }


@router.post(
    "/train",
    response_model=TrainResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_train(body: TrainRequest) -> dict[str, Any]:
    if not body.target:
        raise HTTPException(status_code=400, detail="target must not be empty")
    if len(body.target) != len(body.features):
        raise HTTPException(
            status_code=400,
            detail=(
                f"target series count ({len(body.target)}) must match "
                f"features series count ({len(body.features)})"
            ),
        )
    if not body.overwrite and storage.exists(body.model_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"model_id {body.model_id!r} already exists; pass overwrite=true "
                "to replace"
            ),
        )
    try:
        backend = models.get_tabular_backend(body.backend)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if body.config.mode not in backend.SUPPORTED_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"backend {body.backend!r} does not support mode "
                f"{body.config.mode!r}; valid: {sorted(backend.SUPPORTED_MODES)}"
            ),
        )

    # Concatenate series into one big training pool (per the module
    # docstring). Feature dicts must use the same key set across series.
    try:
        Xs, ys, feature_names, sample_weight = _concat_series(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    t0 = time.time()

    def _do_train():
        return backend.train(
            Xs, ys, feature_names, body.config,
            sample_weight=sample_weight,
        )

    try:
        result = await asyncio.to_thread(_do_train)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("tabular train failed (model_id=%s)", body.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    elapsed = time.time() - t0
    meta = storage.TabularMeta(
        model_id=body.model_id,
        backend=body.backend,
        mode=body.config.mode,
        horizon=body.config.horizon,
        feature_names=feature_names,
        n_training_rows=int(Xs.shape[0]),
        trained_at_unix=storage.now_unix(),
    )
    storage.save(meta, result["blob"])

    return {
        "modelId": body.model_id,
        "backend": body.backend,
        "mode": body.config.mode,
        "horizon": body.config.horizon,
        "nTrainingRows": meta.n_training_rows,
        "nFeatures": len(feature_names),
        "featureNames": feature_names,
        "featureImportance": result["importance"],
        "trainSecs": elapsed,
    }


def _concat_series(
    body: TrainRequest,
) -> tuple[Any, Any, list[str], Any]:
    """Build one (X, y, feature_names, sample_weight) from a list of
    (target, features) series. All series must expose the same feature
    names; sample_weight is propagated and pruned alongside.
    """
    import numpy as np

    if not body.features:
        raise ValueError("features must not be empty")
    expected_names: list[str] | None = None
    Xs_list = []
    ys_list = []
    sws_list: list[np.ndarray] = []
    sw_seen = False
    for i, (tgt, feats) in enumerate(zip(body.target, body.features)):
        X_i, y_i, names, sw_i = features.build_training_matrix(
            target=tgt,
            feature_channels=feats,
            horizon=body.config.horizon,
            mode=body.config.mode,
            min_samples=body.config.min_samples,
            sample_weight=body.config.sample_weight,
        )
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(
                f"series {i} feature names {names} differ from series 0 "
                f"names {expected_names}; all series must share feature set"
            )
        Xs_list.append(X_i)
        ys_list.append(y_i)
        if sw_i is not None:
            sws_list.append(sw_i)
            sw_seen = True
    Xs = np.concatenate(Xs_list, axis=0)
    ys = np.concatenate(ys_list, axis=0)
    sws = np.concatenate(sws_list, axis=0) if sw_seen else None
    return Xs, ys, expected_names or [], sws


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast(body: ForecastRequest) -> dict[str, Any]:
    try:
        meta, blob = storage.load(body.model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        backend = models.get_tabular_backend(meta.backend)
    except KeyError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc

    # Build a single-row feature matrix per series; stack into N rows.
    try:
        rows = [
            features.build_forecast_matrix(feats, meta.feature_names)
            for feats in body.features
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    import numpy as np
    X = np.concatenate(rows, axis=0) if rows else np.empty((0, len(meta.feature_names)))

    def _do_predict():
        return backend.predict(blob, X, meta.mode, None)

    try:
        pred = await asyncio.to_thread(_do_predict)
    except Exception as exc:  # noqa: BLE001
        log.exception("tabular forecast failed (model_id=%s)", body.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    resp: dict[str, Any] = {
        "modelId": body.model_id,
        "backend": meta.backend,
        "mode": meta.mode,
        "horizon": meta.horizon,
    }
    if meta.mode == "direction":
        prob = pred["prob_up"].tolist()
        resp["probUp"] = prob
        resp["confidence"] = [abs(p - 0.5) * 2.0 for p in prob]
    elif meta.mode == "value":
        resp["predicted"] = pred["predicted"].tolist()
    elif meta.mode == "quantile":
        resp["median"] = [[float(v)] for v in pred["median"].tolist()]
        resp["quantiles"] = {
            k: [[float(v)] for v in arr.tolist()]
            for k, arr in pred["quantiles"].items()
        }
    return resp


@router.post(
    "/forecast/ensemble",
    response_model=EnsembleForecastResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_ensemble(body: EnsembleForecastRequest) -> dict[str, Any]:
    """Run multiple stored models on the same features, return a
    weighted-mean combination + each member's individual response.

    Mirrors /v1/<type>/forecast/ensemble on the FM side: weights
    normalize across active (weight > 0) members; mode/horizon/
    feature_names must agree across members.
    """
    if not body.model_ids:
        raise HTTPException(status_code=400, detail="model_ids must not be empty")

    # Load every member + its metadata. Reject early on missing ids.
    members: dict[str, tuple[storage.TabularMeta, bytes]] = {}
    for mid in body.model_ids:
        try:
            members[mid] = storage.load(mid)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Cross-validate: same mode, same horizon, same feature_names.
    metas = [m[0] for m in members.values()]
    ref = metas[0]
    for m in metas[1:]:
        if m.mode != ref.mode:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"member {m.model_id!r} has mode {m.mode!r}; "
                    f"member {ref.model_id!r} has mode {ref.mode!r}; "
                    "all ensemble members must share mode"
                ),
            )
        if m.horizon != ref.horizon:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"member {m.model_id!r} has horizon {m.horizon}; "
                    f"member {ref.model_id!r} has horizon {ref.horizon}; "
                    "all ensemble members must share horizon"
                ),
            )
        if m.feature_names != ref.feature_names:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"member {m.model_id!r} feature_names differ from "
                    f"member {ref.model_id!r}; all ensemble members must "
                    "share feature_names"
                ),
            )

    # Resolve + normalize weights.
    try:
        norm = _resolve_weights(body.model_ids, body.weights)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Build the feature matrix once.
    try:
        rows = [
            features.build_forecast_matrix(feats, ref.feature_names)
            for feats in body.features
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    import numpy as np
    X = np.concatenate(rows, axis=0) if rows else np.empty((0, len(ref.feature_names)))

    # Run each active member in a worker thread; gather.
    active = [mid for mid in body.model_ids if norm.get(mid, 0.0) > 0]

    def _run_one(mid: str) -> dict[str, Any]:
        meta, blob = members[mid]
        backend = models.get_tabular_backend(meta.backend)
        return backend.predict(blob, X, meta.mode, None)

    try:
        individual_preds = await asyncio.gather(
            *[asyncio.to_thread(_run_one, mid) for mid in active]
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("tabular ensemble forecast failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Per-member response shape (matches single-forecast format) so the
    # caller can inspect each member alongside the combination.
    individual: dict[str, dict[str, Any]] = {}
    for mid, pred in zip(active, individual_preds):
        individual[mid] = _shape_member(pred, members[mid][0], norm[mid])

    # Combine.
    out: dict[str, Any] = {
        "mode": ref.mode,
        "horizon": ref.horizon,
        "ensembleMembers": active,
        "weights": norm,
        "individual": individual,
    }
    if ref.mode == "direction":
        # Weighted mean of prob_up; confidence is |p-0.5|*2 on the combo.
        prob_arr = np.add.reduce([
            norm[mid] * np.asarray(p["prob_up"], dtype=np.float64)
            for mid, p in zip(active, individual_preds)
        ])
        prob_list = list(prob_arr)
        out["probUp"] = [float(x) for x in prob_list]
        out["confidence"] = [abs(float(x) - 0.5) * 2.0 for x in prob_list]
    elif ref.mode == "value":
        pred_arr = np.add.reduce([
            norm[mid] * np.asarray(p["predicted"], dtype=np.float64)
            for mid, p in zip(active, individual_preds)
        ])
        out["predicted"] = [float(x) for x in pred_arr]
    elif ref.mode == "quantile":
        # Combine per-quantile level. All members must share the same
        # quantile key set.
        first_q = individual_preds[0]["quantiles"]
        for p in individual_preds[1:]:
            if set(p["quantiles"]) != set(first_q):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "member quantile-level sets differ; "
                        "all ensemble members must share quantile levels"
                    ),
                )
        median_arr = np.add.reduce([
            norm[mid] * np.asarray(p["median"], dtype=np.float64)
            for mid, p in zip(active, individual_preds)
        ])
        out["median"] = [[float(v)] for v in median_arr.tolist()]
        quantiles_out: dict[str, list[list[float]]] = {}
        for q_key in first_q:
            q_arr = np.add.reduce([
                norm[mid]
                * np.asarray(p["quantiles"][q_key], dtype=np.float64)
                for mid, p in zip(active, individual_preds)
            ])
            quantiles_out[q_key] = [[float(v)] for v in q_arr.tolist()]
        out["quantiles"] = quantiles_out

    return out


def _resolve_weights(
    model_ids: list[str], weights: dict[str, float] | None,
) -> dict[str, float]:
    """Mirror of dispatch._resolve_weights for tabular ensembles.

    None weights → uniform across members. Returns normalized weights
    keyed by model_id, summing to 1.0 across active (weight > 0)
    members.
    """
    raw: dict[str, float] = {mid: 1.0 for mid in model_ids}
    if weights is not None:
        for mid, w in weights.items():
            if mid not in raw:
                raise ValueError(
                    f"weights contains {mid!r} which is not in model_ids "
                    f"({list(model_ids)})"
                )
            wf = float(w)
            if not math.isfinite(wf):
                raise ValueError(
                    f"weight for {mid!r} must be a finite non-negative number, got {w}"
                )
            if wf < 0:
                raise ValueError(f"weight for {mid!r} must be >= 0, got {w}")
            raw[mid] = wf
    active = [(mid, w) for mid, w in raw.items() if w > 0]
    if not active:
        raise ValueError("every member weight is 0 — at least one must be > 0")
    total = sum(w for _, w in active)
    return {mid: w / total for mid, w in active}


def _shape_member(
    pred: dict[str, Any], meta: storage.TabularMeta, weight: float,
) -> dict[str, Any]:
    """Reshape a backend's raw PredictionDict into the same wire shape
    individual single-forecast responses have, plus the member weight."""
    out: dict[str, Any] = {
        "backend": meta.backend,
        "mode": meta.mode,
        "horizon": meta.horizon,
        "weight": weight,
    }
    if meta.mode == "direction":
        prob = pred["prob_up"].tolist()
        out["probUp"] = prob
        out["confidence"] = [abs(p - 0.5) * 2.0 for p in prob]
    elif meta.mode == "value":
        out["predicted"] = pred["predicted"].tolist()
    elif meta.mode == "quantile":
        out["median"] = [[float(v)] for v in pred["median"].tolist()]
        out["quantiles"] = {
            k: [[float(v)] for v in arr.tolist()]
            for k, arr in pred["quantiles"].items()
        }
    return out


@router.get(
    "/models",
    response_model=TabularModelsResponse,
    dependencies=[Depends(check_bearer)],
)
def list_models() -> dict[str, Any]:
    items = [
        TabularModelInfo(
            model_id=m.model_id,
            backend=m.backend,
            mode=m.mode,
            horizon=m.horizon,
            n_features=len(m.feature_names),
            feature_names=m.feature_names,
            n_training_rows=m.n_training_rows,
            trained_at_unix=m.trained_at_unix,
        )
        for m in storage.list_ids()
    ]
    return {"models": items}


@router.delete("/models/{model_id}", dependencies=[Depends(check_bearer)])
def delete_model(model_id: str) -> dict[str, Any]:
    removed = storage.delete(model_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no model {model_id!r}")
    return {"modelId": model_id, "removed": True}
