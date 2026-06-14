"""Meta-learners over the tabular backends:

* /v1/tabular/train/calibrated   + /v1/tabular/forecast/calibrated
* /v1/tabular/train/stacking     + /v1/tabular/forecast/stacking
* /v1/tabular/train/diversified  + /v1/tabular/forecast/diversified

All three store a SINGLE blob via ``tabular_storage`` with backend
slug ``"meta:<kind>"`` so the regular /v1/tabular/forecast handler
won't match them. The forecast handlers here decompress + re-route
through the underlying backend modules at predict time.

Why composite endpoints rather than client-side ensembling:
  * calibrated — calibration data has to be HELD OUT from training,
    so we need atomic train+holdout-fit semantics. Doing it client
    side requires two /train calls + manual state.
  * stacking   — OOF predictions require K-fold orchestration on the
    backend; trivial to get wrong client side.
  * diversified — candidate selection requires correlation analysis
    of K-fold OOF predictions; we want this happening once at train
    time, not per forecast.
"""

from __future__ import annotations

import asyncio
import io
import logging
import pickle
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..auth import check_bearer
from .. import models, tabular_features as features, tabular_storage as storage
from .tabular_schemas import (
    CalibratedTrainRequest,
    DiversifiedTrainRequest,
    MetaForecastRequest,
    MetaForecastResponse,
    MetaTrainResponse,
    StackingTrainRequest,
)

log = logging.getLogger("predictalot.tabular.meta")

router = APIRouter(prefix="/v1/tabular", tags=["tabular-meta"])


# ── helpers ────────────────────────────────────────────────────────────
def _build_training_pool(
    target: list[list[float]],
    feats: list[dict[str, list[float]]],
    horizon: int,
    mode: str,
    min_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Multi-series build: produces (X, y, feature_names) with labels
    already shifted by ``horizon`` and warmup rows pruned. Mirrors the
    regular tabular /train path so meta-learners see identical data
    semantics (same horizon-label encoding, same all-zero-row pruning).
    """
    if not target:
        raise ValueError("target must not be empty")
    if len(target) != len(feats):
        raise ValueError("target / features series count mismatch")
    if not feats:
        raise ValueError("features must not be empty")

    expected_names: list[str] | None = None
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i, (tgt, fset) in enumerate(zip(target, feats)):
        X_i, y_i, names, _ = features.build_training_matrix(
            target=tgt,
            feature_channels=fset,
            horizon=horizon,
            mode=mode,
            min_samples=min_samples,
            sample_weight=None,
        )
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(
                f"series {i} feature names {names} differ from series 0"
            )
        Xs.append(X_i)
        ys.append(y_i)
    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(ys, axis=0),
        expected_names or [],
    )


def _persist_meta_blob(
    model_id: str,
    backend: str,
    mode: str,
    horizon: int,
    feature_names: list[str],
    n_training_rows: int,
    blob: bytes,
) -> None:
    """Persist via tabular_storage with a meta-tagged backend slug."""
    meta = storage.TabularMeta(
        model_id=model_id,
        backend=backend,
        mode=mode,
        horizon=horizon,
        feature_names=feature_names,
        n_training_rows=n_training_rows,
        trained_at_unix=storage.now_unix(),
    )
    storage.save(meta, blob)


def _build_forecast_X(
    feats: list[dict[str, list[float]]], feature_names: list[str],
) -> np.ndarray:
    rows = [features.build_forecast_matrix(f, feature_names) for f in feats]
    if not rows:
        return np.empty((0, len(feature_names)))
    return np.concatenate(rows, axis=0)


# ── 1. CALIBRATED ──────────────────────────────────────────────────────
@router.post(
    "/train/calibrated",
    response_model=MetaTrainResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_train_calibrated(body: CalibratedTrainRequest) -> dict[str, Any]:
    if body.config.mode != "direction":
        raise HTTPException(
            status_code=400,
            detail=(
                "calibrated meta-learner only supports mode='direction' "
                f"(got {body.config.mode!r}); probability calibration is "
                "meaningless for value/quantile."
            ),
        )
    if not body.overwrite and storage.exists(body.model_id):
        raise HTTPException(
            status_code=409,
            detail=f"model_id {body.model_id!r} already exists",
        )
    try:
        backend = models.get_tabular_backend(body.base_backend)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        X_dir, y_dir, feature_names = _build_training_pool(
            body.target, body.features,
            horizon=body.config.horizon,
            mode="direction",
            min_samples=body.config.min_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h = body.config.horizon
    # Hold out the last calibration_fraction for calibrator fitting —
    # preserve time order (no shuffle).
    n = X_dir.shape[0]
    n_cal = max(1, int(round(n * body.calibration_fraction)))
    n_base = n - n_cal
    if n_base < 10 or n_cal < 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"split too small (base={n_base}, calibration={n_cal}); "
                "need more training rows or a smaller calibration_fraction"
            ),
        )
    X_base, X_cal = X_dir[:n_base], X_dir[n_base:]
    y_base, y_cal = y_dir[:n_base], y_dir[n_base:]

    t0 = time.time()

    def _do_train():
        from sklearn.calibration import _SigmoidCalibration  # noqa: PLC0415
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

        base_out = backend.train(
            X_base, y_base, feature_names, body.config,
        )
        prob_cal = backend.predict(
            base_out["blob"], X_cal, "direction",
        )["prob_up"]

        if body.calibration_method == "sigmoid":
            calibrator = _SigmoidCalibration()
            calibrator.fit(prob_cal, y_cal)
        else:
            calibrator = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0,
            )
            calibrator.fit(prob_cal, y_cal)
        return base_out, calibrator

    try:
        base_out, calibrator = await asyncio.to_thread(_do_train)
    except Exception as exc:  # noqa: BLE001
        log.exception("calibrated meta train failed (model_id=%s)", body.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed = time.time() - t0

    payload = {
        "kind": "calibrated",
        "base_backend": body.base_backend,
        "base_blob": base_out["blob"],
        "calibrator": calibrator,
        "method": body.calibration_method,
        "feature_names": feature_names,
        "mode": "direction",
        "horizon": h,
    }
    buf = io.BytesIO()
    pickle.dump(payload, buf)
    _persist_meta_blob(
        model_id=body.model_id,
        backend="meta:calibrated",
        mode="direction",
        horizon=h,
        feature_names=feature_names,
        n_training_rows=n,
        blob=buf.getvalue(),
    )

    return {
        "modelId": body.model_id,
        "kind": "calibrated",
        "mode": "direction",
        "horizon": h,
        "membersUsed": [body.base_backend],
        "nTrainingRows": n,
        "nFeatures": len(feature_names),
        "featureNames": feature_names,
        "trainSecs": elapsed,
    }


@router.post(
    "/forecast/calibrated",
    response_model=MetaForecastResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast_calibrated(body: MetaForecastRequest) -> dict[str, Any]:
    payload, meta = _load_meta(body.model_id, expected_kind="calibrated")
    X = _build_forecast_X(body.features, payload["feature_names"])
    backend = models.get_tabular_backend(payload["base_backend"])

    def _do_predict():
        raw = backend.predict(payload["base_blob"], X, "direction")["prob_up"]
        cal_obj = payload["calibrator"]
        # Both _SigmoidCalibration and IsotonicRegression expose .predict
        # with the same shape contract, so we don't need to discriminate.
        calibrated = cal_obj.predict(raw)
        return raw, np.asarray(calibrated, dtype=np.float64)

    try:
        raw_prob, cal_prob = await asyncio.to_thread(_do_predict)
    except Exception as exc:  # noqa: BLE001
        log.exception("calibrated forecast failed (model_id=%s)", body.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "modelId": body.model_id,
        "kind": "calibrated",
        "mode": "direction",
        "horizon": meta.horizon,
        "members": {
            payload["base_backend"]: {
                "probUp": [float(p) for p in raw_prob],
            },
        },
        "probUp": [float(p) for p in cal_prob],
        "confidence": [abs(float(p) - 0.5) * 2.0 for p in cal_prob],
    }


# ── 2. STACKING ────────────────────────────────────────────────────────
@router.post(
    "/train/stacking",
    response_model=MetaTrainResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_train_stacking(body: StackingTrainRequest) -> dict[str, Any]:
    for m in body.members:
        if m.config.mode != "direction":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"stacking v1 only supports direction-mode members "
                    f"(member {m.backend!r} has mode={m.config.mode!r})"
                ),
            )
        if m.config.horizon != body.horizon:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"member {m.backend!r} horizon {m.config.horizon} "
                    f"!= top-level horizon {body.horizon}"
                ),
            )
    if not body.overwrite and storage.exists(body.model_id):
        raise HTTPException(
            status_code=409,
            detail=f"model_id {body.model_id!r} already exists",
        )
    try:
        member_backends = [
            (m, models.get_tabular_backend(m.backend)) for m in body.members
        ]
        meta_backend = models.get_tabular_backend(body.meta_backend)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        X_dir, y_dir, feature_names = _build_training_pool(
            body.target, body.features,
            horizon=body.horizon,
            mode="direction",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h = body.horizon
    n = X_dir.shape[0]
    if n < body.n_folds * 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"need >= n_folds*5 ({body.n_folds * 5}) training rows; "
                f"got {n}"
            ),
        )

    t0 = time.time()

    def _do_train():
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        from sklearn.model_selection import KFold  # noqa: PLC0415

        kf = KFold(n_splits=body.n_folds, shuffle=False)
        oof_probs = np.zeros((n, len(member_backends)))
        for tr_idx, te_idx in kf.split(X_dir):
            for col, (spec, be) in enumerate(member_backends):
                out = be.train(
                    X_dir[tr_idx], y_dir[tr_idx], feature_names, spec.config,
                )
                p = be.predict(
                    out["blob"], X_dir[te_idx], "direction",
                )["prob_up"]
                oof_probs[te_idx, col] = p
        # Train meta on OOF predictions.
        meta_feat_names = [f"oof_{spec.backend}" for spec, _ in member_backends]
        # Reuse the first member's config for hyperparams of the meta (it's
        # mostly defaults — fine for the meta-learner).
        meta_cfg = body.members[0].config
        meta_out = meta_backend.train(
            oof_probs, y_dir, meta_feat_names, meta_cfg,
        )
        # Score the meta on its own OOF inputs (informative, not perfect).
        meta_oof_prob = meta_backend.predict(
            meta_out["blob"], oof_probs, "direction",
        )["prob_up"]
        try:
            oof_auc = float(roc_auc_score(y_dir, meta_oof_prob))
        except ValueError:
            oof_auc = float("nan")
        # Retrain each member on the FULL pool — that's what we'll
        # actually ship.
        full_member_blobs = []
        for spec, be in member_backends:
            out = be.train(X_dir, y_dir, feature_names, spec.config)
            full_member_blobs.append(out["blob"])
        return full_member_blobs, meta_out["blob"], meta_feat_names, oof_auc

    try:
        full_member_blobs, meta_blob, meta_feat_names, oof_auc = (
            await asyncio.to_thread(_do_train)
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("stacking meta train failed (model_id=%s)", body.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed = time.time() - t0

    payload = {
        "kind": "stacking",
        "mode": "direction",
        "horizon": h,
        "feature_names": feature_names,
        "members": [
            {"backend": spec.backend, "blob": blob}
            for (spec, _), blob in zip(member_backends, full_member_blobs)
        ],
        "meta_backend": body.meta_backend,
        "meta_blob": meta_blob,
        "meta_feature_names": meta_feat_names,
    }
    buf = io.BytesIO()
    pickle.dump(payload, buf)
    _persist_meta_blob(
        model_id=body.model_id,
        backend="meta:stacking",
        mode="direction",
        horizon=h,
        feature_names=feature_names,
        n_training_rows=n,
        blob=buf.getvalue(),
    )

    return {
        "modelId": body.model_id,
        "kind": "stacking",
        "mode": "direction",
        "horizon": h,
        "membersUsed": [m.backend for m in body.members],
        "nTrainingRows": n,
        "nFeatures": len(feature_names),
        "featureNames": feature_names,
        "trainSecs": elapsed,
        "oofScore": oof_auc,
    }


@router.post(
    "/forecast/stacking",
    response_model=MetaForecastResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast_stacking(body: MetaForecastRequest) -> dict[str, Any]:
    payload, meta = _load_meta(body.model_id, expected_kind="stacking")
    X = _build_forecast_X(body.features, payload["feature_names"])

    def _do_predict():
        member_probs = []
        member_resp: dict[str, dict[str, Any]] = {}
        for m in payload["members"]:
            be = models.get_tabular_backend(m["backend"])
            p = be.predict(m["blob"], X, "direction")["prob_up"]
            member_probs.append(p)
            member_resp[m["backend"]] = {"probUp": [float(v) for v in p]}
        stacked = np.stack(member_probs, axis=1)
        meta_be = models.get_tabular_backend(payload["meta_backend"])
        final = meta_be.predict(payload["meta_blob"], stacked, "direction")["prob_up"]
        return member_resp, np.asarray(final, dtype=np.float64)

    try:
        members_resp, final_prob = await asyncio.to_thread(_do_predict)
    except Exception as exc:  # noqa: BLE001
        log.exception("stacking forecast failed (model_id=%s)", body.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "modelId": body.model_id,
        "kind": "stacking",
        "mode": "direction",
        "horizon": meta.horizon,
        "members": members_resp,
        "probUp": [float(p) for p in final_prob],
        "confidence": [abs(float(p) - 0.5) * 2.0 for p in final_prob],
    }


# ── 3. DIVERSIFIED ─────────────────────────────────────────────────────
@router.post(
    "/train/diversified",
    response_model=MetaTrainResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_train_diversified(body: DiversifiedTrainRequest) -> dict[str, Any]:
    for c in body.candidates:
        if c.config.mode != body.mode:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"candidate {c.backend!r} mode {c.config.mode!r} != "
                    f"top-level mode {body.mode!r}"
                ),
            )
        if c.config.horizon != body.horizon:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"candidate {c.backend!r} horizon {c.config.horizon} "
                    f"!= top-level horizon {body.horizon}"
                ),
            )
    if body.mode == "quantile" and not body.quantile_levels:
        raise HTTPException(
            status_code=400,
            detail="quantile mode requires quantile_levels",
        )
    if not body.overwrite and storage.exists(body.model_id):
        raise HTTPException(
            status_code=409,
            detail=f"model_id {body.model_id!r} already exists",
        )
    try:
        candidates = [
            (c, models.get_tabular_backend(c.backend)) for c in body.candidates
        ]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        X_eff, y_eff, feature_names = _build_training_pool(
            body.target, body.features,
            horizon=body.horizon,
            mode=body.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h = body.horizon
    n = X_eff.shape[0]
    if n < body.n_folds * 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"need >= n_folds*5 ({body.n_folds * 5}) rows; got {n}"
            ),
        )

    t0 = time.time()

    def _do_train():
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        from sklearn.model_selection import KFold  # noqa: PLC0415

        kf = KFold(n_splits=body.n_folds, shuffle=False)
        # Build OOF [n, n_candidates] of comparable scalars per row.
        oof = np.zeros((n, len(candidates)))
        for tr_idx, te_idx in kf.split(X_eff):
            for col, (spec, be) in enumerate(candidates):
                out = be.train(X_eff[tr_idx], y_eff[tr_idx], feature_names, spec.config)
                pred = be.predict(out["blob"], X_eff[te_idx], body.mode,
                                  body.quantile_levels)
                if body.mode == "direction":
                    oof[te_idx, col] = pred["prob_up"]
                elif body.mode == "value":
                    oof[te_idx, col] = pred["predicted"]
                else:
                    oof[te_idx, col] = pred["median"]
        # Per-candidate score (higher = better).
        scores = np.zeros(len(candidates))
        for col in range(len(candidates)):
            if body.mode == "direction":
                try:
                    scores[col] = roc_auc_score(y_eff, oof[:, col])
                except ValueError:
                    scores[col] = 0.5
            else:
                # Negative MAE so higher is better.
                scores[col] = -float(np.mean(np.abs(oof[:, col] - y_eff)))
        # Pairwise correlation (Pearson) of OOF predictions.
        corr = np.corrcoef(oof.T)
        if corr.ndim == 0:
            corr = np.array([[1.0]])
        slugs = [spec.backend for spec, _ in candidates]
        # Greedy selection by score, reject if pairwise corr > threshold.
        order = np.argsort(-scores)
        selected_idx: list[int] = []
        for idx in order:
            if len(selected_idx) >= body.max_members:
                break
            ok = True
            for s in selected_idx:
                if abs(corr[idx, s]) > body.max_pairwise_corr:
                    ok = False
                    break
            if ok:
                selected_idx.append(int(idx))
        # Honor min_members: pad by best-scoring leftovers if needed.
        if len(selected_idx) < body.min_members:
            for idx in order:
                if int(idx) in selected_idx:
                    continue
                selected_idx.append(int(idx))
                if len(selected_idx) >= body.min_members:
                    break
        # Retrain selected members on FULL pool.
        selected_members: list[dict[str, Any]] = []
        for idx in selected_idx:
            spec, be = candidates[idx]
            out = be.train(X_eff, y_eff, feature_names, spec.config)
            selected_members.append({"backend": spec.backend, "blob": out["blob"]})
        # Serialize per-candidate correlation map.
        corr_map: dict[str, dict[str, float]] = {}
        for i, si in enumerate(slugs):
            corr_map[si] = {sj: float(corr[i, j]) for j, sj in enumerate(slugs)}
        return selected_members, [slugs[i] for i in selected_idx], corr_map

    try:
        selected_members, selected_slugs, corr_map = await asyncio.to_thread(_do_train)
    except Exception as exc:  # noqa: BLE001
        log.exception("diversified meta train failed (model_id=%s)", body.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed = time.time() - t0

    payload = {
        "kind": "diversified",
        "mode": body.mode,
        "horizon": h,
        "feature_names": feature_names,
        "members": selected_members,
        "quantile_levels": body.quantile_levels,
    }
    buf = io.BytesIO()
    pickle.dump(payload, buf)
    _persist_meta_blob(
        model_id=body.model_id,
        backend="meta:diversified",
        mode=body.mode,
        horizon=h,
        feature_names=feature_names,
        n_training_rows=n,
        blob=buf.getvalue(),
    )

    return {
        "modelId": body.model_id,
        "kind": "diversified",
        "mode": body.mode,
        "horizon": h,
        "membersUsed": selected_slugs,
        "nTrainingRows": n,
        "nFeatures": len(feature_names),
        "featureNames": feature_names,
        "trainSecs": elapsed,
        "candidateCorr": corr_map,
    }


@router.post(
    "/forecast/diversified",
    response_model=MetaForecastResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast_diversified(body: MetaForecastRequest) -> dict[str, Any]:
    payload, meta = _load_meta(body.model_id, expected_kind="diversified")
    X = _build_forecast_X(body.features, payload["feature_names"])
    mode = payload["mode"]
    q_levels = payload.get("quantile_levels")

    def _do_predict():
        member_resp: dict[str, dict[str, Any]] = {}
        if mode == "direction":
            probs = []
            for m in payload["members"]:
                be = models.get_tabular_backend(m["backend"])
                p = be.predict(m["blob"], X, "direction")["prob_up"]
                probs.append(p)
                member_resp[m["backend"]] = {"probUp": [float(v) for v in p]}
            combined = np.mean(np.stack(probs, axis=0), axis=0)
            return {"prob": combined.astype(np.float64), "members": member_resp}
        if mode == "value":
            preds = []
            for m in payload["members"]:
                be = models.get_tabular_backend(m["backend"])
                p = be.predict(m["blob"], X, "value")["predicted"]
                preds.append(p)
                member_resp[m["backend"]] = {"predicted": [float(v) for v in p]}
            combined = np.mean(np.stack(preds, axis=0), axis=0)
            return {"pred": combined.astype(np.float64), "members": member_resp}
        # quantile
        medians = []
        quantile_sums: dict[str, np.ndarray] = {}
        for m in payload["members"]:
            be = models.get_tabular_backend(m["backend"])
            pred = be.predict(m["blob"], X, "quantile", q_levels)
            medians.append(pred["median"])
            for k, arr in pred["quantiles"].items():
                quantile_sums.setdefault(k, np.zeros_like(arr))
                quantile_sums[k] = quantile_sums[k] + np.asarray(arr, dtype=np.float64)
            member_resp[m["backend"]] = {
                "median": [[float(v)] for v in pred["median"]],
            }
        k = len(payload["members"])
        median = np.mean(np.stack(medians, axis=0), axis=0)
        quantile_avg = {key: v / k for key, v in quantile_sums.items()}
        return {
            "median": median,
            "quantiles": quantile_avg,
            "members": member_resp,
        }

    try:
        out = await asyncio.to_thread(_do_predict)
    except Exception as exc:  # noqa: BLE001
        log.exception("diversified forecast failed (model_id=%s)", body.model_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    resp: dict[str, Any] = {
        "modelId": body.model_id,
        "kind": "diversified",
        "mode": mode,
        "horizon": meta.horizon,
        "members": out["members"],
        "selectedMembers": [m["backend"] for m in payload["members"]],
    }
    if mode == "direction":
        prob_up = out["prob"]
        resp["probUp"] = [float(p) for p in prob_up]
        resp["confidence"] = [abs(float(p) - 0.5) * 2.0 for p in prob_up]
    elif mode == "value":
        resp["predicted"] = [float(v) for v in out["pred"]]
    else:
        resp["median"] = [[float(v)] for v in out["median"]]
        resp["quantiles"] = {
            k: [[float(v)] for v in arr] for k, arr in out["quantiles"].items()
        }
    return resp


# ── shared loader ──────────────────────────────────────────────────────
def _load_meta(
    model_id: str, expected_kind: str,
) -> tuple[dict[str, Any], storage.TabularMeta]:
    try:
        meta, blob = storage.load(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if meta.backend != f"meta:{expected_kind}":
        raise HTTPException(
            status_code=400,
            detail=(
                f"model_id {model_id!r} has backend={meta.backend!r}; "
                f"expected meta:{expected_kind}"
            ),
        )
    try:
        payload = pickle.loads(blob)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"blob unpickle failed: {exc}",
        ) from exc
    if payload.get("kind") != expected_kind:
        raise HTTPException(
            status_code=500,
            detail=(
                f"blob kind {payload.get('kind')!r} != expected "
                f"{expected_kind!r}"
            ),
        )
    return payload, meta
