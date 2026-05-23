"""POST /v1/univariate/{forecast,forecast/ensemble} + GET /v1/univariate/models."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import dispatch, models, types
from ..auth import check_bearer
from .schemas import (
    UnivariateEnsembleRequest,
    UnivariateEnsembleResponse,
    UnivariateRequest,
    UnivariateResponse,
)
from ._common import build_type_models_response

log = logging.getLogger("predictalot.routers.univariate")

router = APIRouter(prefix="/v1/univariate", tags=["univariate"])


@router.post(
    "/forecast/ensemble",
    response_model=UnivariateEnsembleResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_ensemble(body: UnivariateEnsembleRequest) -> dict[str, Any]:
    try:
        return await dispatch.ensemble_univariate(
            context=body.context,
            horizon=body.config.horizon,
            quantile_levels=body.config.quantile_levels,
            context_length=body.config.context_length,
            weights=body.weights,
            unload_after=body.unload,
        )
    except (dispatch.BadQuantileLevelsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("univariate ensemble failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/forecast",
    response_model=UnivariateResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast(body: UnivariateRequest) -> dict[str, Any]:
    try:
        return await dispatch.dispatch_univariate(
            model=body.model,
            context=body.context,
            horizon=body.config.horizon,
            quantile_levels=body.config.quantile_levels,
            context_length=body.config.context_length,
            unload_after=body.unload,
        )
    except (dispatch.UnknownModelError, types.UnknownTypeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except types.ModelDoesNotSupportTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (dispatch.BadQuantileLevelsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("univariate forecast failed for model=%s", body.model)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models", dependencies=[Depends(check_bearer)])
def list_models() -> dict[str, Any]:
    return build_type_models_response(types.TYPE_UNIVARIATE, models)
