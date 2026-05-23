"""POST /v1/covariates/future/{forecast,forecast/ensemble} + GET /v1/covariates/future/models."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import dispatch, models, types
from ..auth import check_bearer
from ._common import build_type_models_response
from .schemas import (
    CovariatesFutureEnsembleRequest,
    CovariatesFutureEnsembleResponse,
    CovariatesFutureRequest,
    CovariatesFutureResponse,
)

log = logging.getLogger("predictalot.routers.covariates_future")

router = APIRouter(prefix="/v1/covariates/future", tags=["covariates-future"])


@router.post(
    "/forecast/ensemble",
    response_model=CovariatesFutureEnsembleResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_ensemble(body: CovariatesFutureEnsembleRequest) -> dict[str, Any]:
    try:
        return await dispatch.ensemble_covariates_future(
            context=body.context,
            future_covariates=body.future_covariates,
            horizon=body.config.horizon,
            quantile_levels=body.config.quantile_levels,
            context_length=body.config.context_length,
            weights=body.weights,
            unload_after=body.unload,
        )
    except (dispatch.BadQuantileLevelsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("covariates-future ensemble failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/forecast",
    response_model=CovariatesFutureResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast(body: CovariatesFutureRequest) -> dict[str, Any]:
    try:
        return await dispatch.dispatch_covariates_future(
            model=body.model,
            context=body.context,
            future_covariates=body.future_covariates,
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
        log.exception("covariates-future forecast failed for model=%s", body.model)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models")
def list_models() -> dict[str, Any]:
    return build_type_models_response(types.TYPE_COVARIATES_FUTURE, models)
