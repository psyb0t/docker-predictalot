"""POST /v1/covariates/past/{forecast,forecast/ensemble} + GET /v1/covariates/past/models."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import dispatch, models, types
from ..auth import check_bearer
from ._common import build_type_models_response
from .schemas import (
    CovariatesPastEnsembleRequest,
    CovariatesPastEnsembleResponse,
    CovariatesPastRequest,
    CovariatesPastResponse,
)

log = logging.getLogger("predictalot.routers.covariates_past")

router = APIRouter(prefix="/v1/covariates/past", tags=["covariates-past"])


@router.post(
    "/forecast/ensemble",
    response_model=CovariatesPastEnsembleResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_ensemble(body: CovariatesPastEnsembleRequest) -> dict[str, Any]:
    try:
        return await dispatch.ensemble_covariates_past(
            context=body.context,
            past_covariates=body.past_covariates,
            horizon=body.config.horizon,
            quantile_levels=body.config.quantile_levels,
            context_length=body.config.context_length,
            weights=body.weights,
            unload_after=body.unload,
        )
    except (dispatch.BadQuantileLevelsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("covariates-past ensemble failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/forecast",
    response_model=CovariatesPastResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast(body: CovariatesPastRequest) -> dict[str, Any]:
    try:
        return await dispatch.dispatch_covariates_past(
            model=body.model,
            context=body.context,
            past_covariates=body.past_covariates,
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
        log.exception("covariates-past forecast failed for model=%s", body.model)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models", dependencies=[Depends(check_bearer)])
def list_models() -> dict[str, Any]:
    return build_type_models_response(types.TYPE_COVARIATES_PAST, models)
