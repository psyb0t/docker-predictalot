"""POST /v1/multivariate/{forecast,forecast/ensemble} + GET /v1/multivariate/models.

WARNING: moirai-2 multivariate is upstream-untested (see
`.research_files/moirai2-modes.md` §footguns). Verify channel-order
correctness before relying on it.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import dispatch, models, types
from ..auth import check_bearer
from ._common import build_type_models_response
from .schemas import (
    MultivariateEnsembleRequest,
    MultivariateEnsembleResponse,
    MultivariateRequest,
    MultivariateResponse,
)

log = logging.getLogger("predictalot.routers.multivariate")

router = APIRouter(prefix="/v1/multivariate", tags=["multivariate"])


@router.post(
    "/forecast/ensemble",
    response_model=MultivariateEnsembleResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_ensemble(body: MultivariateEnsembleRequest) -> dict[str, Any]:
    try:
        return await dispatch.ensemble_multivariate(
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
        log.exception("multivariate ensemble failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/forecast",
    response_model=MultivariateResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast(body: MultivariateRequest) -> dict[str, Any]:
    try:
        return await dispatch.dispatch_multivariate(
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
        log.exception("multivariate forecast failed for model=%s", body.model)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models", dependencies=[Depends(check_bearer)])
def list_models() -> dict[str, Any]:
    return build_type_models_response(types.TYPE_MULTIVARIATE, models)
