"""Foundation-model lifecycle endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..auth import check_bearer
from ..errors import ModelBusyError, ModelUnloadError
from ..model_lifecycle import model_lifecycle

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.post("/unload", dependencies=[Depends(check_bearer)])
async def post_unload_models() -> dict[str, Any]:
    """Release every idle foundation model and its runtime-owned memory."""
    try:
        return await model_lifecycle.unload_all()
    except ModelBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelUnloadError as exc:
        raise HTTPException(status_code=503, detail="foundation-model unload failed") from exc
