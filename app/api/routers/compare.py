"""Raw compare endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.api._helpers import get_state
from app.services.raw_compare_validation import (
    validate_raw_compare_grid as validate_raw_compare_grid_service,
)

router = APIRouter()


@router.get('/compare/raw/validate')
def validate_raw_compare_grid(
    request: Request,
    file_id_a: Annotated[str, Query(...)],
    file_id_b: Annotated[str, Query(...)],
    key1_byte: Annotated[int, Query()] = 189,
    key2_byte: Annotated[int, Query()] = 193,
) -> JSONResponse:
    """Validate raw SGY A/B grid compatibility before compare execution."""
    state = get_state(request.app)
    try:
        payload = validate_raw_compare_grid_service(
            file_id_a=file_id_a,
            file_id_b=file_id_b,
            key1_byte=int(key1_byte),
            key2_byte=int(key2_byte),
            state=state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=payload)
