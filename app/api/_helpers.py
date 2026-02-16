"""Shared API-layer state access and request guards."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from app.core.state import AppState

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def reject_legacy_key1_query_params(request: Request) -> None:
    """Reject legacy key1 query parameters for non-compatible endpoints."""
    legacy_val = 'key1' + '_val'
    legacy_idx = 'key1' + '_idx'
    present = [
        name for name in (legacy_val, legacy_idx) if name in request.query_params
    ]
    if present:
        names = ', '.join(sorted(present))
        raise HTTPException(
            status_code=422,
            detail=f'Legacy query parameter(s) are not supported: {names}; use key1',
        )


def get_state(app: FastAPI) -> AppState:
    """Return app-scoped state from ``app.state.sv``."""
    sv = getattr(getattr(app, 'state', None), 'sv', None)
    if isinstance(sv, AppState):
        return sv
    msg = 'Application state is not initialized (app.state.sv)'
    logger.error(msg)
    raise RuntimeError(msg)


def _resolve_state(
    *, app: FastAPI | None = None, state: AppState | None = None
) -> AppState:
    if state is not None:
        return state
    if app is None:
        raise RuntimeError('Either app or state must be provided')
    return get_state(app)


__all__ = ['get_state', 'reject_legacy_key1_query_params']
