"""Static correction job lifecycle APIs."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.job_lifecycle_router import build_job_lifecycle_router
from app.contracts.statics.common import (
    StaticJobFilesResponse,
    StaticJobStatusResponse,
)

router = APIRouter()
router.include_router(
    build_job_lifecycle_router(
        route_prefix='/statics/job',
        allowed_job_types={'statics'},
        status_response_model=StaticJobStatusResponse,
        files_response_model=StaticJobFilesResponse,
    )
)
