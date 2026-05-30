"""Static correction router package."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routers.statics import jobs, legacy
from app.api.routers.statics.legacy import (
    run_datum_static_apply_job,
    run_first_break_qc_job,
    run_geometry_linkage_build_job,
    run_refraction_static_apply_job,
    run_refraction_static_export_job,
    run_residual_static_apply_job,
    run_time_term_static_apply_job,
)
from app.services.job_runner import start_job_thread

router = APIRouter()
router.include_router(legacy.router)
router.include_router(jobs.router)

__all__ = [
    'router',
    'run_datum_static_apply_job',
    'run_first_break_qc_job',
    'run_geometry_linkage_build_job',
    'run_refraction_static_apply_job',
    'run_refraction_static_export_job',
    'run_residual_static_apply_job',
    'run_time_term_static_apply_job',
    'start_job_thread',
]
