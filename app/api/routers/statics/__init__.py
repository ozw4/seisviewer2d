"""Static correction router package."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routers.statics import (
    core_jobs,
    jobs,
    refraction_apply,
    refraction_qc,
    refraction_table_export,
)
from app.services.datum_static_service import run_datum_static_apply_job
from app.services.first_break_qc_service import run_first_break_qc_job
from app.services.geometry_linkage_service import run_geometry_linkage_build_job
from app.services.job_runner import start_job_thread
from app.services.refraction_static_export_service import run_refraction_static_export_job
from app.services.refraction_static_service import run_refraction_static_apply_job
from app.services.residual_static_service import run_residual_static_apply_job
from app.services.time_term_static_service import run_time_term_static_apply_job

router = APIRouter()
router.include_router(core_jobs.router)
router.include_router(refraction_apply.router)
router.include_router(refraction_qc.router)
router.include_router(refraction_table_export.router)
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
