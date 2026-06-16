"""Static correction router package."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routers.statics import (
    core_jobs,
    jobs,
)
from app.statics.refraction.api import router as refraction_router

router = APIRouter()
router.include_router(core_jobs.router)
router.include_router(refraction_router)
router.include_router(jobs.router)

__all__ = ['router']
