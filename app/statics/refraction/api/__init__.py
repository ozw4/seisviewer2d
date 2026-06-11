"""Refraction statics API package."""

from __future__ import annotations

from fastapi import APIRouter

from app.statics.refraction.api.apply import router as apply_router
from app.statics.refraction.api.qc import router as qc_router
from app.statics.refraction.api.table_export import router as table_export_router

router = APIRouter()
router.include_router(apply_router)
router.include_router(qc_router)
router.include_router(table_export_router)

__all__ = ['router']
