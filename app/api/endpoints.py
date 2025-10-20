"""Legacy endpoints module.

This module now aggregates routers from :mod:`app.api.routers` while keeping
historical symbols available for tests and external imports.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api._helpers import (
	EXPECTED_SECTION_NDIM,
	OFFSET_BYTE_FIXED,
	USE_FBPICK_OFFSET,
	PipelineTapNotFoundError,
	_filename_for_file_id,
	_maybe_attach_fbpick_offsets,
	_pipeline_payload_to_array,
	_spec_uses_fbpick,
	_update_file_registry,
	cached_readers,
	fbpick_cache,
	get_raw_section,
	get_reader,
	get_section_from_pipeline_tap,
	jobs,
	pipeline_tap_cache,
	window_section_cache,
)
from app.api.routers import (
	fbpick_router,
	picks_router,
	pipeline_router,
	section_router,
	upload_router,
)
from app.utils.segy_meta import (  # noqa: F401
	FILE_REGISTRY,
	get_dt_for_file,
	read_segy_dt_seconds,
)
from app.utils.utils import (  # noqa: F401
	SegySectionReader,
	TraceStoreSectionReader,
	quantize_float32,
	to_builtin,
)

router = APIRouter()
router.include_router(upload_router)
router.include_router(section_router)
router.include_router(fbpick_router)
router.include_router(pipeline_router)
router.include_router(picks_router)

__all__ = [
	'EXPECTED_SECTION_NDIM',
	'OFFSET_BYTE_FIXED',
	'USE_FBPICK_OFFSET',
	'PipelineTapNotFoundError',
	'_filename_for_file_id',
	'_maybe_attach_fbpick_offsets',
	'_pipeline_payload_to_array',
	'_spec_uses_fbpick',
	'_update_file_registry',
	'cached_readers',
	'fbpick_cache',
	'get_raw_section',
	'get_reader',
	'get_section_from_pipeline_tap',
	'jobs',
	'pipeline_tap_cache',
	'router',
	'window_section_cache',
]
