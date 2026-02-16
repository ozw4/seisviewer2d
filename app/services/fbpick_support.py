"""fbpick-related metadata helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from app.utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from app.utils.utils import TraceStoreSectionReader

if TYPE_CHECKING:
    from app.api.schemas import PipelineSpec

USE_FBPICK_OFFSET = 'offset' in FBPICK_MODEL_PATH.name.lower()
OFFSET_BYTE_FIXED: int = 37


def _spec_uses_fbpick(spec: PipelineSpec) -> bool:
    """Return ``True`` when ``spec`` contains an fbpick analyzer step."""
    return any(step.kind == 'analyzer' and step.name == 'fbpick' for step in spec.steps)


def _maybe_attach_fbpick_offsets(
    meta: dict[str, Any],
    *,
    spec: PipelineSpec,
    reader: TraceStoreSectionReader,
    key1: int,
    offset_byte: int | None,
    trace_slice: slice | None = None,
) -> dict[str, Any]:
    """Add offset metadata when the fbpick model expects it."""
    if not USE_FBPICK_OFFSET or offset_byte is None:
        return meta
    if not _spec_uses_fbpick(spec):
        return meta
    get_offsets = getattr(reader, 'get_offsets_for_section', None)
    if get_offsets is None:
        return meta
    offsets = get_offsets(key1, offset_byte)
    if trace_slice is not None:
        offsets = offsets[trace_slice]
    offsets = np.ascontiguousarray(offsets, dtype=np.float32)
    if not meta:
        return {'offsets': offsets}
    meta_with_offsets = dict(meta)
    meta_with_offsets['offsets'] = offsets
    return meta_with_offsets


__all__ = [
    'OFFSET_BYTE_FIXED',
    'USE_FBPICK_OFFSET',
    '_maybe_attach_fbpick_offsets',
    '_spec_uses_fbpick',
]
