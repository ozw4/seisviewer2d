"""fbpick-related metadata helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from app.services.errors import UnprocessableError
from app.utils.fbpick_models import validate_model_id
from app.utils.utils import TraceStoreSectionReader

if TYPE_CHECKING:
    from app.api.schemas import PipelineSpec

DEFAULT_FBPICK_MODEL_ID = 'fbpick_edgenext_small.pt'
OFFSET_BYTE_FIXED: int = 37


def _get_fbpick_model_id_from_spec(spec: PipelineSpec) -> str | None:
    """Return model_id from fbpick analyzer params if present."""
    for step in spec.steps:
        if step.kind != 'analyzer' or step.name != 'fbpick':
            continue
        params = step.params if isinstance(step.params, dict) else {}
        model_id = params.get('model_id')
        if model_id is None:
            return DEFAULT_FBPICK_MODEL_ID
        if not isinstance(model_id, str):
            raise UnprocessableError('model_id must be a string')
        return validate_model_id(model_id)
    return None


def _spec_uses_fbpick(spec: PipelineSpec) -> bool:
    """Return ``True`` when ``spec`` contains an fbpick analyzer step."""
    return _get_fbpick_model_id_from_spec(spec) is not None


def _spec_uses_offset(spec: PipelineSpec) -> bool:
    """Return True when fbpick in ``spec`` requires offset channel."""
    model_id = _get_fbpick_model_id_from_spec(spec)
    if model_id is None:
        return False
    return 'offset' in model_id.lower()


def _maybe_attach_fbpick_offsets(
    meta: dict[str, Any],
    *,
    spec: PipelineSpec,
    reader: TraceStoreSectionReader,
    key1: int,
    offset_byte: int | None,
    trace_slice: slice | None = None,
    section_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Add offset metadata when the fbpick model expects it."""
    if not _spec_uses_offset(spec) or offset_byte is None:
        return meta

    get_offsets = getattr(reader, 'get_offsets_for_section', None)
    if not callable(get_offsets):
        raise UnprocessableError(
            'Offset-enabled model requires get_offsets_for_section'
        )

    try:
        offsets = get_offsets(key1, offset_byte)
    except Exception as exc:  # noqa: BLE001
        raise UnprocessableError(
            'Failed to read offsets for offset-enabled model'
        ) from exc
    if offsets is None:
        raise UnprocessableError('Offsets are required for offset-enabled model')

    try:
        sliced = offsets[trace_slice] if trace_slice is not None else offsets
    except Exception as exc:  # noqa: BLE001
        raise UnprocessableError(
            'Failed to slice offsets for offset-enabled model'
        ) from exc

    try:
        offsets_arr = np.ascontiguousarray(sliced, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise UnprocessableError(
            'Offsets must be numeric for offset-enabled model'
        ) from exc

    if not np.all(np.isfinite(offsets_arr)):
        raise UnprocessableError('Offsets contain NaN or Inf for offset-enabled model')

    if section_shape is not None:
        h, w = int(section_shape[0]), int(section_shape[1])
        if offsets_arr.ndim == 1 and offsets_arr.shape[0] != h:
            raise UnprocessableError('Offsets length does not match trace slice shape')
        if offsets_arr.ndim == 2 and offsets_arr.shape not in {(h, w), (w, h)}:
            raise UnprocessableError('Offsets shape does not match section shape')

    if not meta:
        return {'offsets': offsets_arr}
    meta_with_offsets = dict(meta)
    meta_with_offsets['offsets'] = offsets_arr
    return meta_with_offsets


__all__ = [
    'DEFAULT_FBPICK_MODEL_ID',
    'OFFSET_BYTE_FIXED',
    '_get_fbpick_model_id_from_spec',
    '_maybe_attach_fbpick_offsets',
    '_spec_uses_fbpick',
    '_spec_uses_offset',
]
