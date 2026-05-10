"""Layer-specific observation gates for multi-layer refraction statics."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.refraction_static_layer_config import (
    RefractionStaticLayerConfig,
    normalize_refraction_static_layers,
)
from app.services.refraction_static_types import (
    RefractionLayerKind,
    RefractionLayerObservationMasks,
    RefractionStaticInputModel,
)

LAYER_REJECTION_OK = 'ok'
LAYER_REJECTION_OUTSIDE_GATE = 'outside_layer_offset_gate'
LAYER_REJECTION_DISABLED = 'layer_disabled'
LAYER_REJECTION_MISSING_OFFSET = 'layer_missing_offset'
LAYER_REJECTION_NOT_CONFIGURED = 'layer_not_configured'

_LAYER_KINDS: tuple[RefractionLayerKind, ...] = ('v2_t1', 'v3_t2', 'vsub_t3')
_REASON_DTYPE = '<U32'


class RefractionLayerObservationMaskError(ValueError):
    """Raised when layer observation masks cannot be built."""


def build_refraction_layer_observation_masks(
    *,
    input_model: RefractionStaticInputModel,
    model: Any,
) -> RefractionLayerObservationMasks:
    """Build sorted-order layer observation masks from an input model."""
    offset = getattr(input_model, 'distance_m_sorted', None)
    if offset is None:
        offset = getattr(input_model, 'offset_m_sorted', None)
    return build_refraction_layer_observation_masks_from_arrays(
        base_valid_mask_sorted=getattr(input_model, 'valid_observation_mask_sorted'),
        offset_m_sorted=offset,
        rejection_reason_sorted=getattr(input_model, 'rejection_reason_sorted'),
        model=model,
    )


def build_refraction_layer_observation_masks_from_arrays(
    *,
    base_valid_mask_sorted: np.ndarray,
    offset_m_sorted: np.ndarray,
    rejection_reason_sorted: np.ndarray,
    model: Any,
) -> RefractionLayerObservationMasks:
    """Build deterministic layer masks from sorted observation arrays."""
    base_valid = _coerce_1d_bool(base_valid_mask_sorted, name='base_valid_mask_sorted')
    expected_shape = base_valid.shape
    offset = _coerce_1d_float(
        offset_m_sorted,
        name='offset_m_sorted',
        expected_shape=expected_shape,
    )
    base_reason = _coerce_1d_reason(
        rejection_reason_sorted,
        name='rejection_reason_sorted',
        expected_shape=expected_shape,
    )
    configs = _layer_slots(model)
    allow_overlap = bool(getattr(model, 'allow_overlapping_layer_gates', False))
    if not allow_overlap:
        _validate_configured_gate_overlap(configs)

    layer_kind: list[str] = []
    layer_enabled: list[bool] = []
    layer_min_offset_m: list[float] = []
    layer_max_offset_m: list[float] = []
    used_by_kind: dict[str, np.ndarray] = {}
    reason_by_kind: dict[str, np.ndarray] = {}
    count_by_kind: dict[str, int] = {}

    finite_offset = np.isfinite(offset)
    for kind, config in configs:
        layer_kind.append(kind)
        enabled = bool(config.enabled) if config is not None else False
        layer_enabled.append(enabled)
        layer_min_offset_m.append(_stored_min_offset(config))
        layer_max_offset_m.append(_stored_max_offset(config))

        if config is None:
            used = np.zeros(expected_shape, dtype=bool)
            reason = np.full(
                expected_shape,
                LAYER_REJECTION_NOT_CONFIGURED,
                dtype=_REASON_DTYPE,
            )
        elif not config.enabled:
            used = np.zeros(expected_shape, dtype=bool)
            reason = np.full(
                expected_shape,
                LAYER_REJECTION_DISABLED,
                dtype=_REASON_DTYPE,
            )
        else:
            in_gate = _layer_gate_mask(offset, config)
            used = np.ascontiguousarray(base_valid & finite_offset & in_gate, dtype=bool)
            reason = np.asarray(base_reason, dtype=_REASON_DTYPE).copy()
            reason[used] = LAYER_REJECTION_OK
            missing_offset = base_valid & ~finite_offset
            reason[missing_offset] = LAYER_REJECTION_MISSING_OFFSET
            outside_gate = base_valid & finite_offset & ~in_gate
            reason[outside_gate] = LAYER_REJECTION_OUTSIDE_GATE

        used_by_kind[kind] = np.ascontiguousarray(used, dtype=bool)
        reason_by_kind[kind] = np.ascontiguousarray(reason, dtype=_REASON_DTYPE)
        count_by_kind[kind] = int(np.count_nonzero(used))

    if not allow_overlap:
        _validate_observation_overlap(used_by_kind)

    return RefractionLayerObservationMasks(
        layer_kind=np.asarray(layer_kind, dtype='<U16'),
        layer_enabled=np.asarray(layer_enabled, dtype=bool),
        layer_min_offset_m=np.asarray(layer_min_offset_m, dtype=np.float64),
        layer_max_offset_m=np.asarray(layer_max_offset_m, dtype=np.float64),
        layer_used_mask_sorted=used_by_kind,
        layer_rejection_reason_sorted=reason_by_kind,
        layer_observation_count=count_by_kind,
    )


def refraction_layer_observation_qc(
    masks: RefractionLayerObservationMasks,
) -> dict[str, dict[str, Any]]:
    """Return strict-JSON per-layer observation-gate QC."""
    kinds = [str(value) for value in masks.layer_kind.tolist()]
    enabled = np.asarray(masks.layer_enabled, dtype=bool)
    min_offset = np.asarray(masks.layer_min_offset_m, dtype=np.float64)
    max_offset = np.asarray(masks.layer_max_offset_m, dtype=np.float64)
    payload: dict[str, dict[str, Any]] = {}
    for index, kind in enumerate(kinds):
        reasons = np.asarray(
            masks.layer_rejection_reason_sorted[kind],
        ).astype(str, copy=False)
        candidate = int(np.count_nonzero(reasons == LAYER_REJECTION_OK))
        payload[kind] = {
            'enabled': bool(enabled[index]),
            'n_candidate_observations': candidate if bool(enabled[index]) else 0,
            'n_used_observations': int(masks.layer_observation_count[kind]),
            'min_offset_m': _json_optional_gate_value(min_offset[index]),
            'max_offset_m': _json_optional_gate_value(max_offset[index]),
            'rejection_counts': _reason_counts(reasons),
        }
    return payload


def _layer_slots(
    model: Any,
) -> tuple[tuple[RefractionLayerKind, RefractionStaticLayerConfig | None], ...]:
    configs = normalize_refraction_static_layers(model, enabled_only=False)
    by_kind = {config.kind: config for config in configs}
    if getattr(model, 'method', None) == 'multilayer_time_term':
        return tuple((kind, by_kind.get(kind)) for kind in _LAYER_KINDS)
    return tuple((config.kind, config) for config in configs)


def _layer_gate_mask(
    offset_m: np.ndarray,
    config: RefractionStaticLayerConfig,
) -> np.ndarray:
    gate = np.ones(offset_m.shape, dtype=bool)
    finite_offset = np.isfinite(offset_m)
    gate &= finite_offset
    if config.min_offset_m is not None:
        gate &= offset_m >= float(config.min_offset_m)
    if config.max_offset_m is not None:
        gate &= offset_m <= float(config.max_offset_m)
    return np.ascontiguousarray(gate, dtype=bool)


def _validate_configured_gate_overlap(
    configs: tuple[tuple[RefractionLayerKind, RefractionStaticLayerConfig | None], ...],
) -> None:
    enabled = [
        (kind, _stored_min_offset(config), _stored_max_offset(config))
        for kind, config in configs
        if config is not None and config.enabled
    ]
    for index, (kind_a, min_a, max_a) in enumerate(enabled):
        for kind_b, min_b, max_b in enabled[index + 1 :]:
            if _gates_overlap(min_a, max_a, min_b, max_b):
                raise RefractionLayerObservationMaskError(
                    'overlapping refraction layer offset gates are not allowed '
                    f'by default: {kind_a} overlaps {kind_b}'
                )


def _validate_observation_overlap(used_by_kind: dict[str, np.ndarray]) -> None:
    overlap_count: np.ndarray | None = None
    for used in used_by_kind.values():
        enabled = np.asarray(used, dtype=bool)
        if overlap_count is None:
            overlap_count = enabled.astype(np.int16, copy=True)
        else:
            overlap_count += enabled.astype(np.int16, copy=False)
    if overlap_count is not None and np.any(overlap_count > 1):
        raise RefractionLayerObservationMaskError(
            'overlapping refraction layer offset gates selected the same '
            'observation'
        )


def _gates_overlap(
    min_a: float,
    max_a: float,
    min_b: float,
    max_b: float,
) -> bool:
    lower = max(min_a, min_b)
    upper = min(max_a, max_b)
    return lower <= upper


def _stored_min_offset(config: RefractionStaticLayerConfig | None) -> float:
    if config is None or config.min_offset_m is None:
        return float('-inf')
    return float(config.min_offset_m)


def _stored_max_offset(config: RefractionStaticLayerConfig | None) -> float:
    if config is None or config.max_offset_m is None:
        return float('inf')
    return float(config.max_offset_m)


def _json_optional_gate_value(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _reason_counts(reasons: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(reasons.astype(str, copy=False), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}


def _coerce_1d_bool(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=bool)
    if arr.ndim != 1:
        raise RefractionLayerObservationMaskError(f'{name} must be 1-D')
    return np.ascontiguousarray(arr, dtype=bool)


def _coerce_1d_float(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise RefractionLayerObservationMaskError(f'{name} must be 1-D')
    if arr.shape != expected_shape:
        raise RefractionLayerObservationMaskError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _coerce_1d_reason(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values).astype(_REASON_DTYPE, copy=False)
    if arr.ndim != 1:
        raise RefractionLayerObservationMaskError(f'{name} must be 1-D')
    if arr.shape != expected_shape:
        raise RefractionLayerObservationMaskError(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        )
    return np.ascontiguousarray(arr, dtype=_REASON_DTYPE)


__all__ = [
    'LAYER_REJECTION_DISABLED',
    'LAYER_REJECTION_MISSING_OFFSET',
    'LAYER_REJECTION_NOT_CONFIGURED',
    'LAYER_REJECTION_OK',
    'LAYER_REJECTION_OUTSIDE_GATE',
    'RefractionLayerObservationMaskError',
    'build_refraction_layer_observation_masks',
    'build_refraction_layer_observation_masks_from_arrays',
    'refraction_layer_observation_qc',
]
