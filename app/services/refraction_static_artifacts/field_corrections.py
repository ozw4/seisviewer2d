"""Field-correction QC helpers for refraction static artifacts."""

from __future__ import annotations

from typing import Any

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_types import RefractionDatumStaticsResult
from app.services.refraction_static_artifacts.contract import (
    SIGN_CONVENTION,
    RefractionStaticArtifactError,
)
from app.services.refraction_static_artifacts.solution import (
    _has_field_correction_composition,
    _has_manual_static_field_correction,
    _has_source_depth_field_correction,
    _has_uphole_field_correction,
)


def _source_depth_field_correction_qc(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    if not _has_source_depth_field_correction(result):
        if req.field_corrections.source_depth.mode != 'none':
            raise RefractionStaticArtifactError(
                'source-depth field correction artifacts require source-depth '
                'component arrays'
            )
        return {}
    qc = result.source_depth_field_correction_qc
    if not isinstance(qc, dict):
        raise RefractionStaticArtifactError(
            'source_depth_field_correction_qc is required when source-depth '
            'component arrays are present'
        )
    payload = dict(qc)
    payload.setdefault('source_depth_mode', req.field_corrections.source_depth.mode)
    payload.setdefault('component_name', 'source_depth_shift_s')
    payload.setdefault('source_depth_double_count_guard', 'checked')
    payload.setdefault('warnings', [])
    return payload


def _uphole_field_correction_qc(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    if not _has_uphole_field_correction(result):
        if req.field_corrections.uphole.mode != 'none':
            raise RefractionStaticArtifactError(
                'uphole field correction artifacts require uphole component arrays'
            )
        return {}
    qc = result.source_uphole_field_correction_qc
    if not isinstance(qc, dict):
        raise RefractionStaticArtifactError(
            'source_uphole_field_correction_qc is required when uphole '
            'component arrays are present'
        )
    payload = dict(qc)
    payload.setdefault('uphole_mode', req.field_corrections.uphole.mode)
    payload.setdefault('component_name', 'uphole_shift_s')
    return payload


def _manual_static_field_correction_qc(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    if not _has_manual_static_field_correction(result):
        if req.field_corrections.manual_static.mode != 'none':
            raise RefractionStaticArtifactError(
                'manual static field correction artifacts require manual '
                'static component arrays'
            )
        return {}
    qc = result.manual_static_field_correction_qc
    if not isinstance(qc, dict):
        raise RefractionStaticArtifactError(
            'manual_static_field_correction_qc is required when manual static '
            'component arrays are present'
        )
    payload = dict(qc)
    payload.setdefault(
        'manual_static_mode',
        req.field_corrections.manual_static.mode,
    )
    payload.setdefault('component_name', 'manual_static_shift_s')
    return payload


def _field_correction_composition_qc(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    if not _has_field_correction_composition(result):
        if _field_correction_component_requested(req):
            return {
                'composition_enabled': bool(
                    req.field_corrections.composition.enabled
                ),
                'apply_to_trace_shift': bool(
                    req.field_corrections.composition.apply_to_trace_shift
                ),
                'invalid_component_policy': (
                    req.field_corrections.composition.invalid_component_policy
                ),
                'sign_convention': SIGN_CONVENTION,
                'status': 'not_composed',
            }
        return {}
    qc = result.field_composition_qc
    if not isinstance(qc, dict):
        raise RefractionStaticArtifactError(
            'field_composition_qc is required when field-composition arrays '
            'are present'
        )
    payload = dict(qc)
    payload.setdefault('composition_enabled', True)
    payload.setdefault(
        'apply_to_trace_shift',
        bool(req.field_corrections.composition.apply_to_trace_shift),
    )
    payload.setdefault(
        'invalid_component_policy',
        req.field_corrections.composition.invalid_component_policy,
    )
    payload.setdefault('sign_convention', SIGN_CONVENTION)
    return payload


def _field_correction_component_requested(
    req: RefractionStaticApplyRequest,
) -> bool:
    return (
        req.field_corrections.source_depth.mode != 'none'
        or req.field_corrections.uphole.mode != 'none'
        or req.field_corrections.manual_static.mode != 'none'
    )


__all__ = [
    '_field_correction_component_requested',
    '_field_correction_composition_qc',
    '_manual_static_field_correction_qc',
    '_source_depth_field_correction_qc',
    '_uphole_field_correction_qc',
]
