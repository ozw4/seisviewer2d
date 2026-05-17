"""Static-history and double-application QC for refraction artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
)
from app.services.refraction_static_types import RefractionDatumStaticsResult
from app.services.refraction_static_uphole import REFRACTION_UPHOLE_SOURCES_CSV_NAME
from app.services.refraction_static_artifacts.contract import (
    ARTIFACT_VERSION,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SIGN_CONVENTION,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    WORKFLOW,
)
from app.services.refraction_static_artifacts.field_corrections import (
    _field_correction_component_requested,
)
from app.services.refraction_static_artifacts.io import (
    _assert_strict_json,
    _write_json_atomic,
)
from app.services.refraction_static_artifacts.validation import _validate_result


def write_refraction_static_history_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    output_file_id: str | None = None,
) -> dict[str, Any]:
    """Write and return the strict-JSON static-component history artifact."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    payload = build_refraction_static_history_payload(
        result=values.result,
        req=request,
        output_file_id=output_file_id,
    )
    _write_json_atomic(Path(path), payload)
    return payload


def build_refraction_static_history_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    output_file_id: str | None = None,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    duplicate_qc = _static_history_qc_from_result(values.result, request)
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'artifact_kind': 'refraction_static_history',
        'workflow': WORKFLOW,
        'sign_convention': SIGN_CONVENTION,
        'input_file_id': request.file_id,
        'output_file_id': output_file_id,
        'double_application_policy': (
            request.field_corrections.composition.double_application_policy
        ),
        'components': _static_history_components(request),
        'cumulative_shift_artifact': REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        'cumulative_shift_field': _static_history_cumulative_shift_field(request),
        'double_application': duplicate_qc,
        'warnings': list(duplicate_qc.get('warnings', [])),
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_STATIC_HISTORY_JSON_NAME)
    return payload


def refraction_static_trace_shift_component_names(
    req: RefractionStaticApplyRequest,
) -> tuple[str, ...]:
    request = RefractionStaticApplyRequest.model_validate(req)
    components = ['refraction']
    if _field_components_applied_to_trace_shift(request):
        components.extend(_requested_field_component_names(request))
    return tuple(components)


def refraction_static_double_application_qc(
    *,
    req: RefractionStaticApplyRequest,
    source_meta: Mapping[str, object],
) -> dict[str, Any]:
    request = RefractionStaticApplyRequest.model_validate(req)
    return static_history_double_application_qc(
        input_file_id=request.file_id,
        policy=request.field_corrections.composition.double_application_policy,
        requested_components=refraction_static_trace_shift_component_names(request),
        source_meta=source_meta,
    )


def static_history_double_application_qc(
    *,
    input_file_id: str,
    policy: str,
    requested_components: Iterable[str],
    source_meta: Mapping[str, object],
) -> dict[str, Any]:
    requested = {
        canonical
        for canonical in (
            _canonical_static_history_component(component)
            for component in requested_components
        )
        if canonical is not None
    }
    existing, suspected = _lineage_component_names(source_meta)
    duplicate_components = sorted(requested.intersection(existing))
    suspected_components = sorted(
        requested.intersection(suspected) - set(duplicate_components)
    )
    warnings: list[str] = []
    message = ''
    status = 'checked'
    if duplicate_components or suspected_components:
        status = 'duplicate_rejected' if policy == 'fail' else (
            'duplicate_allowed' if policy == 'allow' else 'duplicate_warned'
        )
        message = _double_application_message(
            input_file_id=input_file_id,
            duplicate_components=duplicate_components,
            suspected_components=suspected_components,
            policy=policy,
        )
        if policy != 'fail':
            warnings.append(message)

    return {
        'policy': policy,
        'status': status,
        'checked_components': sorted(requested),
        'existing_components': sorted(existing),
        'suspected_existing_components': sorted(suspected),
        'duplicate_components': duplicate_components,
        'suspected_duplicate_components': suspected_components,
        'message': message,
        'warnings': warnings,
    }


def _static_history_components(
    req: RefractionStaticApplyRequest,
) -> list[dict[str, object]]:
    field_components_applied = _field_components_applied_to_trace_shift(req)
    components: list[dict[str, object]] = [
        {
            'name': _refraction_history_component_name(req),
            'applied_to_trace_shift': True,
            'artifact': REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        }
    ]
    if req.field_corrections.source_depth.mode != 'none':
        components.append(
            {
                'name': 'source_depth',
                'applied_to_trace_shift': field_components_applied,
                'artifact': REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
            }
        )
    if req.field_corrections.uphole.mode != 'none':
        components.append(
            {
                'name': 'uphole',
                'applied_to_trace_shift': field_components_applied,
                'artifact': REFRACTION_UPHOLE_SOURCES_CSV_NAME,
            }
        )
    if req.field_corrections.manual_static.mode != 'none':
        components.append(
            {
                'name': 'manual_static',
                'applied_to_trace_shift': field_components_applied,
                'artifact': SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
            }
        )
    return components


def _refraction_history_component_name(req: RefractionStaticApplyRequest) -> str:
    if req.conversion.mode in {'t1lsst_1layer', 't1lsst_multilayer'}:
        return 'refraction_t1lsst'
    return 'refraction'


def _static_history_cumulative_shift_field(req: RefractionStaticApplyRequest) -> str:
    if _field_components_applied_to_trace_shift(req):
        return 'final_trace_shift_s_sorted'
    return 'refraction_trace_shift_s_sorted'


def _field_components_applied_to_trace_shift(
    req: RefractionStaticApplyRequest,
) -> bool:
    return bool(
        _field_correction_component_requested(req)
        and req.field_corrections.composition.enabled
        and req.field_corrections.composition.apply_to_trace_shift
    )


def _requested_field_component_names(
    req: RefractionStaticApplyRequest,
) -> tuple[str, ...]:
    components: list[str] = []
    if req.field_corrections.source_depth.mode != 'none':
        components.append('source_depth')
    if req.field_corrections.uphole.mode != 'none':
        components.append('uphole')
    if req.field_corrections.manual_static.mode != 'none':
        components.append('manual_static')
    return tuple(components)


def _static_history_qc_from_result(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    raw = result.qc.get('static_history')
    if isinstance(raw, dict):
        payload = dict(raw)
        payload.setdefault(
            'policy',
            req.field_corrections.composition.double_application_policy,
        )
        payload.setdefault('warnings', [])
        return payload
    return {
        'policy': req.field_corrections.composition.double_application_policy,
        'status': 'not_checked',
        'checked_components': list(refraction_static_trace_shift_component_names(req)),
        'existing_components': [],
        'suspected_existing_components': [],
        'duplicate_components': [],
        'suspected_duplicate_components': [],
        'message': '',
        'warnings': [],
    }


def _lineage_component_names(
    source_meta: Mapping[str, object],
) -> tuple[set[str], set[str]]:
    existing: set[str] = set()
    suspected: set[str] = set()
    derived = source_meta.get('derived')
    if isinstance(derived, Mapping):
        _collect_lineage_components(derived, existing=existing, suspected=suspected)
        components = derived.get('components')
        if isinstance(components, list):
            for component in components:
                if isinstance(component, Mapping):
                    _collect_lineage_components(
                        component,
                        existing=existing,
                        suspected=suspected,
                    )
        history = derived.get('static_history')
        if isinstance(history, Mapping):
            _collect_history_components(history, existing=existing)

    history = source_meta.get('static_history')
    if isinstance(history, Mapping):
        _collect_history_components(history, existing=existing)
    return existing, suspected


def _collect_lineage_components(
    values: Mapping[str, object],
    *,
    existing: set[str],
    suspected: set[str],
) -> None:
    for item in _string_items(values.get('static_components_applied')):
        canonical = _canonical_static_history_component(item)
        if canonical:
            existing.add(canonical)

    component_name = _canonical_static_history_component(values.get('name'))
    if component_name and values.get('applied_to_trace_shift') is not False:
        existing.add(component_name)

    field_applied = values.get('field_corrections_applied_to_trace_shift')
    if field_applied is True:
        requested_fields = [
            item
            for item in (
                _canonical_static_history_component(raw)
                for raw in _string_items(
                    values.get('field_correction_components_requested')
                )
            )
            if item in {'source_depth', 'uphole', 'manual_static'}
        ]
        if requested_fields:
            existing.update(requested_fields)
        else:
            suspected.update({'source_depth', 'uphole', 'manual_static'})


def _collect_history_components(
    history: Mapping[str, object],
    *,
    existing: set[str],
) -> None:
    components = history.get('components')
    if not isinstance(components, list):
        return
    for component in components:
        if not isinstance(component, Mapping):
            continue
        if component.get('applied_to_trace_shift') is not True:
            continue
        canonical = _canonical_static_history_component(component.get('name'))
        if canonical:
            existing.add(canonical)


def _string_items(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _canonical_static_history_component(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    name = value.strip().lower()
    if name in {
        'refraction',
        'refraction_t1lsst',
        'refraction_static',
        'refraction_static_correction',
        'refraction_static_table_apply',
        'static_table_apply',
        'source_static_table',
        'receiver_static_table',
    }:
        return 'refraction'
    if name in {'source_depth', 'source_depth_shift_s'}:
        return 'source_depth'
    if name in {'uphole', 'uphole_time', 'uphole_shift_s'}:
        return 'uphole'
    if name in {'manual_static', 'manual_static_shift_s'}:
        return 'manual_static'
    return None


def _double_application_message(
    *,
    input_file_id: str,
    duplicate_components: list[str],
    suspected_components: list[str],
    policy: str,
) -> str:
    parts: list[str] = []
    if duplicate_components:
        parts.append(
            'duplicate static components already applied: '
            + ', '.join(duplicate_components)
        )
    if suspected_components:
        parts.append(
            'static components may already be applied: '
            + ', '.join(suspected_components)
        )
    detail = '; '.join(parts) if parts else 'duplicate static components detected'
    return (
        f'static history check for input file_id {input_file_id!r}: {detail}; '
        f'double_application_policy={policy}'
    )


__all__ = [
    'build_refraction_static_history_payload',
    'refraction_static_double_application_qc',
    'refraction_static_trace_shift_component_names',
    'static_history_double_application_qc',
    'write_refraction_static_history_json',
]
