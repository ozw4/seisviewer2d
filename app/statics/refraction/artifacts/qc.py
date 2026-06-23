"""JSON QC and static-history artifact builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.statics.refraction.artifacts.cell_velocity import (
    _cell_velocity_component,
    _layer_velocity_modes_for_request,
    _request_summary,
)
from app.statics.refraction.artifacts.contract import (
    ARTIFACT_VERSION,
    NEGATIVE_SHIFT_DESCRIPTION,
    POSITIVE_SHIFT_DESCRIPTION,
    REFRACTION_GRID_MAP_QC_CSV_NAME,
    REFRACTION_GRID_MAP_QC_JSON_NAME,
    REFRACTION_GRID_MAP_QC_NPZ_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SIGN_CONVENTION,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    STATIC_COMPONENT,
    WORKFLOW,
    RefractionStaticArtifactError,
)
from app.statics.refraction.artifacts.field_corrections import (
    _has_field_correction_composition,
    _has_manual_static_field_correction,
    _has_source_depth_field_correction,
    _has_uphole_field_correction,
)
from app.statics.refraction.artifacts.formatters import _json_float
from app.statics.refraction.artifacts.io import _assert_strict_json, _write_json_atomic
from app.statics.refraction.artifacts.registry import (
    _artifact_entries_for_request,
    _artifact_list_for_qc,
    _cell_velocity_artifact_names,
    _request_cell_velocity_layer_kinds,
    _validate_upstream_artifact_names,
)
from app.statics.refraction.artifacts.stats import _fraction, _residual_stat, _stat, _status_counts
from app.statics.refraction.artifacts.validation import (
    _validate_resolved_first_layer,
    _validate_result,
)
from seis_statics.refraction.cell_coordinates import refraction_cell_coordinate_metadata_from_config
from app.statics.refraction.core_options import (
    refractor_cell_options_from_request,
)
from app.statics.refraction.artifacts.source_depth import REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME
from app.statics.refraction.artifacts.uphole import REFRACTION_UPHOLE_SOURCES_CSV_NAME
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    ResolvedRefractionFirstLayer,
)

def write_refraction_static_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    qc: dict[str, Any] | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> dict[str, Any]:
    """Write and return the strict-JSON QC summary artifact."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    first_layer = _validate_resolved_first_layer(
        result=values.result,
        req=request,
        resolved_first_layer=resolved_first_layer,
    )
    payload = qc if qc is not None else build_refraction_static_qc_payload(
        result=values.result,
        req=request,
        resolved_first_layer=first_layer,
    )
    _write_json_atomic(Path(path), payload)
    return payload

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

def build_refraction_static_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    upstream_artifact_names: Iterable[str] = (),
) -> dict[str, Any]:
    values = _validate_result(result)
    r = values.result
    first_layer = _validate_resolved_first_layer(
        result=r,
        req=req,
        resolved_first_layer=resolved_first_layer,
    )
    residual_ms = r.residual_time_s[r.used_row_mask] * 1000.0
    refraction_ms = r.refraction_trace_shift_s_sorted * 1000.0
    valid_refraction_ms = refraction_ms[r.trace_static_valid_mask_sorted]
    floating_values = np.concatenate(
        [
            r.source_floating_datum_elevation_m,
            r.receiver_floating_datum_elevation_m,
        ]
    )
    request = _request_summary(req)
    upstream_names = _validate_upstream_artifact_names(
        upstream_artifact_names,
        resolved_first_layer=first_layer,
        req=req,
    )
    artifact_entries = _artifact_entries_for_request(
        req,
        first_layer,
        upstream_artifact_names=upstream_names,
    )
    method = r.qc.get('method')
    if not isinstance(method, str) or not method:
        method = req.model.method
    conversion_mode = r.qc.get('conversion_mode')
    if not isinstance(conversion_mode, str) or not conversion_mode:
        conversion_mode = req.conversion.mode
    layer_count = r.qc.get('layer_count')
    if layer_count is None:
        layer_count = req.conversion.layer_count
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'method': method,
        'workflow': WORKFLOW,
        'static_component': STATIC_COMPONENT,
        'conversion_mode': conversion_mode,
        'sign_convention': _sign_convention_qc_payload(req),
        'request': request,
        'velocity': {
            'v1_mode': first_layer.mode,
            'v1_status': first_layer.status,
            'weathering_velocity_m_s': _json_float(r.weathering_velocity_m_s),
            'resolved_weathering_velocity_m_s': _json_float(
                first_layer.weathering_velocity_m_s
            ),
            'bedrock_velocity_mode': r.bedrock_velocity_mode,
            'bedrock_velocity_m_s': _json_float(r.bedrock_velocity_m_s),
            'bedrock_slowness_s_per_m': _json_float(r.bedrock_slowness_s_per_m),
            'replacement_slowness_delta_s_per_m': _json_float(
                r.replacement_slowness_delta_s_per_m
            ),
            'bedrock_velocity_status': (
                'per_cell'
                if r.bedrock_velocity_mode == 'solve_cell'
                else ('solved' if r.bedrock_velocity_mode == 'solve_global' else 'fixed')
            ),
        },
        'datum': {
            'datum_mode': r.datum_mode,
            'floating_datum_mode': r.floating_datum_mode,
            'flat_datum_elevation_m': _json_float(r.flat_datum_elevation_m),
            'floating_datum_elevation_min_m': _stat(floating_values, 'min'),
            'floating_datum_elevation_max_m': _stat(floating_values, 'max'),
            'floating_datum_elevation_median_m': _stat(floating_values, 'median'),
            'floating_datum_below_refractor_count': int(
                r.qc.get('floating_datum_below_refractor_count', 0)
            ),
            'flat_datum_below_refractor_count': int(
                r.qc.get('flat_datum_below_refractor_count', 0)
            ),
        },
        'observations': {
            'n_traces': values.n_traces,
            'n_valid_observations': values.n_rows,
            'n_used_observations': int(np.count_nonzero(r.used_row_mask)),
            'n_rejected_by_robust': int(
                np.count_nonzero(r.rejected_by_robust_mask)
            ),
            'used_fraction': _fraction(np.count_nonzero(r.used_row_mask), values.n_rows),
        },
        'nodes': {
            'n_nodes': values.n_nodes,
            'n_active_nodes': int(np.count_nonzero(r.node_solution_status != 'inactive')),
            'n_inactive_nodes': int(np.count_nonzero(r.node_solution_status == 'inactive')),
            'node_pick_count_min': _stat(r.node_pick_count, 'min'),
            'node_pick_count_max': _stat(r.node_pick_count, 'max'),
            'node_pick_count_median': _stat(r.node_pick_count, 'median'),
            'low_fold_node_count': int(np.count_nonzero(r.node_solution_status == 'low_fold')),
        },
        'endpoints': {
            'n_source_endpoints': values.n_source_endpoints,
            'n_receiver_endpoints': values.n_receiver_endpoints,
            'source_datum_status_counts': _status_counts(r.source_datum_status),
            'receiver_datum_status_counts': _status_counts(r.receiver_datum_status),
        },
        'first_break_fit': {
            'residual_rms_ms': _residual_stat(residual_ms, 'rms'),
            'residual_mad_ms': _residual_stat(residual_ms, 'mad'),
            'residual_mean_ms': _residual_stat(residual_ms, 'mean'),
            'residual_median_ms': _residual_stat(residual_ms, 'median'),
            'residual_p95_abs_ms': _residual_stat(residual_ms, 'p95_abs'),
            'residual_max_abs_ms': _residual_stat(residual_ms, 'max_abs'),
            'robust_enabled': bool(req.solver.robust.enabled),
            'robust_method': req.solver.robust.method,
            'robust_iteration_count': int(
                r.qc.get(
                    'robust_iteration_count',
                    0 if not req.solver.robust.enabled else int(np.count_nonzero(r.rejected_by_robust_mask) > 0),
                )
            ),
        },
        'statics': {
            'weathering_replacement_shift_min_ms': _stat(
                r.weathering_replacement_trace_shift_s_sorted * 1000.0,
                'min',
            ),
            'weathering_replacement_shift_max_ms': _stat(
                r.weathering_replacement_trace_shift_s_sorted * 1000.0,
                'max',
            ),
            'floating_datum_shift_min_ms': _stat(
                r.floating_datum_elevation_shift_s_sorted * 1000.0,
                'min',
            ),
            'floating_datum_shift_max_ms': _stat(
                r.floating_datum_elevation_shift_s_sorted * 1000.0,
                'max',
            ),
            'flat_datum_shift_min_ms': _stat(
                r.flat_datum_shift_s_sorted * 1000.0,
                'min',
            ),
            'flat_datum_shift_max_ms': _stat(
                r.flat_datum_shift_s_sorted * 1000.0,
                'max',
            ),
            'refraction_trace_shift_min_ms': _stat(valid_refraction_ms, 'min'),
            'refraction_trace_shift_max_ms': _stat(valid_refraction_ms, 'max'),
            'refraction_trace_shift_median_ms': _stat(valid_refraction_ms, 'median'),
            'refraction_trace_shift_p95_abs_ms': _stat(np.abs(valid_refraction_ms), 'p95'),
            'refraction_trace_shift_max_abs_ms': _stat(np.abs(valid_refraction_ms), 'max'),
            'negative_refraction_shift_count': int(
                np.count_nonzero(valid_refraction_ms < 0.0)
            ),
            'positive_refraction_shift_count': int(
                np.count_nonzero(valid_refraction_ms > 0.0)
            ),
            'zero_refraction_shift_count': int(
                np.count_nonzero(valid_refraction_ms == 0.0)
            ),
            'invalid_refraction_shift_count': int(
                np.count_nonzero(~r.trace_static_valid_mask_sorted)
            ),
            'exceeds_max_abs_shift_count': int(
                np.count_nonzero(r.trace_static_status_sorted == 'exceeds_max_abs_shift')
            ),
        },
        'status_counts': {
            'trace_static_status': _status_counts(r.trace_static_status_sorted),
            'node_solution_status': _status_counts(r.node_solution_status),
            'node_weathering_status': _status_counts(r.node_weathering_status),
            'node_datum_status': _status_counts(r.node_datum_status),
        },
        'artifacts': _artifact_list_for_qc(artifact_entries),
        'warnings': [],
    }
    source_depth_qc = _source_depth_field_correction_qc(r, req)
    uphole_qc = _uphole_field_correction_qc(r, req)
    manual_static_qc = _manual_static_field_correction_qc(r, req)
    composition_qc = _field_correction_composition_qc(r, req)
    field_corrections_qc: dict[str, Any] = {}
    if source_depth_qc:
        field_corrections_qc['source_depth'] = source_depth_qc
        payload['source_depth_double_count_guard'] = source_depth_qc[
            'source_depth_double_count_guard'
        ]
        warnings = source_depth_qc.get('warnings')
        if isinstance(warnings, list):
            payload['warnings'].extend(str(item) for item in warnings)
    else:
        payload['source_depth_double_count_guard'] = 'not_applicable'
    if uphole_qc:
        field_corrections_qc['uphole'] = uphole_qc
    if manual_static_qc:
        field_corrections_qc['manual_static'] = manual_static_qc
    if composition_qc:
        field_corrections_qc['composition'] = composition_qc
    if field_corrections_qc:
        payload['field_corrections'] = field_corrections_qc
    static_history_qc = _static_history_qc_from_result(r, req)
    if static_history_qc.get('status') not in {'not_checked', 'checked'}:
        payload['static_history'] = static_history_qc
        warnings = static_history_qc.get('warnings')
        if isinstance(warnings, list):
            payload['warnings'].extend(str(item) for item in warnings)
    if layer_count is not None:
        payload['layer_count'] = int(layer_count)
    layer_velocity_modes = _layer_velocity_modes_for_request(req)
    if layer_velocity_modes:
        payload['velocity']['layer_velocity_modes'] = layer_velocity_modes
    enabled_layer_kinds = r.qc.get('enabled_layer_kinds')
    if isinstance(enabled_layer_kinds, (list, tuple)):
        payload['enabled_layer_kinds'] = [
            str(layer_kind) for layer_kind in enabled_layer_kinds
        ]
    observation_gates = r.qc.get('observation_gates')
    raw_layer_container = r.qc.get('layers')
    if (
        not isinstance(observation_gates, dict)
        and isinstance(raw_layer_container, dict)
        and isinstance(raw_layer_container.get('observation_gates'), dict)
    ):
        observation_gates = raw_layer_container.get('observation_gates')
    if isinstance(observation_gates, dict):
        payload['observation_gates'] = observation_gates
    cell_velocity_layer_kinds = _request_cell_velocity_layer_kinds(req)
    if cell_velocity_layer_kinds:
        refractor_cell = req.model.refractor_cell
        if refractor_cell is None:
            raise RefractionStaticArtifactError(
                'model.refractor_cell is required for solve_cell QC'
            )
        layer_kind = cell_velocity_layer_kinds[0]
        component = _cell_velocity_component(layer_kind)
        cell_artifact_names = _cell_velocity_artifact_names(layer_kind)
        coordinate_metadata = refraction_cell_coordinate_metadata_from_config(
            refractor_cell_options_from_request(refractor_cell)
        )
        cell_artifacts_by_layer = {}
        for item_layer_kind in cell_velocity_layer_kinds:
            item_component = _cell_velocity_component(item_layer_kind)
            item_names = _cell_velocity_artifact_names(item_layer_kind)
            cell_artifacts_by_layer[item_layer_kind] = {
                **coordinate_metadata,
                'cell_velocity_layer_kind': item_layer_kind,
                'cell_velocity_component': item_component,
                'cells_csv_artifact': item_names.cells_csv,
                'grid_npz_artifact': item_names.grid_npz,
                'qc_json_artifact': item_names.qc_json,
                'solver_history_csv_artifact': item_names.solver_history_csv,
            }
        payload['velocity']['cell_velocity_qc_artifact'] = (
            cell_artifact_names.qc_json
        )
        payload['velocity']['cell_velocity_layer_kind'] = layer_kind
        payload['velocity']['cell_velocity_component'] = component
        payload['velocity']['cell_velocity_layer_kinds'] = list(
            cell_velocity_layer_kinds
        )
        payload['velocity']['cell_velocity_qc_artifacts_by_layer'] = {
            item_layer_kind: item['qc_json_artifact']
            for item_layer_kind, item in cell_artifacts_by_layer.items()
        }
        payload['velocity']['grid_map_qc_artifacts'] = {
            'csv': REFRACTION_GRID_MAP_QC_CSV_NAME,
            'npz': REFRACTION_GRID_MAP_QC_NPZ_NAME,
            'json': REFRACTION_GRID_MAP_QC_JSON_NAME,
        }
        payload['refractor_velocity_cells'] = {
            **coordinate_metadata,
            'cell_velocity_layer_kind': layer_kind,
            'cell_velocity_component': component,
            'cells_csv_artifact': cell_artifact_names.cells_csv,
            'grid_npz_artifact': cell_artifact_names.grid_npz,
            'qc_json_artifact': cell_artifact_names.qc_json,
            'solver_history_csv_artifact': cell_artifact_names.solver_history_csv,
        }
        payload['refractor_velocity_cells_by_layer'] = cell_artifacts_by_layer
        payload['refractor_grid_map_qc'] = {
            'csv_artifact': REFRACTION_GRID_MAP_QC_CSV_NAME,
            'npz_artifact': REFRACTION_GRID_MAP_QC_NPZ_NAME,
            'json_artifact': REFRACTION_GRID_MAP_QC_JSON_NAME,
            'global_velocity_layer_behavior': 'omitted_from_grid_map_qc_rows',
        }
    layer_qc = _final_layer_qc_payload(r.qc.get('layers'))
    if layer_qc:
        payload['layers'] = layer_qc
    _assert_strict_json(payload, artifact_name=REFRACTION_STATIC_QC_JSON_NAME)
    return payload

def _final_layer_qc_payload(raw_layers: object) -> dict[str, Any]:
    if not isinstance(raw_layers, dict):
        return {}
    nested_layers = raw_layers.get('layers')
    if isinstance(nested_layers, dict):
        return dict(nested_layers)
    return dict(raw_layers)

def _sign_convention_qc_payload(
    req: RefractionStaticApplyRequest,
) -> str | dict[str, str]:
    if req.conversion.mode != 't1lsst_1layer':
        return SIGN_CONVENTION
    return {
        'trace_shift_s': SIGN_CONVENTION,
        'positive_shift': POSITIVE_SHIFT_DESCRIPTION,
        'negative_shift': NEGATIVE_SHIFT_DESCRIPTION,
    }

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
    'build_refraction_static_qc_payload',
    'refraction_static_double_application_qc',
    'refraction_static_trace_shift_component_names',
    'static_history_double_application_qc',
    'write_refraction_static_history_json',
    'write_refraction_static_qc_json',
]
