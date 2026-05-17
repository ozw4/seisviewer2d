"""Main refraction static QC JSON payload builder."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_cell_coordinates import (
    refraction_cell_coordinate_metadata_from_config,
)
from app.services.refraction_static_layer_config import (
    normalize_refraction_static_layers,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionLayerKind,
    ResolvedRefractionFirstLayer,
)
from app.services.refraction_static_artifacts.contract import (
    ARTIFACT_VERSION,
    NEGATIVE_SHIFT_DESCRIPTION,
    POSITIVE_SHIFT_DESCRIPTION,
    REFRACTION_GRID_MAP_QC_CSV_NAME,
    REFRACTION_GRID_MAP_QC_JSON_NAME,
    REFRACTION_GRID_MAP_QC_NPZ_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    SIGN_CONVENTION,
    STATIC_COMPONENT,
    WORKFLOW,
    RefractionStaticArtifactError,
)
from app.services.refraction_static_artifacts.field_corrections import (
    _field_correction_composition_qc,
    _manual_static_field_correction_qc,
    _source_depth_field_correction_qc,
    _uphole_field_correction_qc,
)
from app.services.refraction_static_artifacts.formatters import _json_float
from app.services.refraction_static_artifacts.history import (
    _static_history_qc_from_result,
)
from app.services.refraction_static_artifacts.io import (
    _assert_strict_json,
    _write_json_atomic,
)
from app.services.refraction_static_artifacts.registry import (
    _CELL_VELOCITY_COMPONENT_BY_LAYER,
    _artifact_entries_for_request,
    _artifact_list_for_qc,
    _cell_velocity_artifact_names,
    _request_cell_velocity_layer_kinds,
    _validate_upstream_artifact_names,
)
from app.services.refraction_static_artifacts.stats import (
    _fraction,
    _residual_stat,
    _stat,
    _status_counts,
)
from app.services.refraction_static_artifacts.validation import (
    _validate_resolved_first_layer,
    _validate_result,
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


def build_refraction_static_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    upstream_artifact_names: Iterable[str] = (),
    artifact_entries: Iterable[dict[str, object]] | None = None,
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
    if artifact_entries is None:
        resolved_artifact_entries = _artifact_entries_for_request(
            req,
            first_layer,
            upstream_artifact_names=upstream_names,
        )
    else:
        resolved_artifact_entries = tuple(artifact_entries)
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
        'artifacts': _artifact_list_for_qc(resolved_artifact_entries),
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
            refractor_cell
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


def _request_summary(req: RefractionStaticApplyRequest) -> dict[str, Any]:
    return {
        'file_id': req.file_id,
        'key1_byte': int(req.key1_byte),
        'key2_byte': int(req.key2_byte),
        'pick_source_kind': req.pick_source.kind,
        'model_method': req.model.method,
        'apply_mode': req.apply.mode,
        'register_corrected_file': bool(req.apply.register_corrected_file),
    }


def _layer_velocity_modes_for_request(
    req: RefractionStaticApplyRequest,
) -> dict[str, str]:
    if req.model.method != 'multilayer_time_term':
        return {}
    return {
        str(config.kind): str(config.velocity_mode)
        for config in normalize_refraction_static_layers(req.model)
    }


def _cell_velocity_component(layer_kind: RefractionLayerKind) -> str:
    try:
        return _CELL_VELOCITY_COMPONENT_BY_LAYER[layer_kind]
    except KeyError as exc:
        raise RefractionStaticArtifactError(
            f'unsupported cell velocity layer kind: {layer_kind}'
        ) from exc


__all__ = [
    'build_refraction_static_qc_payload',
    'write_refraction_static_qc_json',
]
