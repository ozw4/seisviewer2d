"""Final artifact package writer for GLI refraction statics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_datum import RefractionDatumStaticsResult

REFRACTION_STATIC_SOLUTION_NPZ_NAME = 'refraction_static_solution.npz'
REFRACTION_STATIC_QC_JSON_NAME = 'refraction_static_qc.json'
REFRACTION_STATICS_CSV_NAME = 'refraction_statics.csv'
NEAR_SURFACE_MODEL_CSV_NAME = 'near_surface_model.csv'
FIRST_BREAK_RESIDUALS_CSV_NAME = 'first_break_residuals.csv'
REFRACTION_STATIC_COMPONENTS_CSV_NAME = 'refraction_static_components.csv'
REFRACTION_STATIC_ARTIFACTS_JSON_NAME = 'refraction_static_artifacts.json'

ARTIFACT_VERSION = '1.0'
METHOD = 'gli_variable_thickness'
WORKFLOW = 'refraction_statics'
STATIC_COMPONENT = 'final_refraction'
SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'

_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        'kind': 'npz',
        'required': True,
        'description': 'Machine-readable final refraction statics solution',
    },
    {
        'name': REFRACTION_STATIC_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Human-readable final refraction statics QC summary',
    },
    {
        'name': REFRACTION_STATICS_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Trace-level final refraction statics table',
    },
    {
        'name': NEAR_SURFACE_MODEL_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Node-level near-surface model table',
    },
    {
        'name': FIRST_BREAK_RESIDUALS_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'GLI first-break residual table',
    },
    {
        'name': REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Source/receiver endpoint static component table',
    },
)

_TRACE_STATICS_COLUMNS = (
    'sorted_trace_index',
    'valid_observation',
    'used_observation',
    'trace_static_valid',
    'trace_static_status',
    'source_node_id',
    'receiver_node_id',
    'source_surface_elevation_m',
    'receiver_surface_elevation_m',
    'source_floating_datum_elevation_m',
    'receiver_floating_datum_elevation_m',
    'source_weathering_thickness_m',
    'receiver_weathering_thickness_m',
    'source_refractor_elevation_m',
    'receiver_refractor_elevation_m',
    'source_half_intercept_time_ms',
    'receiver_half_intercept_time_ms',
    'weathering_replacement_trace_shift_ms',
    'floating_datum_elevation_shift_ms',
    'flat_datum_shift_ms',
    'refraction_trace_shift_ms',
    'estimated_first_break_time_ms',
    'first_break_residual_ms',
    'source_weathering_replacement_shift_ms',
    'receiver_weathering_replacement_shift_ms',
    'source_floating_datum_elevation_shift_ms',
    'receiver_floating_datum_elevation_shift_ms',
    'source_flat_datum_shift_ms',
    'receiver_flat_datum_shift_ms',
    'source_refraction_shift_ms',
    'receiver_refraction_shift_ms',
)

_NEAR_SURFACE_COLUMNS = (
    'node_id',
    'node_kind',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'refractor_elevation_m',
    'weathering_thickness_m',
    'half_intercept_time_ms',
    'weathering_replacement_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'pick_count',
    'used_pick_count',
    'rejected_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)

_RESIDUAL_COLUMNS = (
    'row_index',
    'sorted_trace_index',
    'source_node_id',
    'receiver_node_id',
    'distance_m',
    'observed_pick_time_ms',
    'modeled_pick_time_ms',
    'residual_ms',
    'used',
    'rejected_by_robust',
)

_COMPONENT_COLUMNS = (
    'kind',
    'endpoint_key',
    'station_id',
    'node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'refractor_elevation_m',
    'weathering_thickness_m',
    'half_intercept_time_ms',
    'weathering_replacement_shift_ms',
    'floating_datum_elevation_shift_ms',
    'flat_datum_shift_ms',
    'refraction_shift_ms',
    'datum_status',
    'pick_count',
    'residual_rms_ms',
)


class RefractionStaticArtifactError(ValueError):
    """Raised when final refraction static artifacts cannot be written."""


@dataclass(frozen=True)
class RefractionStaticArtifactSet:
    job_dir: Path
    solution_npz: Path
    qc_json: Path
    refraction_statics_csv: Path
    near_surface_model_csv: Path
    first_break_residuals_csv: Path
    refraction_static_components_csv: Path
    manifest_json: Path | None
    artifact_names: tuple[str, ...]
    qc: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedResult:
    result: RefractionDatumStaticsResult
    n_traces: int
    n_nodes: int
    n_source_endpoints: int
    n_receiver_endpoints: int
    n_rows: int


def write_refraction_static_artifacts(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    job_dir: Path,
) -> RefractionStaticArtifactSet:
    """Write the final refraction statics NPZ, QC, CSV, and manifest artifacts."""
    root = _validate_job_dir(job_dir)
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    qc = build_refraction_static_qc_payload(result=values.result, req=request)
    manifest = _build_manifest_payload()

    paths = RefractionStaticArtifactSet(
        job_dir=root,
        solution_npz=root / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        qc_json=root / REFRACTION_STATIC_QC_JSON_NAME,
        refraction_statics_csv=root / REFRACTION_STATICS_CSV_NAME,
        near_surface_model_csv=root / NEAR_SURFACE_MODEL_CSV_NAME,
        first_break_residuals_csv=root / FIRST_BREAK_RESIDUALS_CSV_NAME,
        refraction_static_components_csv=root / REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        manifest_json=root / REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        artifact_names=tuple(item['name'] for item in _ARTIFACTS),
        qc=qc,
    )

    write_refraction_static_solution_npz(
        result=values.result,
        req=request,
        path=paths.solution_npz,
    )
    write_refraction_static_qc_json(
        result=values.result,
        req=request,
        path=paths.qc_json,
        qc=qc,
    )
    write_refraction_statics_csv(result=values.result, path=paths.refraction_statics_csv)
    write_near_surface_model_csv(result=values.result, path=paths.near_surface_model_csv)
    write_first_break_residuals_csv(
        result=values.result,
        path=paths.first_break_residuals_csv,
    )
    write_refraction_static_components_csv(
        result=values.result,
        path=paths.refraction_static_components_csv,
    )
    _write_json_atomic(paths.manifest_json, manifest)

    for artifact_path in (
        paths.solution_npz,
        paths.qc_json,
        paths.refraction_statics_csv,
        paths.near_surface_model_csv,
        paths.first_break_residuals_csv,
        paths.refraction_static_components_csv,
        paths.manifest_json,
    ):
        if not artifact_path.is_file():
            raise RefractionStaticArtifactError(
                f'artifact file missing after write: {artifact_path.name}'
            )
    return paths


def write_refraction_static_solution_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    """Write the compressed, pickle-free machine-readable solution artifact."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    payload = build_refraction_static_solution_arrays(
        result=values.result,
        req=request,
    )
    _write_npz_atomic(Path(path), payload)


def write_refraction_static_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    qc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write and return the strict-JSON QC summary artifact."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    payload = qc if qc is not None else build_refraction_static_qc_payload(
        result=values.result,
        req=request,
    )
    _write_json_atomic(Path(path), payload)
    return payload


def write_refraction_statics_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _trace_statics_rows(values.result)
    _write_csv_atomic(Path(path), _TRACE_STATICS_COLUMNS, rows)


def write_near_surface_model_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _near_surface_model_rows(values.result)
    _write_csv_atomic(Path(path), _NEAR_SURFACE_COLUMNS, rows)


def write_first_break_residuals_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _first_break_residual_rows(values.result)
    _write_csv_atomic(Path(path), _RESIDUAL_COLUMNS, rows)


def write_refraction_static_components_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _component_rows(values.result)
    _write_csv_atomic(Path(path), _COMPONENT_COLUMNS, rows)


def build_refraction_static_solution_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    values = _validate_result(result)
    r = values.result
    arrays: dict[str, np.ndarray] = {
        'artifact_version': _scalar_str(ARTIFACT_VERSION),
        'method': _scalar_str(METHOD),
        'bedrock_velocity_mode': _scalar_str(r.bedrock_velocity_mode),
        'datum_mode': _scalar_str(r.datum_mode),
        'floating_datum_mode': _scalar_str(r.floating_datum_mode),
        'sign_convention': _scalar_str(SIGN_CONVENTION),
        'n_traces': _scalar_int(values.n_traces),
        'n_nodes': _scalar_int(values.n_nodes),
        'n_source_endpoints': _scalar_int(values.n_source_endpoints),
        'n_receiver_endpoints': _scalar_int(values.n_receiver_endpoints),
        'n_valid_observations': _scalar_int(values.n_rows),
        'n_used_observations': _scalar_int(np.count_nonzero(r.used_row_mask)),
        'n_rejected_by_robust': _scalar_int(
            np.count_nonzero(r.rejected_by_robust_mask)
        ),
        'weathering_velocity_m_s': _scalar_float(r.weathering_velocity_m_s),
        'bedrock_velocity_m_s': _scalar_float(r.bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': _scalar_float(r.bedrock_slowness_s_per_m),
        'replacement_slowness_delta_s_per_m': _scalar_float(
            r.replacement_slowness_delta_s_per_m
        ),
        'flat_datum_elevation_m': _scalar_float(_nan_if_none(r.flat_datum_elevation_m)),
        'max_abs_shift_ms': _scalar_float(req.apply.max_abs_shift_ms),
        'sorted_trace_index': _int_array(r.sorted_trace_index),
        'valid_observation_mask_sorted': _bool_array(
            r.valid_observation_mask_sorted
        ),
        'used_observation_mask_sorted': _bool_array(r.used_observation_mask_sorted),
        'trace_static_valid_mask_sorted': _bool_array(
            r.trace_static_valid_mask_sorted
        ),
        'source_node_id_sorted': _int_array(r.source_node_id_sorted),
        'receiver_node_id_sorted': _int_array(r.receiver_node_id_sorted),
        'source_surface_elevation_m_sorted': _float_array(
            r.source_surface_elevation_m_sorted
        ),
        'receiver_surface_elevation_m_sorted': _float_array(
            r.receiver_surface_elevation_m_sorted
        ),
        'source_floating_datum_elevation_m_sorted': _float_array(
            r.source_floating_datum_elevation_m_sorted
        ),
        'receiver_floating_datum_elevation_m_sorted': _float_array(
            r.receiver_floating_datum_elevation_m_sorted
        ),
        'source_weathering_thickness_m_sorted': _float_array(
            r.source_weathering_thickness_m_sorted
        ),
        'receiver_weathering_thickness_m_sorted': _float_array(
            r.receiver_weathering_thickness_m_sorted
        ),
        'source_refractor_elevation_m_sorted': _float_array(
            r.source_refractor_elevation_m_sorted
        ),
        'receiver_refractor_elevation_m_sorted': _float_array(
            r.receiver_refractor_elevation_m_sorted
        ),
        'source_half_intercept_time_s_sorted': _float_array(
            r.source_half_intercept_time_s_sorted
        ),
        'receiver_half_intercept_time_s_sorted': _float_array(
            r.receiver_half_intercept_time_s_sorted
        ),
        'weathering_replacement_trace_shift_s_sorted': _float_array(
            r.weathering_replacement_trace_shift_s_sorted
        ),
        'floating_datum_elevation_shift_s_sorted': _float_array(
            r.floating_datum_elevation_shift_s_sorted
        ),
        'flat_datum_shift_s_sorted': _float_array(r.flat_datum_shift_s_sorted),
        'refraction_trace_shift_s_sorted': _float_array(
            r.refraction_trace_shift_s_sorted
        ),
        'estimated_first_break_time_s_sorted': _float_array(
            r.estimated_first_break_time_s_sorted
        ),
        'first_break_residual_s_sorted': _float_array(
            r.first_break_residual_s_sorted
        ),
        'trace_static_status_sorted': _string_array(r.trace_static_status_sorted),
        'source_weathering_replacement_shift_s_sorted': _float_array(
            r.source_weathering_replacement_shift_s_sorted
        ),
        'receiver_weathering_replacement_shift_s_sorted': _float_array(
            r.receiver_weathering_replacement_shift_s_sorted
        ),
        'source_floating_datum_elevation_shift_s_sorted': _float_array(
            r.source_floating_datum_elevation_shift_s_sorted
        ),
        'receiver_floating_datum_elevation_shift_s_sorted': _float_array(
            r.receiver_floating_datum_elevation_shift_s_sorted
        ),
        'source_flat_datum_shift_s_sorted': _float_array(
            r.source_flat_datum_shift_s_sorted
        ),
        'receiver_flat_datum_shift_s_sorted': _float_array(
            r.receiver_flat_datum_shift_s_sorted
        ),
        'source_refraction_shift_s_sorted': _float_array(
            r.source_refraction_shift_s_sorted
        ),
        'receiver_refraction_shift_s_sorted': _float_array(
            r.receiver_refraction_shift_s_sorted
        ),
        'node_id': _int_array(r.node_id),
        'node_x_m': _float_array(r.node_x_m),
        'node_y_m': _float_array(r.node_y_m),
        'node_surface_elevation_m': _float_array(r.node_surface_elevation_m),
        'node_floating_datum_elevation_m': _float_array(
            r.node_floating_datum_elevation_m
        ),
        'node_refractor_elevation_m': _float_array(r.node_refractor_elevation_m),
        'node_weathering_thickness_m': _float_array(
            r.node_weathering_thickness_m
        ),
        'node_half_intercept_time_s': _float_array(
            r.node_half_intercept_time_s
        ),
        'node_weathering_replacement_shift_s': _float_array(
            r.node_weathering_replacement_shift_s
        ),
        'node_solution_status': _string_array(r.node_solution_status),
        'node_weathering_status': _string_array(r.node_weathering_status),
        'node_datum_status': _string_array(r.node_datum_status),
        'node_pick_count': _int_array(r.node_pick_count),
        'node_used_pick_count': _int_array(r.node_used_pick_count),
        'node_rejected_pick_count': _int_array(r.node_rejected_pick_count),
        'node_residual_rms_s': _float_array(r.node_residual_rms_s),
        'node_residual_mad_s': _float_array(r.node_residual_mad_s),
        'source_endpoint_key': _string_array(r.source_endpoint_key),
        'source_id': _int_array(r.source_id),
        'source_node_id': _int_array(r.source_node_id),
        'source_x_m': _float_array(r.source_x_m),
        'source_y_m': _float_array(r.source_y_m),
        'source_surface_elevation_m': _float_array(r.source_surface_elevation_m),
        'source_floating_datum_elevation_m': _float_array(
            r.source_floating_datum_elevation_m
        ),
        'source_refractor_elevation_m': _float_array(
            r.source_refractor_elevation_m
        ),
        'source_weathering_thickness_m': _float_array(
            r.source_weathering_thickness_m
        ),
        'source_half_intercept_time_s': _float_array(
            r.source_half_intercept_time_s
        ),
        'source_weathering_replacement_shift_s': _float_array(
            r.source_weathering_replacement_shift_s
        ),
        'source_floating_datum_elevation_shift_s': _float_array(
            r.source_floating_datum_elevation_shift_s
        ),
        'source_flat_datum_shift_s': _float_array(r.source_flat_datum_shift_s),
        'source_refraction_shift_s': _float_array(r.source_refraction_shift_s),
        'source_datum_status': _string_array(r.source_datum_status),
        'receiver_endpoint_key': _string_array(r.receiver_endpoint_key),
        'receiver_id': _int_array(r.receiver_id),
        'receiver_node_id': _int_array(r.receiver_node_id),
        'receiver_x_m': _float_array(r.receiver_x_m),
        'receiver_y_m': _float_array(r.receiver_y_m),
        'receiver_surface_elevation_m': _float_array(
            r.receiver_surface_elevation_m
        ),
        'receiver_floating_datum_elevation_m': _float_array(
            r.receiver_floating_datum_elevation_m
        ),
        'receiver_refractor_elevation_m': _float_array(
            r.receiver_refractor_elevation_m
        ),
        'receiver_weathering_thickness_m': _float_array(
            r.receiver_weathering_thickness_m
        ),
        'receiver_half_intercept_time_s': _float_array(
            r.receiver_half_intercept_time_s
        ),
        'receiver_weathering_replacement_shift_s': _float_array(
            r.receiver_weathering_replacement_shift_s
        ),
        'receiver_floating_datum_elevation_shift_s': _float_array(
            r.receiver_floating_datum_elevation_shift_s
        ),
        'receiver_flat_datum_shift_s': _float_array(r.receiver_flat_datum_shift_s),
        'receiver_refraction_shift_s': _float_array(
            r.receiver_refraction_shift_s
        ),
        'receiver_datum_status': _string_array(r.receiver_datum_status),
        'row_trace_index_sorted': _int_array(r.row_trace_index_sorted),
        'row_source_node_id': _int_array(r.row_source_node_id),
        'row_receiver_node_id': _int_array(r.row_receiver_node_id),
        'row_distance_m': _float_array(r.row_distance_m),
        'observed_pick_time_s': _float_array(r.observed_pick_time_s),
        'modeled_pick_time_s': _float_array(r.modeled_pick_time_s),
        'residual_time_s': _float_array(r.residual_time_s),
        'used_row_mask': _bool_array(r.used_row_mask),
        'rejected_by_robust_mask': _bool_array(r.rejected_by_robust_mask),
    }
    _validate_no_object_arrays(arrays, artifact_name=REFRACTION_STATIC_SOLUTION_NPZ_NAME)
    return arrays


def build_refraction_static_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    values = _validate_result(result)
    r = values.result
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
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'method': METHOD,
        'workflow': WORKFLOW,
        'static_component': STATIC_COMPONENT,
        'sign_convention': SIGN_CONVENTION,
        'request': request,
        'velocity': {
            'weathering_velocity_m_s': _json_float(r.weathering_velocity_m_s),
            'bedrock_velocity_mode': r.bedrock_velocity_mode,
            'bedrock_velocity_m_s': _json_float(r.bedrock_velocity_m_s),
            'bedrock_slowness_s_per_m': _json_float(r.bedrock_slowness_s_per_m),
            'replacement_slowness_delta_s_per_m': _json_float(
                r.replacement_slowness_delta_s_per_m
            ),
            'bedrock_velocity_status': (
                'solved'
                if r.bedrock_velocity_mode == 'solve_global'
                else 'fixed'
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
        'artifacts': _artifact_list_for_qc(),
        'warnings': [],
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_STATIC_QC_JSON_NAME)
    return payload


def _validate_job_dir(job_dir: Path) -> Path:
    try:
        root = Path(job_dir)
    except TypeError as exc:
        raise RefractionStaticArtifactError('job_dir must be path-like') from exc
    if not root.exists():
        raise RefractionStaticArtifactError('missing job directory')
    if not root.is_dir():
        raise RefractionStaticArtifactError('job_dir is not a directory')
    if not os.access(root, os.W_OK):
        raise RefractionStaticArtifactError('job directory is not writable')
    return root


def _validate_result(result: RefractionDatumStaticsResult) -> _ValidatedResult:
    if not isinstance(result, RefractionDatumStaticsResult):
        raise RefractionStaticArtifactError(
            'result must be a RefractionDatumStaticsResult instance'
        )
    n_traces = _length(result.sorted_trace_index, name='sorted_trace_index')
    if n_traces <= 0:
        raise RefractionStaticArtifactError('sorted_trace_index must not be empty')
    for name in _TRACE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_traces:
            raise RefractionStaticArtifactError(
                f'trace-order array length mismatch for {name}'
            )
    n_nodes = _length(result.node_id, name='node_id')
    for name in _NODE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_nodes:
            raise RefractionStaticArtifactError(f'node array length mismatch for {name}')
    n_source = _length(result.source_endpoint_key, name='source_endpoint_key')
    for name in _SOURCE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_source:
            raise RefractionStaticArtifactError(
                f'source endpoint array length mismatch for {name}'
            )
    n_receiver = _length(result.receiver_endpoint_key, name='receiver_endpoint_key')
    for name in _RECEIVER_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_receiver:
            raise RefractionStaticArtifactError(
                f'receiver endpoint array length mismatch for {name}'
            )
    n_rows = _length(result.row_trace_index_sorted, name='row_trace_index_sorted')
    for name in _ROW_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_rows:
            raise RefractionStaticArtifactError(
                f'residual array length mismatch for {name}'
            )
    if np.any((result.row_trace_index_sorted < 0) | (result.row_trace_index_sorted >= n_traces)):
        raise RefractionStaticArtifactError(
            'row_trace_index_sorted contains out-of-range trace indices'
        )
    for name in (
        'weathering_velocity_m_s',
        'bedrock_velocity_m_s',
        'bedrock_slowness_s_per_m',
        'replacement_slowness_delta_s_per_m',
    ):
        if not np.isfinite(float(getattr(result, name))):
            raise RefractionStaticArtifactError(f'non-finite required scalar {name}')
    for name in _STATUS_ARRAY_NAMES:
        _validate_status_array(getattr(result, name), name=name)
    return _ValidatedResult(
        result=result,
        n_traces=n_traces,
        n_nodes=n_nodes,
        n_source_endpoints=n_source,
        n_receiver_endpoints=n_receiver,
        n_rows=n_rows,
    )


_TRACE_ARRAY_NAMES = (
    'valid_observation_mask_sorted',
    'used_observation_mask_sorted',
    'trace_static_valid_mask_sorted',
    'source_node_id_sorted',
    'receiver_node_id_sorted',
    'source_surface_elevation_m_sorted',
    'receiver_surface_elevation_m_sorted',
    'source_floating_datum_elevation_m_sorted',
    'receiver_floating_datum_elevation_m_sorted',
    'source_weathering_thickness_m_sorted',
    'receiver_weathering_thickness_m_sorted',
    'source_refractor_elevation_m_sorted',
    'receiver_refractor_elevation_m_sorted',
    'source_half_intercept_time_s_sorted',
    'receiver_half_intercept_time_s_sorted',
    'source_weathering_replacement_shift_s_sorted',
    'receiver_weathering_replacement_shift_s_sorted',
    'source_floating_datum_elevation_shift_s_sorted',
    'receiver_floating_datum_elevation_shift_s_sorted',
    'source_flat_datum_shift_s_sorted',
    'receiver_flat_datum_shift_s_sorted',
    'source_refraction_shift_s_sorted',
    'receiver_refraction_shift_s_sorted',
    'weathering_replacement_trace_shift_s_sorted',
    'floating_datum_elevation_shift_s_sorted',
    'flat_datum_shift_s_sorted',
    'refraction_trace_shift_s_sorted',
    'trace_static_status_sorted',
    'estimated_first_break_time_s_sorted',
    'first_break_residual_s_sorted',
)

_NODE_ARRAY_NAMES = (
    'node_x_m',
    'node_y_m',
    'node_surface_elevation_m',
    'node_kind',
    'node_weathering_thickness_m',
    'node_refractor_elevation_m',
    'node_half_intercept_time_s',
    'node_weathering_replacement_shift_s',
    'node_floating_datum_elevation_m',
    'node_solution_status',
    'node_datum_status',
    'node_weathering_status',
    'node_pick_count',
    'node_used_pick_count',
    'node_rejected_pick_count',
    'node_residual_rms_s',
    'node_residual_mad_s',
)

_SOURCE_ARRAY_NAMES = (
    'source_id',
    'source_node_id',
    'source_x_m',
    'source_y_m',
    'source_surface_elevation_m',
    'source_half_intercept_time_s',
    'source_weathering_thickness_m',
    'source_refractor_elevation_m',
    'source_floating_datum_elevation_m',
    'source_weathering_replacement_shift_s',
    'source_floating_datum_elevation_shift_s',
    'source_flat_datum_shift_s',
    'source_refraction_shift_s',
    'source_datum_status',
)

_RECEIVER_ARRAY_NAMES = (
    'receiver_id',
    'receiver_node_id',
    'receiver_x_m',
    'receiver_y_m',
    'receiver_surface_elevation_m',
    'receiver_half_intercept_time_s',
    'receiver_weathering_thickness_m',
    'receiver_refractor_elevation_m',
    'receiver_floating_datum_elevation_m',
    'receiver_weathering_replacement_shift_s',
    'receiver_floating_datum_elevation_shift_s',
    'receiver_flat_datum_shift_s',
    'receiver_refraction_shift_s',
    'receiver_datum_status',
)

_ROW_ARRAY_NAMES = (
    'row_source_node_id',
    'row_receiver_node_id',
    'row_distance_m',
    'observed_pick_time_s',
    'modeled_pick_time_s',
    'residual_time_s',
    'used_row_mask',
    'rejected_by_robust_mask',
)

_STATUS_ARRAY_NAMES = (
    'trace_static_status_sorted',
    'node_solution_status',
    'node_weathering_status',
    'node_datum_status',
    'source_datum_status',
    'receiver_datum_status',
)

_KNOWN_STATUSES = {
    'ok',
    'not_observed',
    'inactive',
    'low_fold',
    'clipped_lower',
    'clipped_upper',
    'clipped_half_intercept_lower',
    'clipped_half_intercept_upper',
    'exceeds_max_abs_shift',
    'exceeds_max_thickness',
    'invalid_shift',
    'invalid_solution',
    'missing_solution',
    'invalid_velocity',
    'invalid_bedrock_velocity',
    'invalid_surface_elevation',
    'invalid_floating_datum_elevation',
    'invalid_flat_datum_elevation',
    'invalid_weathering_replacement',
    'invalid_weathering_thickness',
    'negative_weathering_thickness',
    'negative_thickness',
    'invalid_half_intercept',
    'invalid_refractor_elevation',
    'invalid_datum_shift',
    'floating_datum_below_refractor',
    'flat_datum_below_refractor',
    'missing_endpoint',
    'missing_node',
}


def _trace_statics_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(int(result.sorted_trace_index.shape[0])):
        rows.append(
            {
                'sorted_trace_index': int(result.sorted_trace_index[index]),
                'valid_observation': _csv_bool(result.valid_observation_mask_sorted[index]),
                'used_observation': _csv_bool(result.used_observation_mask_sorted[index]),
                'trace_static_valid': _csv_bool(result.trace_static_valid_mask_sorted[index]),
                'trace_static_status': str(result.trace_static_status_sorted[index]),
                'source_node_id': int(result.source_node_id_sorted[index]),
                'receiver_node_id': int(result.receiver_node_id_sorted[index]),
                'source_surface_elevation_m': _csv_float(result.source_surface_elevation_m_sorted[index]),
                'receiver_surface_elevation_m': _csv_float(result.receiver_surface_elevation_m_sorted[index]),
                'source_floating_datum_elevation_m': _csv_float(result.source_floating_datum_elevation_m_sorted[index]),
                'receiver_floating_datum_elevation_m': _csv_float(result.receiver_floating_datum_elevation_m_sorted[index]),
                'source_weathering_thickness_m': _csv_float(result.source_weathering_thickness_m_sorted[index]),
                'receiver_weathering_thickness_m': _csv_float(result.receiver_weathering_thickness_m_sorted[index]),
                'source_refractor_elevation_m': _csv_float(result.source_refractor_elevation_m_sorted[index]),
                'receiver_refractor_elevation_m': _csv_float(result.receiver_refractor_elevation_m_sorted[index]),
                'source_half_intercept_time_ms': _csv_ms(result.source_half_intercept_time_s_sorted[index]),
                'receiver_half_intercept_time_ms': _csv_ms(result.receiver_half_intercept_time_s_sorted[index]),
                'weathering_replacement_trace_shift_ms': _csv_ms(result.weathering_replacement_trace_shift_s_sorted[index]),
                'floating_datum_elevation_shift_ms': _csv_ms(result.floating_datum_elevation_shift_s_sorted[index]),
                'flat_datum_shift_ms': _csv_ms(result.flat_datum_shift_s_sorted[index]),
                'refraction_trace_shift_ms': _csv_ms(result.refraction_trace_shift_s_sorted[index]),
                'estimated_first_break_time_ms': _csv_ms(result.estimated_first_break_time_s_sorted[index]),
                'first_break_residual_ms': _csv_ms(result.first_break_residual_s_sorted[index]),
                'source_weathering_replacement_shift_ms': _csv_ms(result.source_weathering_replacement_shift_s_sorted[index]),
                'receiver_weathering_replacement_shift_ms': _csv_ms(result.receiver_weathering_replacement_shift_s_sorted[index]),
                'source_floating_datum_elevation_shift_ms': _csv_ms(result.source_floating_datum_elevation_shift_s_sorted[index]),
                'receiver_floating_datum_elevation_shift_ms': _csv_ms(result.receiver_floating_datum_elevation_shift_s_sorted[index]),
                'source_flat_datum_shift_ms': _csv_ms(result.source_flat_datum_shift_s_sorted[index]),
                'receiver_flat_datum_shift_ms': _csv_ms(result.receiver_flat_datum_shift_s_sorted[index]),
                'source_refraction_shift_ms': _csv_ms(result.source_refraction_shift_s_sorted[index]),
                'receiver_refraction_shift_ms': _csv_ms(result.receiver_refraction_shift_s_sorted[index]),
            }
        )
    return rows


def _near_surface_model_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(int(result.node_id.shape[0])):
        rows.append(
            {
                'node_id': int(result.node_id[index]),
                'node_kind': str(result.node_kind[index]),
                'x_m': _csv_float(result.node_x_m[index]),
                'y_m': _csv_float(result.node_y_m[index]),
                'surface_elevation_m': _csv_float(result.node_surface_elevation_m[index]),
                'floating_datum_elevation_m': _csv_float(result.node_floating_datum_elevation_m[index]),
                'refractor_elevation_m': _csv_float(result.node_refractor_elevation_m[index]),
                'weathering_thickness_m': _csv_float(result.node_weathering_thickness_m[index]),
                'half_intercept_time_ms': _csv_ms(result.node_half_intercept_time_s[index]),
                'weathering_replacement_shift_ms': _csv_ms(result.node_weathering_replacement_shift_s[index]),
                'solution_status': str(result.node_solution_status[index]),
                'weathering_status': str(result.node_weathering_status[index]),
                'datum_status': str(result.node_datum_status[index]),
                'pick_count': int(result.node_pick_count[index]),
                'used_pick_count': int(result.node_used_pick_count[index]),
                'rejected_pick_count': int(result.node_rejected_pick_count[index]),
                'residual_rms_ms': _csv_ms(result.node_residual_rms_s[index]),
                'residual_mad_ms': _csv_ms(result.node_residual_mad_s[index]),
            }
        )
    return rows


def _first_break_residual_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index in range(int(result.row_trace_index_sorted.shape[0])):
        rows.append(
            {
                'row_index': row_index,
                'sorted_trace_index': int(result.row_trace_index_sorted[row_index]),
                'source_node_id': int(result.row_source_node_id[row_index]),
                'receiver_node_id': int(result.row_receiver_node_id[row_index]),
                'distance_m': _csv_float(result.row_distance_m[row_index]),
                'observed_pick_time_ms': _csv_ms(result.observed_pick_time_s[row_index]),
                'modeled_pick_time_ms': _csv_ms(result.modeled_pick_time_s[row_index]),
                'residual_ms': _csv_ms(result.residual_time_s[row_index]),
                'used': _csv_bool(result.used_row_mask[row_index]),
                'rejected_by_robust': _csv_bool(result.rejected_by_robust_mask[row_index]),
            }
        )
    return rows


def _component_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    node_pick_count = _node_lookup(result.node_id, result.node_pick_count)
    node_residual_rms = _node_lookup(result.node_id, result.node_residual_rms_s)
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        node_id = int(result.source_node_id[index])
        rows.append(
            {
                'kind': 'source',
                'endpoint_key': str(result.source_endpoint_key[index]),
                'station_id': int(result.source_id[index]),
                'node_id': node_id,
                'x_m': _csv_float(result.source_x_m[index]),
                'y_m': _csv_float(result.source_y_m[index]),
                'surface_elevation_m': _csv_float(result.source_surface_elevation_m[index]),
                'floating_datum_elevation_m': _csv_float(result.source_floating_datum_elevation_m[index]),
                'refractor_elevation_m': _csv_float(result.source_refractor_elevation_m[index]),
                'weathering_thickness_m': _csv_float(result.source_weathering_thickness_m[index]),
                'half_intercept_time_ms': _csv_ms(result.source_half_intercept_time_s[index]),
                'weathering_replacement_shift_ms': _csv_ms(result.source_weathering_replacement_shift_s[index]),
                'floating_datum_elevation_shift_ms': _csv_ms(result.source_floating_datum_elevation_shift_s[index]),
                'flat_datum_shift_ms': _csv_ms(result.source_flat_datum_shift_s[index]),
                'refraction_shift_ms': _csv_ms(result.source_refraction_shift_s[index]),
                'datum_status': str(result.source_datum_status[index]),
                'pick_count': _csv_int(node_pick_count.get(node_id)),
                'residual_rms_ms': _csv_ms(node_residual_rms.get(node_id)),
            }
        )
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        node_id = int(result.receiver_node_id[index])
        rows.append(
            {
                'kind': 'receiver',
                'endpoint_key': str(result.receiver_endpoint_key[index]),
                'station_id': int(result.receiver_id[index]),
                'node_id': node_id,
                'x_m': _csv_float(result.receiver_x_m[index]),
                'y_m': _csv_float(result.receiver_y_m[index]),
                'surface_elevation_m': _csv_float(result.receiver_surface_elevation_m[index]),
                'floating_datum_elevation_m': _csv_float(result.receiver_floating_datum_elevation_m[index]),
                'refractor_elevation_m': _csv_float(result.receiver_refractor_elevation_m[index]),
                'weathering_thickness_m': _csv_float(result.receiver_weathering_thickness_m[index]),
                'half_intercept_time_ms': _csv_ms(result.receiver_half_intercept_time_s[index]),
                'weathering_replacement_shift_ms': _csv_ms(result.receiver_weathering_replacement_shift_s[index]),
                'floating_datum_elevation_shift_ms': _csv_ms(result.receiver_floating_datum_elevation_shift_s[index]),
                'flat_datum_shift_ms': _csv_ms(result.receiver_flat_datum_shift_s[index]),
                'refraction_shift_ms': _csv_ms(result.receiver_refraction_shift_s[index]),
                'datum_status': str(result.receiver_datum_status[index]),
                'pick_count': _csv_int(node_pick_count.get(node_id)),
                'residual_rms_ms': _csv_ms(node_residual_rms.get(node_id)),
            }
        )
    return rows


def _node_lookup(node_id: np.ndarray, values: np.ndarray) -> dict[int, Any]:
    return {
        int(raw_node): values[index]
        for index, raw_node in enumerate(np.asarray(node_id).tolist())
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


def _build_manifest_payload() -> dict[str, Any]:
    return {
        'artifact_version': ARTIFACT_VERSION,
        'job_kind': 'statics',
        'statics_kind': 'refraction',
        'artifacts': [
            {
                'name': str(item['name']),
                'kind': str(item['kind']),
                'required': bool(item['required']),
            }
            for item in _ARTIFACTS
        ],
    }


def _artifact_list_for_qc() -> list[dict[str, str]]:
    return [
        {
            'name': str(item['name']),
            'kind': str(item['kind']),
            'description': str(item['description']),
        }
        for item in _ARTIFACTS
    ]


def _length(value: object, *, name: str) -> int:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise RefractionStaticArtifactError(f'{name} must be one-dimensional')
    return int(arr.shape[0])


def _validate_status_array(value: object, *, name: str) -> None:
    unknown = sorted(
        {
            str(item)
            for item in np.asarray(value).tolist()
            if str(item) not in _KNOWN_STATUSES
        }
    )
    if unknown:
        raise RefractionStaticArtifactError(
            f'unknown status array values in {name}: {unknown}'
        )


def _scalar_str(value: object) -> np.ndarray:
    text = '' if value is None else str(value)
    return np.asarray(text, dtype=f'<U{max(1, len(text))}')


def _scalar_int(value: object) -> np.ndarray:
    return np.asarray(int(value), dtype=np.int64)


def _scalar_float(value: object) -> np.ndarray:
    return np.asarray(float(value), dtype=np.float64)


def _int_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def _float_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def _bool_array(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=bool)


def _string_array(value: object) -> np.ndarray:
    raw = [str(item) for item in np.asarray(value).tolist()]
    max_len = max([1, *(len(item) for item in raw)])
    return np.ascontiguousarray(raw, dtype=f'<U{max_len}')


def _validate_no_object_arrays(
    arrays: dict[str, np.ndarray],
    *,
    artifact_name: str,
) -> None:
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise RefractionStaticArtifactError(
                f'{artifact_name}: object array is not allowed for {key}'
            )


def _nan_if_none(value: float | None) -> float:
    return float('nan') if value is None else float(value)


def _json_float(value: object) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _csv_float(value: object) -> str | float:
    if value is None:
        return ''
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ''
    return out if np.isfinite(out) else ''


def _csv_ms(value_s: object) -> str | float:
    out = _csv_float(value_s)
    return '' if out == '' else float(out) * 1000.0


def _csv_bool(value: object) -> str:
    return 'true' if bool(value) else 'false'


def _csv_int(value: object) -> str | int:
    if value is None:
        return ''
    try:
        return int(value)
    except (TypeError, ValueError):
        return ''


def _stat(values: object, stat: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if stat == 'min':
        return float(np.min(arr))
    if stat == 'max':
        return float(np.max(arr))
    if stat == 'median':
        return float(np.median(arr))
    if stat == 'p95':
        return float(np.percentile(arr, 95.0))
    raise RefractionStaticArtifactError(f'unsupported statistic: {stat}')


def _residual_stat(values_ms: np.ndarray, stat: str) -> float | None:
    arr = np.asarray(values_ms, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if stat == 'rms':
        return float(np.sqrt(np.mean(arr * arr)))
    if stat == 'mad':
        median = float(np.median(arr))
        return float(np.median(np.abs(arr - median)))
    if stat == 'mean':
        return float(np.mean(arr))
    if stat == 'median':
        return float(np.median(arr))
    if stat == 'p95_abs':
        return float(np.percentile(np.abs(arr), 95.0))
    if stat == 'max_abs':
        return float(np.max(np.abs(arr)))
    raise RefractionStaticArtifactError(f'unsupported residual statistic: {stat}')


def _fraction(numerator: int | np.integer, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _status_counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in np.asarray(values).tolist():
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    _validate_no_object_arrays(payload, artifact_name=path.name)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('wb') as handle:
            np.savez_compressed(handle, **payload)
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RefractionStaticArtifactError(
            f'{path.name}: failed to write NPZ artifact'
        ) from exc


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _assert_strict_json(payload, artifact_name=path.name)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8') as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            handle.write('\n')
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RefractionStaticArtifactError(
            f'{path.name}: failed to write JSON artifact'
        ) from exc


def _write_csv_atomic(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='raise')
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RefractionStaticArtifactError(
            f'{path.name}: failed to write CSV artifact'
        ) from exc


def _assert_strict_json(payload: dict[str, Any], *, artifact_name: str) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'{artifact_name}: payload is not strict JSON serializable'
        ) from exc


__all__ = [
    'FIRST_BREAK_RESIDUALS_CSV_NAME',
    'NEAR_SURFACE_MODEL_CSV_NAME',
    'REFRACTION_STATICS_CSV_NAME',
    'REFRACTION_STATIC_ARTIFACTS_JSON_NAME',
    'REFRACTION_STATIC_COMPONENTS_CSV_NAME',
    'REFRACTION_STATIC_QC_JSON_NAME',
    'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
    'RefractionStaticArtifactError',
    'RefractionStaticArtifactSet',
    'build_refraction_static_qc_payload',
    'build_refraction_static_solution_arrays',
    'write_first_break_residuals_csv',
    'write_near_surface_model_csv',
    'write_refraction_static_artifacts',
    'write_refraction_static_components_csv',
    'write_refraction_static_qc_json',
    'write_refraction_static_solution_npz',
    'write_refraction_statics_csv',
]
