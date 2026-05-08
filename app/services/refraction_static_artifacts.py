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
from app.services.refraction_static_status import REFRACTION_STATIC_STATUSES
from app.services.refraction_static_t1lsst import (
    REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
    write_refraction_t1lsst_1layer_components_csv,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionStaticArtifactSet,
    ResolvedRefractionFirstLayer,
)
from app.services.refraction_static_v1 import (
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
)

REFRACTION_STATIC_SOLUTION_NPZ_NAME = 'refraction_static_solution.npz'
REFRACTION_STATIC_QC_JSON_NAME = 'refraction_static_qc.json'
REFRACTION_STATICS_CSV_NAME = 'refraction_statics.csv'
NEAR_SURFACE_MODEL_CSV_NAME = 'near_surface_model.csv'
FIRST_BREAK_RESIDUALS_CSV_NAME = 'first_break_residuals.csv'
REFRACTION_STATIC_COMPONENTS_CSV_NAME = 'refraction_static_components.csv'
SOURCE_STATIC_TABLE_CSV_NAME = 'source_static_table.csv'
RECEIVER_STATIC_TABLE_CSV_NAME = 'receiver_static_table.csv'
SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME = 'source_receiver_static_table.npz'
REFRACTION_STATIC_ARTIFACTS_JSON_NAME = 'refraction_static_artifacts.json'

ARTIFACT_VERSION = '1.0'
METHOD = 'gli_variable_thickness'
WORKFLOW = 'refraction_statics'
STATIC_COMPONENT = 'final_refraction'
SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'
POSITIVE_SHIFT_DESCRIPTION = 'event appears later in corrected data'
NEGATIVE_SHIFT_DESCRIPTION = 'event appears earlier in corrected data'

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
    {
        'name': SOURCE_STATIC_TABLE_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'IRAS-style source endpoint final static table',
    },
    {
        'name': RECEIVER_STATIC_TABLE_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'IRAS-style receiver endpoint final static table',
    },
    {
        'name': SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        'kind': 'npz',
        'required': True,
        'description': 'Machine-readable source/receiver endpoint static tables',
    },
)

_V1_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_V1_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Direct-arrival V1 estimation QC summary',
    },
    {
        'name': REFRACTION_V1_ESTIMATES_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Per-source direct-arrival V1 estimates',
    },
)

_OPTIONAL_CONSTANT_V1_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_V1_QC_JSON_NAME,
        'kind': 'json',
        'required': False,
        'description': (
            'Optional standalone V1 QC summary; constant V1 details are recorded '
            f'in {REFRACTION_STATIC_QC_JSON_NAME}'
        ),
    },
    {
        'name': REFRACTION_V1_ESTIMATES_CSV_NAME,
        'kind': 'csv',
        'required': False,
        'description': (
            'Optional per-source direct-arrival V1 estimates; not produced for '
            'constant first-layer mode'
        ),
    },
)

_T1LSST_1LAYER_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'T1LSST-compatible one-layer source/receiver components',
    },
)

REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES = frozenset(
    str(item['name'])
    for item in _ARTIFACTS + _V1_ARTIFACTS + _T1LSST_1LAYER_ARTIFACTS
) | {REFRACTION_STATIC_ARTIFACTS_JSON_NAME}

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

_SOURCE_STATIC_TABLE_COLUMNS = (
    'endpoint_kind',
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'sh1_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)

_RECEIVER_STATIC_TABLE_COLUMNS = (
    'endpoint_kind',
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'sh1_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'pick_count',
    'used_pick_count',
    'residual_rms_ms',
    'residual_mad_ms',
)


class RefractionStaticArtifactError(ValueError):
    """Raised when final refraction static artifacts cannot be written."""


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
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> RefractionStaticArtifactSet:
    """Write the final refraction statics NPZ, QC, CSV, and manifest artifacts."""
    root = _validate_job_dir(job_dir)
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    first_layer = _validate_resolved_first_layer(
        result=values.result,
        req=request,
        resolved_first_layer=resolved_first_layer,
    )
    artifact_entries = _artifact_entries_for_request(request, first_layer)
    qc = build_refraction_static_qc_payload(
        result=values.result,
        req=request,
        resolved_first_layer=first_layer,
    )
    manifest = _build_manifest_payload(artifact_entries)
    _assert_strict_json(manifest, artifact_name=REFRACTION_STATIC_ARTIFACTS_JSON_NAME)
    t1lsst_components_path = (
        root / REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME
        if request.conversion.mode == 't1lsst_1layer'
        else None
    )

    paths = RefractionStaticArtifactSet(
        job_dir=root,
        solution_npz=root / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        qc_json=root / REFRACTION_STATIC_QC_JSON_NAME,
        refraction_statics_csv=root / REFRACTION_STATICS_CSV_NAME,
        near_surface_model_csv=root / NEAR_SURFACE_MODEL_CSV_NAME,
        first_break_residuals_csv=root / FIRST_BREAK_RESIDUALS_CSV_NAME,
        refraction_static_components_csv=root / REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        source_static_table_csv=root / SOURCE_STATIC_TABLE_CSV_NAME,
        receiver_static_table_csv=root / RECEIVER_STATIC_TABLE_CSV_NAME,
        source_receiver_static_table_npz=root / SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        manifest_json=root / REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        artifact_names=tuple(str(item['name']) for item in artifact_entries),
        qc=qc,
        refraction_t1lsst_1layer_components_csv=t1lsst_components_path,
    )

    write_refraction_static_solution_npz(
        result=values.result,
        req=request,
        path=paths.solution_npz,
        resolved_first_layer=first_layer,
    )
    write_refraction_static_qc_json(
        result=values.result,
        req=request,
        path=paths.qc_json,
        qc=qc,
        resolved_first_layer=first_layer,
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
    write_source_static_table_csv(
        result=values.result,
        path=paths.source_static_table_csv,
    )
    write_receiver_static_table_csv(
        result=values.result,
        path=paths.receiver_static_table_csv,
    )
    write_source_receiver_static_table_npz(
        result=values.result,
        path=paths.source_receiver_static_table_npz,
    )
    if paths.refraction_t1lsst_1layer_components_csv is not None:
        write_refraction_t1lsst_1layer_components_csv(
            result=values.result,
            path=paths.refraction_t1lsst_1layer_components_csv,
        )
    _write_json_atomic(paths.manifest_json, manifest)

    artifact_paths = (
        paths.solution_npz,
        paths.qc_json,
        paths.refraction_statics_csv,
        paths.near_surface_model_csv,
        paths.first_break_residuals_csv,
        paths.refraction_static_components_csv,
        paths.source_static_table_csv,
        paths.receiver_static_table_csv,
        paths.source_receiver_static_table_npz,
        paths.manifest_json,
    )
    if paths.refraction_t1lsst_1layer_components_csv is not None:
        artifact_paths = artifact_paths + (
            paths.refraction_t1lsst_1layer_components_csv,
        )
    artifact_paths = artifact_paths + tuple(
        root / str(item['name']) for item in _v1_artifact_entries(request, first_layer)
        if bool(item['required'])
    )
    for artifact_path in artifact_paths:
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
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> None:
    """Write the compressed, pickle-free machine-readable solution artifact."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    payload = build_refraction_static_solution_arrays(
        result=values.result,
        req=request,
        resolved_first_layer=resolved_first_layer,
    )
    _write_npz_atomic(Path(path), payload)


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


def write_source_static_table_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _source_static_table_rows(values.result)
    _write_csv_atomic(Path(path), _SOURCE_STATIC_TABLE_COLUMNS, rows)


def write_receiver_static_table_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _receiver_static_table_rows(values.result)
    _write_csv_atomic(Path(path), _RECEIVER_STATIC_TABLE_COLUMNS, rows)


def write_source_receiver_static_table_npz(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    payload = build_source_receiver_static_table_arrays(result=values.result)
    _write_npz_atomic(Path(path), payload)


def build_source_receiver_static_table_arrays(
    *,
    result: RefractionDatumStaticsResult,
) -> dict[str, np.ndarray]:
    values = _validate_result(result)
    r = values.result
    source_t1_s = _endpoint_node_values(
        r.source_node_id,
        r.node_id,
        r.node_half_intercept_time_s,
    )
    source_sh1_m = _endpoint_node_values(
        r.source_node_id,
        r.node_id,
        r.node_weathering_thickness_m,
    )
    source_weathering_correction_s = _endpoint_node_values(
        r.source_node_id,
        r.node_id,
        r.node_weathering_replacement_shift_s,
    )
    receiver_t1_s = _endpoint_node_values(
        r.receiver_node_id,
        r.node_id,
        r.node_half_intercept_time_s,
    )
    receiver_sh1_m = _endpoint_node_values(
        r.receiver_node_id,
        r.node_id,
        r.node_weathering_thickness_m,
    )
    receiver_weathering_correction_s = _endpoint_node_values(
        r.receiver_node_id,
        r.node_id,
        r.node_weathering_replacement_shift_s,
    )
    source_static_status = _source_static_status_array(r)
    receiver_static_status = _receiver_static_status_array(r)
    arrays: dict[str, np.ndarray] = {
        'source_endpoint_key': _string_array(r.source_endpoint_key),
        'source_id': _int_array(r.source_id),
        'source_node_id': _int_array(r.source_node_id),
        'source_x_m': _float_array(r.source_x_m),
        'source_y_m': _float_array(r.source_y_m),
        'source_surface_elevation_m': _float_array(r.source_surface_elevation_m),
        'source_t1_s': source_t1_s,
        'source_v1_m_s': _filled_float_array(
            r.weathering_velocity_m_s,
            values.n_source_endpoints,
        ),
        'source_v2_m_s': _filled_float_array(
            r.bedrock_velocity_m_s,
            values.n_source_endpoints,
        ),
        'source_sh1_m': source_sh1_m,
        'source_weathering_correction_s': source_weathering_correction_s,
        'source_elevation_correction_s': _sum_float_arrays(
            r.source_floating_datum_elevation_shift_s,
            r.source_flat_datum_shift_s,
        ),
        'source_total_static_s': _float_array(r.source_refraction_shift_s),
        'source_total_applied_shift_s': _float_array(r.source_refraction_shift_s),
        'source_static_status': source_static_status,
        'receiver_endpoint_key': _string_array(r.receiver_endpoint_key),
        'receiver_id': _int_array(r.receiver_id),
        'receiver_node_id': _int_array(r.receiver_node_id),
        'receiver_x_m': _float_array(r.receiver_x_m),
        'receiver_y_m': _float_array(r.receiver_y_m),
        'receiver_surface_elevation_m': _float_array(
            r.receiver_surface_elevation_m
        ),
        'receiver_t1_s': receiver_t1_s,
        'receiver_v1_m_s': _filled_float_array(
            r.weathering_velocity_m_s,
            values.n_receiver_endpoints,
        ),
        'receiver_v2_m_s': _filled_float_array(
            r.bedrock_velocity_m_s,
            values.n_receiver_endpoints,
        ),
        'receiver_sh1_m': receiver_sh1_m,
        'receiver_weathering_correction_s': receiver_weathering_correction_s,
        'receiver_elevation_correction_s': _sum_float_arrays(
            r.receiver_floating_datum_elevation_shift_s,
            r.receiver_flat_datum_shift_s,
        ),
        'receiver_total_static_s': _float_array(r.receiver_refraction_shift_s),
        'receiver_total_applied_shift_s': _float_array(r.receiver_refraction_shift_s),
        'receiver_static_status': receiver_static_status,
    }
    _validate_no_object_arrays(
        arrays,
        artifact_name=SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    )
    return arrays


def build_refraction_static_solution_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> dict[str, np.ndarray]:
    values = _validate_result(result)
    r = values.result
    first_layer = _validate_resolved_first_layer(
        result=r,
        req=req,
        resolved_first_layer=resolved_first_layer,
    )
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
        'v1_mode': _scalar_str(first_layer.mode),
        'v1_weathering_velocity_m_s': _scalar_float(
            first_layer.weathering_velocity_m_s
        ),
        'weathering_velocity_m_s': _scalar_float(r.weathering_velocity_m_s),
        'resolved_weathering_velocity_m_s': _scalar_float(
            first_layer.weathering_velocity_m_s
        ),
        'bedrock_velocity_m_s': _scalar_float(r.bedrock_velocity_m_s),
        'v2_refractor_velocity_m_s': _scalar_float(r.bedrock_velocity_m_s),
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
        'node_t1_time_s': _float_array(r.node_half_intercept_time_s),
        'node_sh1_weathering_thickness_m': _float_array(
            r.node_weathering_thickness_m
        ),
        'node_weathering_correction_s': _float_array(
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
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
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
    artifact_entries = _artifact_entries_for_request(req, first_layer)
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'method': METHOD,
        'workflow': WORKFLOW,
        'static_component': STATIC_COMPONENT,
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
        'artifacts': _artifact_list_for_qc(artifact_entries),
        'warnings': [],
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_STATIC_QC_JSON_NAME)
    return payload


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


def _validate_resolved_first_layer(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> ResolvedRefractionFirstLayer:
    expected_mode = req.model.first_layer_mode
    result_velocity = float(result.weathering_velocity_m_s)
    if resolved_first_layer is None:
        try:
            velocity = float(req.model.resolved_weathering_velocity_m_s)
        except ValueError as exc:
            raise RefractionStaticArtifactError(
                'resolved first-layer weathering velocity is required'
            ) from exc
        status = (
            'estimated'
            if expected_mode == 'estimate_direct_arrival'
            else 'resolved_constant'
        )
        resolved_first_layer = ResolvedRefractionFirstLayer(
            mode=expected_mode,
            weathering_velocity_m_s=velocity,
            status=status,
            qc={
                'v1_mode': expected_mode,
                'weathering_velocity_m_s': velocity,
                'resolved_weathering_velocity_m_s': velocity,
                'v1_status': status,
            },
        )

    if resolved_first_layer.mode != expected_mode:
        raise RefractionStaticArtifactError(
            'resolved first-layer mode does not match request'
        )
    velocity = float(resolved_first_layer.weathering_velocity_m_s)
    if not np.isfinite(velocity) or velocity <= 0.0:
        raise RefractionStaticArtifactError(
            'resolved first-layer weathering velocity must be finite and positive'
        )
    if not _velocities_close(velocity, result_velocity):
        raise RefractionStaticArtifactError(
            'resolved first-layer weathering velocity does not match result'
        )
    return resolved_first_layer


def _velocities_close(left: float, right: float) -> bool:
    return bool(
        np.isclose(
            float(left),
            float(right),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
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


def _source_static_table_rows(
    result: RefractionDatumStaticsResult,
) -> list[dict[str, object]]:
    node_context = _node_context(result)
    static_status = _source_static_status_array(result)
    flat_datum = _nan_if_none(result.flat_datum_elevation_m)
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        node_id = int(result.source_node_id[index])
        t1_s = node_context['t1_s'].get(node_id)
        sh1_m = node_context['weathering_thickness'].get(node_id)
        weathering_correction_s = node_context['weathering_correction'].get(node_id)
        elevation_correction_s = _sum_correction_s(
            result.source_floating_datum_elevation_shift_s[index],
            result.source_flat_datum_shift_s[index],
        )
        rows.append(
            {
                'endpoint_kind': 'source',
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_id': int(result.source_id[index]),
                'source_node_id': node_id,
                'x_m': _csv_float(result.source_x_m[index]),
                'y_m': _csv_float(result.source_y_m[index]),
                'surface_elevation_m': _csv_float(
                    result.source_surface_elevation_m[index]
                ),
                'floating_datum_elevation_m': _csv_float(
                    result.source_floating_datum_elevation_m[index]
                ),
                'flat_datum_elevation_m': _csv_float(flat_datum),
                't1_ms': _csv_ms(t1_s),
                'v1_m_s': _csv_float(result.weathering_velocity_m_s),
                'v2_m_s': _csv_float(result.bedrock_velocity_m_s),
                'sh1_weathering_thickness_m': _csv_float(sh1_m),
                'refractor_elevation_m': _csv_float(
                    result.source_refractor_elevation_m[index]
                ),
                'weathering_correction_ms': _csv_ms(weathering_correction_s),
                'floating_datum_correction_ms': _csv_ms(
                    result.source_floating_datum_elevation_shift_s[index]
                ),
                'flat_datum_correction_ms': _csv_ms(
                    result.source_flat_datum_shift_s[index]
                ),
                'elevation_correction_ms': _csv_ms(elevation_correction_s),
                'total_static_ms': _csv_ms(result.source_refraction_shift_s[index]),
                'total_applied_shift_ms': _csv_ms(
                    result.source_refraction_shift_s[index]
                ),
                'solution_status': str(
                    node_context['solution_status'].get(node_id, 'missing_solution')
                ),
                'weathering_status': str(
                    node_context['weathering_status'].get(node_id, 'missing_node')
                ),
                'datum_status': str(result.source_datum_status[index]),
                'static_status': str(static_status[index]),
                'pick_count': _csv_int(node_context['pick_count'].get(node_id)),
                'used_pick_count': _csv_int(
                    node_context['used_pick_count'].get(node_id)
                ),
                'residual_rms_ms': _csv_ms(node_context['residual_rms'].get(node_id)),
                'residual_mad_ms': _csv_ms(node_context['residual_mad'].get(node_id)),
            }
        )
    return rows


def _receiver_static_table_rows(
    result: RefractionDatumStaticsResult,
) -> list[dict[str, object]]:
    node_context = _node_context(result)
    static_status = _receiver_static_status_array(result)
    flat_datum = _nan_if_none(result.flat_datum_elevation_m)
    rows: list[dict[str, object]] = []
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        node_id = int(result.receiver_node_id[index])
        t1_s = node_context['t1_s'].get(node_id)
        sh1_m = node_context['weathering_thickness'].get(node_id)
        weathering_correction_s = node_context['weathering_correction'].get(node_id)
        elevation_correction_s = _sum_correction_s(
            result.receiver_floating_datum_elevation_shift_s[index],
            result.receiver_flat_datum_shift_s[index],
        )
        rows.append(
            {
                'endpoint_kind': 'receiver',
                'receiver_endpoint_key': str(result.receiver_endpoint_key[index]),
                'receiver_id': int(result.receiver_id[index]),
                'receiver_node_id': node_id,
                'x_m': _csv_float(result.receiver_x_m[index]),
                'y_m': _csv_float(result.receiver_y_m[index]),
                'surface_elevation_m': _csv_float(
                    result.receiver_surface_elevation_m[index]
                ),
                'floating_datum_elevation_m': _csv_float(
                    result.receiver_floating_datum_elevation_m[index]
                ),
                'flat_datum_elevation_m': _csv_float(flat_datum),
                't1_ms': _csv_ms(t1_s),
                'v1_m_s': _csv_float(result.weathering_velocity_m_s),
                'v2_m_s': _csv_float(result.bedrock_velocity_m_s),
                'sh1_weathering_thickness_m': _csv_float(sh1_m),
                'refractor_elevation_m': _csv_float(
                    result.receiver_refractor_elevation_m[index]
                ),
                'weathering_correction_ms': _csv_ms(weathering_correction_s),
                'floating_datum_correction_ms': _csv_ms(
                    result.receiver_floating_datum_elevation_shift_s[index]
                ),
                'flat_datum_correction_ms': _csv_ms(
                    result.receiver_flat_datum_shift_s[index]
                ),
                'elevation_correction_ms': _csv_ms(elevation_correction_s),
                'total_static_ms': _csv_ms(result.receiver_refraction_shift_s[index]),
                'total_applied_shift_ms': _csv_ms(
                    result.receiver_refraction_shift_s[index]
                ),
                'solution_status': str(
                    node_context['solution_status'].get(node_id, 'missing_solution')
                ),
                'weathering_status': str(
                    node_context['weathering_status'].get(node_id, 'missing_node')
                ),
                'datum_status': str(result.receiver_datum_status[index]),
                'static_status': str(static_status[index]),
                'pick_count': _csv_int(node_context['pick_count'].get(node_id)),
                'used_pick_count': _csv_int(
                    node_context['used_pick_count'].get(node_id)
                ),
                'residual_rms_ms': _csv_ms(node_context['residual_rms'].get(node_id)),
                'residual_mad_ms': _csv_ms(node_context['residual_mad'].get(node_id)),
            }
        )
    return rows


def _node_lookup(node_id: np.ndarray, values: np.ndarray) -> dict[int, Any]:
    return {
        int(raw_node): values[index]
        for index, raw_node in enumerate(np.asarray(node_id).tolist())
    }


def _endpoint_node_values(
    endpoint_node_id: np.ndarray,
    node_id: np.ndarray,
    node_values: np.ndarray,
) -> np.ndarray:
    lookup = _node_lookup(node_id, node_values)
    out = np.full(np.asarray(endpoint_node_id).shape, np.nan, dtype=np.float64)
    for index, raw_node in enumerate(np.asarray(endpoint_node_id).tolist()):
        value = lookup.get(int(raw_node))
        if value is not None:
            out[index] = _float_or_nan(value)
    return np.ascontiguousarray(out, dtype=np.float64)


def _node_context(result: RefractionDatumStaticsResult) -> dict[str, dict[int, Any]]:
    return {
        'solution_status': _node_lookup(result.node_id, result.node_solution_status),
        'weathering_status': _node_lookup(result.node_id, result.node_weathering_status),
        't1_s': _node_lookup(result.node_id, result.node_half_intercept_time_s),
        'weathering_thickness': _node_lookup(
            result.node_id,
            result.node_weathering_thickness_m,
        ),
        'weathering_correction': _node_lookup(
            result.node_id,
            result.node_weathering_replacement_shift_s,
        ),
        'pick_count': _node_lookup(result.node_id, result.node_pick_count),
        'used_pick_count': _node_lookup(result.node_id, result.node_used_pick_count),
        'residual_rms': _node_lookup(result.node_id, result.node_residual_rms_s),
        'residual_mad': _node_lookup(result.node_id, result.node_residual_mad_s),
    }


def _source_static_status_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    node_context = _node_context(result)
    return _endpoint_static_status_array(
        node_id=result.source_node_id,
        x_m=result.source_x_m,
        y_m=result.source_y_m,
        surface_elevation_m=result.source_surface_elevation_m,
        t1_s=_endpoint_node_values(
            result.source_node_id,
            result.node_id,
            result.node_half_intercept_time_s,
        ),
        weathering_thickness_m=_endpoint_node_values(
            result.source_node_id,
            result.node_id,
            result.node_weathering_thickness_m,
        ),
        total_shift_s=result.source_refraction_shift_s,
        datum_status=result.source_datum_status,
        node_solution_status=node_context['solution_status'],
        node_weathering_status=node_context['weathering_status'],
    )


def _receiver_static_status_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    node_context = _node_context(result)
    return _endpoint_static_status_array(
        node_id=result.receiver_node_id,
        x_m=result.receiver_x_m,
        y_m=result.receiver_y_m,
        surface_elevation_m=result.receiver_surface_elevation_m,
        t1_s=_endpoint_node_values(
            result.receiver_node_id,
            result.node_id,
            result.node_half_intercept_time_s,
        ),
        weathering_thickness_m=_endpoint_node_values(
            result.receiver_node_id,
            result.node_id,
            result.node_weathering_thickness_m,
        ),
        total_shift_s=result.receiver_refraction_shift_s,
        datum_status=result.receiver_datum_status,
        node_solution_status=node_context['solution_status'],
        node_weathering_status=node_context['weathering_status'],
    )


def _endpoint_static_status_array(
    *,
    node_id: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    surface_elevation_m: np.ndarray,
    t1_s: np.ndarray,
    weathering_thickness_m: np.ndarray,
    total_shift_s: np.ndarray,
    datum_status: np.ndarray,
    node_solution_status: dict[int, Any],
    node_weathering_status: dict[int, Any],
) -> np.ndarray:
    statuses: list[str] = []
    for index, raw_node_id in enumerate(np.asarray(node_id).tolist()):
        endpoint_node_id = int(raw_node_id)
        solution_status = str(
            node_solution_status.get(endpoint_node_id, 'missing_solution')
        )
        weathering_status = str(
            node_weathering_status.get(endpoint_node_id, 'missing_node')
        )
        statuses.append(
            _endpoint_static_status(
                node_missing=endpoint_node_id not in node_solution_status,
                x_m=x_m[index],
                y_m=y_m[index],
                surface_elevation_m=surface_elevation_m[index],
                t1_s=t1_s[index],
                weathering_thickness_m=weathering_thickness_m[index],
                total_shift_s=total_shift_s[index],
                solution_status=solution_status,
                weathering_status=weathering_status,
                datum_status=datum_status[index],
            )
        )
    return _string_array(statuses)


def _endpoint_static_status(
    *,
    node_missing: bool,
    x_m: object,
    y_m: object,
    surface_elevation_m: object,
    t1_s: object,
    weathering_thickness_m: object,
    total_shift_s: object,
    solution_status: object,
    weathering_status: object,
    datum_status: object,
) -> str:
    solution = str(solution_status)
    weathering = str(weathering_status)
    datum = str(datum_status)
    if node_missing or 'missing_node' in {solution, weathering, datum}:
        return 'missing_linkage'
    if not all(
        np.isfinite(_float_or_nan(value))
        for value in (x_m, y_m, surface_elevation_m)
    ):
        return 'missing_geometry'
    if 'inactive' in {solution, weathering, datum}:
        return 'inactive_endpoint'
    if 'low_fold' in {solution, weathering, datum}:
        return 'insufficient_pick_fold'
    if (
        not np.isfinite(_float_or_nan(t1_s))
        or solution in {'invalid_solution', 'missing_solution'}
        or weathering == 'invalid_half_intercept'
    ):
        return 'invalid_t1'
    if (
        not np.isfinite(_float_or_nan(weathering_thickness_m))
        or weathering
        in {
            'invalid_weathering_thickness',
            'negative_weathering_thickness',
            'negative_thickness',
            'exceeds_max_thickness',
            'invalid_weathering_replacement',
        }
        or datum == 'invalid_weathering_replacement'
    ):
        return 'invalid_weathering_thickness'
    if datum in {
        'invalid_datum_shift',
        'invalid_floating_datum_elevation',
        'invalid_flat_datum_elevation',
        'floating_datum_below_refractor',
        'flat_datum_below_refractor',
    }:
        return 'invalid_datum'
    for status in (datum, weathering, solution):
        if status not in {'ok', 'solved', 'zero_thickness'}:
            return status
    if not np.isfinite(_float_or_nan(total_shift_s)):
        return 'not_applied'
    return 'ok'


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


def _artifact_entries_for_request(
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> tuple[dict[str, str | bool], ...]:
    return (
        _ARTIFACTS
        + _t1lsst_artifact_entries(req)
        + _v1_artifact_entries(req, resolved_first_layer)
    )


def _t1lsst_artifact_entries(
    req: RefractionStaticApplyRequest,
) -> tuple[dict[str, str | bool], ...]:
    if req.conversion.mode == 't1lsst_1layer':
        return _T1LSST_1LAYER_ARTIFACTS
    return ()


def _v1_artifact_entries(
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
) -> tuple[dict[str, str | bool], ...]:
    mode = (
        resolved_first_layer.mode
        if resolved_first_layer is not None
        else req.model.first_layer_mode
    )
    if mode == 'estimate_direct_arrival':
        return _V1_ARTIFACTS
    return _OPTIONAL_CONSTANT_V1_ARTIFACTS


def _build_manifest_payload(
    artifact_entries: tuple[dict[str, str | bool], ...],
) -> dict[str, Any]:
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
            for item in artifact_entries
        ],
    }


def _artifact_list_for_qc(
    artifact_entries: tuple[dict[str, str | bool], ...],
) -> list[dict[str, str]]:
    return [
        {
            'name': str(item['name']),
            'kind': str(item['kind']),
            'description': str(item['description']),
        }
        for item in artifact_entries
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
            if str(item) not in REFRACTION_STATIC_STATUSES
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


def _filled_float_array(value: object, shape: int) -> np.ndarray:
    return np.full(int(shape), float(value), dtype=np.float64)


def _sum_float_arrays(left: object, right: object) -> np.ndarray:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    out = np.full(left_arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(left_arr) & np.isfinite(right_arr)
    out[finite] = left_arr[finite] + right_arr[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


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


def _float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out if np.isfinite(out) else float('nan')


def _sum_correction_s(left: object, right: object) -> float:
    left_value = _float_or_nan(left)
    right_value = _float_or_nan(right)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float('nan')
    return float(left_value + right_value)


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
    'REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES',
    'REFRACTION_STATIC_QC_JSON_NAME',
    'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
    'REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME',
    'REFRACTION_V1_ESTIMATES_CSV_NAME',
    'REFRACTION_V1_QC_JSON_NAME',
    'RECEIVER_STATIC_TABLE_CSV_NAME',
    'SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME',
    'SOURCE_STATIC_TABLE_CSV_NAME',
    'RefractionStaticArtifactError',
    'RefractionStaticArtifactSet',
    'build_refraction_static_qc_payload',
    'build_refraction_static_solution_arrays',
    'build_source_receiver_static_table_arrays',
    'write_first_break_residuals_csv',
    'write_near_surface_model_csv',
    'write_refraction_static_artifacts',
    'write_refraction_static_components_csv',
    'write_refraction_static_qc_json',
    'write_refraction_static_solution_npz',
    'write_refraction_statics_csv',
    'write_receiver_static_table_csv',
    'write_source_receiver_static_table_npz',
    'write_source_static_table_csv',
]
