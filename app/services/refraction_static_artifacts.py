"""Final artifact package writer for GLI refraction statics."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_cell_coordinates import (
    effective_refraction_cell_grid_config,
    refraction_cell_coordinate_metadata_from_config,
)
from app.services.refraction_static_cell_grid import build_refraction_cell_grid
from app.services.refraction_static_design_matrix import (
    LOW_FOLD_CELL_REJECTION_REASON,
    LOW_FOLD_CELL_VELOCITY_STATUS,
)
from app.services.refraction_static_status import (
    REFRACTION_STATIC_STATUSES,
    classify_refraction_endpoint_static_status,
)
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
REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME = (
    'refraction_refractor_velocity_cells.csv'
)
REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME = (
    'refraction_refractor_velocity_grid.npz'
)
REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME = 'refraction_refractor_velocity_qc.json'
REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME = 'refraction_cell_solver_history.csv'
REFRACTION_STATIC_ARTIFACTS_JSON_NAME = 'refraction_static_artifacts.json'
REFRACTION_STATIC_REQUEST_JSON_NAME = 'refraction_static_request.json'

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
        'origin': 'upstream',
        'description': 'Direct-arrival V1 estimation QC summary',
    },
    {
        'name': REFRACTION_V1_ESTIMATES_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'origin': 'upstream',
        'description': 'Per-source direct-arrival V1 estimates',
    },
)

_V1_ARTIFACT_NAMES = frozenset(
    {
        REFRACTION_V1_QC_JSON_NAME,
        REFRACTION_V1_ESTIMATES_CSV_NAME,
    }
)

_T1LSST_1LAYER_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'T1LSST-compatible one-layer source/receiver components',
    },
)

_REFRACTOR_CELL_VELOCITY_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Per-cell refractor velocity grid and QC metrics',
    },
    {
        'name': REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        'kind': 'npz',
        'required': True,
        'description': 'Machine-readable refractor velocity cell grid',
    },
    {
        'name': REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Refractor velocity cell QC summary',
    },
    {
        'name': REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Cell V2/T1 solver convergence and history summary',
    },
)

REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES = frozenset(
    str(item['name'])
    for item in (
        _ARTIFACTS
        + _V1_ARTIFACTS
        + _T1LSST_1LAYER_ARTIFACTS
        + _REFRACTOR_CELL_VELOCITY_ARTIFACTS
    )
) | {
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
}

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

# Keep the legacy millisecond residual columns and append explicit seconds/cell
# aliases so residual rows can be joined to refractor-cell QC artifacts.
_RESIDUAL_COLUMNS = (
    'row_index',
    'observation_index',
    'sorted_trace_index',
    'source_node_id',
    'receiver_node_id',
    'distance_m',
    'observed_pick_time_ms',
    'observed_pick_time_s',
    'modeled_pick_time_ms',
    'modeled_pick_time_s',
    'residual_ms',
    'residual_s',
    'used',
    'used_in_solve',
    'rejected_by_robust',
    'rejection_reason',
    'cell_id',
    'cell_ix',
    'cell_iy',
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
    'source_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
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

_SOURCE_STATIC_TABLE_2LAYER_COLUMNS = (
    'endpoint_kind',
    'source_endpoint_key',
    'source_id',
    'source_node_id',
    'source_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    't2_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'v3_m_s',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
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
    'receiver_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
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

_RECEIVER_STATIC_TABLE_2LAYER_COLUMNS = (
    'endpoint_kind',
    'receiver_endpoint_key',
    'receiver_id',
    'receiver_node_id',
    'receiver_v2_cell_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    't2_ms',
    'v1_m_s',
    'v2_m_s',
    'v2_status',
    'v3_m_s',
    'sh1_weathering_thickness_m',
    'sh2_weathering_thickness_m',
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

# Keep the original Phase 2 cell columns and add self-describing aliases used
# by downstream QC checks; existing artifact names and column meanings remain.
_REFRACTOR_VELOCITY_CELL_COLUMNS = (
    'cell_id',
    'ix',
    'iy',
    'cell_ix',
    'cell_iy',
    'coordinate_mode',
    'x_min_m',
    'x_max_m',
    'y_min_m',
    'y_max_m',
    'x_center_m',
    'y_center_m',
    'cell_center_x_m',
    'cell_center_y_m',
    'cell_center_inline_m',
    'cell_center_crossline_m',
    'active',
    'n_observations',
    'n_used_observations',
    'n_rejected_observations',
    'n_sources',
    'n_receivers',
    'v2_m_s',
    'slowness_s_per_m',
    'initial_v2_m_s',
    'v2_update_from_initial_m_s',
    'velocity_status',
    'status_reason',
    'residual_rms_ms',
    'residual_mad_ms',
    'residual_mean_ms',
    'residual_p95_abs_ms',
    'smoothing_enabled',
    'smoothing_weight',
    'smoothing_neighbor_count',
)

_CELL_SOLVER_HISTORY_COLUMNS = (
    'iteration',
    'stage',
    'n_candidate_observations',
    'n_used_observations',
    'n_rejected_observations',
    'n_active_cells',
    'n_low_fold_cells',
    'n_empty_cells',
    'residual_rms_ms',
    'residual_mad_ms',
    'max_abs_residual_ms',
    'median_v2_m_s',
    'min_v2_m_s',
    'max_v2_m_s',
    'max_abs_v2_update_m_s',
    'smoothing_weight',
    'damping_weight',
    'robust_threshold',
    'converged',
    'convergence_reason',
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


@dataclass(frozen=True)
class RefractionCellSolverHistoryRow:
    iteration: int
    stage: str
    n_candidate_observations: int
    n_used_observations: int
    n_rejected_observations: int
    n_active_cells: int
    n_low_fold_cells: int
    n_empty_cells: int
    residual_rms_ms: float | None
    residual_mad_ms: float | None
    max_abs_residual_ms: float | None
    median_v2_m_s: float | None
    min_v2_m_s: float | None
    max_v2_m_s: float | None
    max_abs_v2_update_m_s: float | None
    smoothing_weight: float
    damping_weight: float
    robust_threshold: float
    converged: bool
    convergence_reason: str


def write_refraction_static_artifacts(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    job_dir: Path,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    upstream_artifact_names: Iterable[str] = (),
) -> RefractionStaticArtifactSet:
    """Write final refraction statics artifacts and an artifact manifest.

    This writer owns only the final refraction statics artifacts it writes in
    this function.  Upstream artifacts, currently the direct-arrival V1 QC and
    estimates files, are included in the manifest only when their plain file
    names are passed via ``upstream_artifact_names``.  Declared upstream
    artifacts must already exist in ``job_dir`` and are validated with a
    dedicated upstream-artifact error.
    """
    root = _validate_job_dir(job_dir)
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    first_layer = _validate_resolved_first_layer(
        result=values.result,
        req=request,
        resolved_first_layer=resolved_first_layer,
    )
    upstream_names = _validate_upstream_artifact_names(
        upstream_artifact_names,
        resolved_first_layer=first_layer,
    )
    _validate_declared_upstream_artifacts(root, upstream_names)
    artifact_entries = _artifact_entries_for_request(
        request,
        first_layer,
        upstream_artifact_names=upstream_names,
    )
    qc = build_refraction_static_qc_payload(
        result=values.result,
        req=request,
        resolved_first_layer=first_layer,
        upstream_artifact_names=upstream_names,
    )
    manifest = _build_manifest_payload(artifact_entries)
    _assert_strict_json(manifest, artifact_name=REFRACTION_STATIC_ARTIFACTS_JSON_NAME)
    t1lsst_components_path = (
        root / REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME
        if request.conversion.mode == 't1lsst_1layer'
        else None
    )
    cell_velocity_artifacts_enabled = request.model.bedrock_velocity_mode == 'solve_cell'
    cell_velocity_cells_path = (
        root / REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME
        if cell_velocity_artifacts_enabled
        else None
    )
    cell_velocity_grid_path = (
        root / REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME
        if cell_velocity_artifacts_enabled
        else None
    )
    cell_velocity_qc_path = (
        root / REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME
        if cell_velocity_artifacts_enabled
        else None
    )
    cell_solver_history_path = (
        root / REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME
        if cell_velocity_artifacts_enabled
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
        artifact_names=tuple(
            str(item['name']) for item in artifact_entries if bool(item['required'])
        ),
        qc=qc,
        refraction_t1lsst_1layer_components_csv=t1lsst_components_path,
        refraction_refractor_velocity_cells_csv=cell_velocity_cells_path,
        refraction_refractor_velocity_grid_npz=cell_velocity_grid_path,
        refraction_refractor_velocity_qc_json=cell_velocity_qc_path,
        refraction_cell_solver_history_csv=cell_solver_history_path,
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
        req=request,
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
    if (
        paths.refraction_refractor_velocity_cells_csv is not None
        and paths.refraction_refractor_velocity_grid_npz is not None
        and paths.refraction_refractor_velocity_qc_json is not None
        and paths.refraction_cell_solver_history_csv is not None
    ):
        write_refraction_refractor_velocity_cells_csv(
            result=values.result,
            req=request,
            path=paths.refraction_refractor_velocity_cells_csv,
        )
        write_refraction_refractor_velocity_grid_npz(
            result=values.result,
            req=request,
            path=paths.refraction_refractor_velocity_grid_npz,
        )
        write_refraction_refractor_velocity_qc_json(
            result=values.result,
            req=request,
            path=paths.refraction_refractor_velocity_qc_json,
        )
        write_refraction_cell_solver_history_csv(
            result=values.result,
            req=request,
            path=paths.refraction_cell_solver_history_csv,
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
    if (
        paths.refraction_refractor_velocity_cells_csv is not None
        and paths.refraction_refractor_velocity_grid_npz is not None
        and paths.refraction_refractor_velocity_qc_json is not None
        and paths.refraction_cell_solver_history_csv is not None
    ):
        artifact_paths = artifact_paths + (
            paths.refraction_refractor_velocity_cells_csv,
            paths.refraction_refractor_velocity_grid_npz,
            paths.refraction_refractor_velocity_qc_json,
            paths.refraction_cell_solver_history_csv,
        )
    for artifact_path in artifact_paths:
        if not artifact_path.is_file():
            raise RefractionStaticArtifactError(
                f'artifact file missing after write: {artifact_path.name}'
            )
    _validate_declared_upstream_artifacts(root, upstream_names)
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
    req: RefractionStaticApplyRequest | None = None,
) -> None:
    values = _validate_result(result)
    rows = _first_break_residual_rows(values.result, req=req)
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
    _write_csv_atomic(Path(path), _source_static_table_columns(values.result), rows)


def write_receiver_static_table_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _receiver_static_table_rows(values.result)
    _write_csv_atomic(Path(path), _receiver_static_table_columns(values.result), rows)


def write_source_receiver_static_table_npz(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    payload = build_source_receiver_static_table_arrays(result=values.result)
    _write_npz_atomic(Path(path), payload)


def write_refraction_refractor_velocity_cells_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=req,
    )
    rows = _refractor_velocity_cell_rows(arrays)
    _write_csv_atomic(Path(path), _REFRACTOR_VELOCITY_CELL_COLUMNS, rows)


def write_refraction_refractor_velocity_grid_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=req,
    )
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    )
    _write_npz_atomic(Path(path), arrays)


def write_refraction_refractor_velocity_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> dict[str, Any]:
    payload = build_refraction_refractor_velocity_qc_payload(
        result=result,
        req=req,
    )
    _write_json_atomic(Path(path), payload)
    return payload


def write_refraction_cell_solver_history_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    rows = build_refraction_cell_solver_history_rows(result=result, req=req)
    csv_rows = [_cell_solver_history_csv_row(row) for row in rows]
    _write_csv_atomic(Path(path), _CELL_SOLVER_HISTORY_COLUMNS, csv_rows)


def build_refraction_refractor_velocity_grid_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    if request.model.bedrock_velocity_mode != 'solve_cell':
        raise RefractionStaticArtifactError(
            'refractor cell velocity artifacts require solve_cell request mode'
        )
    if values.result.bedrock_velocity_mode != 'solve_cell':
        raise RefractionStaticArtifactError(
            'refractor cell velocity artifacts require solve_cell result mode'
        )
    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )

    grid_config = effective_refraction_cell_grid_config(refractor_cell)
    grid = build_refraction_cell_grid(grid_config)
    n_total_cells = int(grid.cell_id.shape[0])
    active_cell_id = _required_cell_int_array(
        values.result.active_cell_id,
        name='active_cell_id',
    )
    inactive_cell_id = _required_cell_int_array(
        values.result.inactive_cell_id,
        name='inactive_cell_id',
    )
    cell_slowness = _required_cell_float_array(
        values.result.cell_bedrock_slowness_s_per_m,
        name='cell_bedrock_slowness_s_per_m',
    )
    cell_velocity = _required_cell_float_array(
        values.result.cell_bedrock_velocity_m_s,
        name='cell_bedrock_velocity_m_s',
    )
    cell_status = _required_cell_status_array(
        values.result.cell_velocity_status,
        name='cell_velocity_status',
    )
    row_midpoint_cell_id = _required_cell_int_array(
        values.result.row_midpoint_cell_id,
        name='row_midpoint_cell_id',
    )
    if row_midpoint_cell_id.shape != (values.n_rows,):
        raise RefractionStaticArtifactError(
            'row_midpoint_cell_id length must match residual rows'
        )
    active_shape = active_cell_id.shape
    for name, array in (
        ('cell_bedrock_slowness_s_per_m', cell_slowness),
        ('cell_bedrock_velocity_m_s', cell_velocity),
        ('cell_velocity_status', cell_status),
    ):
        if array.shape != active_shape:
            raise RefractionStaticArtifactError(
                f'{name} length must match active_cell_id'
            )
    _validate_refractor_velocity_cell_ids(
        grid_cell_id=grid.cell_id,
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
    )

    active_cell_mask = np.zeros(n_total_cells, dtype=bool)
    v2_m_s = np.full(n_total_cells, np.nan, dtype=np.float64)
    slowness_s_per_m = np.full(n_total_cells, np.nan, dtype=np.float64)
    velocity_status = np.full(n_total_cells, 'inactive', dtype='<U32')
    smoothing_neighbor_count = _active_neighbor_count_by_cell(
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        number_of_cell_x=grid.number_of_cell_x,
        number_of_cell_y=grid.number_of_cell_y,
    )
    for index, raw_cell_id in enumerate(active_cell_id.tolist()):
        cell_id = int(raw_cell_id)
        active_cell_mask[cell_id] = True
        v2_m_s[cell_id] = float(cell_velocity[index])
        slowness_s_per_m[cell_id] = float(cell_slowness[index])
        velocity_status[cell_id] = str(cell_status[index])
    low_fold_cell_id = _qc_cell_id_array(
        values.result.qc,
        'low_fold_cell_id',
        n_total_cells=n_total_cells,
    )
    if low_fold_cell_id.size:
        velocity_status[low_fold_cell_id] = LOW_FOLD_CELL_VELOCITY_STATUS

    coordinate_mode = str(refractor_cell.coordinate_mode)
    center_aliases = _refractor_cell_center_alias_arrays(
        x_center_m=grid.x_center_m,
        y_center_m=grid.y_center_m,
        coordinate_mode=coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    initial_v2 = np.full(n_total_cells, np.nan, dtype=np.float64)
    if request.model.initial_bedrock_velocity_m_s is not None:
        initial_v2.fill(float(request.model.initial_bedrock_velocity_m_s))
    v2_update = np.full(n_total_cells, np.nan, dtype=np.float64)
    finite_update = np.isfinite(v2_m_s) & np.isfinite(initial_v2)
    v2_update[finite_update] = v2_m_s[finite_update] - initial_v2[finite_update]
    smoothing_weight = float(refractor_cell.velocity_smoothing_weight)
    smoothing_enabled = bool(smoothing_weight > 0.0)

    n_observations = np.zeros(n_total_cells, dtype=np.int64)
    n_used_observations = np.zeros(n_total_cells, dtype=np.int64)
    valid_row_cell = (
        (row_midpoint_cell_id >= 0)
        & (row_midpoint_cell_id < n_total_cells)
    )
    np.add.at(n_observations, row_midpoint_cell_id[valid_row_cell], 1)
    qc_observation_count = _qc_cell_count_array(
        values.result.qc,
        'cell_observation_count',
        n_total_cells=n_total_cells,
    )
    if qc_observation_count is not None:
        n_observations = qc_observation_count
    used_row = valid_row_cell & np.asarray(values.result.used_row_mask, dtype=bool)
    np.add.at(n_used_observations, row_midpoint_cell_id[used_row], 1)
    if np.any(n_used_observations > n_observations):
        raise RefractionStaticArtifactError(
            'used observations per cell cannot exceed total observations per cell'
        )
    n_sources = _unique_observation_endpoint_count_by_cell(
        row_midpoint_cell_id=row_midpoint_cell_id,
        endpoint_id=values.result.row_source_node_id,
        n_total_cells=n_total_cells,
    )
    n_receivers = _unique_observation_endpoint_count_by_cell(
        row_midpoint_cell_id=row_midpoint_cell_id,
        endpoint_id=values.result.row_receiver_node_id,
        n_total_cells=n_total_cells,
    )
    residual_stats = _per_cell_residual_stats_ms(
        row_midpoint_cell_id=row_midpoint_cell_id,
        residual_time_s=values.result.residual_time_s,
        used_row_mask=values.result.used_row_mask,
        n_total_cells=n_total_cells,
    )
    status_reason = _refractor_cell_status_reasons(
        velocity_status=velocity_status,
        n_observations=n_observations,
    )

    arrays = {
        'cell_id': np.ascontiguousarray(grid.cell_id, dtype=np.int64),
        'ix': np.ascontiguousarray(grid.ix, dtype=np.int64),
        'iy': np.ascontiguousarray(grid.iy, dtype=np.int64),
        'cell_ix': np.ascontiguousarray(grid.ix, dtype=np.int64),
        'cell_iy': np.ascontiguousarray(grid.iy, dtype=np.int64),
        'coordinate_mode': _string_array(
            np.full(n_total_cells, coordinate_mode, dtype=f'<U{len(coordinate_mode)}')
        ),
        'x_min_m': np.ascontiguousarray(grid.x_min_m, dtype=np.float64),
        'x_max_m': np.ascontiguousarray(grid.x_max_m, dtype=np.float64),
        'y_min_m': np.ascontiguousarray(grid.y_min_m, dtype=np.float64),
        'y_max_m': np.ascontiguousarray(grid.y_max_m, dtype=np.float64),
        'x_center_m': np.ascontiguousarray(grid.x_center_m, dtype=np.float64),
        'y_center_m': np.ascontiguousarray(grid.y_center_m, dtype=np.float64),
        'cell_center_x_m': center_aliases['x_m'],
        'cell_center_y_m': center_aliases['y_m'],
        'cell_center_inline_m': center_aliases['inline_m'],
        'cell_center_crossline_m': center_aliases['crossline_m'],
        'active_cell_mask': np.ascontiguousarray(active_cell_mask, dtype=bool),
        'n_observations_per_cell': np.ascontiguousarray(
            n_observations,
            dtype=np.int64,
        ),
        'n_used_observations_per_cell': np.ascontiguousarray(
            n_used_observations,
            dtype=np.int64,
        ),
        'n_rejected_observations_per_cell': np.ascontiguousarray(
            n_observations - n_used_observations,
            dtype=np.int64,
        ),
        'n_sources_per_cell': np.ascontiguousarray(n_sources, dtype=np.int64),
        'n_receivers_per_cell': np.ascontiguousarray(n_receivers, dtype=np.int64),
        'v2_m_s': np.ascontiguousarray(v2_m_s, dtype=np.float64),
        'slowness_s_per_m': np.ascontiguousarray(slowness_s_per_m, dtype=np.float64),
        'initial_v2_m_s': np.ascontiguousarray(initial_v2, dtype=np.float64),
        'v2_update_from_initial_m_s': np.ascontiguousarray(
            v2_update,
            dtype=np.float64,
        ),
        'velocity_status': _string_array(velocity_status),
        'status_reason': _string_array(status_reason),
        'residual_rms_ms': residual_stats['rms'],
        'residual_mad_ms': residual_stats['mad'],
        'residual_mean_ms': residual_stats['mean'],
        'residual_p95_abs_ms': residual_stats['p95_abs'],
        'smoothing_enabled': np.full(n_total_cells, smoothing_enabled, dtype=bool),
        'smoothing_weight': np.full(
            n_total_cells,
            smoothing_weight,
            dtype=np.float64,
        ),
        'smoothing_neighbor_count': np.ascontiguousarray(
            smoothing_neighbor_count,
            dtype=np.int64,
        ),
    }
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    )
    return arrays


def build_refraction_refractor_velocity_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=request,
    )
    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity QC'
        )
    grid_config = effective_refraction_cell_grid_config(refractor_cell)
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)
    velocity = np.asarray(arrays['v2_m_s'], dtype=np.float64)
    active_velocity = velocity[active_mask & np.isfinite(velocity)]
    n_total = int(arrays['cell_id'].shape[0])
    n_active = int(np.count_nonzero(active_mask))
    n_observations_in_grid = int(np.sum(arrays['n_observations_per_cell']))
    n_valid_observations = int(np.count_nonzero(result.valid_observation_mask_sorted))
    default_outside_observations = max(0, n_valid_observations - n_observations_in_grid)
    n_used = int(np.sum(arrays['n_used_observations_per_cell']))
    n_low_fold = int(
        np.count_nonzero(
            np.asarray(arrays['velocity_status']).astype(str, copy=False)
            == LOW_FOLD_CELL_VELOCITY_STATUS
        )
    )
    n_low_fold_rejected = int(
        np.sum(
            np.asarray(arrays['n_rejected_observations_per_cell'], dtype=np.int64)[
                np.asarray(arrays['velocity_status']).astype(str, copy=False)
                == LOW_FOLD_CELL_VELOCITY_STATUS
            ]
        )
    )
    n_smoothing_rows = _qc_int(
        result.qc,
        'n_cell_smoothing_rows',
        default=_estimated_cell_smoothing_rows(
            active_cell_mask=active_mask,
            number_of_cell_x=int(grid_config.number_of_cell_x),
            number_of_cell_y=int(grid_config.number_of_cell_y),
            velocity_smoothing_weight=float(
                refractor_cell.velocity_smoothing_weight
            ),
        ),
    )
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'bedrock_velocity_mode': 'solve_cell',
        'cell_assignment_mode': refractor_cell.assignment_mode,
        **refraction_cell_coordinate_metadata_from_config(refractor_cell),
        'outside_grid_policy': refractor_cell.outside_grid_policy,
        'number_of_cell_x': int(grid_config.number_of_cell_x),
        'number_of_cell_y': int(grid_config.number_of_cell_y),
        'size_of_cell_x_m': float(grid_config.size_of_cell_x_m),
        'size_of_cell_y_m': _json_float(grid_config.size_of_cell_y_m),
        'n_total_cells': n_total,
        'n_active_cells': n_active,
        'n_inactive_cells': int(n_total - n_active),
        'min_observations_per_cell': _qc_int(
            result.qc,
            'min_observations_per_cell',
            default=int(refractor_cell.min_observations_per_cell),
        ),
        'n_low_fold_cells': _qc_int(
            result.qc,
            'n_low_fold_cells',
            default=n_low_fold,
        ),
        'n_observations_outside_grid': _qc_int(
            result.qc,
            'n_observations_outside_grid',
            default=default_outside_observations,
        ),
        'n_observations_rejected_by_low_fold_cell': _qc_int(
            result.qc,
            'n_observations_rejected_by_low_fold_cell',
            default=n_low_fold_rejected,
        ),
        'low_fold_cell_rejection_reason': str(
            result.qc.get(
                'low_fold_cell_rejection_reason',
                LOW_FOLD_CELL_REJECTION_REASON,
            )
        ),
        'n_used_observations': n_used,
        'velocity_min_m_s': _stat(active_velocity, 'min'),
        'velocity_median_m_s': _stat(active_velocity, 'median'),
        'velocity_max_m_s': _stat(active_velocity, 'max'),
        'velocity_smoothing_weight': float(
            refractor_cell.velocity_smoothing_weight
        ),
        'smoothing_reference_distance_m': _qc_optional_float(
            result.qc,
            'smoothing_reference_distance_m',
            default=refractor_cell.smoothing_reference_distance_m,
        ),
        'n_cell_smoothing_rows': n_smoothing_rows,
    }
    _assert_strict_json(
        payload,
        artifact_name=REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    )
    return payload


def build_refraction_cell_solver_history_rows(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> list[RefractionCellSolverHistoryRow]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    if request.model.bedrock_velocity_mode != 'solve_cell':
        raise RefractionStaticArtifactError(
            'cell solver history artifact requires solve_cell request mode'
        )
    if values.result.bedrock_velocity_mode != 'solve_cell':
        raise RefractionStaticArtifactError(
            'cell solver history artifact requires solve_cell result mode'
        )

    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=values.result,
        req=request,
    )
    cell_counts = _cell_solver_history_cell_counts(arrays)
    initial_v2 = _initial_cell_v2_m_s(request)
    final_velocity = np.asarray(arrays['v2_m_s'], dtype=np.float64)
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)
    active_final_v2 = final_velocity[active_mask & np.isfinite(final_velocity)]
    update = np.asarray(
        arrays['v2_update_from_initial_m_s'],
        dtype=np.float64,
    )
    active_update = update[active_mask & np.isfinite(update)]

    n_candidate = values.n_rows
    n_used = int(np.count_nonzero(values.result.used_row_mask))
    n_robust_rejected = int(np.count_nonzero(values.result.rejected_by_robust_mask))
    smoothing_weight = _history_smoothing_weight(request)
    damping_weight = float(request.solver.damping)
    robust_threshold = float(request.solver.robust.threshold)
    robust_iteration_count = _history_robust_iteration_count(
        values.result,
        request,
    )
    residual_stats = _history_residual_stats_ms(values.result)

    return [
        RefractionCellSolverHistoryRow(
            iteration=0,
            stage='initial',
            n_candidate_observations=n_candidate,
            n_used_observations=n_candidate,
            n_rejected_observations=0,
            n_active_cells=cell_counts['active'],
            n_low_fold_cells=cell_counts['low_fold'],
            n_empty_cells=cell_counts['empty'],
            residual_rms_ms=None,
            residual_mad_ms=None,
            max_abs_residual_ms=None,
            median_v2_m_s=initial_v2,
            min_v2_m_s=initial_v2,
            max_v2_m_s=initial_v2,
            max_abs_v2_update_m_s=0.0,
            smoothing_weight=smoothing_weight,
            damping_weight=damping_weight,
            robust_threshold=robust_threshold,
            converged=False,
            convergence_reason='initial_state',
        ),
        RefractionCellSolverHistoryRow(
            iteration=max(1, robust_iteration_count),
            stage='final',
            n_candidate_observations=n_candidate,
            n_used_observations=n_used,
            n_rejected_observations=n_robust_rejected,
            n_active_cells=cell_counts['active'],
            n_low_fold_cells=cell_counts['low_fold'],
            n_empty_cells=cell_counts['empty'],
            residual_rms_ms=residual_stats['rms'],
            residual_mad_ms=residual_stats['mad'],
            max_abs_residual_ms=residual_stats['max_abs'],
            median_v2_m_s=_stat(active_final_v2, 'median'),
            min_v2_m_s=_stat(active_final_v2, 'min'),
            max_v2_m_s=_stat(active_final_v2, 'max'),
            max_abs_v2_update_m_s=_history_max_abs(active_update),
            smoothing_weight=smoothing_weight,
            damping_weight=damping_weight,
            robust_threshold=robust_threshold,
            converged=True,
            convergence_reason=_history_convergence_reason(
                robust_iteration_count=robust_iteration_count,
                n_robust_rejected_observations=n_robust_rejected,
                smoothing_weight=smoothing_weight,
            ),
        ),
    ]


def build_source_receiver_static_table_arrays(
    *,
    result: RefractionDatumStaticsResult,
) -> dict[str, np.ndarray]:
    values = _validate_result(result)
    r = values.result
    source_t1_s = _float_array(r.source_half_intercept_time_s)
    source_sh1_m = _float_array(r.source_weathering_thickness_m)
    source_weathering_correction_s = _float_array(
        r.source_weathering_replacement_shift_s
    )
    receiver_t1_s = _float_array(r.receiver_half_intercept_time_s)
    receiver_sh1_m = _float_array(r.receiver_weathering_thickness_m)
    receiver_weathering_correction_s = _float_array(
        r.receiver_weathering_replacement_shift_s
    )
    source_static_status = _source_static_status_array(r)
    receiver_static_status = _receiver_static_status_array(r)
    source_v2 = _endpoint_v2_m_s(
        r.source_v2_m_s,
        shape=values.n_source_endpoints,
        scalar_v2_m_s=r.bedrock_velocity_m_s,
    )
    receiver_v2 = _endpoint_v2_m_s(
        r.receiver_v2_m_s,
        shape=values.n_receiver_endpoints,
        scalar_v2_m_s=r.bedrock_velocity_m_s,
    )
    arrays: dict[str, np.ndarray] = {
        'source_endpoint_key': _string_array(r.source_endpoint_key),
        'source_id': _int_array(r.source_id),
        'source_node_id': _int_array(r.source_node_id),
        'source_v2_cell_id': _endpoint_cell_id_array(
            r.source_v2_cell_id,
            values.n_source_endpoints,
        ),
        'source_v2_status': _endpoint_v2_status_array(
            r.source_v2_status,
            values.n_source_endpoints,
        ),
        'source_x_m': _float_array(r.source_x_m),
        'source_y_m': _float_array(r.source_y_m),
        'source_surface_elevation_m': _float_array(r.source_surface_elevation_m),
        'source_t1_s': source_t1_s,
        'source_v1_m_s': _filled_float_array(
            r.weathering_velocity_m_s,
            values.n_source_endpoints,
        ),
        'source_v2_m_s': source_v2,
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
        'receiver_v2_cell_id': _endpoint_cell_id_array(
            r.receiver_v2_cell_id,
            values.n_receiver_endpoints,
        ),
        'receiver_v2_status': _endpoint_v2_status_array(
            r.receiver_v2_status,
            values.n_receiver_endpoints,
        ),
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
        'receiver_v2_m_s': receiver_v2,
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
    if _has_source_2layer_static_fields(r):
        assert r.source_t2_time_s is not None
        assert r.source_v3_m_s is not None
        assert r.source_sh2_weathering_thickness_m is not None
        arrays.update(
            {
                'source_t2_s': _float_array(r.source_t2_time_s),
                'source_v3_m_s': _float_array(r.source_v3_m_s),
                'source_sh2_m': _float_array(
                    r.source_sh2_weathering_thickness_m
                ),
            }
        )
    if _has_receiver_2layer_static_fields(r):
        assert r.receiver_t2_time_s is not None
        assert r.receiver_v3_m_s is not None
        assert r.receiver_sh2_weathering_thickness_m is not None
        arrays.update(
            {
                'receiver_t2_s': _float_array(r.receiver_t2_time_s),
                'receiver_v3_m_s': _float_array(r.receiver_v3_m_s),
                'receiver_sh2_m': _float_array(
                    r.receiver_sh2_weathering_thickness_m
                ),
            }
        )
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
        'source_v2_cell_id_sorted': _endpoint_cell_id_array(
            r.source_v2_cell_id_sorted,
            values.n_traces,
        ),
        'source_v2_m_s_sorted': _endpoint_v2_m_s(
            r.source_v2_m_s_sorted,
            shape=values.n_traces,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'source_v2_status_sorted': _endpoint_v2_status_array(
            r.source_v2_status_sorted,
            values.n_traces,
        ),
        'receiver_weathering_replacement_shift_s_sorted': _float_array(
            r.receiver_weathering_replacement_shift_s_sorted
        ),
        'receiver_v2_cell_id_sorted': _endpoint_cell_id_array(
            r.receiver_v2_cell_id_sorted,
            values.n_traces,
        ),
        'receiver_v2_m_s_sorted': _endpoint_v2_m_s(
            r.receiver_v2_m_s_sorted,
            shape=values.n_traces,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'receiver_v2_status_sorted': _endpoint_v2_status_array(
            r.receiver_v2_status_sorted,
            values.n_traces,
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
        'node_v2_cell_id': _endpoint_cell_id_array(r.node_v2_cell_id, values.n_nodes),
        'node_v2_m_s': _endpoint_v2_m_s(
            r.node_v2_m_s,
            shape=values.n_nodes,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'node_v2_status': _endpoint_v2_status_array(
            r.node_v2_status,
            values.n_nodes,
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
        'source_v2_cell_id': _endpoint_cell_id_array(
            r.source_v2_cell_id,
            values.n_source_endpoints,
        ),
        'source_v2_m_s': _endpoint_v2_m_s(
            r.source_v2_m_s,
            shape=values.n_source_endpoints,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'source_v2_status': _endpoint_v2_status_array(
            r.source_v2_status,
            values.n_source_endpoints,
        ),
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
        'receiver_v2_cell_id': _endpoint_cell_id_array(
            r.receiver_v2_cell_id,
            values.n_receiver_endpoints,
        ),
        'receiver_v2_m_s': _endpoint_v2_m_s(
            r.receiver_v2_m_s,
            shape=values.n_receiver_endpoints,
            scalar_v2_m_s=r.bedrock_velocity_m_s,
        ),
        'receiver_v2_status': _endpoint_v2_status_array(
            r.receiver_v2_status,
            values.n_receiver_endpoints,
        ),
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
    if _has_source_2layer_static_fields(r):
        assert r.source_t2_time_s is not None
        assert r.source_v3_m_s is not None
        assert r.source_sh2_weathering_thickness_m is not None
        arrays.update(
            {
                'source_t2_time_s': _float_array(r.source_t2_time_s),
                'source_v3_m_s': _float_array(r.source_v3_m_s),
                'source_sh2_weathering_thickness_m': _float_array(
                    r.source_sh2_weathering_thickness_m
                ),
            }
        )
    if _has_receiver_2layer_static_fields(r):
        assert r.receiver_t2_time_s is not None
        assert r.receiver_v3_m_s is not None
        assert r.receiver_sh2_weathering_thickness_m is not None
        arrays.update(
            {
                'receiver_t2_time_s': _float_array(r.receiver_t2_time_s),
                'receiver_v3_m_s': _float_array(r.receiver_v3_m_s),
                'receiver_sh2_weathering_thickness_m': _float_array(
                    r.receiver_sh2_weathering_thickness_m
                ),
            }
        )
    _validate_no_object_arrays(arrays, artifact_name=REFRACTION_STATIC_SOLUTION_NPZ_NAME)
    return arrays


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
    )
    artifact_entries = _artifact_entries_for_request(
        req,
        first_layer,
        upstream_artifact_names=upstream_names,
    )
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
    if r.bedrock_velocity_mode == 'solve_cell':
        refractor_cell = req.model.refractor_cell
        if refractor_cell is None:
            raise RefractionStaticArtifactError(
                'model.refractor_cell is required for solve_cell QC'
            )
        coordinate_metadata = refraction_cell_coordinate_metadata_from_config(
            refractor_cell
        )
        payload['velocity']['cell_velocity_qc_artifact'] = (
            REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME
        )
        payload['refractor_velocity_cells'] = {
            **coordinate_metadata,
            'cells_csv_artifact': REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
            'grid_npz_artifact': REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
            'qc_json_artifact': REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
            'solver_history_csv_artifact': REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
        }
    layer_qc = r.qc.get('layers')
    if isinstance(layer_qc, dict):
        payload['layers'] = layer_qc
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
    _validate_optional_arrays(
        result=result,
        names=_SOURCE_2LAYER_STATIC_ARRAY_NAMES,
        expected_length=n_source,
        label='source two-layer endpoint',
    )
    n_receiver = _length(result.receiver_endpoint_key, name='receiver_endpoint_key')
    for name in _RECEIVER_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_receiver:
            raise RefractionStaticArtifactError(
                f'receiver endpoint array length mismatch for {name}'
            )
    _validate_optional_arrays(
        result=result,
        names=_RECEIVER_2LAYER_STATIC_ARRAY_NAMES,
        expected_length=n_receiver,
        label='receiver two-layer endpoint',
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
    if result.bedrock_velocity_mode == 'solve_cell':
        _validate_solve_cell_local_v2_arrays(
            result=result,
            n_traces=n_traces,
            n_nodes=n_nodes,
            n_source=n_source,
            n_receiver=n_receiver,
            n_rows=n_rows,
        )
    return _ValidatedResult(
        result=result,
        n_traces=n_traces,
        n_nodes=n_nodes,
        n_source_endpoints=n_source,
        n_receiver_endpoints=n_receiver,
        n_rows=n_rows,
    )


def _validate_optional_arrays(
    *,
    result: RefractionDatumStaticsResult,
    names: tuple[str, ...],
    expected_length: int,
    label: str,
) -> None:
    present = [name for name in names if getattr(result, name) is not None]
    if not present:
        return
    if len(present) != len(names):
        missing = ', '.join(name for name in names if name not in present)
        raise RefractionStaticArtifactError(
            f'{label} arrays must be provided together; missing {missing}'
        )
    for name in names:
        if _length(getattr(result, name), name=name) != expected_length:
            raise RefractionStaticArtifactError(
                f'{label} array length mismatch for {name}'
            )


def _validate_solve_cell_local_v2_arrays(
    *,
    result: RefractionDatumStaticsResult,
    n_traces: int,
    n_nodes: int,
    n_source: int,
    n_receiver: int,
    n_rows: int,
) -> None:
    expected_lengths = {
        'node_v2_cell_id': n_nodes,
        'node_v2_m_s': n_nodes,
        'node_v2_status': n_nodes,
        'source_v2_cell_id': n_source,
        'source_v2_m_s': n_source,
        'source_v2_status': n_source,
        'receiver_v2_cell_id': n_receiver,
        'receiver_v2_m_s': n_receiver,
        'receiver_v2_status': n_receiver,
        'source_v2_cell_id_sorted': n_traces,
        'source_v2_m_s_sorted': n_traces,
        'source_v2_status_sorted': n_traces,
        'receiver_v2_cell_id_sorted': n_traces,
        'receiver_v2_m_s_sorted': n_traces,
        'receiver_v2_status_sorted': n_traces,
    }
    for name, expected_length in expected_lengths.items():
        value = getattr(result, name)
        if value is None:
            raise RefractionStaticArtifactError(
                f'solve_cell result requires {name}'
            )
        if _length(value, name=name) != expected_length:
            raise RefractionStaticArtifactError(
                f'solve_cell local V2 array length mismatch for {name}'
            )
        if name.endswith('_status'):
            _validate_status_array(value, name=name)
    active_cell_id = _required_cell_int_array(
        result.active_cell_id,
        name='active_cell_id',
    )
    for name in (
        'cell_bedrock_slowness_s_per_m',
        'cell_bedrock_velocity_m_s',
        'cell_velocity_status',
    ):
        value = getattr(result, name)
        if value is None:
            raise RefractionStaticArtifactError(f'solve_cell result requires {name}')
        if _length(value, name=name) != int(active_cell_id.shape[0]):
            raise RefractionStaticArtifactError(
                f'solve_cell cell array length mismatch for {name}'
            )
        if name.endswith('_status'):
            _validate_status_array(value, name=name)
    inactive_cell_id = _required_cell_int_array(
        result.inactive_cell_id,
        name='inactive_cell_id',
    )
    if np.intersect1d(active_cell_id, inactive_cell_id).size:
        raise RefractionStaticArtifactError(
            'active_cell_id and inactive_cell_id must not overlap'
        )
    row_midpoint_cell_id = _required_cell_int_array(
        result.row_midpoint_cell_id,
        name='row_midpoint_cell_id',
    )
    if int(row_midpoint_cell_id.shape[0]) != n_rows:
        raise RefractionStaticArtifactError(
            'solve_cell row_midpoint_cell_id length mismatch'
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

_SOURCE_2LAYER_STATIC_ARRAY_NAMES = (
    'source_t2_time_s',
    'source_v3_m_s',
    'source_sh2_weathering_thickness_m',
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

_RECEIVER_2LAYER_STATIC_ARRAY_NAMES = (
    'receiver_t2_time_s',
    'receiver_v3_m_s',
    'receiver_sh2_weathering_thickness_m',
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


def _first_break_residual_rows(
    result: RefractionDatumStaticsResult,
    *,
    req: RefractionStaticApplyRequest | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cell_id_by_row, cell_ix_by_row, cell_iy_by_row = _residual_row_cell_context(
        result,
        req=req,
    )
    for row_index in range(int(result.row_trace_index_sorted.shape[0])):
        rejected_by_robust = bool(result.rejected_by_robust_mask[row_index])
        used = bool(result.used_row_mask[row_index])
        rows.append(
            {
                'row_index': row_index,
                'observation_index': row_index,
                'sorted_trace_index': int(result.row_trace_index_sorted[row_index]),
                'source_node_id': int(result.row_source_node_id[row_index]),
                'receiver_node_id': int(result.row_receiver_node_id[row_index]),
                'distance_m': _csv_float(result.row_distance_m[row_index]),
                'observed_pick_time_ms': _csv_ms(result.observed_pick_time_s[row_index]),
                'observed_pick_time_s': _csv_float(result.observed_pick_time_s[row_index]),
                'modeled_pick_time_ms': _csv_ms(result.modeled_pick_time_s[row_index]),
                'modeled_pick_time_s': _csv_float(result.modeled_pick_time_s[row_index]),
                'residual_ms': _csv_ms(result.residual_time_s[row_index]),
                'residual_s': _csv_float(result.residual_time_s[row_index]),
                'used': _csv_bool(used),
                'used_in_solve': _csv_bool(used),
                'rejected_by_robust': _csv_bool(rejected_by_robust),
                'rejection_reason': _residual_rejection_reason(
                    used=used,
                    rejected_by_robust=rejected_by_robust,
                ),
                'cell_id': _csv_cell_id(cell_id_by_row[row_index]),
                'cell_ix': _csv_cell_id(cell_ix_by_row[row_index]),
                'cell_iy': _csv_cell_id(cell_iy_by_row[row_index]),
            }
        )
    return rows


def _residual_row_cell_context(
    result: RefractionDatumStaticsResult,
    *,
    req: RefractionStaticApplyRequest | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    empty = np.full(n_rows, -1, dtype=np.int64)
    if result.bedrock_velocity_mode != 'solve_cell':
        return empty, empty.copy(), empty.copy()

    if result.row_midpoint_cell_id is None:
        raise RefractionStaticArtifactError(
            'solve_cell residual rows require row_midpoint_cell_id'
        )
    cell_id = np.ascontiguousarray(result.row_midpoint_cell_id, dtype=np.int64)
    if cell_id.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            'row_midpoint_cell_id length must match residual rows'
        )
    number_of_cell_x = _residual_cell_x_count(result=result, req=req)
    cell_ix = np.full(n_rows, -1, dtype=np.int64)
    cell_iy = np.full(n_rows, -1, dtype=np.int64)
    if number_of_cell_x is not None:
        valid = cell_id >= 0
        cell_ix[valid] = cell_id[valid] % number_of_cell_x
        cell_iy[valid] = cell_id[valid] // number_of_cell_x
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(cell_ix, dtype=np.int64),
        np.ascontiguousarray(cell_iy, dtype=np.int64),
    )


def _residual_cell_x_count(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None,
) -> int | None:
    if req is not None:
        request = RefractionStaticApplyRequest.model_validate(req)
        refractor_cell = request.model.refractor_cell
        if (
            request.model.bedrock_velocity_mode == 'solve_cell'
            and refractor_cell is not None
        ):
            return int(
                effective_refraction_cell_grid_config(
                    refractor_cell
                ).number_of_cell_x
            )
    raw = result.qc.get('number_of_cell_x')
    if raw is None:
        return None
    return _required_positive_qc_int(result.qc, 'number_of_cell_x')


def _residual_rejection_reason(
    *,
    used: bool,
    rejected_by_robust: bool,
) -> str:
    if rejected_by_robust:
        return 'robust_outlier'
    if not used:
        return 'not_used'
    return 'ok'


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


def _source_static_table_columns(
    result: RefractionDatumStaticsResult,
) -> tuple[str, ...]:
    if _has_source_2layer_static_fields(result):
        return _SOURCE_STATIC_TABLE_2LAYER_COLUMNS
    return _SOURCE_STATIC_TABLE_COLUMNS


def _receiver_static_table_columns(
    result: RefractionDatumStaticsResult,
) -> tuple[str, ...]:
    if _has_receiver_2layer_static_fields(result):
        return _RECEIVER_STATIC_TABLE_2LAYER_COLUMNS
    return _RECEIVER_STATIC_TABLE_COLUMNS


def _has_source_2layer_static_fields(result: RefractionDatumStaticsResult) -> bool:
    return all(
        getattr(result, name) is not None for name in _SOURCE_2LAYER_STATIC_ARRAY_NAMES
    )


def _has_receiver_2layer_static_fields(result: RefractionDatumStaticsResult) -> bool:
    return all(
        getattr(result, name) is not None
        for name in _RECEIVER_2LAYER_STATIC_ARRAY_NAMES
    )


def _source_static_table_rows(
    result: RefractionDatumStaticsResult,
) -> list[dict[str, object]]:
    node_context = _node_context(result)
    static_status = _source_static_status_array(result)
    flat_datum = _nan_if_none(result.flat_datum_elevation_m)
    source_v2 = _endpoint_v2_m_s(
        result.source_v2_m_s,
        shape=int(result.source_endpoint_key.shape[0]),
        scalar_v2_m_s=result.bedrock_velocity_m_s,
    )
    source_v2_cell_id = _endpoint_cell_id_array(
        result.source_v2_cell_id,
        int(result.source_endpoint_key.shape[0]),
    )
    source_v2_status = _endpoint_v2_status_array(
        result.source_v2_status,
        int(result.source_endpoint_key.shape[0]),
    )
    has_2layer_fields = _has_source_2layer_static_fields(result)
    source_t2_time_s = result.source_t2_time_s
    source_v3_m_s = result.source_v3_m_s
    source_sh2_m = result.source_sh2_weathering_thickness_m
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        node_id = int(result.source_node_id[index])
        t1_s = result.source_half_intercept_time_s[index]
        sh1_m = result.source_weathering_thickness_m[index]
        weathering_correction_s = result.source_weathering_replacement_shift_s[index]
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
                'source_v2_cell_id': _csv_cell_id(source_v2_cell_id[index]),
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
                'v2_m_s': _csv_float(source_v2[index]),
                'v2_status': str(source_v2_status[index]),
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
        if has_2layer_fields:
            assert source_t2_time_s is not None
            assert source_v3_m_s is not None
            assert source_sh2_m is not None
            rows[-1].update(
                {
                    't2_ms': _csv_ms(source_t2_time_s[index]),
                    'v3_m_s': _csv_float(source_v3_m_s[index]),
                    'sh2_weathering_thickness_m': _csv_float(source_sh2_m[index]),
                }
            )
    return rows


def _receiver_static_table_rows(
    result: RefractionDatumStaticsResult,
) -> list[dict[str, object]]:
    node_context = _node_context(result)
    static_status = _receiver_static_status_array(result)
    flat_datum = _nan_if_none(result.flat_datum_elevation_m)
    receiver_v2 = _endpoint_v2_m_s(
        result.receiver_v2_m_s,
        shape=int(result.receiver_endpoint_key.shape[0]),
        scalar_v2_m_s=result.bedrock_velocity_m_s,
    )
    receiver_v2_cell_id = _endpoint_cell_id_array(
        result.receiver_v2_cell_id,
        int(result.receiver_endpoint_key.shape[0]),
    )
    receiver_v2_status = _endpoint_v2_status_array(
        result.receiver_v2_status,
        int(result.receiver_endpoint_key.shape[0]),
    )
    has_2layer_fields = _has_receiver_2layer_static_fields(result)
    receiver_t2_time_s = result.receiver_t2_time_s
    receiver_v3_m_s = result.receiver_v3_m_s
    receiver_sh2_m = result.receiver_sh2_weathering_thickness_m
    rows: list[dict[str, object]] = []
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        node_id = int(result.receiver_node_id[index])
        t1_s = result.receiver_half_intercept_time_s[index]
        sh1_m = result.receiver_weathering_thickness_m[index]
        weathering_correction_s = result.receiver_weathering_replacement_shift_s[index]
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
                'receiver_v2_cell_id': _csv_cell_id(receiver_v2_cell_id[index]),
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
                'v2_m_s': _csv_float(receiver_v2[index]),
                'v2_status': str(receiver_v2_status[index]),
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
        if has_2layer_fields:
            assert receiver_t2_time_s is not None
            assert receiver_v3_m_s is not None
            assert receiver_sh2_m is not None
            rows[-1].update(
                {
                    't2_ms': _csv_ms(receiver_t2_time_s[index]),
                    'v3_m_s': _csv_float(receiver_v3_m_s[index]),
                    'sh2_weathering_thickness_m': _csv_float(receiver_sh2_m[index]),
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
        t1_s=result.source_half_intercept_time_s,
        weathering_thickness_m=result.source_weathering_thickness_m,
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
        t1_s=result.receiver_half_intercept_time_s,
        weathering_thickness_m=result.receiver_weathering_thickness_m,
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
    return classify_refraction_endpoint_static_status(
        node_missing=node_missing,
        x_m=x_m,
        y_m=y_m,
        surface_elevation_m=surface_elevation_m,
        t1_s=t1_s,
        weathering_thickness_m=weathering_thickness_m,
        total_shift_s=total_shift_s,
        solution_status=solution_status,
        weathering_status=weathering_status,
        datum_status=datum_status,
    )


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
    *,
    upstream_artifact_names: Iterable[str] = (),
) -> tuple[dict[str, str | bool], ...]:
    return (
        _ARTIFACTS
        + _refractor_cell_velocity_artifact_entries(req)
        + _t1lsst_artifact_entries(req)
        + _upstream_artifact_entries(
            _validate_upstream_artifact_names(
                upstream_artifact_names,
                resolved_first_layer=resolved_first_layer,
            )
        )
    )


def _refractor_cell_velocity_artifact_entries(
    req: RefractionStaticApplyRequest,
) -> tuple[dict[str, str | bool], ...]:
    if req.model.bedrock_velocity_mode == 'solve_cell':
        return _REFRACTOR_CELL_VELOCITY_ARTIFACTS
    return ()


def _t1lsst_artifact_entries(
    req: RefractionStaticApplyRequest,
) -> tuple[dict[str, str | bool], ...]:
    if req.conversion.mode == 't1lsst_1layer':
        return _T1LSST_1LAYER_ARTIFACTS
    return ()


def _validate_upstream_artifact_names(
    names: Iterable[str],
    *,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
) -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for name in names:
        if not isinstance(name, str):
            raise RefractionStaticArtifactError(
                'upstream artifact names must be strings'
            )
        if name in seen:
            continue
        if name not in _V1_ARTIFACT_NAMES:
            raise RefractionStaticArtifactError(
                f'unsupported upstream artifact: {name}'
            )
        seen.add(name)
        values.append(name)

    if not values:
        return ()

    mode = (
        resolved_first_layer.mode
        if resolved_first_layer is not None
        else None
    )
    if mode != 'estimate_direct_arrival':
        raise RefractionStaticArtifactError(
            'upstream V1 artifacts are only valid when first-layer mode is '
            'estimate_direct_arrival'
        )

    value_set = set(values)
    if value_set != _V1_ARTIFACT_NAMES:
        expected = ', '.join(sorted(_V1_ARTIFACT_NAMES))
        raise RefractionStaticArtifactError(
            f'upstream V1 artifacts must include both: {expected}'
        )
    return tuple(
        str(item['name']) for item in _V1_ARTIFACTS if str(item['name']) in value_set
    )


def _validate_declared_upstream_artifacts(
    root: Path,
    names: tuple[str, ...],
) -> None:
    for name in names:
        artifact_path = root / name
        if not artifact_path.is_file():
            raise RefractionStaticArtifactError(
                f'declared upstream artifact missing: {name}'
            )


def _upstream_artifact_entries(
    names: tuple[str, ...],
) -> tuple[dict[str, str | bool], ...]:
    name_set = set(names)
    return tuple(item for item in _V1_ARTIFACTS if str(item['name']) in name_set)


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
                'origin': str(item.get('origin', 'final')),
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


def _refractor_velocity_cell_rows(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_cells = int(arrays['cell_id'].shape[0])
    for index in range(n_cells):
        rows.append(
            {
                'cell_id': int(arrays['cell_id'][index]),
                'ix': int(arrays['ix'][index]),
                'iy': int(arrays['iy'][index]),
                'cell_ix': int(arrays['cell_ix'][index]),
                'cell_iy': int(arrays['cell_iy'][index]),
                'coordinate_mode': str(arrays['coordinate_mode'][index]),
                'x_min_m': _csv_grid_float(arrays['x_min_m'][index]),
                'x_max_m': _csv_grid_float(arrays['x_max_m'][index]),
                'y_min_m': _csv_grid_float(arrays['y_min_m'][index]),
                'y_max_m': _csv_grid_float(arrays['y_max_m'][index]),
                'x_center_m': _csv_grid_float(arrays['x_center_m'][index]),
                'y_center_m': _csv_grid_float(arrays['y_center_m'][index]),
                'cell_center_x_m': _csv_grid_float(
                    arrays['cell_center_x_m'][index]
                ),
                'cell_center_y_m': _csv_grid_float(
                    arrays['cell_center_y_m'][index]
                ),
                'cell_center_inline_m': _csv_grid_float(
                    arrays['cell_center_inline_m'][index]
                ),
                'cell_center_crossline_m': _csv_grid_float(
                    arrays['cell_center_crossline_m'][index]
                ),
                'active': _csv_bool(arrays['active_cell_mask'][index]),
                'n_observations': int(
                    arrays['n_observations_per_cell'][index]
                ),
                'n_used_observations': int(
                    arrays['n_used_observations_per_cell'][index]
                ),
                'n_rejected_observations': int(
                    arrays['n_rejected_observations_per_cell'][index]
                ),
                'n_sources': int(arrays['n_sources_per_cell'][index]),
                'n_receivers': int(arrays['n_receivers_per_cell'][index]),
                'v2_m_s': _csv_float(arrays['v2_m_s'][index]),
                'slowness_s_per_m': _csv_float(
                    arrays['slowness_s_per_m'][index]
                ),
                'initial_v2_m_s': _csv_float(arrays['initial_v2_m_s'][index]),
                'v2_update_from_initial_m_s': _csv_float(
                    arrays['v2_update_from_initial_m_s'][index]
                ),
                'velocity_status': str(arrays['velocity_status'][index]),
                'status_reason': str(arrays['status_reason'][index]),
                'residual_rms_ms': _csv_float(arrays['residual_rms_ms'][index]),
                'residual_mad_ms': _csv_float(arrays['residual_mad_ms'][index]),
                'residual_mean_ms': _csv_float(arrays['residual_mean_ms'][index]),
                'residual_p95_abs_ms': _csv_float(
                    arrays['residual_p95_abs_ms'][index]
                ),
                'smoothing_enabled': _csv_bool(arrays['smoothing_enabled'][index]),
                'smoothing_weight': _csv_float(arrays['smoothing_weight'][index]),
                'smoothing_neighbor_count': int(
                    arrays['smoothing_neighbor_count'][index]
                ),
            }
        )
    return rows


def _cell_solver_history_csv_row(
    row: RefractionCellSolverHistoryRow,
) -> dict[str, object]:
    return {
        'iteration': int(row.iteration),
        'stage': row.stage,
        'n_candidate_observations': int(row.n_candidate_observations),
        'n_used_observations': int(row.n_used_observations),
        'n_rejected_observations': int(row.n_rejected_observations),
        'n_active_cells': int(row.n_active_cells),
        'n_low_fold_cells': int(row.n_low_fold_cells),
        'n_empty_cells': int(row.n_empty_cells),
        'residual_rms_ms': _csv_float(row.residual_rms_ms),
        'residual_mad_ms': _csv_float(row.residual_mad_ms),
        'max_abs_residual_ms': _csv_float(row.max_abs_residual_ms),
        'median_v2_m_s': _csv_float(row.median_v2_m_s),
        'min_v2_m_s': _csv_float(row.min_v2_m_s),
        'max_v2_m_s': _csv_float(row.max_v2_m_s),
        'max_abs_v2_update_m_s': _csv_float(row.max_abs_v2_update_m_s),
        'smoothing_weight': _csv_float(row.smoothing_weight),
        'damping_weight': _csv_float(row.damping_weight),
        'robust_threshold': _csv_float(row.robust_threshold),
        'converged': _csv_bool(row.converged),
        'convergence_reason': row.convergence_reason,
    }


def _cell_solver_history_cell_counts(
    arrays: dict[str, np.ndarray],
) -> dict[str, int]:
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)
    velocity_status = np.asarray(arrays['velocity_status']).astype(str, copy=False)
    n_observations = np.asarray(arrays['n_observations_per_cell'], dtype=np.int64)
    return {
        'active': int(np.count_nonzero(active_mask)),
        'low_fold': int(
            np.count_nonzero(velocity_status == LOW_FOLD_CELL_VELOCITY_STATUS)
        ),
        'empty': int(np.count_nonzero(n_observations == 0)),
    }


def _history_residual_stats_ms(
    result: RefractionDatumStaticsResult,
) -> dict[str, float | None]:
    residual_ms = np.asarray(result.residual_time_s, dtype=np.float64)[
        np.asarray(result.used_row_mask, dtype=bool)
    ] * 1000.0
    return {
        'rms': _residual_stat(residual_ms, 'rms'),
        'mad': _residual_stat(residual_ms, 'mad'),
        'max_abs': _residual_stat(residual_ms, 'max_abs'),
    }


def _history_max_abs(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.max(np.abs(finite)))


def _initial_cell_v2_m_s(req: RefractionStaticApplyRequest) -> float:
    value = req.model.initial_bedrock_velocity_m_s
    if value is not None:
        return float(value)
    return 0.5 * (
        float(req.model.min_bedrock_velocity_m_s)
        + float(req.model.max_bedrock_velocity_m_s)
    )


def _history_smoothing_weight(req: RefractionStaticApplyRequest) -> float:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell solver history'
        )
    return float(refractor_cell.velocity_smoothing_weight)


def _history_robust_iteration_count(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> int:
    raw = result.qc.get('robust_iteration_count')
    if raw is not None:
        return int(raw)
    if req.solver.robust.enabled and np.count_nonzero(result.rejected_by_robust_mask):
        return 1
    return 0


def _history_convergence_reason(
    *,
    robust_iteration_count: int,
    n_robust_rejected_observations: int,
    smoothing_weight: float,
) -> str:
    if robust_iteration_count > 0 or n_robust_rejected_observations > 0:
        return 'robust_reweight_converged'
    if smoothing_weight > 0.0:
        return 'smoothed_least_squares_converged'
    return 'least_squares_converged'


def _refractor_cell_center_alias_arrays(
    *,
    x_center_m: np.ndarray,
    y_center_m: np.ndarray,
    coordinate_mode: str,
    line_origin_x_m: float | None,
    line_origin_y_m: float | None,
    line_azimuth_deg: float | None,
) -> dict[str, np.ndarray]:
    x_center = np.asarray(x_center_m, dtype=np.float64)
    y_center = np.asarray(y_center_m, dtype=np.float64)
    if x_center.shape != y_center.shape:
        raise RefractionStaticArtifactError(
            'refractor cell center coordinate shape mismatch'
        )

    inline = np.full(x_center.shape, np.nan, dtype=np.float64)
    crossline = np.full(x_center.shape, np.nan, dtype=np.float64)
    center_x = np.ascontiguousarray(x_center, dtype=np.float64)
    center_y = np.ascontiguousarray(y_center, dtype=np.float64)
    if coordinate_mode == 'line_2d_projected':
        origin_x = _required_finite_float(
            line_origin_x_m,
            name='model.refractor_cell.line_origin_x_m',
        )
        origin_y = _required_finite_float(
            line_origin_y_m,
            name='model.refractor_cell.line_origin_y_m',
        )
        azimuth_deg = _required_finite_float(
            line_azimuth_deg,
            name='model.refractor_cell.line_azimuth_deg',
        )
        inline = np.ascontiguousarray(x_center, dtype=np.float64)
        crossline = np.ascontiguousarray(y_center, dtype=np.float64)
        azimuth_rad = np.deg2rad(azimuth_deg)
        inline_unit_x = float(np.sin(azimuth_rad))
        inline_unit_y = float(np.cos(azimuth_rad))
        center_x = np.ascontiguousarray(
            origin_x + inline * inline_unit_x + crossline * inline_unit_y,
            dtype=np.float64,
        )
        center_y = np.ascontiguousarray(
            origin_y + inline * inline_unit_y - crossline * inline_unit_x,
            dtype=np.float64,
        )

    return {
        'x_m': center_x,
        'y_m': center_y,
        'inline_m': np.ascontiguousarray(inline, dtype=np.float64),
        'crossline_m': np.ascontiguousarray(crossline, dtype=np.float64),
    }


def _unique_observation_endpoint_count_by_cell(
    *,
    row_midpoint_cell_id: np.ndarray,
    endpoint_id: np.ndarray,
    n_total_cells: int,
) -> np.ndarray:
    row_cell = np.asarray(row_midpoint_cell_id, dtype=np.int64)
    endpoint = np.asarray(endpoint_id, dtype=np.int64)
    if row_cell.shape != endpoint.shape:
        raise RefractionStaticArtifactError(
            'row endpoint arrays must match row_midpoint_cell_id shape'
        )
    counts = np.zeros(int(n_total_cells), dtype=np.int64)
    for cell_id in range(int(n_total_cells)):
        in_cell = row_cell == cell_id
        if np.any(in_cell):
            counts[cell_id] = int(np.unique(endpoint[in_cell]).shape[0])
    return np.ascontiguousarray(counts, dtype=np.int64)


def _refractor_cell_status_reasons(
    *,
    velocity_status: np.ndarray,
    n_observations: np.ndarray,
) -> np.ndarray:
    status = np.asarray(velocity_status).astype(str, copy=False)
    counts = np.asarray(n_observations, dtype=np.int64)
    if status.shape != counts.shape:
        raise RefractionStaticArtifactError(
            'velocity_status and n_observations shape mismatch'
        )
    reasons: list[str] = []
    for value, count in zip(status.tolist(), counts.tolist(), strict=True):
        if value == LOW_FOLD_CELL_VELOCITY_STATUS:
            reasons.append(LOW_FOLD_CELL_REJECTION_REASON)
        elif value == 'inactive':
            reasons.append('no_observations' if int(count) == 0 else 'inactive_cell')
        else:
            reasons.append(str(value))
    return _string_array(reasons)


def _required_cell_int_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'solve_cell result requires {name}')
    return _int_array(value)


def _required_cell_float_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'solve_cell result requires {name}')
    return _float_array(value)


def _required_cell_status_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'solve_cell result requires {name}')
    status = _string_array(value)
    _validate_status_array(status, name=name)
    return status


def _validate_refractor_velocity_cell_ids(
    *,
    grid_cell_id: np.ndarray,
    active_cell_id: np.ndarray,
    inactive_cell_id: np.ndarray,
) -> None:
    grid_ids = {int(value) for value in np.asarray(grid_cell_id).tolist()}
    active_ids = [int(value) for value in np.asarray(active_cell_id).tolist()]
    inactive_ids = [int(value) for value in np.asarray(inactive_cell_id).tolist()]
    combined = active_ids + inactive_ids
    if len(combined) != len(set(combined)):
        raise RefractionStaticArtifactError(
            'active and inactive refractor cell IDs must be unique'
        )
    combined_ids = set(combined)
    if combined_ids != grid_ids:
        missing = sorted(grid_ids - combined_ids)
        extra = sorted(combined_ids - grid_ids)
        raise RefractionStaticArtifactError(
            'solve_cell refractor cell IDs do not cover the configured grid: '
            f'missing={missing}, extra={extra}'
        )


def _active_neighbor_count_by_cell(
    *,
    active_cell_id: np.ndarray,
    n_total_cells: int,
    number_of_cell_x: int,
    number_of_cell_y: int,
) -> np.ndarray:
    active = {int(value) for value in np.asarray(active_cell_id).tolist()}
    counts = np.zeros(int(n_total_cells), dtype=np.int64)
    for cell_id in active:
        ix = cell_id % int(number_of_cell_x)
        iy = cell_id // int(number_of_cell_x)
        neighbors = []
        if ix > 0:
            neighbors.append(cell_id - 1)
        if ix + 1 < int(number_of_cell_x):
            neighbors.append(cell_id + 1)
        if iy > 0:
            neighbors.append(cell_id - int(number_of_cell_x))
        if iy + 1 < int(number_of_cell_y):
            neighbors.append(cell_id + int(number_of_cell_x))
        counts[cell_id] = sum(1 for neighbor in neighbors if neighbor in active)
    return np.ascontiguousarray(counts, dtype=np.int64)


def _estimated_cell_smoothing_rows(
    *,
    active_cell_mask: np.ndarray,
    number_of_cell_x: int,
    number_of_cell_y: int,
    velocity_smoothing_weight: float,
) -> int:
    if float(velocity_smoothing_weight) == 0.0:
        return 0
    active = {
        index
        for index, enabled in enumerate(np.asarray(active_cell_mask, dtype=bool))
        if bool(enabled)
    }
    n_edges = 0
    for cell_id in active:
        ix = cell_id % int(number_of_cell_x)
        iy = cell_id // int(number_of_cell_x)
        right = cell_id + 1
        down = cell_id + int(number_of_cell_x)
        if ix + 1 < int(number_of_cell_x) and right in active:
            n_edges += 1
        if iy + 1 < int(number_of_cell_y) and down in active:
            n_edges += 1
    return n_edges


def _per_cell_residual_stats_ms(
    *,
    row_midpoint_cell_id: np.ndarray,
    residual_time_s: np.ndarray,
    used_row_mask: np.ndarray,
    n_total_cells: int,
) -> dict[str, np.ndarray]:
    row_cell = np.asarray(row_midpoint_cell_id, dtype=np.int64)
    residual_ms = np.asarray(residual_time_s, dtype=np.float64) * 1000.0
    used = np.asarray(used_row_mask, dtype=bool)
    out = {
        'rms': np.full(int(n_total_cells), np.nan, dtype=np.float64),
        'mad': np.full(int(n_total_cells), np.nan, dtype=np.float64),
        'mean': np.full(int(n_total_cells), np.nan, dtype=np.float64),
        'p95_abs': np.full(int(n_total_cells), np.nan, dtype=np.float64),
    }
    for cell_id in range(int(n_total_cells)):
        values = residual_ms[(row_cell == cell_id) & used]
        for stat, key in (
            ('rms', 'rms'),
            ('mad', 'mad'),
            ('mean', 'mean'),
            ('p95_abs', 'p95_abs'),
        ):
            stat_value = _residual_stat(values, stat)
            if stat_value is not None:
                out[key][cell_id] = float(stat_value)
    return {
        key: np.ascontiguousarray(value, dtype=np.float64)
        for key, value in out.items()
    }


def _qc_int(qc: dict[str, Any], key: str, *, default: int) -> int:
    raw = qc.get(key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'QC field {key} must be an integer'
        ) from exc


def _required_positive_qc_int(qc: dict[str, Any], key: str) -> int:
    raw = qc.get(key)
    if raw is None:
        raise RefractionStaticArtifactError(f'QC field {key} is required')
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'QC field {key} must be an integer'
        ) from exc
    if value <= 0:
        raise RefractionStaticArtifactError(
            f'QC field {key} must be a positive integer'
        )
    return value


def _qc_cell_id_array(
    qc: dict[str, Any],
    key: str,
    *,
    n_total_cells: int,
) -> np.ndarray:
    raw = qc.get(key, [])
    arr = np.asarray(raw)
    if arr.ndim != 1:
        raise RefractionStaticArtifactError(f'QC field {key} must be one-dimensional')
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    try:
        out = np.ascontiguousarray(arr, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'QC field {key} must contain integer cell IDs'
        ) from exc
    if np.any(out < 0) or np.any(out >= n_total_cells):
        raise RefractionStaticArtifactError(
            f'QC field {key} contains out-of-range cell IDs'
        )
    return out


def _qc_cell_count_array(
    qc: dict[str, Any],
    key: str,
    *,
    n_total_cells: int,
) -> np.ndarray | None:
    raw = qc.get(key)
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.shape != (n_total_cells,):
        raise RefractionStaticArtifactError(
            f'QC field {key} must have one count per refractor cell'
        )
    try:
        out = np.ascontiguousarray(arr, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'QC field {key} must contain integer counts'
        ) from exc
    if np.any(out < 0):
        raise RefractionStaticArtifactError(
            f'QC field {key} must not contain negative counts'
        )
    return out


def _qc_optional_float(
    qc: dict[str, Any],
    key: str,
    *,
    default: float | None,
) -> float | None:
    raw = qc.get(key, default)
    return _json_float(raw)


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


def _endpoint_v2_m_s(
    value: object,
    *,
    shape: int,
    scalar_v2_m_s: float,
) -> np.ndarray:
    if value is None:
        return _filled_float_array(scalar_v2_m_s, shape)
    return np.ascontiguousarray(value, dtype=np.float64)


def _endpoint_cell_id_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        return np.full(int(shape), -1, dtype=np.int64)
    return np.ascontiguousarray(value, dtype=np.int64)


def _endpoint_v2_status_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        return _string_array(np.full(int(shape), 'ok', dtype='<U2'))
    return _string_array(value)


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


def _required_finite_float(value: object, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticArtifactError(f'{name} must be finite') from exc
    if not np.isfinite(out):
        raise RefractionStaticArtifactError(f'{name} must be finite')
    return out


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


def _csv_grid_float(value: object) -> str | float:
    if value is None:
        return ''
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ''
    if np.isnan(out):
        return ''
    if np.isposinf(out):
        return 'inf'
    if np.isneginf(out):
        return '-inf'
    return out


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


def _csv_cell_id(value: object) -> str | int:
    out = _csv_int(value)
    if out == '' or int(out) < 0:
        return ''
    return out


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
    'REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME',
    'REFRACTION_STATICS_CSV_NAME',
    'REFRACTION_STATIC_ARTIFACTS_JSON_NAME',
    'REFRACTION_STATIC_COMPONENTS_CSV_NAME',
    'REFRACTION_STATIC_REQUEST_JSON_NAME',
    'REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES',
    'REFRACTION_STATIC_QC_JSON_NAME',
    'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
    'REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME',
    'REFRACTION_V1_ESTIMATES_CSV_NAME',
    'REFRACTION_V1_QC_JSON_NAME',
    'RECEIVER_STATIC_TABLE_CSV_NAME',
    'SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME',
    'SOURCE_STATIC_TABLE_CSV_NAME',
    'RefractionCellSolverHistoryRow',
    'RefractionStaticArtifactError',
    'RefractionStaticArtifactSet',
    'build_refraction_cell_solver_history_rows',
    'build_refraction_refractor_velocity_grid_arrays',
    'build_refraction_refractor_velocity_qc_payload',
    'build_refraction_static_qc_payload',
    'build_refraction_static_solution_arrays',
    'build_source_receiver_static_table_arrays',
    'write_first_break_residuals_csv',
    'write_near_surface_model_csv',
    'write_refraction_cell_solver_history_csv',
    'write_refraction_refractor_velocity_cells_csv',
    'write_refraction_refractor_velocity_grid_npz',
    'write_refraction_refractor_velocity_qc_json',
    'write_refraction_static_artifacts',
    'write_refraction_static_components_csv',
    'write_refraction_static_qc_json',
    'write_refraction_static_solution_npz',
    'write_refraction_statics_csv',
    'write_receiver_static_table_csv',
    'write_source_receiver_static_table_npz',
    'write_source_static_table_csv',
]
