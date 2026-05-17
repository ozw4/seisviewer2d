"""Final artifact package writer for GLI refraction statics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
    refraction_cell_coordinate_metadata_from_config,
)
from app.services.refraction_static_cell_grid import build_refraction_cell_grid
from app.services.refraction_static_cell_grid import assign_observation_midpoint_cells
from app.services.refraction_static_design_matrix import (
    LOW_FOLD_CELL_REJECTION_REASON,
    LOW_FOLD_CELL_VELOCITY_STATUS,
)
from app.services.refraction_static_layer_config import (
    normalize_refraction_static_layers,
)
from app.services.refraction_static_layer_observations import (
    build_refraction_layer_observation_masks_from_arrays,
)
from app.services.refraction_static_status import (
    REFRACTION_STATIC_STATUSES,
    classify_refraction_endpoint_static_status,
)
from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
)
from app.services.refraction_static_t1lsst import (
    REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
    write_refraction_t1lsst_1layer_components_csv,
)
from app.services.refraction_static_types import (
    RefractionLayerKind,
    RefractionDatumStaticsResult,
    RefractionLayerSolveResult,
    RefractionStaticArtifactSet,
    ResolvedRefractionFirstLayer,
)
from app.services.refraction_static_uphole import (
    REFRACTION_UPHOLE_SOURCES_CSV_NAME,
)
from app.services.refraction_static_v1 import (
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
)

from app.services.refraction_static_artifacts.contract import (
    _CELL_SOLVER_HISTORY_COLUMNS,
    _COMPONENT_COLUMNS,
    _FIRST_BREAK_FIT_QC_COLUMNS,
    _FIRST_BREAK_TIME_EXPORT_COLUMNS,
    _GRID_MAP_QC_COLUMNS,
    _LINE_PROFILE_QC_COLUMNS,
    _NEAR_SURFACE_2LAYER_COLUMNS,
    _NEAR_SURFACE_3LAYER_COLUMNS,
    _NEAR_SURFACE_COLUMNS,
    _REDUCED_TIME_QC_COLUMNS,
    _REFRACTOR_VELOCITY_CELL_COLUMNS,
    _RESIDUAL_COLUMNS,
    _STATIC_COMPONENT_QC_ENDPOINT_COLUMNS,
    _STATIC_COMPONENT_QC_TRACE_COLUMNS,
    _TRACE_STATICS_COLUMNS,
    _ValidatedResult,
    ARTIFACT_VERSION,
    FIRST_BREAK_FIT_QC_RESIDUAL_SIGN,
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    FIRST_BREAK_TIME_EXPORT_FORMAT_NAME,
    FIRST_BREAK_TIME_EXPORT_FORMAT_VERSION,
    FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
    LINE_PROFILE_QC_SCHEMA_VERSION,
    METHOD,
    NEAR_SURFACE_MODEL_CSV_NAME,
    NEGATIVE_SHIFT_DESCRIPTION,
    POSITIVE_SHIFT_DESCRIPTION,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REDUCED_TIME_QC_FORMULA,
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_GRID_MAP_QC_CSV_NAME,
    REFRACTION_GRID_MAP_QC_JSON_NAME,
    REFRACTION_GRID_MAP_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
    SIGN_CONVENTION,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    STATIC_COMPONENT,
    UPLOADED_REFRACTION_PICKS_NPZ_NAME,
    WORKFLOW,
    RefractionCellSolverHistoryRow,
    RefractionStaticArtifactError,
)
from app.services.refraction_static_artifacts.formatters import (
    _csv_bool,
    _csv_cell_id,
    _csv_float,
    _csv_grid_float,
    _csv_identifier,
    _csv_int,
    _csv_json_object,
    _csv_layer_index,
    _csv_meters,
    _csv_ms,
    _float_or_nan,
    _json_float,
    _nan_if_none,
    _required_finite_float,
)
from app.services.refraction_static_artifacts.io import (
    _assert_strict_json,
    _validate_no_object_arrays,
    _write_csv_atomic,
    _write_json_atomic,
    _write_npz_atomic,
)
from app.services.refraction_static_artifacts.stats import (
    _fraction,
    _residual_stat,
    _stat,
    _status_counts,
)
from app.services.refraction_static_artifacts.registry import (
    _CELL_VELOCITY_COMPONENT_BY_LAYER,
    _artifact_entries_for_request,
    _artifact_list_for_qc,
    _build_manifest_payload,
    _cell_velocity_artifact_names,
    _cell_velocity_artifact_names_for_request as _cell_velocity_artifact_names_for_request,
    _cell_velocity_artifact_paths_for_request,
    _cell_velocity_layer_kind,
    _request_cell_velocity_layer_kinds,
    _validate_declared_upstream_artifacts,
    _validate_upstream_artifact_names,
    REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES,
)
import app.services.refraction_static_artifacts.registry as _artifact_registry

_artifact_content_type = _artifact_registry._artifact_content_type
_grid_map_qc_artifact_entries = _artifact_registry._grid_map_qc_artifact_entries
_refractor_cell_velocity_artifact_entries = (
    _artifact_registry._refractor_cell_velocity_artifact_entries
)
_t1lsst_artifact_entries = _artifact_registry._t1lsst_artifact_entries
_upstream_artifact_entries = _artifact_registry._upstream_artifact_entries


def write_refraction_static_artifacts(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    job_dir: Path,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    upstream_artifact_names: Iterable[str] = (),
    source_job_id: str | None = None,
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
        req=request,
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
    cell_velocity_artifact_paths = _cell_velocity_artifact_paths_for_request(
        root,
        request,
    )
    first_cell_velocity_artifacts = (
        cell_velocity_artifact_paths[0]
        if cell_velocity_artifact_paths
        else None
    )
    has_grid_map_qc_artifacts = bool(cell_velocity_artifact_paths)

    paths = RefractionStaticArtifactSet(
        job_dir=root,
        solution_npz=root / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        qc_json=root / REFRACTION_STATIC_QC_JSON_NAME,
        refraction_statics_csv=root / REFRACTION_STATICS_CSV_NAME,
        near_surface_model_csv=root / NEAR_SURFACE_MODEL_CSV_NAME,
        first_break_residuals_csv=root / FIRST_BREAK_RESIDUALS_CSV_NAME,
        refraction_first_break_time_export_csv=(
            root / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME
        ),
        refraction_first_break_fit_qc_csv=(
            root / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME
        ),
        refraction_first_break_fit_qc_npz=(
            root / REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME
        ),
        refraction_first_break_fit_qc_json=(
            root / REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME
        ),
        refraction_reduced_time_qc_csv=root / REFRACTION_REDUCED_TIME_QC_CSV_NAME,
        refraction_reduced_time_qc_npz=root / REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
        refraction_reduced_time_qc_json=root / REFRACTION_REDUCED_TIME_QC_JSON_NAME,
        refraction_static_components_csv=root / REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        refraction_static_component_qc_trace_csv=(
            root / REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME
        ),
        refraction_static_component_qc_endpoint_csv=(
            root / REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME
        ),
        refraction_static_component_qc_npz=(
            root / REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME
        ),
        refraction_static_component_qc_json=(
            root / REFRACTION_STATIC_COMPONENT_QC_JSON_NAME
        ),
        source_static_table_csv=root / SOURCE_STATIC_TABLE_CSV_NAME,
        receiver_static_table_csv=root / RECEIVER_STATIC_TABLE_CSV_NAME,
        source_receiver_static_table_npz=root / SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        refraction_line_profile_qc_source_csv=(
            root / REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME
        ),
        refraction_line_profile_qc_receiver_csv=(
            root / REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME
        ),
        refraction_line_profile_qc_combined_csv=(
            root / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME
        ),
        refraction_line_profile_qc_npz=root / REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
        refraction_line_profile_qc_json=root / REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        refraction_time_term_spreadsheet_csv=(
            root / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME
        ),
        static_history_json=root / REFRACTION_STATIC_HISTORY_JSON_NAME,
        manifest_json=root / REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        artifact_names=tuple(
            str(item['name']) for item in artifact_entries if bool(item['required'])
        ),
        qc=qc,
        refraction_t1lsst_1layer_components_csv=t1lsst_components_path,
        refraction_refractor_velocity_cells_csv=(
            first_cell_velocity_artifacts.cells_csv
            if first_cell_velocity_artifacts is not None
            else None
        ),
        refraction_refractor_velocity_grid_npz=(
            first_cell_velocity_artifacts.grid_npz
            if first_cell_velocity_artifacts is not None
            else None
        ),
        refraction_refractor_velocity_qc_json=(
            first_cell_velocity_artifacts.qc_json
            if first_cell_velocity_artifacts is not None
            else None
        ),
        refraction_cell_solver_history_csv=(
            first_cell_velocity_artifacts.solver_history_csv
            if first_cell_velocity_artifacts is not None
            else None
        ),
        refraction_grid_map_qc_csv=(
            root / REFRACTION_GRID_MAP_QC_CSV_NAME
            if has_grid_map_qc_artifacts
            else None
        ),
        refraction_grid_map_qc_npz=(
            root / REFRACTION_GRID_MAP_QC_NPZ_NAME
            if has_grid_map_qc_artifacts
            else None
        ),
        refraction_grid_map_qc_json=(
            root / REFRACTION_GRID_MAP_QC_JSON_NAME
            if has_grid_map_qc_artifacts
            else None
        ),
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
    write_refraction_first_break_time_export_csv(
        result=values.result,
        path=paths.refraction_first_break_time_export_csv,
        req=request,
        source_job_id=source_job_id,
    )
    write_refraction_first_break_fit_qc_csv(
        result=values.result,
        req=request,
        path=paths.refraction_first_break_fit_qc_csv,
    )
    write_refraction_first_break_fit_qc_npz(
        result=values.result,
        req=request,
        path=paths.refraction_first_break_fit_qc_npz,
    )
    write_refraction_first_break_fit_qc_json(
        result=values.result,
        req=request,
        path=paths.refraction_first_break_fit_qc_json,
    )
    write_refraction_reduced_time_qc_csv(
        result=values.result,
        req=request,
        path=paths.refraction_reduced_time_qc_csv,
    )
    write_refraction_reduced_time_qc_npz(
        result=values.result,
        req=request,
        path=paths.refraction_reduced_time_qc_npz,
    )
    write_refraction_reduced_time_qc_json(
        result=values.result,
        req=request,
        path=paths.refraction_reduced_time_qc_json,
    )
    write_refraction_static_components_csv(
        result=values.result,
        path=paths.refraction_static_components_csv,
    )
    write_refraction_static_component_qc_artifacts(
        result=values.result,
        req=request,
        trace_csv_path=paths.refraction_static_component_qc_trace_csv,
        endpoint_csv_path=paths.refraction_static_component_qc_endpoint_csv,
        npz_path=paths.refraction_static_component_qc_npz,
        json_path=paths.refraction_static_component_qc_json,
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
    write_refraction_line_profile_qc_artifacts(
        result=values.result,
        req=request,
        source_csv_path=paths.refraction_line_profile_qc_source_csv,
        receiver_csv_path=paths.refraction_line_profile_qc_receiver_csv,
        combined_csv_path=paths.refraction_line_profile_qc_combined_csv,
        npz_path=paths.refraction_line_profile_qc_npz,
        json_path=paths.refraction_line_profile_qc_json,
    )
    write_refraction_time_term_spreadsheet_csv(
        result=values.result,
        path=paths.refraction_time_term_spreadsheet_csv,
        source_job_id=source_job_id,
    )
    write_refraction_static_history_json(
        result=values.result,
        req=request,
        path=paths.static_history_json,
    )
    for cell_artifacts in cell_velocity_artifact_paths:
        write_refraction_refractor_velocity_cells_csv(
            result=values.result,
            req=request,
            path=cell_artifacts.cells_csv,
            layer_kind=cell_artifacts.layer_kind,
        )
        write_refraction_refractor_velocity_grid_npz(
            result=values.result,
            req=request,
            path=cell_artifacts.grid_npz,
            layer_kind=cell_artifacts.layer_kind,
        )
        write_refraction_refractor_velocity_qc_json(
            result=values.result,
            req=request,
            path=cell_artifacts.qc_json,
            layer_kind=cell_artifacts.layer_kind,
        )
        write_refraction_cell_solver_history_csv(
            result=values.result,
            req=request,
            path=cell_artifacts.solver_history_csv,
            layer_kind=cell_artifacts.layer_kind,
        )
    if (
        paths.refraction_grid_map_qc_csv is not None
        and paths.refraction_grid_map_qc_npz is not None
        and paths.refraction_grid_map_qc_json is not None
    ):
        write_refraction_grid_map_qc_csv(
            result=values.result,
            req=request,
            path=paths.refraction_grid_map_qc_csv,
        )
        write_refraction_grid_map_qc_npz(
            result=values.result,
            req=request,
            path=paths.refraction_grid_map_qc_npz,
        )
        write_refraction_grid_map_qc_json(
            result=values.result,
            req=request,
            path=paths.refraction_grid_map_qc_json,
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
        paths.refraction_first_break_time_export_csv,
        paths.refraction_first_break_fit_qc_csv,
        paths.refraction_first_break_fit_qc_npz,
        paths.refraction_first_break_fit_qc_json,
        paths.refraction_reduced_time_qc_csv,
        paths.refraction_reduced_time_qc_npz,
        paths.refraction_reduced_time_qc_json,
        paths.refraction_static_components_csv,
        paths.refraction_static_component_qc_trace_csv,
        paths.refraction_static_component_qc_endpoint_csv,
        paths.refraction_static_component_qc_npz,
        paths.refraction_static_component_qc_json,
        paths.source_static_table_csv,
        paths.receiver_static_table_csv,
        paths.source_receiver_static_table_npz,
        paths.refraction_line_profile_qc_source_csv,
        paths.refraction_line_profile_qc_receiver_csv,
        paths.refraction_line_profile_qc_combined_csv,
        paths.refraction_line_profile_qc_npz,
        paths.refraction_line_profile_qc_json,
        paths.refraction_time_term_spreadsheet_csv,
        paths.static_history_json,
        paths.manifest_json,
    )
    if paths.refraction_t1lsst_1layer_components_csv is not None:
        artifact_paths = artifact_paths + (
            paths.refraction_t1lsst_1layer_components_csv,
        )
    for cell_artifacts in cell_velocity_artifact_paths:
        artifact_paths = artifact_paths + (
            cell_artifacts.cells_csv,
            cell_artifacts.grid_npz,
            cell_artifacts.qc_json,
            cell_artifacts.solver_history_csv,
        )
    if (
        paths.refraction_grid_map_qc_csv is not None
        and paths.refraction_grid_map_qc_npz is not None
        and paths.refraction_grid_map_qc_json is not None
    ):
        artifact_paths = artifact_paths + (
            paths.refraction_grid_map_qc_csv,
            paths.refraction_grid_map_qc_npz,
            paths.refraction_grid_map_qc_json,
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


def write_refraction_statics_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _trace_statics_rows(values.result)
    _write_csv_atomic(Path(path), _trace_statics_columns(values.result), rows)


def write_near_surface_model_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _near_surface_model_rows(values.result)
    _write_csv_atomic(Path(path), _near_surface_columns(values.result), rows)


def write_first_break_residuals_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
    req: RefractionStaticApplyRequest | None = None,
) -> None:
    values = _validate_result(result)
    rows = _first_break_residual_rows(values.result, req=req)
    _write_csv_atomic(Path(path), _RESIDUAL_COLUMNS, rows)


def write_refraction_first_break_time_export_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
    req: RefractionStaticApplyRequest | None = None,
    source_job_id: str | None = None,
) -> None:
    values = _validate_result(result)
    rows = _first_break_time_export_rows(
        values.result,
        req=req,
        source_job_id=source_job_id,
    )
    _write_csv_atomic(Path(path), _FIRST_BREAK_TIME_EXPORT_COLUMNS, rows)


def write_refraction_first_break_fit_qc_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
    path: Path,
) -> None:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    arrays = build_refraction_first_break_fit_qc_arrays(
        result=values.result,
        req=request,
    )
    rows = _first_break_fit_qc_rows(arrays)
    _write_csv_atomic(Path(path), _FIRST_BREAK_FIT_QC_COLUMNS, rows)


def write_refraction_first_break_fit_qc_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
    path: Path,
) -> None:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    arrays = build_refraction_first_break_fit_qc_arrays(
        result=values.result,
        req=request,
    )
    _write_npz_atomic(Path(path), arrays)


def write_refraction_first_break_fit_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
    path: Path,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    payload = build_refraction_first_break_fit_qc_payload(
        result=values.result,
        req=request,
    )
    _write_json_atomic(Path(path), payload)
    return payload


def write_refraction_reduced_time_qc_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_reduced_time_qc_arrays(
        result=values.result,
        req=request,
    )
    rows = _reduced_time_qc_rows(arrays)
    _write_csv_atomic(Path(path), _REDUCED_TIME_QC_COLUMNS, rows)


def write_refraction_reduced_time_qc_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_reduced_time_qc_arrays(
        result=values.result,
        req=request,
    )
    _write_npz_atomic(Path(path), arrays)


def write_refraction_reduced_time_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    payload = build_refraction_reduced_time_qc_payload(
        result=values.result,
        req=request,
    )
    _write_json_atomic(Path(path), payload)
    return payload


def write_refraction_static_components_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    values = _validate_result(result)
    rows = _component_rows(values.result)
    _write_csv_atomic(Path(path), _component_columns(values.result), rows)


def write_refraction_static_component_qc_artifacts(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    trace_csv_path: Path,
    endpoint_csv_path: Path,
    npz_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_static_component_qc_arrays(
        result=values.result,
        req=request,
    )
    _write_csv_atomic(
        Path(trace_csv_path),
        _STATIC_COMPONENT_QC_TRACE_COLUMNS,
        _static_component_qc_trace_rows(arrays),
    )
    _write_csv_atomic(
        Path(endpoint_csv_path),
        _STATIC_COMPONENT_QC_ENDPOINT_COLUMNS,
        _static_component_qc_endpoint_rows(arrays),
    )
    _write_npz_atomic(Path(npz_path), arrays)
    payload = build_refraction_static_component_qc_payload(
        arrays=arrays,
    )
    _write_json_atomic(Path(json_path), payload)
    return payload


def write_refraction_line_profile_qc_artifacts(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    source_csv_path: Path,
    receiver_csv_path: Path,
    combined_csv_path: Path,
    npz_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_line_profile_qc_arrays(
        result=values.result,
        req=request,
    )
    _write_csv_atomic(
        Path(source_csv_path),
        _LINE_PROFILE_QC_COLUMNS,
        _line_profile_qc_rows(arrays, endpoint_kind='source'),
    )
    _write_csv_atomic(
        Path(receiver_csv_path),
        _LINE_PROFILE_QC_COLUMNS,
        _line_profile_qc_rows(arrays, endpoint_kind='receiver'),
    )
    _write_csv_atomic(
        Path(combined_csv_path),
        _LINE_PROFILE_QC_COLUMNS,
        _line_profile_qc_rows(arrays),
    )
    _write_npz_atomic(Path(npz_path), arrays)
    payload = build_refraction_line_profile_qc_payload(
        result=values.result,
        req=request,
        arrays=arrays,
    )
    _write_json_atomic(Path(json_path), payload)
    return payload


def write_refraction_refractor_velocity_cells_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    layer_kind: RefractionLayerKind | None = None,
) -> None:
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=req,
        layer_kind=layer_kind,
    )
    rows = _refractor_velocity_cell_rows(arrays)
    _write_csv_atomic(Path(path), _REFRACTOR_VELOCITY_CELL_COLUMNS, rows)


def write_refraction_refractor_velocity_grid_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    layer_kind: RefractionLayerKind | None = None,
) -> None:
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=req,
        layer_kind=layer_kind,
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
    layer_kind: RefractionLayerKind | None = None,
) -> dict[str, Any]:
    payload = build_refraction_refractor_velocity_qc_payload(
        result=result,
        req=req,
        layer_kind=layer_kind,
    )
    _write_json_atomic(Path(path), payload)
    return payload


def write_refraction_cell_solver_history_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
    layer_kind: RefractionLayerKind | None = None,
) -> None:
    rows = build_refraction_cell_solver_history_rows(
        result=result,
        req=req,
        layer_kind=layer_kind,
    )
    csv_rows = [_cell_solver_history_csv_row(row) for row in rows]
    _write_csv_atomic(Path(path), _CELL_SOLVER_HISTORY_COLUMNS, csv_rows)


def write_refraction_grid_map_qc_csv(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    arrays = build_refraction_grid_map_qc_arrays(result=result, req=req)
    rows = _grid_map_qc_rows(arrays)
    _write_csv_atomic(Path(path), _GRID_MAP_QC_COLUMNS, rows)


def write_refraction_grid_map_qc_npz(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> None:
    arrays = build_refraction_grid_map_qc_arrays(result=result, req=req)
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_GRID_MAP_QC_NPZ_NAME,
    )
    _write_npz_atomic(Path(path), arrays)


def write_refraction_grid_map_qc_json(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    path: Path,
) -> dict[str, Any]:
    payload = build_refraction_grid_map_qc_payload(result=result, req=req)
    _write_json_atomic(Path(path), payload)
    return payload


def build_refraction_grid_map_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    request = RefractionStaticApplyRequest.model_validate(req)
    layer_kinds = _request_cell_velocity_layer_kinds(request)
    if not layer_kinds:
        raise RefractionStaticArtifactError(
            'grid map QC artifacts require a solve_cell velocity layer'
        )

    pieces: dict[str, list[np.ndarray]] = {
        'layer_kind': [],
        'cell_id': [],
        'cell_ix': [],
        'cell_iy': [],
        'cell_center_x_m': [],
        'cell_center_y_m': [],
        'cell_center_inline_m': [],
        'cell_center_crossline_m': [],
        'x_min_m': [],
        'x_max_m': [],
        'y_min_m': [],
        'y_max_m': [],
        'velocity_m_s': [],
        'initial_velocity_m_s': [],
        'velocity_update_from_initial_m_s': [],
        'slowness_s_per_m': [],
        'n_observations': [],
        'n_used_observations': [],
        'n_rejected_observations': [],
        'n_sources': [],
        'n_receivers': [],
        'residual_rms_ms': [],
        'residual_mad_ms': [],
        'status': [],
        'status_reason': [],
        'active_cell_mask': [],
        'coordinate_mode': [],
        'cell_velocity_component': [],
    }

    for layer_kind in layer_kinds:
        layer_arrays = build_refraction_refractor_velocity_grid_arrays(
            result=result,
            req=request,
            layer_kind=layer_kind,
        )
        mapped = {
            'layer_kind': layer_arrays['cell_velocity_layer_kind'],
            'cell_id': layer_arrays['cell_id'],
            'cell_ix': layer_arrays['cell_ix'],
            'cell_iy': layer_arrays['cell_iy'],
            'cell_center_x_m': layer_arrays['cell_center_x_m'],
            'cell_center_y_m': layer_arrays['cell_center_y_m'],
            'cell_center_inline_m': layer_arrays['cell_center_inline_m'],
            'cell_center_crossline_m': layer_arrays['cell_center_crossline_m'],
            'x_min_m': layer_arrays['x_min_m'],
            'x_max_m': layer_arrays['x_max_m'],
            'y_min_m': layer_arrays['y_min_m'],
            'y_max_m': layer_arrays['y_max_m'],
            'velocity_m_s': layer_arrays['velocity_m_s'],
            'initial_velocity_m_s': layer_arrays['initial_velocity_m_s'],
            'velocity_update_from_initial_m_s': (
                layer_arrays['velocity_update_from_initial_m_s']
            ),
            'slowness_s_per_m': layer_arrays['slowness_s_per_m'],
            'n_observations': layer_arrays['n_observations_per_cell'],
            'n_used_observations': layer_arrays['n_used_observations_per_cell'],
            'n_rejected_observations': (
                layer_arrays['n_rejected_observations_per_cell']
            ),
            'n_sources': layer_arrays['n_sources_per_cell'],
            'n_receivers': layer_arrays['n_receivers_per_cell'],
            'residual_rms_ms': layer_arrays['residual_rms_ms'],
            'residual_mad_ms': layer_arrays['residual_mad_ms'],
            'status': layer_arrays['velocity_status'],
            'status_reason': layer_arrays['status_reason'],
            'active_cell_mask': layer_arrays['active_cell_mask'],
            'coordinate_mode': layer_arrays['coordinate_mode'],
            'cell_velocity_component': layer_arrays['cell_velocity_component'],
        }
        for key, value in mapped.items():
            pieces[key].append(np.asarray(value))

    arrays: dict[str, np.ndarray] = {}
    for key, values in pieces.items():
        merged = np.concatenate(values)
        if merged.dtype.kind in {'U', 'S'}:
            arrays[key] = _string_array(merged)
        elif merged.dtype == bool:
            arrays[key] = np.ascontiguousarray(merged, dtype=bool)
        elif np.issubdtype(merged.dtype, np.integer):
            arrays[key] = np.ascontiguousarray(merged, dtype=np.int64)
        else:
            arrays[key] = np.ascontiguousarray(merged, dtype=np.float64)

    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for grid map QC artifacts'
        )
    grid_config = effective_refraction_cell_grid_config(refractor_cell)
    arrays.update(
        {
            'artifact_version': _scalar_str(ARTIFACT_VERSION),
            'artifact_kind': _scalar_str('refraction_grid_map_qc'),
            'global_velocity_layer_behavior': _scalar_str(
                'omitted_from_grid_map_qc_rows'
            ),
            'coordinate_mode': _string_array(arrays['coordinate_mode']),
            'number_of_cell_x': _scalar_int(grid_config.number_of_cell_x),
            'number_of_cell_y': _scalar_int(grid_config.number_of_cell_y),
            'size_of_cell_x_m': _scalar_float(grid_config.size_of_cell_x_m),
            'size_of_cell_y_m': _scalar_float(
                _nan_if_none(grid_config.size_of_cell_y_m)
            ),
            'x_coordinate_origin_m': _scalar_float(
                grid_config.x_coordinate_origin_m
            ),
            'y_coordinate_origin_m': _scalar_float(
                grid_config.y_coordinate_origin_m
            ),
        }
    )
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_GRID_MAP_QC_NPZ_NAME,
    )
    return arrays


def build_refraction_grid_map_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    request = RefractionStaticApplyRequest.model_validate(req)
    layer_kinds = _request_cell_velocity_layer_kinds(request)
    if not layer_kinds:
        raise RefractionStaticArtifactError(
            'grid map QC artifacts require a solve_cell velocity layer'
        )
    arrays = build_refraction_grid_map_qc_arrays(result=result, req=request)
    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for grid map QC artifacts'
        )
    grid_config = effective_refraction_cell_grid_config(refractor_cell)

    layers = {}
    raw_layer_kind = np.asarray(arrays['layer_kind']).astype(str, copy=False)
    for layer_kind in layer_kinds:
        mask = raw_layer_kind == layer_kind
        layers[layer_kind] = _grid_map_qc_layer_summary(arrays, mask=mask)

    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'artifact_kind': 'refraction_grid_map_qc',
        'row_count': int(raw_layer_kind.size),
        'cell_layer_count': len(layer_kinds),
        'cell_velocity_layer_kinds': list(layer_kinds),
        'global_velocity_layer_behavior': 'omitted_from_grid_map_qc_rows',
        'omitted_global_velocity_layers': _grid_map_qc_global_velocity_layers(
            request
        ),
        'artifacts': {
            'csv': REFRACTION_GRID_MAP_QC_CSV_NAME,
            'npz': REFRACTION_GRID_MAP_QC_NPZ_NAME,
            'json': REFRACTION_GRID_MAP_QC_JSON_NAME,
        },
        'grid': {
            'cell_assignment_mode': refractor_cell.assignment_mode,
            **refraction_cell_coordinate_metadata_from_config(refractor_cell),
            'outside_grid_policy': refractor_cell.outside_grid_policy,
            'number_of_cell_x': int(grid_config.number_of_cell_x),
            'number_of_cell_y': int(grid_config.number_of_cell_y),
            'size_of_cell_x_m': float(grid_config.size_of_cell_x_m),
            'size_of_cell_y_m': _json_float(grid_config.size_of_cell_y_m),
            'x_coordinate_origin_m': float(grid_config.x_coordinate_origin_m),
            'y_coordinate_origin_m': float(grid_config.y_coordinate_origin_m),
            'y_axis_unbounded': grid_config.size_of_cell_y_m is None,
        },
        'layers': layers,
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_GRID_MAP_QC_JSON_NAME)
    return payload


def build_refraction_refractor_velocity_grid_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_kind: RefractionLayerKind | None = None,
) -> dict[str, np.ndarray]:
    request = RefractionStaticApplyRequest.model_validate(req)
    resolved_layer_kind = _cell_velocity_layer_kind(request, layer_kind=layer_kind)
    result = _cell_velocity_artifact_result_for_layer(
        result=result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    values = _validate_result(result)
    if not _request_has_cell_velocity_layer(request):
        raise RefractionStaticArtifactError(
            'refractor cell velocity artifacts require solve_cell request mode'
        )
    if not _result_has_cell_velocity_arrays(values.result):
        raise RefractionStaticArtifactError(
            'refractor cell velocity artifacts require cell velocity result arrays'
        )
    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )
    component = _cell_velocity_component(resolved_layer_kind)

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
    cell_slowness = _layer_cell_values_for_active_cells(
        cell_slowness,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name='cell_bedrock_slowness_s_per_m',
    )
    cell_velocity = _layer_cell_values_for_active_cells(
        cell_velocity,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name='cell_bedrock_velocity_m_s',
    )
    cell_status = _layer_cell_status_for_active_cells(
        cell_status,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name='cell_velocity_status',
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
    initial_v2 = np.full(
        n_total_cells,
        _initial_cell_v2_m_s(request, layer_kind=resolved_layer_kind),
        dtype=np.float64,
    )
    v2_update = np.full(n_total_cells, np.nan, dtype=np.float64)
    finite_update = np.isfinite(v2_m_s) & np.isfinite(initial_v2)
    v2_update[finite_update] = v2_m_s[finite_update] - initial_v2[finite_update]
    smoothing_weight = _history_smoothing_weight(
        request,
        layer_kind=resolved_layer_kind,
    )
    smoothing_enabled = bool(smoothing_weight > 0.0)

    n_observations = np.zeros(n_total_cells, dtype=np.int64)
    n_used_observations = np.zeros(n_total_cells, dtype=np.int64)
    cell_candidate_row = _cell_velocity_candidate_row_mask(
        values.result,
        request,
        layer_kind=resolved_layer_kind,
    )
    cell_row_midpoint_cell_id = np.where(
        cell_candidate_row,
        row_midpoint_cell_id,
        -1,
    )
    valid_row_cell = (
        (cell_row_midpoint_cell_id >= 0)
        & (cell_row_midpoint_cell_id < n_total_cells)
    )
    np.add.at(n_observations, cell_row_midpoint_cell_id[valid_row_cell], 1)
    qc_observation_count = _qc_cell_count_array(
        values.result.qc,
        'cell_observation_count',
        n_total_cells=n_total_cells,
    )
    if qc_observation_count is not None:
        n_observations = qc_observation_count
    used_row = valid_row_cell & np.asarray(values.result.used_row_mask, dtype=bool)
    np.add.at(n_used_observations, cell_row_midpoint_cell_id[used_row], 1)
    if np.any(n_used_observations > n_observations):
        raise RefractionStaticArtifactError(
            'used observations per cell cannot exceed total observations per cell'
        )
    n_sources = _unique_observation_endpoint_count_by_cell(
        row_midpoint_cell_id=cell_row_midpoint_cell_id,
        endpoint_id=values.result.row_source_node_id,
        n_total_cells=n_total_cells,
    )
    n_receivers = _unique_observation_endpoint_count_by_cell(
        row_midpoint_cell_id=cell_row_midpoint_cell_id,
        endpoint_id=values.result.row_receiver_node_id,
        n_total_cells=n_total_cells,
    )
    residual_stats = _per_cell_residual_stats_ms(
        row_midpoint_cell_id=cell_row_midpoint_cell_id,
        residual_time_s=values.result.residual_time_s,
        used_row_mask=used_row,
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
        'cell_velocity_layer_kind': _string_array(
            np.full(
                n_total_cells,
                resolved_layer_kind,
                dtype=f'<U{len(resolved_layer_kind)}',
            )
        ),
        'cell_velocity_component': _string_array(
            np.full(n_total_cells, component, dtype=f'<U{len(component)}')
        ),
        'velocity_m_s': np.ascontiguousarray(v2_m_s, dtype=np.float64),
        'v2_m_s': np.ascontiguousarray(v2_m_s, dtype=np.float64),
        'slowness_s_per_m': np.ascontiguousarray(slowness_s_per_m, dtype=np.float64),
        'initial_velocity_m_s': np.ascontiguousarray(initial_v2, dtype=np.float64),
        'initial_v2_m_s': np.ascontiguousarray(initial_v2, dtype=np.float64),
        'velocity_update_from_initial_m_s': np.ascontiguousarray(
            v2_update,
            dtype=np.float64,
        ),
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
    layer_kind: RefractionLayerKind | None = None,
) -> dict[str, Any]:
    request = RefractionStaticApplyRequest.model_validate(req)
    resolved_layer_kind = _cell_velocity_layer_kind(request, layer_kind=layer_kind)
    result = _cell_velocity_artifact_result_for_layer(
        result=result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    refractor_cell = request.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity QC'
        )
    grid_config = effective_refraction_cell_grid_config(refractor_cell)
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)
    velocity = np.asarray(arrays['velocity_m_s'], dtype=np.float64)
    active_velocity = velocity[active_mask & np.isfinite(velocity)]
    n_total = int(arrays['cell_id'].shape[0])
    n_active = int(np.count_nonzero(active_mask))
    n_observations_in_grid = int(np.sum(arrays['n_observations_per_cell']))
    if request.model.method == 'multilayer_time_term':
        n_valid_observations = int(
            np.count_nonzero(
                _cell_velocity_candidate_row_mask(
                    result,
                    request,
                    layer_kind=resolved_layer_kind,
                )
            )
        )
    else:
        n_valid_observations = int(
            np.count_nonzero(result.valid_observation_mask_sorted)
        )
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
                _history_smoothing_weight(
                    request,
                    layer_kind=resolved_layer_kind,
                )
            ),
        ),
    )
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'bedrock_velocity_mode': 'solve_cell',
        'cell_velocity_layer_kind': resolved_layer_kind,
        'cell_velocity_component': _cell_velocity_component(resolved_layer_kind),
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
            default=_cell_velocity_min_observations_per_cell(
                request,
                layer_kind=resolved_layer_kind,
            ),
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
        'velocity_smoothing_weight': _history_smoothing_weight(
            request,
            layer_kind=resolved_layer_kind,
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
        artifact_name=_cell_velocity_artifact_names(resolved_layer_kind).qc_json,
    )
    return payload


def build_refraction_cell_solver_history_rows(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_kind: RefractionLayerKind | None = None,
) -> list[RefractionCellSolverHistoryRow]:
    request = RefractionStaticApplyRequest.model_validate(req)
    resolved_layer_kind = _cell_velocity_layer_kind(request, layer_kind=layer_kind)
    result = _cell_velocity_artifact_result_for_layer(
        result=result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    values = _validate_result(result)
    if not _request_has_cell_velocity_layer(request):
        raise RefractionStaticArtifactError(
            'cell solver history artifact requires solve_cell request mode'
        )
    if not _result_has_cell_velocity_arrays(values.result):
        raise RefractionStaticArtifactError(
            'cell solver history artifact requires cell velocity result arrays'
        )

    arrays = build_refraction_refractor_velocity_grid_arrays(
        result=values.result,
        req=request,
        layer_kind=resolved_layer_kind,
    )
    cell_counts = _cell_solver_history_cell_counts(arrays)
    initial_v2 = _initial_cell_v2_m_s(request, layer_kind=resolved_layer_kind)
    final_velocity = np.asarray(arrays['velocity_m_s'], dtype=np.float64)
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)
    active_final_v2 = final_velocity[active_mask & np.isfinite(final_velocity)]
    update = np.asarray(
        arrays['velocity_update_from_initial_m_s'],
        dtype=np.float64,
    )
    active_update = update[active_mask & np.isfinite(update)]

    cell_candidate_row = _cell_velocity_candidate_row_mask(
        values.result,
        request,
        layer_kind=resolved_layer_kind,
    )
    cell_used_row = cell_candidate_row & np.asarray(
        values.result.used_row_mask,
        dtype=bool,
    )
    n_candidate = int(np.count_nonzero(cell_candidate_row))
    n_used = int(np.count_nonzero(cell_used_row))
    n_robust_rejected = int(
        np.count_nonzero(
            cell_candidate_row
            & np.asarray(values.result.rejected_by_robust_mask, dtype=bool)
        )
    )
    smoothing_weight = _history_smoothing_weight(
        request,
        layer_kind=resolved_layer_kind,
    )
    damping_weight = float(request.solver.damping)
    robust_threshold = float(request.solver.robust.threshold)
    robust_iteration_count = _history_robust_iteration_count(
        values.result,
        request,
    )
    residual_stats = _history_residual_stats_ms(
        values.result,
        used_row_mask=cell_used_row,
    )

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
            median_velocity_m_s=initial_v2,
            median_v2_m_s=initial_v2,
            min_velocity_m_s=initial_v2,
            min_v2_m_s=initial_v2,
            max_velocity_m_s=initial_v2,
            max_v2_m_s=initial_v2,
            max_abs_velocity_update_m_s=0.0,
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
            median_velocity_m_s=_stat(active_final_v2, 'median'),
            median_v2_m_s=_stat(active_final_v2, 'median'),
            min_velocity_m_s=_stat(active_final_v2, 'min'),
            min_v2_m_s=_stat(active_final_v2, 'min'),
            max_velocity_m_s=_stat(active_final_v2, 'max'),
            max_v2_m_s=_stat(active_final_v2, 'max'),
            max_abs_velocity_update_m_s=_history_max_abs(active_update),
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


def build_refraction_line_profile_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    """Build combined source/receiver endpoint arrays for line-profile QC."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    if not _line_profile_qc_available(request):
        return _empty_line_profile_qc_arrays()

    source = _line_profile_endpoint_arrays(
        values.result,
        request,
        endpoint='source',
    )
    receiver = _line_profile_endpoint_arrays(
        values.result,
        request,
        endpoint='receiver',
    )
    arrays = {
        column: np.concatenate((source[column], receiver[column]))
        for column in _LINE_PROFILE_QC_COLUMNS
    }
    order = np.lexsort(
        (
            np.asarray(arrays['endpoint_key']).astype(str, copy=False),
            np.asarray(arrays['inline_m'], dtype=np.float64),
            np.asarray(arrays['endpoint_kind']).astype(str, copy=False),
        )
    )
    out = {
        column: np.ascontiguousarray(np.asarray(values)[order])
        for column, values in arrays.items()
    }
    _validate_no_object_arrays(out, artifact_name=REFRACTION_LINE_PROFILE_QC_NPZ_NAME)
    return out


def build_refraction_line_profile_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    profile_arrays = (
        build_refraction_line_profile_qc_arrays(result=values.result, req=request)
        if arrays is None
        else arrays
    )
    endpoint_kind = np.asarray(profile_arrays['endpoint_kind']).astype(str, copy=False)
    inline_m = np.asarray(profile_arrays['inline_m'], dtype=np.float64)
    crossline_m = np.asarray(profile_arrays['crossline_m'], dtype=np.float64)
    status = (
        'available' if _line_profile_qc_available(request) else 'unavailable'
    )
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'schema_version': LINE_PROFILE_QC_SCHEMA_VERSION,
        'kind': 'refraction_line_profile_qc',
        'workflow': WORKFLOW,
        'status': status,
        'availability_reason': _line_profile_qc_availability_reason(request),
        'sign_convention': SIGN_CONVENTION,
        **_line_profile_coordinate_metadata(request),
        'sort_order': ['endpoint_kind', 'inline_m', 'endpoint_key'],
        'columns': list(_LINE_PROFILE_QC_COLUMNS),
        'row_count': int(endpoint_kind.shape[0]),
        'source_row_count': int(np.count_nonzero(endpoint_kind == 'source')),
        'receiver_row_count': int(np.count_nonzero(endpoint_kind == 'receiver')),
        'endpoint_kind_counts': _status_counts(endpoint_kind),
        'static_status_counts': _status_counts(profile_arrays['static_status']),
        'solution_status_counts': _status_counts(profile_arrays['solution_status']),
        'inline_m_min': _stat(inline_m, 'min'),
        'inline_m_max': _stat(inline_m, 'max'),
        'crossline_m_min': _stat(crossline_m, 'min'),
        'crossline_m_max': _stat(crossline_m, 'max'),
        'artifacts': {
            'source_csv': REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
            'receiver_csv': REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
            'combined_csv': REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
            'npz': REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
            'json': REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        },
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_LINE_PROFILE_QC_JSON_NAME)
    return payload


def build_refraction_static_component_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    """Build trace and endpoint static component waterfall QC arrays."""
    values = _validate_result(result)
    request = RefractionStaticApplyRequest.model_validate(req)
    r = values.result
    apply_to_trace_shift = bool(
        request.field_corrections.composition.apply_to_trace_shift
    )

    source_endpoint_key_sorted = _trace_endpoint_key_sorted_array(
        r,
        endpoint='source',
    )
    receiver_endpoint_key_sorted = _trace_endpoint_key_sorted_array(
        r,
        endpoint='receiver',
    )
    source_depth_shift_s_sorted = _endpoint_shift_to_trace_order(
        endpoint_key=r.source_endpoint_key,
        endpoint_shift_s=_source_depth_shift_s_array(r),
        endpoint_key_sorted=source_endpoint_key_sorted,
        label='source_depth_shift_s',
    )
    uphole_shift_s_sorted = _endpoint_shift_to_trace_order(
        endpoint_key=r.source_endpoint_key,
        endpoint_shift_s=_source_uphole_shift_s_array(r),
        endpoint_key_sorted=source_endpoint_key_sorted,
        label='uphole_shift_s',
    )
    source_manual_static_shift_s_sorted = _endpoint_shift_to_trace_order(
        endpoint_key=r.source_endpoint_key,
        endpoint_shift_s=_source_manual_static_shift_s_array(r),
        endpoint_key_sorted=source_endpoint_key_sorted,
        label='source_manual_static_shift_s',
    )
    receiver_manual_static_shift_s_sorted = _endpoint_shift_to_trace_order(
        endpoint_key=r.receiver_endpoint_key,
        endpoint_shift_s=_receiver_manual_static_shift_s_array(r),
        endpoint_key_sorted=receiver_endpoint_key_sorted,
        label='receiver_manual_static_shift_s',
    )
    manual_static_shift_s_sorted = _sum_float_arrays(
        source_manual_static_shift_s_sorted,
        receiver_manual_static_shift_s_sorted,
    )
    datum_shift_s_sorted = _sum_float_arrays(
        r.floating_datum_elevation_shift_s_sorted,
        r.flat_datum_shift_s_sorted,
    )
    trace_field_shift_s = _trace_field_shift_s_sorted_array(r)
    trace_field_status = _trace_field_static_status_sorted_array(r)
    applied_field_shift_s = _applied_field_shift_s_sorted_array(r)
    final_trace_shift_s = _final_trace_shift_s_sorted(r)
    source_field_shift_s = _source_field_shift_s_array(r)
    source_field_status = _source_field_static_status_array(r)
    receiver_field_shift_s = _receiver_field_shift_s_array(r)
    receiver_field_status = _receiver_field_static_status_array(r)
    source_depth_status = _source_depth_status_array(r)
    source_uphole_status = _source_uphole_status_array(r)
    source_manual_status = _source_manual_static_status_array(r)
    receiver_manual_status = _receiver_manual_static_status_array(r)
    source_total_with_field_s = _total_with_field_shift_s(
        refraction_shift_s=r.source_refraction_shift_s,
        field_shift_s=source_field_shift_s,
        field_status=source_field_status,
    )
    receiver_total_with_field_s = _total_with_field_shift_s(
        refraction_shift_s=r.receiver_refraction_shift_s,
        field_shift_s=receiver_field_shift_s,
        field_status=receiver_field_status,
    )
    source_applied_field_s = _applied_endpoint_field_shift_s(
        field_shift_s=source_field_shift_s,
        field_status=source_field_status,
        apply_to_trace_shift=apply_to_trace_shift,
    )
    receiver_applied_field_s = _applied_endpoint_field_shift_s(
        field_shift_s=receiver_field_shift_s,
        field_status=receiver_field_status,
        apply_to_trace_shift=apply_to_trace_shift,
    )

    source_count = int(r.source_endpoint_key.shape[0])
    receiver_count = int(r.receiver_endpoint_key.shape[0])
    trace_count = int(r.sorted_trace_index.shape[0])
    endpoint_count = source_count + receiver_count

    endpoint_kind = _string_array(
        np.concatenate(
            (
                np.full(source_count, 'source', dtype='<U8'),
                np.full(receiver_count, 'receiver', dtype='<U8'),
            )
        )
    )
    endpoint_key = _string_array(
        np.concatenate(
            (
                _string_array(r.source_endpoint_key),
                _string_array(r.receiver_endpoint_key),
            )
        )
    )
    endpoint_weathering_correction_s = np.concatenate(
        (
            _float_array(r.source_weathering_replacement_shift_s),
            _float_array(r.receiver_weathering_replacement_shift_s),
        )
    )
    endpoint_elevation_correction_s = np.concatenate(
        (
            _sum_float_arrays(
                r.source_floating_datum_elevation_shift_s,
                r.source_flat_datum_shift_s,
            ),
            _sum_float_arrays(
                r.receiver_floating_datum_elevation_shift_s,
                r.receiver_flat_datum_shift_s,
            ),
        )
    )
    endpoint_source_depth_correction_s = np.concatenate(
        (
            _source_depth_shift_s_array(r),
            np.full(receiver_count, np.nan, dtype=np.float64),
        )
    )
    endpoint_source_depth_status = _string_array(
        np.concatenate(
            (
                source_depth_status,
                np.full(
                    receiver_count,
                    _FIELD_NOT_APPLICABLE_STATUS,
                    dtype='<U48',
                ),
            )
        )
    )
    endpoint_uphole_correction_s = np.concatenate(
        (
            _source_uphole_shift_s_array(r),
            np.full(receiver_count, np.nan, dtype=np.float64),
        )
    )
    endpoint_uphole_status = _string_array(
        np.concatenate(
            (
                source_uphole_status,
                np.full(
                    receiver_count,
                    _FIELD_NOT_APPLICABLE_STATUS,
                    dtype='<U48',
                ),
            )
        )
    )
    endpoint_manual_static_s = np.concatenate(
        (
            _source_manual_static_shift_s_array(r),
            _receiver_manual_static_shift_s_array(r),
        )
    )
    endpoint_manual_static_status = _string_array(
        np.concatenate((source_manual_status, receiver_manual_status))
    )
    endpoint_field_correction_s = np.concatenate(
        (source_field_shift_s, receiver_field_shift_s)
    )
    endpoint_source_field_shift_s = np.concatenate(
        (
            source_field_shift_s,
            np.full(receiver_count, np.nan, dtype=np.float64),
        )
    )
    endpoint_receiver_field_shift_s = np.concatenate(
        (
            np.full(source_count, np.nan, dtype=np.float64),
            receiver_field_shift_s,
        )
    )
    endpoint_source_field_static_status = _string_array(
        np.concatenate(
            (
                source_field_status,
                np.full(
                    receiver_count,
                    _FIELD_NOT_APPLICABLE_STATUS,
                    dtype='<U48',
                ),
            )
        )
    )
    endpoint_receiver_field_static_status = _string_array(
        np.concatenate(
            (
                np.full(
                    source_count,
                    _FIELD_NOT_APPLICABLE_STATUS,
                    dtype='<U48',
                ),
                receiver_field_status,
            )
        )
    )
    endpoint_applied_field_correction_s = np.concatenate(
        (source_applied_field_s, receiver_applied_field_s)
    )
    endpoint_total_static_s = np.concatenate(
        (
            _float_array(r.source_refraction_shift_s),
            _float_array(r.receiver_refraction_shift_s),
        )
    )
    endpoint_total_with_field_shift_s = np.concatenate(
        (source_total_with_field_s, receiver_total_with_field_s)
    )
    endpoint_source_total_with_field_shift_s = np.concatenate(
        (
            source_total_with_field_s,
            np.full(receiver_count, np.nan, dtype=np.float64),
        )
    )
    endpoint_receiver_total_with_field_shift_s = np.concatenate(
        (
            np.full(source_count, np.nan, dtype=np.float64),
            receiver_total_with_field_s,
        )
    )
    endpoint_static_status = _string_array(
        np.concatenate(
            (
                _source_static_status_array(r),
                _receiver_static_status_array(r),
            )
        )
    )

    return {
        'artifact_version': _scalar_str(ARTIFACT_VERSION),
        'sign_convention': _scalar_str(SIGN_CONVENTION),
        'apply_to_trace_shift': np.asarray(apply_to_trace_shift, dtype=bool),
        'trace_index_sorted': _int_array(r.sorted_trace_index),
        'source_endpoint_key': source_endpoint_key_sorted,
        'receiver_endpoint_key': receiver_endpoint_key_sorted,
        'refraction_shift_s': _base_refraction_trace_shift_s_sorted_array(r),
        'weathering_shift_s': _float_array(
            r.weathering_replacement_trace_shift_s_sorted
        ),
        'datum_shift_s': datum_shift_s_sorted,
        'field_shift_s': trace_field_shift_s,
        'trace_field_shift_s': trace_field_shift_s,
        'computed_field_shift_s': trace_field_shift_s,
        'applied_field_shift_s': applied_field_shift_s,
        'trace_field_static_status': trace_field_status,
        'manual_static_shift_s': manual_static_shift_s_sorted,
        'source_depth_shift_s': source_depth_shift_s_sorted,
        'uphole_shift_s': uphole_shift_s_sorted,
        'final_trace_shift_s': final_trace_shift_s,
        'applied_trace_shift_s': final_trace_shift_s,
        'trace_apply_to_trace_shift': np.full(
            trace_count,
            apply_to_trace_shift,
            dtype=bool,
        ),
        'trace_static_status': _final_trace_static_status_sorted_array(r),
        'endpoint_kind': endpoint_kind,
        'endpoint_key': endpoint_key,
        'endpoint_weathering_correction_s': np.ascontiguousarray(
            endpoint_weathering_correction_s,
            dtype=np.float64,
        ),
        'endpoint_elevation_correction_s': np.ascontiguousarray(
            endpoint_elevation_correction_s,
            dtype=np.float64,
        ),
        'endpoint_source_depth_correction_s': np.ascontiguousarray(
            endpoint_source_depth_correction_s,
            dtype=np.float64,
        ),
        'endpoint_source_depth_status': endpoint_source_depth_status,
        'endpoint_uphole_correction_s': np.ascontiguousarray(
            endpoint_uphole_correction_s,
            dtype=np.float64,
        ),
        'endpoint_uphole_status': endpoint_uphole_status,
        'endpoint_manual_static_s': np.ascontiguousarray(
            endpoint_manual_static_s,
            dtype=np.float64,
        ),
        'endpoint_manual_static_status': endpoint_manual_static_status,
        'endpoint_source_field_shift_s': np.ascontiguousarray(
            endpoint_source_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_source_field_static_status': endpoint_source_field_static_status,
        'endpoint_receiver_field_shift_s': np.ascontiguousarray(
            endpoint_receiver_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_receiver_field_static_status': endpoint_receiver_field_static_status,
        'endpoint_field_correction_s': np.ascontiguousarray(
            endpoint_field_correction_s,
            dtype=np.float64,
        ),
        'endpoint_computed_field_correction_s': np.ascontiguousarray(
            endpoint_field_correction_s,
            dtype=np.float64,
        ),
        'endpoint_applied_field_correction_s': np.ascontiguousarray(
            endpoint_applied_field_correction_s,
            dtype=np.float64,
        ),
        'endpoint_total_static_s': np.ascontiguousarray(
            endpoint_total_static_s,
            dtype=np.float64,
        ),
        'endpoint_total_applied_shift_s': np.ascontiguousarray(
            endpoint_total_static_s,
            dtype=np.float64,
        ),
        'endpoint_total_with_field_shift_s': np.ascontiguousarray(
            endpoint_total_with_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_source_total_with_field_shift_s': np.ascontiguousarray(
            endpoint_source_total_with_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_receiver_total_with_field_shift_s': np.ascontiguousarray(
            endpoint_receiver_total_with_field_shift_s,
            dtype=np.float64,
        ),
        'endpoint_apply_to_trace_shift': np.full(
            endpoint_count,
            apply_to_trace_shift,
            dtype=bool,
        ),
        'endpoint_static_status': endpoint_static_status,
    }


def build_refraction_static_component_qc_payload(
    *,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build strict-JSON static component waterfall QC summary."""
    apply_to_trace_shift = bool(np.asarray(arrays['apply_to_trace_shift']).item())
    payload = {
        'artifact_version': ARTIFACT_VERSION,
        'kind': 'refraction_static_component_qc',
        'sign_convention': SIGN_CONVENTION,
        'units': {
            'csv_time_shift_columns': 'milliseconds',
            'npz_time_shift_arrays': 'seconds',
        },
        'apply_to_trace_shift': apply_to_trace_shift,
        'trace': {
            'row_count': int(arrays['trace_index_sorted'].shape[0]),
            'component_summary_ms': _component_qc_stats_ms(
                {
                    'refraction_trace_shift_ms': arrays['refraction_shift_s'],
                    'refraction_shift_ms': arrays['refraction_shift_s'],
                    'weathering_shift_ms': arrays['weathering_shift_s'],
                    'datum_shift_ms': arrays['datum_shift_s'],
                    'trace_field_shift_ms': arrays['trace_field_shift_s'],
                    'field_shift_ms': arrays['field_shift_s'],
                    'computed_field_shift_ms': arrays['computed_field_shift_s'],
                    'applied_field_shift_ms': arrays['applied_field_shift_s'],
                    'manual_static_shift_ms': arrays['manual_static_shift_s'],
                    'source_depth_shift_ms': arrays['source_depth_shift_s'],
                    'uphole_shift_ms': arrays['uphole_shift_s'],
                    'final_trace_shift_ms': arrays['final_trace_shift_s'],
                    'applied_trace_shift_ms': arrays['applied_trace_shift_s'],
                }
            ),
            'status_counts': _status_counts(arrays['trace_static_status']),
        },
        'endpoint': {
            'row_count': int(arrays['endpoint_key'].shape[0]),
            'component_summary_ms': _component_qc_stats_ms(
                {
                    'weathering_correction_ms': arrays[
                        'endpoint_weathering_correction_s'
                    ],
                    'elevation_correction_ms': arrays[
                        'endpoint_elevation_correction_s'
                    ],
                    'source_depth_correction_ms': arrays[
                        'endpoint_source_depth_correction_s'
                    ],
                    'uphole_correction_ms': arrays['endpoint_uphole_correction_s'],
                    'manual_static_shift_ms': arrays['endpoint_manual_static_s'],
                    'manual_static_ms': arrays['endpoint_manual_static_s'],
                    'source_field_shift_ms': arrays[
                        'endpoint_source_field_shift_s'
                    ],
                    'receiver_field_shift_ms': arrays[
                        'endpoint_receiver_field_shift_s'
                    ],
                    'field_correction_ms': arrays['endpoint_field_correction_s'],
                    'computed_field_correction_ms': arrays[
                        'endpoint_computed_field_correction_s'
                    ],
                    'applied_field_correction_ms': arrays[
                        'endpoint_applied_field_correction_s'
                    ],
                    'total_static_ms': arrays['endpoint_total_static_s'],
                    'total_applied_shift_ms': arrays[
                        'endpoint_total_applied_shift_s'
                    ],
                    'source_total_with_field_shift_ms': arrays[
                        'endpoint_source_total_with_field_shift_s'
                    ],
                    'receiver_total_with_field_shift_ms': arrays[
                        'endpoint_receiver_total_with_field_shift_s'
                    ],
                    'total_with_field_shift_ms': arrays[
                        'endpoint_total_with_field_shift_s'
                    ],
                }
            ),
            'status_counts': _status_counts(arrays['endpoint_static_status']),
        },
        'artifacts': {
            'trace_csv': REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
            'endpoint_csv': REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
            'npz': REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
            'json': REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
        },
        'source_artifacts': {
            'solution_npz': REFRACTION_STATIC_SOLUTION_NPZ_NAME,
            'source_static_table_csv': SOURCE_STATIC_TABLE_CSV_NAME,
            'receiver_static_table_csv': RECEIVER_STATIC_TABLE_CSV_NAME,
            'source_receiver_static_table_npz': SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        },
    }
    _assert_strict_json(
        payload,
        artifact_name=REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    )
    return payload


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
        'source_endpoint_key_sorted': _trace_endpoint_key_sorted_array(
            r,
            endpoint='source',
        ),
        'receiver_endpoint_key_sorted': _trace_endpoint_key_sorted_array(
            r,
            endpoint='receiver',
        ),
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
        'node_sh1_weathering_thickness_m': _node_sh1_weathering_thickness_m(r),
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
        'source_t1_s': _float_array(r.source_half_intercept_time_s),
        'source_weathering_replacement_shift_s': _float_array(
            r.source_weathering_replacement_shift_s
        ),
        'source_weathering_correction_s': _float_array(
            r.source_weathering_replacement_shift_s
        ),
        'source_floating_datum_elevation_shift_s': _float_array(
            r.source_floating_datum_elevation_shift_s
        ),
        'source_flat_datum_shift_s': _float_array(r.source_flat_datum_shift_s),
        'source_refraction_shift_s': _float_array(r.source_refraction_shift_s),
        'source_datum_status': _string_array(r.source_datum_status),
        'source_sh1_m': _source_sh1_weathering_thickness_m(r),
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
        'receiver_t1_s': _float_array(r.receiver_half_intercept_time_s),
        'receiver_weathering_replacement_shift_s': _float_array(
            r.receiver_weathering_replacement_shift_s
        ),
        'receiver_weathering_correction_s': _float_array(
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
        'receiver_sh1_m': _receiver_sh1_weathering_thickness_m(r),
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
    source_field_shift = _source_field_shift_s_array(r)
    source_field_status = _source_field_static_status_array(r)
    receiver_field_shift = _receiver_field_shift_s_array(r)
    receiver_field_status = _receiver_field_static_status_array(r)
    trace_field_shift = _trace_field_shift_s_sorted_array(r)
    trace_field_status = _trace_field_static_status_sorted_array(r)
    arrays.update(
        {
            'source_depth_m': _source_depth_m_array(r),
            'source_depth_shift_s': _source_depth_shift_s_array(r),
            'source_depth_status': _source_depth_status_array(r),
            'source_uphole_time_s': _source_uphole_time_s_array(r),
            'source_uphole_shift_s': _source_uphole_shift_s_array(r),
            'source_uphole_status': _source_uphole_status_array(r),
            'source_manual_static_shift_s': _source_manual_static_shift_s_array(r),
            'source_manual_static_status': _source_manual_static_status_array(r),
            'receiver_manual_static_shift_s': _receiver_manual_static_shift_s_array(r),
            'receiver_manual_static_status': _receiver_manual_static_status_array(r),
            'source_field_shift_s': source_field_shift,
            'source_field_static_status': source_field_status,
            'source_total_with_field_shift_s': _total_with_field_shift_s(
                refraction_shift_s=r.source_refraction_shift_s,
                field_shift_s=source_field_shift,
                field_status=source_field_status,
            ),
            'receiver_field_shift_s': receiver_field_shift,
            'receiver_field_static_status': receiver_field_status,
            'receiver_total_with_field_shift_s': _total_with_field_shift_s(
                refraction_shift_s=r.receiver_refraction_shift_s,
                field_shift_s=receiver_field_shift,
                field_status=receiver_field_status,
            ),
            'source_field_shift_s_sorted': _source_field_shift_s_sorted_array(r),
            'receiver_field_shift_s_sorted': _receiver_field_shift_s_sorted_array(r),
            'trace_field_shift_s_sorted': trace_field_shift,
            'trace_field_static_status_sorted': trace_field_status,
            'trace_field_static_valid_mask_sorted': _field_static_valid_mask(
                shift_s=trace_field_shift,
                status=trace_field_status,
            ),
            'base_refraction_trace_shift_s_sorted': _base_refraction_trace_shift_s_sorted_array(
                r
            ),
            'final_trace_shift_s_sorted': _final_trace_shift_s_sorted(r),
            'final_trace_static_status_sorted': (
                _final_trace_static_status_sorted_array(r)
            ),
            'final_trace_static_valid_mask_sorted': (
                _final_trace_static_valid_mask_sorted_array(r)
            ),
            'applied_field_shift_s_sorted': _applied_field_shift_s_sorted_array(r),
        }
    )
    if _has_node_2layer_static_fields(r):
        assert r.node_sh2_weathering_thickness_m is not None
        node_sh1_m = _node_sh1_weathering_thickness_m(r)
        node_layer1_base = r.node_surface_elevation_m - node_sh1_m
        arrays.update(
            {
                'node_sh2_weathering_thickness_m': _float_array(
                    r.node_sh2_weathering_thickness_m
                ),
                'node_layer1_base_elevation_m': _float_array(
                    node_layer1_base
                ),
                'node_final_refractor_elevation_m': _float_array(
                    r.node_refractor_elevation_m
                ),
            }
        )
        if _has_node_3layer_static_fields(r):
            assert r.node_sh3_weathering_thickness_m is not None
            arrays.update(
                {
                    'node_sh3_weathering_thickness_m': _float_array(
                        r.node_sh3_weathering_thickness_m
                    ),
                    'node_layer2_base_elevation_m': _float_array(
                        node_layer1_base - r.node_sh2_weathering_thickness_m
                    ),
                }
            )
    if _has_source_2layer_static_fields(r):
        assert r.source_t2_time_s is not None
        assert r.source_v3_m_s is not None
        assert r.source_sh2_weathering_thickness_m is not None
        source_sh1_m = _source_sh1_weathering_thickness_m(r)
        source_layer1_base = r.source_surface_elevation_m - source_sh1_m
        arrays.update(
            {
                'source_t2_time_s': _float_array(r.source_t2_time_s),
                'source_t2_s': _float_array(r.source_t2_time_s),
                'source_v3_m_s': _float_array(r.source_v3_m_s),
                'source_sh1_weathering_thickness_m': source_sh1_m,
                'source_sh2_weathering_thickness_m': _float_array(
                    r.source_sh2_weathering_thickness_m
                ),
                'source_sh2_m': _float_array(
                    r.source_sh2_weathering_thickness_m
                ),
                'source_layer1_base_elevation_m': _float_array(
                    source_layer1_base
                ),
                'source_final_refractor_elevation_m': _float_array(
                    r.source_refractor_elevation_m
                ),
            }
        )
        if _has_source_3layer_static_fields(r):
            assert r.source_t3_time_s is not None
            assert r.source_vsub_m_s is not None
            assert r.source_sh3_weathering_thickness_m is not None
            arrays.update(
                {
                    'source_t3_time_s': _float_array(r.source_t3_time_s),
                    'source_t3_s': _float_array(r.source_t3_time_s),
                    'source_vsub_m_s': _float_array(r.source_vsub_m_s),
                    'source_sh3_weathering_thickness_m': _float_array(
                        r.source_sh3_weathering_thickness_m
                    ),
                    'source_sh3_m': _float_array(
                        r.source_sh3_weathering_thickness_m
                    ),
                    'source_layer2_base_elevation_m': _float_array(
                        source_layer1_base - r.source_sh2_weathering_thickness_m
                    ),
                }
            )
    if _has_receiver_2layer_static_fields(r):
        assert r.receiver_t2_time_s is not None
        assert r.receiver_v3_m_s is not None
        assert r.receiver_sh2_weathering_thickness_m is not None
        receiver_sh1_m = _receiver_sh1_weathering_thickness_m(r)
        receiver_layer1_base = r.receiver_surface_elevation_m - receiver_sh1_m
        arrays.update(
            {
                'receiver_t2_time_s': _float_array(r.receiver_t2_time_s),
                'receiver_t2_s': _float_array(r.receiver_t2_time_s),
                'receiver_v3_m_s': _float_array(r.receiver_v3_m_s),
                'receiver_sh1_weathering_thickness_m': receiver_sh1_m,
                'receiver_sh2_weathering_thickness_m': _float_array(
                    r.receiver_sh2_weathering_thickness_m
                ),
                'receiver_sh2_m': _float_array(
                    r.receiver_sh2_weathering_thickness_m
                ),
                'receiver_layer1_base_elevation_m': _float_array(
                    receiver_layer1_base
                ),
                'receiver_final_refractor_elevation_m': _float_array(
                    r.receiver_refractor_elevation_m
                ),
            }
        )
        if _has_receiver_3layer_static_fields(r):
            assert r.receiver_t3_time_s is not None
            assert r.receiver_vsub_m_s is not None
            assert r.receiver_sh3_weathering_thickness_m is not None
            arrays.update(
                {
                    'receiver_t3_time_s': _float_array(r.receiver_t3_time_s),
                    'receiver_t3_s': _float_array(r.receiver_t3_time_s),
                    'receiver_vsub_m_s': _float_array(r.receiver_vsub_m_s),
                    'receiver_sh3_weathering_thickness_m': _float_array(
                        r.receiver_sh3_weathering_thickness_m
                    ),
                    'receiver_sh3_m': _float_array(
                        r.receiver_sh3_weathering_thickness_m
                    ),
                    'receiver_layer2_base_elevation_m': _float_array(
                        receiver_layer1_base - r.receiver_sh2_weathering_thickness_m
                    ),
                }
            )
    _validate_no_object_arrays(arrays, artifact_name=REFRACTION_STATIC_SOLUTION_NPZ_NAME)
    return arrays


def build_refraction_first_break_fit_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
) -> dict[str, np.ndarray]:
    """Build the viewer-ready observed-modeled first-break fit QC arrays."""
    values = _validate_result(result)
    r = values.result
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    source_key_by_row = _residual_row_string_context(
        r,
        'row_source_endpoint_key',
    )
    receiver_key_by_row = _residual_row_string_context(
        r,
        'row_receiver_endpoint_key',
    )
    source_id_by_row = _row_endpoint_id_context(
        r,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_id',
    )
    receiver_id_by_row = _row_endpoint_id_context(
        r,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_id',
    )
    source_x_m = _row_endpoint_float_context(
        r,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_x_m',
    )
    source_y_m = _row_endpoint_float_context(
        r,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_y_m',
    )
    receiver_x_m = _row_endpoint_float_context(
        r,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_x_m',
    )
    receiver_y_m = _row_endpoint_float_context(
        r,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_y_m',
    )
    midpoint_x_m = _midpoint_coordinate(source_x_m, receiver_x_m)
    midpoint_y_m = _midpoint_coordinate(source_y_m, receiver_y_m)
    inline_m, crossline_m = _first_break_fit_inline_crossline(
        midpoint_x_m=midpoint_x_m,
        midpoint_y_m=midpoint_y_m,
        req=request,
    )
    layer_kind_by_row, _layer_index_by_row = _residual_row_layer_context(r)
    rejection_reason_by_row = _residual_row_string_context(
        r,
        'row_rejection_reason',
    )
    cell_id_by_row, cell_ix_by_row, cell_iy_by_row = _residual_row_cell_context(
        r,
        req=request,
    )
    used = np.asarray(r.used_row_mask, dtype=bool)
    rejection_reason = np.asarray(
        [
            _residual_rejection_reason(
                used=bool(used[row_index]),
                rejected_by_robust=bool(r.rejected_by_robust_mask[row_index]),
                explicit_reason=rejection_reason_by_row[row_index],
            )
            for row_index in range(values.n_rows)
        ],
        dtype='<U64',
    )
    status = np.where(used, 'ok', 'rejected')
    arrays = {
        'observation_index': np.arange(values.n_rows, dtype=np.int64),
        'sorted_trace_index': _int_array(r.row_trace_index_sorted),
        'trace_index_sorted': _int_array(r.row_trace_index_sorted),
        'source_endpoint_key': _string_array(source_key_by_row),
        'receiver_endpoint_key': _string_array(receiver_key_by_row),
        'source_id': _string_array(source_id_by_row),
        'receiver_id': _string_array(receiver_id_by_row),
        'source_node_id': _int_array(r.row_source_node_id),
        'receiver_node_id': _int_array(r.row_receiver_node_id),
        'source_x_m': _float_array(source_x_m),
        'source_y_m': _float_array(source_y_m),
        'receiver_x_m': _float_array(receiver_x_m),
        'receiver_y_m': _float_array(receiver_y_m),
        'midpoint_x_m': _float_array(midpoint_x_m),
        'midpoint_y_m': _float_array(midpoint_y_m),
        'inline_m': _float_array(inline_m),
        'crossline_m': _float_array(crossline_m),
        'offset_m': _float_array(r.row_distance_m),
        'observed_first_break_time_s': _float_array(r.observed_pick_time_s),
        'modeled_first_break_time_s': _float_array(r.modeled_pick_time_s),
        'residual_time_s': _float_array(r.residual_time_s),
        'residual_s': _float_array(r.residual_time_s),
        'residual_time_ms': _float_array(r.residual_time_s * 1000.0),
        'layer_kind': _string_array(layer_kind_by_row),
        'cell_id': _cell_id_float_array(cell_id_by_row),
        'cell_ix': _cell_id_float_array(cell_ix_by_row),
        'cell_iy': _cell_id_float_array(cell_iy_by_row),
        'used_for_inversion': _bool_array(used),
        'used_in_solve': _bool_array(used),
        'rejection_reason': _string_array(rejection_reason),
        'reject_reason': _string_array(rejection_reason),
        'status': _string_array(status),
        'sign_convention': _string_array(
            np.full(values.n_rows, SIGN_CONVENTION, dtype=f'<U{len(SIGN_CONVENTION)}')
        ),
    }
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    )
    return arrays


def build_refraction_first_break_fit_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest | None = None,
) -> dict[str, Any]:
    """Build the strict-JSON schema and summary for first-break fit QC."""
    arrays = build_refraction_first_break_fit_qc_arrays(result=result, req=req)
    used = np.asarray(arrays['used_for_inversion'], dtype=bool)
    residual_s = np.asarray(arrays['residual_time_s'], dtype=np.float64)
    used_residual_s = residual_s[used]
    residual_ms = residual_s * 1000.0
    used_residual_ms = residual_ms[used]
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'schema_version': 1,
        'kind': 'refraction_first_break_fit_qc',
        'workflow': WORKFLOW,
        'sign_convention': SIGN_CONVENTION,
        'residual_sign': FIRST_BREAK_FIT_QC_RESIDUAL_SIGN,
        'residual_definition': (
            'residual_time_s = observed_first_break_time_s - '
            'modeled_first_break_time_s'
        ),
        'modeled_time_definition': {
            'general': (
                'modeled_first_break_time_s = source_time_term_s + '
                'receiver_time_term_s + moveout_or_cell_path_time_s'
            ),
            'midpoint_cell': (
                'moveout_or_cell_path_time_s = offset_m * '
                'cell_slowness_s_per_m'
            ),
            'global_velocity': (
                'moveout_or_cell_path_time_s = offset_m / velocity_m_s'
            ),
        },
        'columns': list(_FIRST_BREAK_FIT_QC_COLUMNS),
        'row_count': int(arrays['observation_index'].shape[0]),
        'used_count': int(np.count_nonzero(used)),
        'rejected_count': int(np.count_nonzero(~used)),
        'status_counts': _status_counts(arrays['status']),
        'rejection_reason_counts': _status_counts(arrays['rejection_reason']),
        'layer_kind_counts': _status_counts(arrays['layer_kind']),
        'residual_summary': {
            'all_rms_s': _residual_stat(residual_s, 'rms'),
            'all_mad_s': _residual_stat(residual_s, 'mad'),
            'all_mean_s': _residual_stat(residual_s, 'mean'),
            'all_p95_abs_s': _residual_stat(residual_s, 'p95_abs'),
            'all_rms_ms': _residual_stat(residual_ms, 'rms'),
            'all_mad_ms': _residual_stat(residual_ms, 'mad'),
            'all_mean_ms': _residual_stat(residual_ms, 'mean'),
            'all_p95_abs_ms': _residual_stat(residual_ms, 'p95_abs'),
            'used_rms_s': _residual_stat(used_residual_s, 'rms'),
            'used_mad_s': _residual_stat(used_residual_s, 'mad'),
            'used_mean_s': _residual_stat(used_residual_s, 'mean'),
            'used_p95_abs_s': _residual_stat(used_residual_s, 'p95_abs'),
            'used_rms_ms': _residual_stat(used_residual_ms, 'rms'),
            'used_mad_ms': _residual_stat(used_residual_ms, 'mad'),
            'used_mean_ms': _residual_stat(used_residual_ms, 'mean'),
            'used_p95_abs_ms': _residual_stat(used_residual_ms, 'p95_abs'),
        },
        'artifacts': {
            'csv': REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
            'npz': REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
            'json': REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
        },
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME)
    return payload


def build_refraction_reduced_time_qc_arrays(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, np.ndarray]:
    """Build reduced-time first-break QC arrays without changing raw picks."""
    values = _validate_result(result)
    r = values.result
    request = RefractionStaticApplyRequest.model_validate(req)
    source_key_by_row = _residual_row_string_context(
        r,
        'row_source_endpoint_key',
    )
    receiver_key_by_row = _residual_row_string_context(
        r,
        'row_receiver_endpoint_key',
    )
    source_x_m = _row_endpoint_float_context(
        r,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_x_m',
    )
    source_y_m = _row_endpoint_float_context(
        r,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_y_m',
    )
    receiver_x_m = _row_endpoint_float_context(
        r,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_x_m',
    )
    receiver_y_m = _row_endpoint_float_context(
        r,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_y_m',
    )
    midpoint_x_m = _midpoint_coordinate(source_x_m, receiver_x_m)
    midpoint_y_m = _midpoint_coordinate(source_y_m, receiver_y_m)
    inline_m, crossline_m = _first_break_fit_inline_crossline(
        midpoint_x_m=midpoint_x_m,
        midpoint_y_m=midpoint_y_m,
        req=request,
    )
    layer_kind_by_row, _layer_index_by_row = _residual_row_layer_context(r)
    gate_flags = _reduced_time_layer_gate_flags(request, r.row_distance_m)
    layer_gate_kind = _reduced_time_layer_gate_kind(
        layer_kind_by_row=layer_kind_by_row,
        gate_flags=gate_flags,
    )
    reduction_velocity = _reduced_time_reduction_velocity_by_row(
        result=r,
        req=request,
        layer_gate_kind=layer_gate_kind,
    )
    observed = np.asarray(r.observed_pick_time_s, dtype=np.float64)
    offset = np.asarray(r.row_distance_m, dtype=np.float64)
    reduced_time = np.full(values.n_rows, np.nan, dtype=np.float64)
    status = _reduced_time_status(
        observed_time_s=observed,
        offset_m=offset,
        reduction_velocity_m_s=reduction_velocity,
    )
    ok = status == 'ok'
    reduced_time[ok] = observed[ok] - offset[ok] / reduction_velocity[ok]

    arrays = {
        'trace_index_sorted': _int_array(r.row_trace_index_sorted),
        'source_endpoint_key': _string_array(source_key_by_row),
        'receiver_endpoint_key': _string_array(receiver_key_by_row),
        'offset_m': _float_array(offset),
        'inline_m': _float_array(inline_m),
        'crossline_m': _float_array(crossline_m),
        'observed_first_break_time_s': _float_array(observed),
        'reduction_velocity_m_s': _float_array(reduction_velocity),
        'reduced_time_s': _float_array(reduced_time),
        'reduced_time_ms': _float_array(reduced_time * 1000.0),
        'layer_gate_kind': _string_array(layer_gate_kind),
        'within_v1_gate': _bool_array(gate_flags['v1_direct_arrival']),
        'within_v2_t1_gate': _bool_array(gate_flags['v2_t1']),
        'within_v3_t2_gate': _bool_array(gate_flags['v3_t2']),
        'within_vsub_t3_gate': _bool_array(gate_flags['vsub_t3']),
        'used_for_inversion': _bool_array(r.used_row_mask),
        'status': _string_array(status),
        'reduction_velocity_mode': _scalar_str(
            request.reduced_time_qc.reduction_velocity_mode
        ),
    }
    _validate_no_object_arrays(
        arrays,
        artifact_name=REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    )
    return arrays


def build_refraction_reduced_time_qc_payload(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    """Build the strict-JSON schema and summary for reduced-time QC."""
    request = RefractionStaticApplyRequest.model_validate(req)
    arrays = build_refraction_reduced_time_qc_arrays(result=result, req=request)
    status = np.asarray(arrays['status']).astype(str, copy=False)
    velocity = np.asarray(arrays['reduction_velocity_m_s'], dtype=np.float64)
    payload: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'schema_version': 1,
        'kind': 'refraction_reduced_time_qc',
        'workflow': WORKFLOW,
        'sign_convention': SIGN_CONVENTION,
        'formula': REDUCED_TIME_QC_FORMULA,
        'reduction_velocity_mode': (
            request.reduced_time_qc.reduction_velocity_mode
        ),
        'fixed_velocity_m_s': _json_float(
            request.reduced_time_qc.fixed_velocity_m_s
        ),
        'columns': list(_REDUCED_TIME_QC_COLUMNS),
        'row_count': int(arrays['trace_index_sorted'].shape[0]),
        'used_count': int(np.count_nonzero(arrays['used_for_inversion'])),
        'status_counts': _status_counts(status),
        'layer_gate_kind_counts': _status_counts(arrays['layer_gate_kind']),
        'missing_velocity_count': int(
            np.count_nonzero(status == 'missing_reduction_velocity')
        ),
        'reduction_velocity_summary': {
            'min_m_s': _stat(velocity, 'min'),
            'max_m_s': _stat(velocity, 'max'),
            'median_m_s': _stat(velocity, 'median'),
        },
        'offset_gates': _reduced_time_gate_qc(request),
        'artifacts': {
            'csv': REFRACTION_REDUCED_TIME_QC_CSV_NAME,
            'npz': REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
            'json': REFRACTION_REDUCED_TIME_QC_JSON_NAME,
        },
    }
    _assert_strict_json(payload, artifact_name=REFRACTION_REDUCED_TIME_QC_JSON_NAME)
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
    _validate_optional_arrays(
        result=result,
        names=_NODE_2LAYER_STATIC_ARRAY_NAMES,
        expected_length=n_nodes,
        label='node two-layer',
    )
    n_source = _length(result.source_endpoint_key, name='source_endpoint_key')
    for name in _SOURCE_ARRAY_NAMES:
        if _length(getattr(result, name), name=name) != n_source:
            raise RefractionStaticArtifactError(
                f'source endpoint array length mismatch for {name}'
            )
    _validate_optional_arrays(
        result=result,
        names=('source_depth_m', 'source_depth_shift_s', 'source_depth_status'),
        expected_length=n_source,
        label='source-depth endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=(
            'source_uphole_time_s',
            'source_uphole_shift_s',
            'source_uphole_status',
        ),
        expected_length=n_source,
        label='uphole source endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=('source_manual_static_shift_s', 'source_manual_static_status'),
        expected_length=n_source,
        label='manual static source endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=('source_field_shift_s', 'source_field_static_status'),
        expected_length=n_source,
        label='source field-composition endpoint',
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
        names=('receiver_manual_static_shift_s', 'receiver_manual_static_status'),
        expected_length=n_receiver,
        label='manual static receiver endpoint',
    )
    _validate_optional_arrays(
        result=result,
        names=('receiver_field_shift_s', 'receiver_field_static_status'),
        expected_length=n_receiver,
        label='receiver field-composition endpoint',
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
    _validate_optional_arrays(
        result=result,
        names=(
            'source_endpoint_key_sorted',
            'receiver_endpoint_key_sorted',
        ),
        expected_length=n_traces,
        label='trace endpoint key',
    )
    _validate_optional_arrays(
        result=result,
        names=(
            'source_field_shift_s_sorted',
            'receiver_field_shift_s_sorted',
            'trace_field_shift_s_sorted',
            'trace_field_static_status_sorted',
            'trace_field_static_valid_mask_sorted',
            'base_refraction_trace_shift_s_sorted',
        ),
        expected_length=n_traces,
        label='trace field-composition',
    )
    _validate_optional_arrays(
        result=result,
        names=(
            'final_trace_shift_s_sorted',
            'final_trace_static_status_sorted',
            'final_trace_static_valid_mask_sorted',
            'applied_field_shift_s_sorted',
        ),
        expected_length=n_traces,
        label='final trace field-composition',
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

_NODE_2LAYER_STATIC_ARRAY_NAMES = (
    'node_sh1_weathering_thickness_m',
    'node_sh2_weathering_thickness_m',
)

_NODE_3LAYER_STATIC_ARRAY_NAMES = (
    'node_sh1_weathering_thickness_m',
    'node_sh2_weathering_thickness_m',
    'node_sh3_weathering_thickness_m',
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
    'source_sh1_weathering_thickness_m',
    'source_sh2_weathering_thickness_m',
)

_SOURCE_3LAYER_STATIC_ARRAY_NAMES = (
    'source_t2_time_s',
    'source_t3_time_s',
    'source_v3_m_s',
    'source_vsub_m_s',
    'source_sh1_weathering_thickness_m',
    'source_sh2_weathering_thickness_m',
    'source_sh3_weathering_thickness_m',
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
    'receiver_sh1_weathering_thickness_m',
    'receiver_sh2_weathering_thickness_m',
)

_RECEIVER_3LAYER_STATIC_ARRAY_NAMES = (
    'receiver_t2_time_s',
    'receiver_t3_time_s',
    'receiver_v3_m_s',
    'receiver_vsub_m_s',
    'receiver_sh1_weathering_thickness_m',
    'receiver_sh2_weathering_thickness_m',
    'receiver_sh3_weathering_thickness_m',
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

_FIELD_DISABLED_STATUS = 'not_enabled'
_FIELD_NOT_APPLICABLE_STATUS = 'not_applicable'
_FIELD_TOTAL_VALID_STATUSES = frozenset(
    {'ok', _FIELD_DISABLED_STATUS, _FIELD_NOT_APPLICABLE_STATUS}
)


def _trace_statics_columns(
    result: RefractionDatumStaticsResult,
) -> tuple[str, ...]:
    columns = _insert_after(
        _TRACE_STATICS_COLUMNS,
        'flat_datum_shift_ms',
        (
            'source_field_shift_ms',
            'receiver_field_shift_ms',
            'trace_field_shift_ms',
        ),
    )
    return _insert_after(
        columns,
        'refraction_trace_shift_ms',
        ('final_trace_shift_ms', 'trace_field_static_status'),
    )


def _trace_statics_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_field_shift_s = _source_field_shift_s_sorted_array(result)
    receiver_field_shift_s = _receiver_field_shift_s_sorted_array(result)
    trace_field_shift_s = _trace_field_shift_s_sorted_array(result)
    trace_field_status = _trace_field_static_status_sorted_array(result)
    base_refraction_trace_shift_s = _base_refraction_trace_shift_s_sorted_array(result)
    final_trace_shift_s = _final_trace_shift_s_sorted(result)
    for index in range(int(result.sorted_trace_index.shape[0])):
        row = (
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
                'source_field_shift_ms': _csv_ms(source_field_shift_s[index]),
                'receiver_field_shift_ms': _csv_ms(receiver_field_shift_s[index]),
                'trace_field_shift_ms': _csv_ms(trace_field_shift_s[index]),
                'refraction_trace_shift_ms': _csv_ms(base_refraction_trace_shift_s[index]),
                'final_trace_shift_ms': _csv_ms(final_trace_shift_s[index]),
                'trace_field_static_status': str(trace_field_status[index]),
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
        rows.append(row)
    return rows


def _static_component_qc_trace_rows(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(arrays['trace_index_sorted'].shape[0])
    for index in range(n_rows):
        rows.append(
            {
                'trace_index_sorted': int(arrays['trace_index_sorted'][index]),
                'source_endpoint_key': str(arrays['source_endpoint_key'][index]),
                'receiver_endpoint_key': str(arrays['receiver_endpoint_key'][index]),
                'refraction_trace_shift_ms': _csv_ms(
                    arrays['refraction_shift_s'][index]
                ),
                'refraction_shift_ms': _csv_ms(arrays['refraction_shift_s'][index]),
                'weathering_shift_ms': _csv_ms(arrays['weathering_shift_s'][index]),
                'datum_shift_ms': _csv_ms(arrays['datum_shift_s'][index]),
                'trace_field_shift_ms': _csv_ms(
                    arrays['trace_field_shift_s'][index]
                ),
                'field_shift_ms': _csv_ms(arrays['field_shift_s'][index]),
                'computed_field_shift_ms': _csv_ms(
                    arrays['computed_field_shift_s'][index]
                ),
                'applied_field_shift_ms': _csv_ms(
                    arrays['applied_field_shift_s'][index]
                ),
                'trace_field_static_status': str(
                    arrays['trace_field_static_status'][index]
                ),
                'manual_static_shift_ms': _csv_ms(
                    arrays['manual_static_shift_s'][index]
                ),
                'source_depth_shift_ms': _csv_ms(
                    arrays['source_depth_shift_s'][index]
                ),
                'uphole_shift_ms': _csv_ms(arrays['uphole_shift_s'][index]),
                'final_trace_shift_ms': _csv_ms(
                    arrays['final_trace_shift_s'][index]
                ),
                'applied_trace_shift_ms': _csv_ms(
                    arrays['applied_trace_shift_s'][index]
                ),
                'apply_to_trace_shift': _csv_bool(
                    arrays['trace_apply_to_trace_shift'][index]
                ),
                'static_status': str(arrays['trace_static_status'][index]),
                'sign_convention': SIGN_CONVENTION,
            }
        )
    return rows


def _static_component_qc_endpoint_rows(
    arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(arrays['endpoint_key'].shape[0])
    for index in range(n_rows):
        rows.append(
            {
                'endpoint_kind': str(arrays['endpoint_kind'][index]),
                'endpoint_key': str(arrays['endpoint_key'][index]),
                'weathering_correction_ms': _csv_ms(
                    arrays['endpoint_weathering_correction_s'][index]
                ),
                'elevation_correction_ms': _csv_ms(
                    arrays['endpoint_elevation_correction_s'][index]
                ),
                'source_depth_correction_ms': _csv_ms(
                    arrays['endpoint_source_depth_correction_s'][index]
                ),
                'source_depth_status': str(
                    arrays['endpoint_source_depth_status'][index]
                ),
                'uphole_correction_ms': _csv_ms(
                    arrays['endpoint_uphole_correction_s'][index]
                ),
                'uphole_status': str(arrays['endpoint_uphole_status'][index]),
                'manual_static_shift_ms': _csv_ms(
                    arrays['endpoint_manual_static_s'][index]
                ),
                'manual_static_ms': _csv_ms(
                    arrays['endpoint_manual_static_s'][index]
                ),
                'manual_static_status': str(
                    arrays['endpoint_manual_static_status'][index]
                ),
                'source_field_shift_ms': _csv_ms(
                    arrays['endpoint_source_field_shift_s'][index]
                ),
                'source_field_static_status': str(
                    arrays['endpoint_source_field_static_status'][index]
                ),
                'receiver_field_shift_ms': _csv_ms(
                    arrays['endpoint_receiver_field_shift_s'][index]
                ),
                'receiver_field_static_status': str(
                    arrays['endpoint_receiver_field_static_status'][index]
                ),
                'field_correction_ms': _csv_ms(
                    arrays['endpoint_field_correction_s'][index]
                ),
                'computed_field_correction_ms': _csv_ms(
                    arrays['endpoint_computed_field_correction_s'][index]
                ),
                'applied_field_correction_ms': _csv_ms(
                    arrays['endpoint_applied_field_correction_s'][index]
                ),
                'total_static_ms': _csv_ms(arrays['endpoint_total_static_s'][index]),
                'total_applied_shift_ms': _csv_ms(
                    arrays['endpoint_total_applied_shift_s'][index]
                ),
                'source_total_with_field_shift_ms': _csv_ms(
                    arrays['endpoint_source_total_with_field_shift_s'][index]
                ),
                'receiver_total_with_field_shift_ms': _csv_ms(
                    arrays['endpoint_receiver_total_with_field_shift_s'][index]
                ),
                'total_with_field_shift_ms': _csv_ms(
                    arrays['endpoint_total_with_field_shift_s'][index]
                ),
                'apply_to_trace_shift': _csv_bool(
                    arrays['endpoint_apply_to_trace_shift'][index]
                ),
                'static_status': str(arrays['endpoint_static_status'][index]),
                'sign_convention': SIGN_CONVENTION,
            }
        )
    return rows


def _near_surface_model_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    node_sh1_m = _node_sh1_weathering_thickness_m(result)
    node_sh2_m = result.node_sh2_weathering_thickness_m
    node_sh3_m = result.node_sh3_weathering_thickness_m
    has_3layer_fields = _has_node_3layer_static_fields(result)
    has_2layer_fields = _has_node_2layer_static_fields(result)
    for index in range(int(result.node_id.shape[0])):
        row = {
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
        if has_2layer_fields:
            assert node_sh2_m is not None
            layer1_base = result.node_surface_elevation_m[index] - node_sh1_m[index]
            row.update(
                {
                    'sh1_weathering_thickness_m': _csv_float(node_sh1_m[index]),
                    'sh2_weathering_thickness_m': _csv_float(node_sh2_m[index]),
                    'layer1_base_elevation_m': _csv_float(layer1_base),
                    'final_refractor_elevation_m': _csv_float(
                        result.node_refractor_elevation_m[index]
                    ),
                }
            )
            if has_3layer_fields:
                assert node_sh3_m is not None
                layer2_base = layer1_base - node_sh2_m[index]
                row.update(
                    {
                        'sh3_weathering_thickness_m': _csv_float(
                            node_sh3_m[index]
                        ),
                        'layer2_base_elevation_m': _csv_float(layer2_base),
                    }
                )
        rows.append(row)
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
    layer_kind_by_row, layer_index_by_row = _residual_row_layer_context(result)
    source_key_by_row = _residual_row_string_context(
        result,
        'row_source_endpoint_key',
    )
    receiver_key_by_row = _residual_row_string_context(
        result,
        'row_receiver_endpoint_key',
    )
    rejection_reason_by_row = _residual_row_string_context(
        result,
        'row_rejection_reason',
    )
    row_velocity_m_s = _residual_row_velocity_context(result)
    for row_index in range(int(result.row_trace_index_sorted.shape[0])):
        rejected_by_robust = bool(result.rejected_by_robust_mask[row_index])
        used = bool(result.used_row_mask[row_index])
        rejection_reason = _residual_rejection_reason(
            used=used,
            rejected_by_robust=rejected_by_robust,
            explicit_reason=rejection_reason_by_row[row_index],
        )
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
                'rejection_reason': rejection_reason,
                'cell_id': _csv_cell_id(cell_id_by_row[row_index]),
                'cell_ix': _csv_cell_id(cell_ix_by_row[row_index]),
                'cell_iy': _csv_cell_id(cell_iy_by_row[row_index]),
                'trace_index_sorted': int(result.row_trace_index_sorted[row_index]),
                'layer_kind': str(layer_kind_by_row[row_index]),
                'layer_index': _csv_layer_index(layer_index_by_row[row_index]),
                'source_endpoint_key': str(source_key_by_row[row_index]),
                'receiver_endpoint_key': str(receiver_key_by_row[row_index]),
                'offset_m': _csv_float(result.row_distance_m[row_index]),
                'residual_time_s': _csv_float(result.residual_time_s[row_index]),
                'midpoint_cell_id': _csv_cell_id(cell_id_by_row[row_index]),
                'row_velocity_m_s': _csv_float(row_velocity_m_s[row_index]),
            }
        )
    return rows


def _first_break_time_export_rows(
    result: RefractionDatumStaticsResult,
    *,
    req: RefractionStaticApplyRequest | None = None,
    source_job_id: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    layer_kind_by_row, _layer_index_by_row = _first_break_export_layer_context(
        result,
    )
    source_key_by_row = _residual_row_string_context(
        result,
        'row_source_endpoint_key',
    )
    receiver_key_by_row = _residual_row_string_context(
        result,
        'row_receiver_endpoint_key',
    )
    rejection_reason_by_row = _residual_row_string_context(
        result,
        'row_rejection_reason',
    )
    source_id_by_row = _row_endpoint_id_context(
        result,
        endpoint='source',
        row_endpoint_key=source_key_by_row,
        value_field='source_id',
    )
    receiver_id_by_row = _row_endpoint_id_context(
        result,
        endpoint='receiver',
        row_endpoint_key=receiver_key_by_row,
        value_field='receiver_id',
    )
    job_id = _source_job_id(source_job_id, req)

    for row_index in range(int(result.row_trace_index_sorted.shape[0])):
        rejected_by_robust = bool(result.rejected_by_robust_mask[row_index])
        used = bool(result.used_row_mask[row_index])
        rejection_reason = _residual_rejection_reason(
            used=used,
            rejected_by_robust=rejected_by_robust,
            explicit_reason=rejection_reason_by_row[row_index],
        )
        rows.append(
            {
                'format_name': FIRST_BREAK_TIME_EXPORT_FORMAT_NAME,
                'format_version': FIRST_BREAK_TIME_EXPORT_FORMAT_VERSION,
                'source_job_id': job_id,
                'observation_index': row_index,
                'sorted_trace_index': int(result.row_trace_index_sorted[row_index]),
                'source_endpoint_key': str(source_key_by_row[row_index]),
                'receiver_endpoint_key': str(receiver_key_by_row[row_index]),
                'source_id': _csv_identifier(source_id_by_row[row_index]),
                'receiver_id': _csv_identifier(receiver_id_by_row[row_index]),
                'offset_m': _csv_meters(result.row_distance_m[row_index]),
                'layer_kind': str(layer_kind_by_row[row_index]),
                'observed_pick_time_ms': _csv_ms(
                    result.observed_pick_time_s[row_index]
                ),
                'modeled_pick_time_ms': _csv_ms(
                    result.modeled_pick_time_s[row_index]
                ),
                'residual_ms': _csv_ms(result.residual_time_s[row_index]),
                'used_in_solve': _csv_bool(used),
                'reject_reason': rejection_reason,
                'sign_convention': FIRST_BREAK_TIME_EXPORT_SIGN_CONVENTION,
            }
        )
    return rows


def _first_break_fit_qc_rows(
    arrays: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(np.asarray(arrays['observation_index']).shape[0])
    for row_index in range(n_rows):
        rejection_reason = str(arrays['rejection_reason'][row_index])
        rows.append(
            {
                'observation_index': int(arrays['observation_index'][row_index]),
                'sorted_trace_index': int(arrays['sorted_trace_index'][row_index]),
                'trace_index_sorted': int(arrays['trace_index_sorted'][row_index]),
                'source_endpoint_key': str(arrays['source_endpoint_key'][row_index]),
                'receiver_endpoint_key': str(
                    arrays['receiver_endpoint_key'][row_index]
                ),
                'source_id': _csv_identifier(arrays['source_id'][row_index]),
                'receiver_id': _csv_identifier(arrays['receiver_id'][row_index]),
                'source_node_id': int(arrays['source_node_id'][row_index]),
                'receiver_node_id': int(arrays['receiver_node_id'][row_index]),
                'source_x_m': _csv_float(arrays['source_x_m'][row_index]),
                'source_y_m': _csv_float(arrays['source_y_m'][row_index]),
                'receiver_x_m': _csv_float(arrays['receiver_x_m'][row_index]),
                'receiver_y_m': _csv_float(arrays['receiver_y_m'][row_index]),
                'midpoint_x_m': _csv_float(arrays['midpoint_x_m'][row_index]),
                'midpoint_y_m': _csv_float(arrays['midpoint_y_m'][row_index]),
                'inline_m': _csv_float(arrays['inline_m'][row_index]),
                'crossline_m': _csv_float(arrays['crossline_m'][row_index]),
                'offset_m': _csv_float(arrays['offset_m'][row_index]),
                'observed_first_break_time_s': _csv_float(
                    arrays['observed_first_break_time_s'][row_index]
                ),
                'modeled_first_break_time_s': _csv_float(
                    arrays['modeled_first_break_time_s'][row_index]
                ),
                'residual_time_s': _csv_float(arrays['residual_time_s'][row_index]),
                'residual_s': _csv_float(arrays['residual_s'][row_index]),
                'residual_time_ms': _csv_float(arrays['residual_time_ms'][row_index]),
                'layer_kind': str(arrays['layer_kind'][row_index]),
                'cell_id': _csv_cell_id(arrays['cell_id'][row_index]),
                'cell_ix': _csv_cell_id(arrays['cell_ix'][row_index]),
                'cell_iy': _csv_cell_id(arrays['cell_iy'][row_index]),
                'used_for_inversion': _csv_bool(
                    arrays['used_for_inversion'][row_index]
                ),
                'used_in_solve': _csv_bool(arrays['used_in_solve'][row_index]),
                'rejection_reason': rejection_reason,
                'reject_reason': rejection_reason,
                'status': str(arrays['status'][row_index]),
                'sign_convention': str(arrays['sign_convention'][row_index]),
            }
        )
    return rows


def _reduced_time_qc_rows(
    arrays: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(np.asarray(arrays['trace_index_sorted']).shape[0])
    for row_index in range(n_rows):
        rows.append(
            {
                'trace_index_sorted': int(arrays['trace_index_sorted'][row_index]),
                'source_endpoint_key': str(arrays['source_endpoint_key'][row_index]),
                'receiver_endpoint_key': str(
                    arrays['receiver_endpoint_key'][row_index]
                ),
                'offset_m': _csv_float(arrays['offset_m'][row_index]),
                'inline_m': _csv_float(arrays['inline_m'][row_index]),
                'crossline_m': _csv_float(arrays['crossline_m'][row_index]),
                'observed_first_break_time_s': _csv_float(
                    arrays['observed_first_break_time_s'][row_index]
                ),
                'reduction_velocity_m_s': _csv_float(
                    arrays['reduction_velocity_m_s'][row_index]
                ),
                'reduced_time_s': _csv_float(arrays['reduced_time_s'][row_index]),
                'reduced_time_ms': _csv_float(arrays['reduced_time_ms'][row_index]),
                'layer_gate_kind': str(arrays['layer_gate_kind'][row_index]),
                'within_v1_gate': _csv_bool(arrays['within_v1_gate'][row_index]),
                'within_v2_t1_gate': _csv_bool(
                    arrays['within_v2_t1_gate'][row_index]
                ),
                'within_v3_t2_gate': _csv_bool(
                    arrays['within_v3_t2_gate'][row_index]
                ),
                'within_vsub_t3_gate': _csv_bool(
                    arrays['within_vsub_t3_gate'][row_index]
                ),
                'used_for_inversion': _csv_bool(
                    arrays['used_for_inversion'][row_index]
                ),
                'status': str(arrays['status'][row_index]),
            }
        )
    return rows


def _reduced_time_layer_gate_flags(
    req: RefractionStaticApplyRequest,
    offset_m: np.ndarray,
) -> dict[str, np.ndarray]:
    offset = np.asarray(offset_m, dtype=np.float64)
    flags = {
        'v1_direct_arrival': np.zeros(offset.shape, dtype=bool),
        'v2_t1': np.zeros(offset.shape, dtype=bool),
        'v3_t2': np.zeros(offset.shape, dtype=bool),
        'vsub_t3': np.zeros(offset.shape, dtype=bool),
    }
    first_layer = req.model.first_layer
    if first_layer is not None and first_layer.mode == 'estimate_direct_arrival':
        flags['v1_direct_arrival'] = _offset_gate_mask(
            offset,
            min_offset_m=first_layer.min_direct_offset_m,
            max_offset_m=first_layer.max_direct_offset_m,
            enabled=True,
        )
    for config in normalize_refraction_static_layers(req.model, enabled_only=False):
        if not config.enabled:
            continue
        flags[config.kind] = _offset_gate_mask(
            offset,
            min_offset_m=config.min_offset_m,
            max_offset_m=config.max_offset_m,
            enabled=True,
        )
    return {key: np.ascontiguousarray(value, dtype=bool) for key, value in flags.items()}


def _offset_gate_mask(
    offset_m: np.ndarray,
    *,
    min_offset_m: float | None,
    max_offset_m: float | None,
    enabled: bool,
) -> np.ndarray:
    offset = np.asarray(offset_m, dtype=np.float64)
    mask = np.zeros(offset.shape, dtype=bool)
    if not enabled:
        return mask
    mask = np.isfinite(offset)
    if min_offset_m is not None:
        mask &= offset >= float(min_offset_m)
    if max_offset_m is not None:
        mask &= offset <= float(max_offset_m)
    return np.ascontiguousarray(mask, dtype=bool)


def _reduced_time_layer_gate_kind(
    *,
    layer_kind_by_row: np.ndarray,
    gate_flags: Mapping[str, np.ndarray],
) -> np.ndarray:
    out = np.asarray(layer_kind_by_row).astype('<U32', copy=True)
    empty = out == ''
    for kind in ('v2_t1', 'v3_t2', 'vsub_t3', 'v1_direct_arrival'):
        mask = empty & np.asarray(gate_flags[kind], dtype=bool)
        out[mask] = kind
        empty &= ~mask
    return np.ascontiguousarray(out, dtype='<U32')


def _reduced_time_reduction_velocity_by_row(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_gate_kind: np.ndarray,
) -> np.ndarray:
    mode = req.reduced_time_qc.reduction_velocity_mode
    n_rows = int(result.row_trace_index_sorted.shape[0])
    if mode == 'fixed':
        return np.full(
            n_rows,
            float(req.reduced_time_qc.fixed_velocity_m_s),
            dtype=np.float64,
        )
    if mode == 'initial_velocity':
        return _configured_reduction_velocity_by_row(
            req=req,
            layer_gate_kind=layer_gate_kind,
        )
    velocity = _residual_row_velocity_context(result)
    kind = np.asarray(layer_gate_kind).astype(str, copy=False)
    if np.any(kind == 'v1_direct_arrival'):
        velocity = np.asarray(velocity, dtype=np.float64).copy()
        velocity[kind == 'v1_direct_arrival'] = _float_or_nan(
            result.weathering_velocity_m_s
        )
    return np.ascontiguousarray(velocity, dtype=np.float64)


def _configured_reduction_velocity_by_row(
    *,
    req: RefractionStaticApplyRequest,
    layer_gate_kind: np.ndarray,
) -> np.ndarray:
    kind = np.asarray(layer_gate_kind).astype(str, copy=False)
    out = np.full(kind.shape, np.nan, dtype=np.float64)
    velocity_by_kind = _configured_initial_velocity_by_layer(req)
    for layer_kind, velocity in velocity_by_kind.items():
        out[kind == layer_kind] = velocity
    return np.ascontiguousarray(out, dtype=np.float64)


def _configured_initial_velocity_by_layer(
    req: RefractionStaticApplyRequest,
) -> dict[str, float]:
    values: dict[str, float] = {}
    first_layer_velocity = _configured_v1_velocity(req)
    if first_layer_velocity is not None:
        values['v1_direct_arrival'] = first_layer_velocity
    for config in normalize_refraction_static_layers(req.model, enabled_only=False):
        velocity = config.initial_velocity_m_s
        if velocity is None:
            velocity = config.fixed_velocity_m_s
        if velocity is None:
            continue
        velocity_f = _float_or_nan(velocity)
        if np.isfinite(velocity_f) and velocity_f > 0.0:
            values[config.kind] = velocity_f
    return values


def _configured_v1_velocity(req: RefractionStaticApplyRequest) -> float | None:
    first_layer = req.model.first_layer
    if first_layer is not None and first_layer.weathering_velocity_m_s is not None:
        return float(first_layer.weathering_velocity_m_s)
    if req.model.weathering_velocity_m_s is not None:
        return float(req.model.weathering_velocity_m_s)
    return None


def _reduced_time_status(
    *,
    observed_time_s: np.ndarray,
    offset_m: np.ndarray,
    reduction_velocity_m_s: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(observed_time_s, dtype=np.float64)
    offset = np.asarray(offset_m, dtype=np.float64)
    velocity = np.asarray(reduction_velocity_m_s, dtype=np.float64)
    status = np.full(observed.shape, 'ok', dtype='<U32')
    missing_observed = ~np.isfinite(observed)
    missing_offset = np.isfinite(observed) & ~np.isfinite(offset)
    missing_velocity = (
        np.isfinite(observed)
        & np.isfinite(offset)
        & (~np.isfinite(velocity) | (velocity <= 0.0))
    )
    status[missing_observed] = 'missing_observed_time'
    status[missing_offset] = 'missing_offset'
    status[missing_velocity] = 'missing_reduction_velocity'
    return np.ascontiguousarray(status, dtype='<U32')


def _reduced_time_gate_qc(req: RefractionStaticApplyRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'v1_direct_arrival': {
            'enabled': False,
            'min_offset_m': None,
            'max_offset_m': None,
        },
        'v2_t1': {'enabled': False, 'min_offset_m': None, 'max_offset_m': None},
        'v3_t2': {'enabled': False, 'min_offset_m': None, 'max_offset_m': None},
        'vsub_t3': {'enabled': False, 'min_offset_m': None, 'max_offset_m': None},
    }
    first_layer = req.model.first_layer
    if first_layer is not None and first_layer.mode == 'estimate_direct_arrival':
        payload['v1_direct_arrival'] = {
            'enabled': True,
            'min_offset_m': _json_float(first_layer.min_direct_offset_m),
            'max_offset_m': _json_float(first_layer.max_direct_offset_m),
        }
    for config in normalize_refraction_static_layers(req.model, enabled_only=False):
        payload[config.kind] = {
            'enabled': bool(config.enabled),
            'min_offset_m': _json_float(config.min_offset_m),
            'max_offset_m': _json_float(config.max_offset_m),
        }
    return payload


def _source_job_id(
    source_job_id: str | None,
    req: RefractionStaticApplyRequest | None,
) -> str:
    raw = source_job_id
    if raw is None and req is not None:
        raw = getattr(req, 'source_job_id', None)
    return '' if raw is None else str(raw)


def _first_break_export_layer_context(
    result: RefractionDatumStaticsResult,
) -> tuple[np.ndarray, np.ndarray]:
    return _residual_row_layer_context(result)


def _row_endpoint_id_context(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
    row_endpoint_key: np.ndarray,
    value_field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    keys = (
        result.source_endpoint_key
        if endpoint == 'source'
        else result.receiver_endpoint_key
    )
    values = getattr(result, value_field)
    lookup = {
        str(key): value
        for key, value in zip(
            np.asarray(keys).tolist(),
            np.asarray(values).tolist(),
            strict=True,
        )
    }
    out = np.full(n_rows, '', dtype=object)
    for index, raw_key in enumerate(row_endpoint_key.tolist()):
        value = lookup.get(str(raw_key))
        if value is not None:
            out[index] = value
    return np.ascontiguousarray(out, dtype=object)


def _row_endpoint_float_context(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
    row_endpoint_key: np.ndarray,
    value_field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    keys = (
        result.source_endpoint_key
        if endpoint == 'source'
        else result.receiver_endpoint_key
    )
    values = getattr(result, value_field)
    lookup = {
        str(key): float(value)
        for key, value in zip(
            np.asarray(keys).tolist(),
            np.asarray(values).tolist(),
            strict=True,
        )
    }
    out = np.full(n_rows, np.nan, dtype=np.float64)
    for index, raw_key in enumerate(row_endpoint_key.tolist()):
        value = lookup.get(str(raw_key))
        if value is not None:
            out[index] = value
    return np.ascontiguousarray(out, dtype=np.float64)


def _first_break_row_time_terms(
    result: RefractionDatumStaticsResult,
    *,
    layer_kind_by_row: np.ndarray,
    source_key_by_row: np.ndarray,
    receiver_key_by_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    source = np.full(n_rows, np.nan, dtype=np.float64)
    receiver = np.full(n_rows, np.nan, dtype=np.float64)
    source_t1_sorted = np.asarray(
        result.source_half_intercept_time_s_sorted,
        dtype=np.float64,
    )
    receiver_t1_sorted = np.asarray(
        result.receiver_half_intercept_time_s_sorted,
        dtype=np.float64,
    )
    source_t2 = _endpoint_time_lookup(
        result.source_endpoint_key,
        result.source_t2_time_s,
    )
    receiver_t2 = _endpoint_time_lookup(
        result.receiver_endpoint_key,
        result.receiver_t2_time_s,
    )
    source_t3 = _endpoint_time_lookup(
        result.source_endpoint_key,
        result.source_t3_time_s,
    )
    receiver_t3 = _endpoint_time_lookup(
        result.receiver_endpoint_key,
        result.receiver_t3_time_s,
    )
    for row_index, trace_index in enumerate(result.row_trace_index_sorted.tolist()):
        kind = str(layer_kind_by_row[row_index])
        if kind == 'v3_t2':
            source[row_index] = source_t2.get(
                str(source_key_by_row[row_index]),
                np.nan,
            )
            receiver[row_index] = receiver_t2.get(
                str(receiver_key_by_row[row_index]),
                np.nan,
            )
        elif kind == 'vsub_t3':
            source[row_index] = source_t3.get(
                str(source_key_by_row[row_index]),
                np.nan,
            )
            receiver[row_index] = receiver_t3.get(
                str(receiver_key_by_row[row_index]),
                np.nan,
            )
        elif kind == 'v2_t1':
            source[row_index] = source_t1_sorted[int(trace_index)]
            receiver[row_index] = receiver_t1_sorted[int(trace_index)]
    return (
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(receiver, dtype=np.float64),
    )


def _endpoint_time_lookup(
    endpoint_key: np.ndarray,
    values: np.ndarray | None,
) -> dict[str, float]:
    if values is None:
        return {}
    return {
        str(key): float(value)
        for key, value in zip(
            np.asarray(endpoint_key).tolist(),
            np.asarray(values, dtype=np.float64).tolist(),
            strict=True,
        )
    }


def _first_break_moveout_time_s(
    *,
    modeled_pick_time_s: np.ndarray,
    source_time_term_s: np.ndarray,
    receiver_time_term_s: np.ndarray,
) -> np.ndarray:
    modeled = np.asarray(modeled_pick_time_s, dtype=np.float64)
    source = np.asarray(source_time_term_s, dtype=np.float64)
    receiver = np.asarray(receiver_time_term_s, dtype=np.float64)
    out = modeled - source - receiver
    invalid = ~np.isfinite(modeled) | ~np.isfinite(source) | ~np.isfinite(receiver)
    out[invalid] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _midpoint_coordinate(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    out = 0.5 * (left_arr + right_arr)
    out[~np.isfinite(left_arr) | ~np.isfinite(right_arr)] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _first_break_fit_inline_crossline(
    *,
    midpoint_x_m: np.ndarray,
    midpoint_y_m: np.ndarray,
    req: RefractionStaticApplyRequest | None,
) -> tuple[np.ndarray, np.ndarray]:
    midpoint_x = np.asarray(midpoint_x_m, dtype=np.float64)
    midpoint_y = np.asarray(midpoint_y_m, dtype=np.float64)
    inline = np.full(midpoint_x.shape, np.nan, dtype=np.float64)
    crossline = np.full(midpoint_x.shape, np.nan, dtype=np.float64)
    if req is None or req.model.refractor_cell is None:
        return inline, crossline
    refractor_cell = req.model.refractor_cell
    if refractor_cell.coordinate_mode != 'line_2d_projected':
        return inline, crossline
    projected = project_refraction_cell_points(
        x_m=midpoint_x,
        y_m=midpoint_y,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    if projected.projected_inline_m is not None:
        inline = projected.projected_inline_m
    if projected.projected_crossline_m is not None:
        crossline = projected.projected_crossline_m
    return (
        np.ascontiguousarray(inline, dtype=np.float64),
        np.ascontiguousarray(crossline, dtype=np.float64),
    )


def _residual_row_layer_context(
    result: RefractionDatumStaticsResult,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    kind = np.full(n_rows, '', dtype='<U16')
    index = np.zeros(n_rows, dtype=np.int64)
    raw_kind = getattr(result, 'row_layer_kind', None)
    raw_index = getattr(result, 'row_layer_index', None)
    if raw_kind is not None:
        kind = np.asarray(raw_kind).astype('<U16', copy=False)
        if kind.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_layer_kind length must match residual rows'
            )
    if raw_index is not None:
        index = np.asarray(raw_index, dtype=np.int64)
        if index.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_layer_index length must match residual rows'
            )
    return np.ascontiguousarray(kind), np.ascontiguousarray(index)


def _residual_row_string_context(
    result: RefractionDatumStaticsResult,
    field: str,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    raw = getattr(result, field, None)
    if raw is None:
        return np.full(n_rows, '', dtype=object)
    out = np.asarray(raw, dtype=object)
    if out.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            f'{field} length must match residual rows'
        )
    return np.ascontiguousarray(out, dtype=object)


def _residual_row_velocity_context(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    raw = getattr(result, 'row_velocity_m_s', None)
    if raw is not None:
        out = np.asarray(raw, dtype=np.float64)
        if out.shape != (n_rows,):
            raise RefractionStaticArtifactError(
                'row_velocity_m_s length must match residual rows'
            )
        return np.ascontiguousarray(out, dtype=np.float64)
    return _row_velocity_from_cell_or_scalar(result, n_rows)


def _row_velocity_from_cell_or_scalar(
    result: RefractionDatumStaticsResult,
    n_rows: int,
) -> np.ndarray:
    out = np.full(n_rows, _float_or_nan(result.bedrock_velocity_m_s), dtype=np.float64)
    if result.row_midpoint_cell_id is None or result.cell_bedrock_velocity_m_s is None:
        return np.ascontiguousarray(out, dtype=np.float64)
    cell_id = np.asarray(result.row_midpoint_cell_id, dtype=np.int64)
    if cell_id.shape != (n_rows,):
        return np.ascontiguousarray(out, dtype=np.float64)
    velocity = np.asarray(result.cell_bedrock_velocity_m_s, dtype=np.float64)
    active_cell_id = result.active_cell_id
    if active_cell_id is not None:
        active = np.asarray(active_cell_id, dtype=np.int64)
        if active.shape == velocity.shape:
            out.fill(np.nan)
            for raw_cell, raw_velocity in zip(active.tolist(), velocity.tolist(), strict=True):
                rows = cell_id == int(raw_cell)
                out[rows] = float(raw_velocity)
            return np.ascontiguousarray(out, dtype=np.float64)
    if velocity.ndim == 1 and velocity.size > int(np.max(cell_id, initial=-1)):
        valid = (cell_id >= 0) & (cell_id < int(velocity.size))
        out[valid] = velocity[cell_id[valid]]
    return np.ascontiguousarray(out, dtype=np.float64)


def _residual_row_cell_context(
    result: RefractionDatumStaticsResult,
    *,
    req: RefractionStaticApplyRequest | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    empty = np.full(n_rows, -1, dtype=np.int64)
    request = RefractionStaticApplyRequest.model_validate(req) if req is not None else None
    request_cell_layer_kinds = (
        _request_cell_velocity_layer_kinds(request) if request is not None else ()
    )
    request_has_cell_velocity = bool(request_cell_layer_kinds)
    if (
        not request_has_cell_velocity
        and result.bedrock_velocity_mode != 'solve_cell'
    ):
        return empty, empty.copy(), empty.copy()

    if result.row_midpoint_cell_id is None:
        raise RefractionStaticArtifactError(
            'cell velocity residual rows require row_midpoint_cell_id'
        )
    cell_id = np.ascontiguousarray(result.row_midpoint_cell_id, dtype=np.int64)
    if cell_id.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            'row_midpoint_cell_id length must match residual rows'
        )
    if request is not None and request_has_cell_velocity:
        layer_kind_by_row, _layer_index_by_row = _residual_row_layer_context(result)
        if np.any(layer_kind_by_row.astype(str, copy=False) != ''):
            supported_layer = np.isin(
                layer_kind_by_row.astype(str, copy=False),
                np.asarray(request_cell_layer_kinds, dtype=str),
            )
            cell_id = np.where(supported_layer, cell_id, -1)
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
        if _request_has_cell_velocity_layer(request) and refractor_cell is not None:
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
    explicit_reason: object = '',
) -> str:
    if rejected_by_robust:
        return 'robust_outlier'
    reason = str(explicit_reason)
    if reason:
        return reason
    if not used:
        return 'not_used'
    return 'ok'


def _component_rows(result: RefractionDatumStaticsResult) -> list[dict[str, object]]:
    node_pick_count = _node_lookup(result.node_id, result.node_pick_count)
    node_residual_rms = _node_lookup(result.node_id, result.node_residual_rms_s)
    has_source_depth = _has_source_depth_field_correction(result)
    has_uphole = _has_uphole_field_correction(result)
    has_manual_static = _has_manual_static_field_correction(result)
    has_field_composition = _has_field_correction_composition(result)
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        node_id = int(result.source_node_id[index])
        row = (
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
        if has_source_depth:
            assert result.source_depth_shift_s is not None
            assert result.source_depth_status is not None
            row.update(
                {
                    'source_depth_shift_ms': _csv_ms(
                        result.source_depth_shift_s[index]
                    ),
                    'source_depth_status': str(result.source_depth_status[index]),
                }
            )
        if has_uphole:
            assert result.source_uphole_shift_s is not None
            assert result.source_uphole_status is not None
            row.update(
                {
                    'uphole_shift_ms': _csv_ms(
                        result.source_uphole_shift_s[index]
                    ),
                    'uphole_status': str(result.source_uphole_status[index]),
                }
            )
        if has_manual_static:
            assert result.source_manual_static_shift_s is not None
            assert result.source_manual_static_status is not None
            row.update(
                {
                    'manual_static_shift_ms': _csv_ms(
                        result.source_manual_static_shift_s[index]
                    ),
                    'manual_static_status': str(
                        result.source_manual_static_status[index]
                    ),
                }
            )
        if has_field_composition:
            assert result.source_field_shift_s is not None
            assert result.source_field_static_status is not None
            row.update(
                {
                    'field_shift_ms': _csv_ms(result.source_field_shift_s[index]),
                    'field_status': str(result.source_field_static_status[index]),
                }
            )
        rows.append(row)
    for index in range(int(result.receiver_endpoint_key.shape[0])):
        node_id = int(result.receiver_node_id[index])
        row = (
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
        if has_source_depth:
            row.update(
                {
                    'source_depth_shift_ms': '',
                    'source_depth_status': 'not_applicable',
                }
            )
        if has_uphole:
            row.update(
                {
                    'uphole_shift_ms': '',
                    'uphole_status': 'not_applicable',
                }
            )
        if has_manual_static:
            assert result.receiver_manual_static_shift_s is not None
            assert result.receiver_manual_static_status is not None
            row.update(
                {
                    'manual_static_shift_ms': _csv_ms(
                        result.receiver_manual_static_shift_s[index]
                    ),
                    'manual_static_status': str(
                        result.receiver_manual_static_status[index]
                    ),
                }
            )
        if has_field_composition:
            assert result.receiver_field_shift_s is not None
            assert result.receiver_field_static_status is not None
            row.update(
                {
                    'field_shift_ms': _csv_ms(result.receiver_field_shift_s[index]),
                    'field_status': str(result.receiver_field_static_status[index]),
                }
            )
        rows.append(row)
    return rows


def _component_columns(result: RefractionDatumStaticsResult) -> tuple[str, ...]:
    columns = _COMPONENT_COLUMNS
    if _has_source_depth_field_correction(result):
        columns = _insert_after(
            columns,
            'flat_datum_shift_ms',
            ('source_depth_shift_ms', 'source_depth_status'),
        )
    if _has_uphole_field_correction(result):
        anchor = 'source_depth_status' if 'source_depth_status' in columns else 'flat_datum_shift_ms'
        columns = _insert_after(
            columns,
            anchor,
            ('uphole_shift_ms', 'uphole_status'),
        )
    if _has_manual_static_field_correction(result):
        if 'uphole_status' in columns:
            anchor = 'uphole_status'
        elif 'source_depth_status' in columns:
            anchor = 'source_depth_status'
        else:
            anchor = 'flat_datum_shift_ms'
        columns = _insert_after(
            columns,
            anchor,
            ('manual_static_shift_ms', 'manual_static_status'),
        )
    if _has_field_correction_composition(result):
        anchor = (
            'manual_static_status'
            if 'manual_static_status' in columns
            else (
                'uphole_status'
                if 'uphole_status' in columns
                else (
                    'source_depth_status'
                    if 'source_depth_status' in columns
                    else 'flat_datum_shift_ms'
                )
            )
        )
        columns = _insert_after(columns, anchor, ('field_shift_ms', 'field_status'))
    return columns


def _near_surface_columns(result: RefractionDatumStaticsResult) -> tuple[str, ...]:
    if _has_node_3layer_static_fields(result):
        return _NEAR_SURFACE_3LAYER_COLUMNS
    if _has_node_2layer_static_fields(result):
        return _NEAR_SURFACE_2LAYER_COLUMNS
    return _NEAR_SURFACE_COLUMNS


def _has_source_depth_field_correction(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_depth_m is not None,
        result.source_depth_shift_s is not None,
        result.source_depth_status is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionStaticArtifactError(
            'source-depth field correction arrays must be provided together'
        )
    return True


def _has_uphole_field_correction(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_uphole_time_s is not None,
        result.source_uphole_shift_s is not None,
        result.source_uphole_status is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionStaticArtifactError(
            'uphole field correction arrays must be provided together'
        )
    return True


def _has_manual_static_field_correction(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_manual_static_shift_s is not None,
        result.source_manual_static_status is not None,
        result.receiver_manual_static_shift_s is not None,
        result.receiver_manual_static_status is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionStaticArtifactError(
            'manual static field correction arrays must be provided together'
        )
    return True


def _has_field_correction_composition(
    result: RefractionDatumStaticsResult,
) -> bool:
    present = (
        result.source_field_shift_s is not None,
        result.source_field_static_status is not None,
        result.receiver_field_shift_s is not None,
        result.receiver_field_static_status is not None,
        result.source_field_shift_s_sorted is not None,
        result.receiver_field_shift_s_sorted is not None,
        result.trace_field_shift_s_sorted is not None,
        result.trace_field_static_status_sorted is not None,
        result.trace_field_static_valid_mask_sorted is not None,
        result.base_refraction_trace_shift_s_sorted is not None,
        result.field_composition_qc is not None,
    )
    if not any(present):
        return False
    if not all(present):
        raise RefractionStaticArtifactError(
            'field-correction composition arrays must be provided together'
        )
    return True


def _source_depth_m_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_source_depth_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_depth_m,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_depth_shift_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_source_depth_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_depth_shift_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_depth_status_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_source_depth_field_correction(result):
        return _disabled_field_status_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.source_depth_status,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_uphole_time_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_uphole_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_uphole_time_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_uphole_shift_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_uphole_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_uphole_shift_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_uphole_status_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_uphole_field_correction(result):
        return _disabled_field_status_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.source_uphole_status,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_manual_static_shift_s_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_manual_static_field_correction(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_manual_static_shift_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_manual_static_status_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_manual_static_field_correction(result):
        return _disabled_field_status_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.source_manual_static_status,
        int(result.source_endpoint_key.shape[0]),
    )


def _receiver_manual_static_shift_s_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_manual_static_field_correction(result):
        return _disabled_field_float_array(int(result.receiver_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.receiver_manual_static_shift_s,
        int(result.receiver_endpoint_key.shape[0]),
    )


def _receiver_manual_static_status_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_manual_static_field_correction(result):
        return _disabled_field_status_array(int(result.receiver_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.receiver_manual_static_status,
        int(result.receiver_endpoint_key.shape[0]),
    )


def _source_field_shift_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.source_field_shift_s,
        int(result.source_endpoint_key.shape[0]),
    )


def _source_field_static_status_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_status_array(int(result.source_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.source_field_static_status,
        int(result.source_endpoint_key.shape[0]),
    )


def _receiver_field_shift_s_array(result: RefractionDatumStaticsResult) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.receiver_endpoint_key.shape[0]))
    return _optional_field_float_array(
        result.receiver_field_shift_s,
        int(result.receiver_endpoint_key.shape[0]),
    )


def _receiver_field_static_status_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_status_array(int(result.receiver_endpoint_key.shape[0]))
    return _optional_field_status_array(
        result.receiver_field_static_status,
        int(result.receiver_endpoint_key.shape[0]),
    )


def _source_field_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.sorted_trace_index.shape[0]))
    return _optional_field_float_array(
        result.source_field_shift_s_sorted,
        int(result.sorted_trace_index.shape[0]),
    )


def _receiver_field_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.sorted_trace_index.shape[0]))
    return _optional_field_float_array(
        result.receiver_field_shift_s_sorted,
        int(result.sorted_trace_index.shape[0]),
    )


def _trace_field_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_float_array(int(result.sorted_trace_index.shape[0]))
    return _optional_field_float_array(
        result.trace_field_shift_s_sorted,
        int(result.sorted_trace_index.shape[0]),
    )


def _trace_field_static_status_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if not _has_field_correction_composition(result):
        return _disabled_field_status_array(int(result.sorted_trace_index.shape[0]))
    return _optional_field_status_array(
        result.trace_field_static_status_sorted,
        int(result.sorted_trace_index.shape[0]),
    )


def _base_refraction_trace_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.base_refraction_trace_shift_s_sorted is None:
        return _float_array(result.refraction_trace_shift_s_sorted)
    return _float_array(result.base_refraction_trace_shift_s_sorted)


def _optional_field_float_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError('field correction array is missing')
    arr = _float_array(value)
    if arr.shape != (int(shape),):
        raise RefractionStaticArtifactError(
            'field correction array has unexpected shape'
        )
    return arr


def _optional_field_status_array(value: object, shape: int) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError('field correction status array is missing')
    arr = _string_array(value)
    if arr.shape != (int(shape),):
        raise RefractionStaticArtifactError(
            'field correction status array has unexpected shape'
        )
    return arr


def _disabled_field_float_array(shape: int) -> np.ndarray:
    return np.zeros(int(shape), dtype=np.float64)


def _disabled_field_status_array(shape: int) -> np.ndarray:
    return np.full(int(shape), _FIELD_DISABLED_STATUS, dtype='<U16')


def _field_static_valid_mask(
    *,
    shift_s: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    status_text = np.asarray(status).astype(str)
    valid_status = np.isin(status_text, tuple(_FIELD_TOTAL_VALID_STATUSES))
    return np.ascontiguousarray(valid_status & np.isfinite(shift_s), dtype=bool)


def _total_with_field_shift_s(
    *,
    refraction_shift_s: np.ndarray,
    field_shift_s: np.ndarray,
    field_status: np.ndarray,
) -> np.ndarray:
    refraction = np.asarray(refraction_shift_s, dtype=np.float64)
    field = np.asarray(field_shift_s, dtype=np.float64)
    status = np.asarray(field_status).astype(str)
    if refraction.shape != field.shape or refraction.shape != status.shape:
        raise RefractionStaticArtifactError(
            'field total shift arrays must have matching shapes'
        )
    out = np.full(refraction.shape, np.nan, dtype=np.float64)
    valid = (
        np.isin(status, tuple(_FIELD_TOTAL_VALID_STATUSES))
        & np.isfinite(refraction)
        & np.isfinite(field)
    )
    out[valid] = refraction[valid] + field[valid]
    return np.ascontiguousarray(out, dtype=np.float64)


def _final_trace_shift_s_sorted(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.final_trace_shift_s_sorted is not None:
        return _float_array(result.final_trace_shift_s_sorted)
    return _total_with_field_shift_s(
        refraction_shift_s=_base_refraction_trace_shift_s_sorted_array(result),
        field_shift_s=_trace_field_shift_s_sorted_array(result),
        field_status=_trace_field_static_status_sorted_array(result),
    )


def _final_trace_static_status_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.final_trace_static_status_sorted is not None:
        return _string_array(result.final_trace_static_status_sorted)
    return _string_array(result.trace_static_status_sorted)


def _final_trace_static_valid_mask_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.final_trace_static_valid_mask_sorted is not None:
        return _bool_array(result.final_trace_static_valid_mask_sorted)
    return _bool_array(result.trace_static_valid_mask_sorted)


def _applied_field_shift_s_sorted_array(
    result: RefractionDatumStaticsResult,
) -> np.ndarray:
    if result.applied_field_shift_s_sorted is not None:
        return _float_array(result.applied_field_shift_s_sorted)
    return np.zeros(int(result.sorted_trace_index.shape[0]), dtype=np.float64)


def _trace_endpoint_key_sorted_array(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
) -> np.ndarray:
    if endpoint == 'source':
        raw = result.source_endpoint_key_sorted
    elif endpoint == 'receiver':
        raw = result.receiver_endpoint_key_sorted
    else:
        raise RefractionStaticArtifactError(f'unsupported endpoint kind: {endpoint}')

    expected_shape = result.sorted_trace_index.shape
    if raw is None:
        raise RefractionStaticArtifactError(f'{endpoint}_endpoint_key_sorted is required')
    out = _string_array(raw)
    if out.shape != expected_shape:
        raise RefractionStaticArtifactError(
            f'{endpoint}_endpoint_key_sorted shape mismatch'
        )
    return out


def _endpoint_shift_to_trace_order(
    *,
    endpoint_key: np.ndarray,
    endpoint_shift_s: np.ndarray,
    endpoint_key_sorted: np.ndarray,
    label: str,
) -> np.ndarray:
    shift_by_key: dict[str, float] = {}
    for key, shift in zip(endpoint_key.tolist(), endpoint_shift_s.tolist()):
        text = str(key)
        value = float(shift)
        existing = shift_by_key.get(text)
        if existing is not None and not (
            existing == value or (np.isnan(existing) and np.isnan(value))
        ):
            raise RefractionStaticArtifactError(
                f'{label} cannot be mapped to trace order; duplicate endpoint '
                f'key {text!r} has conflicting shifts'
            )
        shift_by_key[text] = value

    out = np.full(endpoint_key_sorted.shape, np.nan, dtype=np.float64)
    for index, key in enumerate(endpoint_key_sorted.tolist()):
        text = str(key)
        if text not in shift_by_key:
            raise RefractionStaticArtifactError(
                f'{label} cannot be mapped to trace order; endpoint key '
                f'{text!r} is missing'
            )
        out[index] = shift_by_key[text]
    return np.ascontiguousarray(out, dtype=np.float64)


def _applied_endpoint_field_shift_s(
    *,
    field_shift_s: np.ndarray,
    field_status: np.ndarray,
    apply_to_trace_shift: bool,
) -> np.ndarray:
    field = np.asarray(field_shift_s, dtype=np.float64)
    if not apply_to_trace_shift:
        return np.zeros(field.shape, dtype=np.float64)
    status = np.asarray(field_status).astype(str)
    out = np.full(field.shape, np.nan, dtype=np.float64)
    valid = np.isin(status, tuple(_FIELD_TOTAL_VALID_STATUSES)) & np.isfinite(field)
    out[valid] = field[valid]
    return np.ascontiguousarray(out, dtype=np.float64)


def _component_qc_stats_ms(
    components_s: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            'min': _stat(values * 1000.0, 'min'),
            'median': _stat(values * 1000.0, 'median'),
            'max': _stat(values * 1000.0, 'max'),
        }
        for name, values in components_s.items()
    }


from app.services.refraction_static_artifacts.field_corrections import (  # noqa: E402
    _applied_endpoint_field_shift_s,
    _applied_field_shift_s_sorted_array,
    _base_refraction_trace_shift_s_sorted_array,
    _disabled_field_float_array,
    _disabled_field_status_array,
    _endpoint_shift_to_trace_order,
    _field_static_valid_mask,
    _final_trace_shift_s_sorted,
    _final_trace_static_status_sorted_array,
    _final_trace_static_valid_mask_sorted_array,
    _has_field_correction_composition,
    _has_manual_static_field_correction,
    _has_source_depth_field_correction,
    _has_uphole_field_correction,
    _optional_field_float_array,
    _optional_field_status_array,
    _receiver_field_shift_s_array,
    _receiver_field_shift_s_sorted_array,
    _receiver_field_static_status_array,
    _receiver_manual_static_shift_s_array,
    _receiver_manual_static_status_array,
    _source_depth_m_array,
    _source_depth_shift_s_array,
    _source_depth_status_array,
    _source_field_shift_s_array,
    _source_field_shift_s_sorted_array,
    _source_field_static_status_array,
    _source_manual_static_shift_s_array,
    _source_manual_static_status_array,
    _source_uphole_shift_s_array,
    _source_uphole_status_array,
    _source_uphole_time_s_array,
    _total_with_field_shift_s,
    _trace_endpoint_key_sorted_array,
    _trace_field_shift_s_sorted_array,
    _trace_field_static_status_sorted_array,
)

_LINE_PROFILE_STRING_COLUMNS = frozenset(
    {'endpoint_kind', 'endpoint_key', 'static_status', 'solution_status'}
)
_LINE_PROFILE_INT_COLUMNS = frozenset(
    {'node_id', 'pick_count', 'used_pick_count'}
)


def _line_profile_qc_available(req: RefractionStaticApplyRequest) -> bool:
    refractor_cell = req.model.refractor_cell
    return (
        refractor_cell is not None
        and refractor_cell.coordinate_mode == 'line_2d_projected'
    )


def _line_profile_qc_availability_reason(
    req: RefractionStaticApplyRequest,
) -> str:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        return 'no_projected_inline_coordinate_model'
    if refractor_cell.coordinate_mode != 'line_2d_projected':
        return 'projected_inline_coordinates_unavailable_for_grid_3d'
    return 'line_2d_projected'


def _line_profile_coordinate_metadata(
    req: RefractionStaticApplyRequest,
) -> dict[str, Any]:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        return {
            'coordinate_mode': 'grid_3d',
            'line_origin_x_m': None,
            'line_origin_y_m': None,
            'line_azimuth_deg': None,
        }
    return refraction_cell_coordinate_metadata_from_config(refractor_cell)


def _empty_line_profile_qc_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for column in _LINE_PROFILE_QC_COLUMNS:
        if column in _LINE_PROFILE_STRING_COLUMNS:
            arrays[column] = np.asarray([], dtype='<U1')
        elif column in _LINE_PROFILE_INT_COLUMNS:
            arrays[column] = np.asarray([], dtype=np.int64)
        else:
            arrays[column] = np.asarray([], dtype=np.float64)
    return arrays


def _line_profile_endpoint_arrays(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    *,
    endpoint: str,
) -> dict[str, np.ndarray]:
    if endpoint == 'source':
        endpoint_key = _string_array(result.source_endpoint_key)
        node_id = _int_array(result.source_node_id)
        x_m = _float_array(result.source_x_m)
        y_m = _float_array(result.source_y_m)
        surface_elevation_m = _float_array(result.source_surface_elevation_m)
        t1_s = _float_array(result.source_half_intercept_time_s)
        t2_s = _optional_line_profile_float_array(
            result.source_t2_time_s,
            shape=endpoint_key.shape[0],
        )
        t3_s = _optional_line_profile_float_array(
            result.source_t3_time_s,
            shape=endpoint_key.shape[0],
        )
        v2_m_s = _endpoint_v2_m_s(
            result.source_v2_m_s,
            shape=endpoint_key.shape[0],
            scalar_v2_m_s=result.bedrock_velocity_m_s,
        )
        v3_m_s = _optional_line_profile_float_array(
            result.source_v3_m_s,
            shape=endpoint_key.shape[0],
        )
        vsub_m_s = _optional_line_profile_float_array(
            result.source_vsub_m_s,
            shape=endpoint_key.shape[0],
        )
        sh1_m = _source_sh1_weathering_thickness_m(result)
        sh2_m = _optional_line_profile_float_array(
            result.source_sh2_weathering_thickness_m,
            shape=endpoint_key.shape[0],
        )
        sh3_m = _optional_line_profile_float_array(
            result.source_sh3_weathering_thickness_m,
            shape=endpoint_key.shape[0],
        )
        weathering_correction_s = _float_array(
            result.source_weathering_replacement_shift_s
        )
        elevation_correction_s = _sum_float_arrays(
            result.source_floating_datum_elevation_shift_s,
            result.source_flat_datum_shift_s,
        )
        field_correction_s = _source_field_shift_s_array(result)
        manual_static_s = _source_manual_static_shift_s_array(result)
        total_static_s = _float_array(result.source_refraction_shift_s)
        source_field_shift_s = field_correction_s
        receiver_field_shift_s = np.full_like(field_correction_s, np.nan)
        source_total_with_field_shift_s = _total_with_field_shift_s(
            refraction_shift_s=result.source_refraction_shift_s,
            field_shift_s=_source_field_shift_s_array(result),
            field_status=_source_field_static_status_array(result),
        )
        receiver_total_with_field_shift_s = np.full_like(
            source_total_with_field_shift_s,
            np.nan,
        )
        static_status = _source_static_status_array(result)
        kind = 'source'
    elif endpoint == 'receiver':
        endpoint_key = _string_array(result.receiver_endpoint_key)
        node_id = _int_array(result.receiver_node_id)
        x_m = _float_array(result.receiver_x_m)
        y_m = _float_array(result.receiver_y_m)
        surface_elevation_m = _float_array(result.receiver_surface_elevation_m)
        t1_s = _float_array(result.receiver_half_intercept_time_s)
        t2_s = _optional_line_profile_float_array(
            result.receiver_t2_time_s,
            shape=endpoint_key.shape[0],
        )
        t3_s = _optional_line_profile_float_array(
            result.receiver_t3_time_s,
            shape=endpoint_key.shape[0],
        )
        v2_m_s = _endpoint_v2_m_s(
            result.receiver_v2_m_s,
            shape=endpoint_key.shape[0],
            scalar_v2_m_s=result.bedrock_velocity_m_s,
        )
        v3_m_s = _optional_line_profile_float_array(
            result.receiver_v3_m_s,
            shape=endpoint_key.shape[0],
        )
        vsub_m_s = _optional_line_profile_float_array(
            result.receiver_vsub_m_s,
            shape=endpoint_key.shape[0],
        )
        sh1_m = _receiver_sh1_weathering_thickness_m(result)
        sh2_m = _optional_line_profile_float_array(
            result.receiver_sh2_weathering_thickness_m,
            shape=endpoint_key.shape[0],
        )
        sh3_m = _optional_line_profile_float_array(
            result.receiver_sh3_weathering_thickness_m,
            shape=endpoint_key.shape[0],
        )
        weathering_correction_s = _float_array(
            result.receiver_weathering_replacement_shift_s
        )
        elevation_correction_s = _sum_float_arrays(
            result.receiver_floating_datum_elevation_shift_s,
            result.receiver_flat_datum_shift_s,
        )
        field_correction_s = _receiver_field_shift_s_array(result)
        manual_static_s = _receiver_manual_static_shift_s_array(result)
        total_static_s = _float_array(result.receiver_refraction_shift_s)
        source_field_shift_s = np.full_like(field_correction_s, np.nan)
        receiver_field_shift_s = field_correction_s
        source_total_with_field_shift_s = np.full_like(field_correction_s, np.nan)
        receiver_total_with_field_shift_s = _total_with_field_shift_s(
            refraction_shift_s=result.receiver_refraction_shift_s,
            field_shift_s=_receiver_field_shift_s_array(result),
            field_status=_receiver_field_static_status_array(result),
        )
        static_status = _receiver_static_status_array(result)
        kind = 'receiver'
    else:
        raise RefractionStaticArtifactError(f'unsupported endpoint kind: {endpoint}')

    projected = _line_profile_projected_coordinates(
        req=req,
        x_m=x_m,
        y_m=y_m,
    )
    node_context = _node_context(result)
    layer1_base = _line_profile_layer1_base_elevation(
        surface_elevation_m=surface_elevation_m,
        sh1_m=sh1_m,
    )
    layer2_base = _line_profile_layer2_base_elevation(
        layer1_base_elevation_m=layer1_base,
        sh2_m=sh2_m,
        has_3layer=bool(np.any(np.isfinite(sh3_m))),
    )
    n_endpoints = int(endpoint_key.shape[0])
    return {
        'endpoint_kind': _string_array(np.full(n_endpoints, kind, dtype=f'<U{len(kind)}')),
        'endpoint_key': endpoint_key,
        'node_id': node_id,
        'inline_m': projected['inline_m'],
        'crossline_m': projected['crossline_m'],
        'x_m': x_m,
        'y_m': y_m,
        'surface_elevation_m': surface_elevation_m,
        'pick_count': _line_profile_node_int_values(
            node_id,
            node_context['pick_count'],
        ),
        'used_pick_count': _line_profile_node_int_values(
            node_id,
            node_context['used_pick_count'],
        ),
        'residual_rms_ms': _line_profile_seconds_to_ms(
            _line_profile_node_float_values(node_id, node_context['residual_rms'])
        ),
        'residual_mad_ms': _line_profile_seconds_to_ms(
            _line_profile_node_float_values(node_id, node_context['residual_mad'])
        ),
        'v1_m_s': _filled_float_array(result.weathering_velocity_m_s, n_endpoints),
        'v2_m_s': v2_m_s,
        'v3_m_s': v3_m_s,
        'vsub_m_s': vsub_m_s,
        't1_ms': _line_profile_seconds_to_ms(t1_s),
        't2_ms': _line_profile_seconds_to_ms(t2_s),
        't3_ms': _line_profile_seconds_to_ms(t3_s),
        'sh1_m': sh1_m,
        'sh2_m': sh2_m,
        'sh3_m': sh3_m,
        'layer1_base_elevation_m': layer1_base,
        'layer2_base_elevation_m': layer2_base,
        'final_refractor_elevation_m': _float_array(
            result.source_refractor_elevation_m
            if endpoint == 'source'
            else result.receiver_refractor_elevation_m
        ),
        'weathering_correction_ms': _line_profile_seconds_to_ms(
            weathering_correction_s
        ),
        'elevation_correction_ms': _line_profile_seconds_to_ms(
            elevation_correction_s
        ),
        'source_field_shift_ms': _line_profile_seconds_to_ms(source_field_shift_s),
        'receiver_field_shift_ms': _line_profile_seconds_to_ms(
            receiver_field_shift_s
        ),
        'field_correction_ms': _line_profile_seconds_to_ms(field_correction_s),
        'manual_static_shift_ms': _line_profile_seconds_to_ms(manual_static_s),
        'manual_static_ms': _line_profile_seconds_to_ms(manual_static_s),
        'total_static_ms': _line_profile_seconds_to_ms(total_static_s),
        'total_applied_shift_ms': _line_profile_seconds_to_ms(total_static_s),
        'source_total_with_field_shift_ms': _line_profile_seconds_to_ms(
            source_total_with_field_shift_s
        ),
        'receiver_total_with_field_shift_ms': _line_profile_seconds_to_ms(
            receiver_total_with_field_shift_s
        ),
        'static_status': _string_array(static_status),
        'solution_status': _line_profile_node_string_values(
            node_id,
            node_context['solution_status'],
            default='missing_solution',
        ),
    }


def _line_profile_projected_coordinates(
    *,
    req: RefractionStaticApplyRequest,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> dict[str, np.ndarray]:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'line-profile QC requires model.refractor_cell'
        )
    projected = project_refraction_cell_points(
        x_m=x_m,
        y_m=y_m,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    if (
        projected.projected_inline_m is None
        or projected.projected_crossline_m is None
    ):
        raise RefractionStaticArtifactError(
            'line-profile QC requires projected inline/crossline coordinates'
        )
    return {
        'inline_m': np.ascontiguousarray(
            projected.projected_inline_m,
            dtype=np.float64,
        ),
        'crossline_m': np.ascontiguousarray(
            projected.projected_crossline_m,
            dtype=np.float64,
        ),
    }


def _optional_line_profile_float_array(
    value: object,
    *,
    shape: int,
) -> np.ndarray:
    if value is None:
        return np.full(int(shape), np.nan, dtype=np.float64)
    array = _float_array(value)
    if array.shape != (int(shape),):
        raise RefractionStaticArtifactError(
            'line-profile optional endpoint array length mismatch'
        )
    return array


def _line_profile_seconds_to_ms(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64) * 1000.0)


def _line_profile_layer1_base_elevation(
    *,
    surface_elevation_m: np.ndarray,
    sh1_m: np.ndarray,
) -> np.ndarray:
    surface = np.asarray(surface_elevation_m, dtype=np.float64)
    sh1 = np.asarray(sh1_m, dtype=np.float64)
    out = np.full(surface.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(surface) & np.isfinite(sh1)
    out[finite] = surface[finite] - sh1[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _line_profile_layer2_base_elevation(
    *,
    layer1_base_elevation_m: np.ndarray,
    sh2_m: np.ndarray,
    has_3layer: bool,
) -> np.ndarray:
    out = np.full(np.asarray(layer1_base_elevation_m).shape, np.nan, dtype=np.float64)
    if not has_3layer:
        return out
    layer1 = np.asarray(layer1_base_elevation_m, dtype=np.float64)
    sh2 = np.asarray(sh2_m, dtype=np.float64)
    finite = np.isfinite(layer1) & np.isfinite(sh2)
    out[finite] = layer1[finite] - sh2[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _line_profile_node_int_values(
    node_id: np.ndarray,
    lookup: Mapping[int, Any],
) -> np.ndarray:
    out = np.full(np.asarray(node_id).shape, -1, dtype=np.int64)
    for index, raw_node_id in enumerate(np.asarray(node_id).tolist()):
        value = lookup.get(int(raw_node_id))
        if value is not None:
            out[index] = int(value)
    return np.ascontiguousarray(out, dtype=np.int64)


def _line_profile_node_float_values(
    node_id: np.ndarray,
    lookup: Mapping[int, Any],
) -> np.ndarray:
    out = np.full(np.asarray(node_id).shape, np.nan, dtype=np.float64)
    for index, raw_node_id in enumerate(np.asarray(node_id).tolist()):
        value = lookup.get(int(raw_node_id))
        if value is not None:
            out[index] = _float_or_nan(value)
    return np.ascontiguousarray(out, dtype=np.float64)


def _line_profile_node_string_values(
    node_id: np.ndarray,
    lookup: Mapping[int, Any],
    *,
    default: str,
) -> np.ndarray:
    values = [
        str(lookup.get(int(raw_node_id), default))
        for raw_node_id in np.asarray(node_id).tolist()
    ]
    return _string_array(values)


def _line_profile_qc_rows(
    arrays: Mapping[str, np.ndarray],
    *,
    endpoint_kind: str | None = None,
) -> list[dict[str, object]]:
    kind = np.asarray(arrays['endpoint_kind']).astype(str, copy=False)
    selected = np.arange(kind.shape[0], dtype=np.int64)
    if endpoint_kind is not None:
        selected = selected[kind == endpoint_kind]
    rows: list[dict[str, object]] = []
    for index in selected.tolist():
        rows.append(
            {
                column: _line_profile_csv_value(column, arrays[column][index])
                for column in _LINE_PROFILE_QC_COLUMNS
            }
        )
    return rows


def _line_profile_csv_value(column: str, value: object) -> object:
    if column in _LINE_PROFILE_STRING_COLUMNS:
        return str(value)
    if column in _LINE_PROFILE_INT_COLUMNS:
        out = _csv_int(value)
        return '' if out != '' and int(out) < 0 else out
    return _csv_float(value)


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


def _endpoint_layer_qc_context(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
) -> dict[str, dict[str, dict[str, int | float]]]:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    layer_kind, _layer_index = _residual_row_layer_context(result)
    endpoint_field = (
        'row_source_endpoint_key'
        if endpoint == 'source'
        else 'row_receiver_endpoint_key'
    )
    endpoint_key = _residual_row_string_context(result, endpoint_field)
    used = np.asarray(result.used_row_mask, dtype=bool)
    residual_s = np.asarray(result.residual_time_s, dtype=np.float64)
    context: dict[str, dict[str, dict[str, Any]]] = {
        'pick_count': {},
        'used_pick_count': {},
        'residual_values_ms': {},
    }
    for row_index in range(n_rows):
        kind = str(layer_kind[row_index])
        key = str(endpoint_key[row_index])
        if not kind or not key:
            continue
        _increment_layer_count(context['pick_count'], key, kind)
        if bool(used[row_index]):
            _increment_layer_count(context['used_pick_count'], key, kind)
            residual = residual_s[row_index]
            if np.isfinite(residual):
                values = context['residual_values_ms'].setdefault(key, {}).setdefault(
                    kind,
                    [],
                )
                values.append(float(residual) * 1000.0)
    return {
        'pick_count': context['pick_count'],
        'used_pick_count': context['used_pick_count'],
        'residual_rms_ms': _endpoint_layer_residual_stat(
            context['residual_values_ms'],
            stat='rms',
        ),
        'residual_mad_ms': _endpoint_layer_residual_stat(
            context['residual_values_ms'],
            stat='mad',
        ),
    }


def _endpoint_layer_qc_row_fields(
    layer_context: dict[str, dict[str, dict[str, int | float]]],
    endpoint_key: str,
) -> dict[str, str]:
    return {
        'pick_count_by_layer': _csv_json_object(
            layer_context['pick_count'].get(endpoint_key, {})
        ),
        'used_pick_count_by_layer': _csv_json_object(
            layer_context['used_pick_count'].get(endpoint_key, {})
        ),
        'residual_rms_by_layer_ms': _csv_json_object(
            layer_context['residual_rms_ms'].get(endpoint_key, {})
        ),
        'residual_mad_by_layer_ms': _csv_json_object(
            layer_context['residual_mad_ms'].get(endpoint_key, {})
        ),
    }


def _increment_layer_count(
    target: dict[str, dict[str, int]],
    endpoint_key: str,
    layer_kind: str,
) -> None:
    by_layer = target.setdefault(endpoint_key, {})
    by_layer[layer_kind] = int(by_layer.get(layer_kind, 0)) + 1


def _endpoint_layer_residual_stat(
    values_by_endpoint: dict[str, dict[str, list[float]]],
    *,
    stat: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for endpoint_key, values_by_layer in values_by_endpoint.items():
        by_layer: dict[str, float] = {}
        for layer_kind, values in values_by_layer.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            if stat == 'rms':
                by_layer[layer_kind] = float(np.sqrt(np.mean(arr * arr)))
            elif stat == 'mad':
                by_layer[layer_kind] = float(np.median(np.abs(arr - np.median(arr))))
            else:
                raise RefractionStaticArtifactError(
                    f'unsupported endpoint layer residual stat: {stat}'
                )
        out[endpoint_key] = by_layer
    return out


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


def _layer_velocity_modes_for_request(
    req: RefractionStaticApplyRequest,
) -> dict[str, str]:
    if req.model.method != 'multilayer_time_term':
        return {}
    return {
        str(config.kind): str(config.velocity_mode)
        for config in normalize_refraction_static_layers(req.model)
    }


def _request_has_cell_velocity_layer(req: RefractionStaticApplyRequest) -> bool:
    return bool(_request_cell_velocity_layer_kinds(req))


def _request_cell_velocity_layer(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> Any | None:
    if req.model.method != 'multilayer_time_term':
        return None
    for layer in normalize_refraction_static_layers(req.model):
        if layer.velocity_mode != 'solve_cell':
            continue
        if layer_kind is None or layer.kind == layer_kind:
            return layer
    return None


def _result_has_cell_velocity_arrays(result: RefractionDatumStaticsResult) -> bool:
    return (
        result.active_cell_id is not None
        and result.inactive_cell_id is not None
        and result.cell_bedrock_velocity_m_s is not None
        and result.cell_bedrock_slowness_s_per_m is not None
        and result.cell_velocity_status is not None
    )


def _cell_velocity_artifact_result_for_layer(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer_kind: RefractionLayerKind,
) -> RefractionDatumStaticsResult:
    if req.model.method != 'multilayer_time_term':
        return result

    layer = _cell_velocity_layer_result(result, layer_kind)
    if layer is None:
        if (
            len(_request_cell_velocity_layer_kinds(req)) == 1
            and _result_has_cell_velocity_arrays(result)
        ):
            return result
        raise RefractionStaticArtifactError(
            f'{layer_kind} cell velocity artifacts require layer solve result arrays'
        )

    return _cell_velocity_artifact_result_from_layer(
        result=result,
        req=req,
        layer=layer,
    )


def _cell_velocity_layer_result(
    result: RefractionDatumStaticsResult,
    layer_kind: RefractionLayerKind,
) -> RefractionLayerSolveResult | None:
    for layer in result.layer_results or ():
        if layer.layer_kind == layer_kind:
            return layer
    return None


def _cell_velocity_artifact_result_from_layer(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer: RefractionLayerSolveResult,
) -> RefractionDatumStaticsResult:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    active_cell_id = _required_layer_cell_id_array(
        layer.active_cell_id,
        name=f'{layer.layer_kind}.active_cell_id',
    )
    inactive_cell_id = _required_layer_cell_id_array(
        layer.inactive_cell_id,
        name=f'{layer.layer_kind}.inactive_cell_id',
    )
    n_total_cells = _cell_velocity_n_total_cells(req)
    cell_velocity = _layer_cell_values_for_active_cells(
        layer.cell_velocity_m_s,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name=f'{layer.layer_kind}.cell_velocity_m_s',
    )
    cell_slowness = _layer_cell_values_for_active_cells(
        layer.cell_slowness_s_per_m,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name=f'{layer.layer_kind}.cell_slowness_s_per_m',
    )
    cell_status = _layer_cell_status_for_active_cells(
        layer.cell_velocity_status,
        active_cell_id=active_cell_id,
        n_total_cells=n_total_cells,
        name=f'{layer.layer_kind}.cell_velocity_status',
    )
    velocity_summary = _stat(cell_velocity, 'median')
    slowness_summary = _stat(cell_slowness, 'median')
    qc = {**result.qc, **layer.qc}
    if 'layers' in result.qc:
        qc['layers'] = result.qc['layers']
    return replace(
        result,
        bedrock_velocity_mode='solve_cell',
        bedrock_velocity_m_s=(
            result.bedrock_velocity_m_s
            if velocity_summary is None
            else velocity_summary
        ),
        bedrock_slowness_s_per_m=(
            result.bedrock_slowness_s_per_m
            if slowness_summary is None
            else slowness_summary
        ),
        active_cell_id=active_cell_id,
        inactive_cell_id=inactive_cell_id,
        cell_bedrock_velocity_m_s=cell_velocity,
        cell_bedrock_slowness_s_per_m=cell_slowness,
        cell_velocity_status=cell_status,
        row_midpoint_cell_id=_row_midpoint_cell_id_for_cell_velocity_layer(
            result=result,
            req=req,
            layer=layer,
            n_rows=n_rows,
        ),
        modeled_pick_time_s=_layer_trace_float_array(
            layer.trace_predicted_time_s_sorted,
            n_rows=n_rows,
            name=f'{layer.layer_kind}.trace_predicted_time_s_sorted',
        ),
        residual_time_s=_layer_trace_float_array(
            layer.trace_residual_s_sorted,
            n_rows=n_rows,
            name=f'{layer.layer_kind}.trace_residual_s_sorted',
        ),
        used_row_mask=_layer_trace_bool_array(
            layer.used_observation_mask_sorted,
            n_rows=n_rows,
            name=f'{layer.layer_kind}.used_observation_mask_sorted',
        ),
        rejected_by_robust_mask=_layer_trace_bool_array(
            (
                np.zeros(n_rows, dtype=bool)
                if layer.rejected_by_robust_mask_sorted is None
                else layer.rejected_by_robust_mask_sorted
            ),
            n_rows=n_rows,
            name=f'{layer.layer_kind}.rejected_by_robust_mask_sorted',
        ),
        qc=qc,
    )


def _cell_velocity_n_total_cells(req: RefractionStaticApplyRequest) -> int:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )
    grid = build_refraction_cell_grid(
        effective_refraction_cell_grid_config(refractor_cell)
    )
    return int(grid.cell_id.shape[0])


def _required_layer_cell_id_array(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'{name} is required')
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 1:
        raise RefractionStaticArtifactError(f'{name} must be one-dimensional')
    return np.ascontiguousarray(array, dtype=np.int64)


def _layer_cell_values_for_active_cells(
    value: object,
    *,
    active_cell_id: np.ndarray,
    n_total_cells: int,
    name: str,
) -> np.ndarray:
    if value is None:
        raise RefractionStaticArtifactError(f'{name} is required')
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise RefractionStaticArtifactError(f'{name} must be one-dimensional')
    if array.shape == active_cell_id.shape:
        return np.ascontiguousarray(array, dtype=np.float64)
    if array.shape == (int(n_total_cells),):
        return np.ascontiguousarray(array[active_cell_id], dtype=np.float64)
    raise RefractionStaticArtifactError(
        f'{name} must match active_cell_id length or total cell count'
    )


def _layer_cell_status_for_active_cells(
    value: object,
    *,
    active_cell_id: np.ndarray,
    n_total_cells: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return _string_array(np.full(active_cell_id.shape, 'solved', dtype='<U6'))
    status = _string_array(value)
    if status.shape == active_cell_id.shape:
        _validate_status_array(status, name=name)
        return status
    if status.shape == (int(n_total_cells),):
        out = _string_array(status[active_cell_id])
        _validate_status_array(out, name=name)
        return out
    raise RefractionStaticArtifactError(
        f'{name} must match active_cell_id length or total cell count'
    )


def _layer_trace_float_array(value: object, *, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (int(n_rows),):
        raise RefractionStaticArtifactError(
            f'{name} length must match residual rows'
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def _layer_trace_bool_array(value: object, *, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != (int(n_rows),):
        raise RefractionStaticArtifactError(
            f'{name} length must match residual rows'
        )
    return np.ascontiguousarray(array, dtype=bool)


def _row_midpoint_cell_id_for_cell_velocity_layer(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    layer: RefractionLayerSolveResult,
    n_rows: int,
) -> np.ndarray:
    if layer.row_midpoint_cell_id is not None:
        array = np.asarray(layer.row_midpoint_cell_id, dtype=np.int64)
        if array.shape == (int(n_rows),):
            return np.ascontiguousarray(array, dtype=np.int64)
    if result.row_midpoint_cell_id is not None:
        array = np.asarray(result.row_midpoint_cell_id, dtype=np.int64)
        if array.shape == (int(n_rows),):
            return np.ascontiguousarray(array, dtype=np.int64)
    return _compute_row_midpoint_cell_id(result=result, req=req, n_rows=n_rows)


def _compute_row_midpoint_cell_id(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    n_rows: int,
) -> np.ndarray:
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )
    source_x, source_y = _row_endpoint_xy(result, endpoint='source', n_rows=n_rows)
    receiver_x, receiver_y = _row_endpoint_xy(
        result,
        endpoint='receiver',
        n_rows=n_rows,
    )
    source_projected = project_refraction_cell_points(
        x_m=source_x,
        y_m=source_y,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    receiver_projected = project_refraction_cell_points(
        x_m=receiver_x,
        y_m=receiver_y,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    grid = build_refraction_cell_grid(
        effective_refraction_cell_grid_config(refractor_cell)
    )
    assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=source_projected.x_m,
        source_y_m=source_projected.y_m,
        receiver_x_m=receiver_projected.x_m,
        receiver_y_m=receiver_projected.y_m,
    )
    return np.ascontiguousarray(assignment.cell_id, dtype=np.int64)


def _row_endpoint_xy(
    result: RefractionDatumStaticsResult,
    *,
    endpoint: str,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if endpoint == 'source':
        row_key = result.row_source_endpoint_key
        endpoint_key = result.source_endpoint_key
        x_m = result.source_x_m
        y_m = result.source_y_m
    elif endpoint == 'receiver':
        row_key = result.row_receiver_endpoint_key
        endpoint_key = result.receiver_endpoint_key
        x_m = result.receiver_x_m
        y_m = result.receiver_y_m
    else:
        raise RefractionStaticArtifactError(f'unsupported endpoint: {endpoint}')
    if row_key is None:
        raise RefractionStaticArtifactError(
            f'row_{endpoint}_endpoint_key is required for layer cell artifacts'
        )
    row_key_array = np.asarray(row_key, dtype=object)
    if row_key_array.shape != (int(n_rows),):
        raise RefractionStaticArtifactError(
            f'row_{endpoint}_endpoint_key length must match residual rows'
        )
    endpoint_key_array = np.asarray(endpoint_key, dtype=object)
    x_array = np.asarray(x_m, dtype=np.float64)
    y_array = np.asarray(y_m, dtype=np.float64)
    if endpoint_key_array.shape != x_array.shape or x_array.shape != y_array.shape:
        raise RefractionStaticArtifactError(
            f'{endpoint} endpoint key and coordinate arrays must match'
        )
    index_by_key = {
        str(key): index for index, key in enumerate(endpoint_key_array.tolist())
    }
    indices: list[int] = []
    for key in row_key_array.tolist():
        try:
            indices.append(index_by_key[str(key)])
        except KeyError as exc:
            raise RefractionStaticArtifactError(
                f'row {endpoint} endpoint key {key!r} is missing from endpoint table'
            ) from exc
    index_array = np.asarray(indices, dtype=np.int64)
    return (
        np.ascontiguousarray(x_array[index_array], dtype=np.float64),
        np.ascontiguousarray(y_array[index_array], dtype=np.float64),
    )


def _cell_velocity_candidate_row_mask(
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> np.ndarray:
    n_rows = int(result.row_trace_index_sorted.shape[0])
    layer = _request_cell_velocity_layer(req, layer_kind=layer_kind)
    if layer is None:
        return np.ones(n_rows, dtype=bool)
    distance = np.asarray(result.row_distance_m, dtype=np.float64)
    if distance.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            'row_distance_m length must match residual rows'
        )
    valid_observation = np.asarray(result.valid_observation_mask_sorted, dtype=bool)
    if valid_observation.shape != (n_rows,):
        raise RefractionStaticArtifactError(
            'valid_observation_mask_sorted length must match residual rows'
        )
    try:
        masks = build_refraction_layer_observation_masks_from_arrays(
            base_valid_mask_sorted=valid_observation,
            offset_m_sorted=distance,
            rejection_reason_sorted=np.full(n_rows, 'ok', dtype='<U32'),
            model=req.model,
        )
        mask = masks.layer_used_mask_sorted[layer.kind]
    except (KeyError, ValueError) as exc:
        raise RefractionStaticArtifactError(
            f'could not build cell velocity layer observation mask for {layer.kind}'
        ) from exc
    return np.ascontiguousarray(mask, dtype=bool)


def _cell_velocity_component(layer_kind: RefractionLayerKind) -> str:
    try:
        return _CELL_VELOCITY_COMPONENT_BY_LAYER[layer_kind]
    except KeyError as exc:
        raise RefractionStaticArtifactError(
            f'unsupported cell velocity layer kind: {layer_kind}'
        ) from exc


def _cell_velocity_layer_bounds(
    req: RefractionStaticApplyRequest,
    layer: Any,
) -> tuple[float | None, float | None]:
    min_velocity = getattr(layer, 'min_velocity_m_s', None)
    max_velocity = getattr(layer, 'max_velocity_m_s', None)
    if layer.kind == 'v2_t1':
        if min_velocity is None:
            min_velocity = req.model.min_bedrock_velocity_m_s
        if max_velocity is None:
            max_velocity = req.model.max_bedrock_velocity_m_s
    return min_velocity, max_velocity


def _cell_velocity_min_observations_per_cell(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> int:
    layer = _request_cell_velocity_layer(req, layer_kind=layer_kind)
    if layer is not None and layer.min_observations_per_cell is not None:
        return int(layer.min_observations_per_cell)
    refractor_cell = req.model.refractor_cell
    if refractor_cell is None:
        raise RefractionStaticArtifactError(
            'model.refractor_cell is required for cell velocity artifacts'
        )
    return int(refractor_cell.min_observations_per_cell)


def _grid_map_qc_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = int(arrays['layer_kind'].shape[0])
    for index in range(n_rows):
        rows.append(
            {
                'layer_kind': str(arrays['layer_kind'][index]),
                'cell_ix': int(arrays['cell_ix'][index]),
                'cell_iy': int(arrays['cell_iy'][index]),
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
                'velocity_m_s': _csv_float(arrays['velocity_m_s'][index]),
                'initial_velocity_m_s': _csv_float(
                    arrays['initial_velocity_m_s'][index]
                ),
                'velocity_update_from_initial_m_s': _csv_float(
                    arrays['velocity_update_from_initial_m_s'][index]
                ),
                'slowness_s_per_m': _csv_float(arrays['slowness_s_per_m'][index]),
                'n_observations': int(arrays['n_observations'][index]),
                'n_sources': int(arrays['n_sources'][index]),
                'n_receivers': int(arrays['n_receivers'][index]),
                'residual_rms_ms': _csv_float(arrays['residual_rms_ms'][index]),
                'residual_mad_ms': _csv_float(arrays['residual_mad_ms'][index]),
                'status': str(arrays['status'][index]),
                'status_reason': str(arrays['status_reason'][index]),
            }
        )
    return rows


def _grid_map_qc_layer_summary(
    arrays: dict[str, np.ndarray],
    *,
    mask: np.ndarray,
) -> dict[str, Any]:
    status = np.asarray(arrays['status']).astype(str, copy=False)[mask]
    status_reason = np.asarray(arrays['status_reason']).astype(str, copy=False)[mask]
    active_mask = np.asarray(arrays['active_cell_mask'], dtype=bool)[mask]
    velocity = np.asarray(arrays['velocity_m_s'], dtype=np.float64)[mask]
    active_velocity = velocity[active_mask & np.isfinite(velocity)]
    return {
        'active_cell_count': int(np.count_nonzero(active_mask)),
        'empty_cell_count': int(np.count_nonzero(status_reason == 'no_observations')),
        'low_fold_cell_count': int(
            np.count_nonzero(status == LOW_FOLD_CELL_VELOCITY_STATUS)
        ),
        'velocity_min_m_s': _stat(active_velocity, 'min'),
        'velocity_median_m_s': _stat(active_velocity, 'median'),
        'velocity_max_m_s': _stat(active_velocity, 'max'),
        'status_counts': _status_counts(status),
    }


def _grid_map_qc_global_velocity_layers(
    req: RefractionStaticApplyRequest,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for layer in normalize_refraction_static_layers(req.model):
        if layer.velocity_mode == 'solve_cell':
            continue
        rows.append(
            {
                'layer_kind': layer.kind,
                'velocity_mode': layer.velocity_mode,
                'row_behavior': 'omitted',
            }
        )
    return rows


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
                'cell_velocity_layer_kind': str(
                    arrays['cell_velocity_layer_kind'][index]
                ),
                'cell_velocity_component': str(
                    arrays['cell_velocity_component'][index]
                ),
                'velocity_m_s': _csv_float(arrays['velocity_m_s'][index]),
                'v2_m_s': _csv_float(arrays['v2_m_s'][index]),
                'slowness_s_per_m': _csv_float(
                    arrays['slowness_s_per_m'][index]
                ),
                'initial_velocity_m_s': _csv_float(
                    arrays['initial_velocity_m_s'][index]
                ),
                'initial_v2_m_s': _csv_float(arrays['initial_v2_m_s'][index]),
                'velocity_update_from_initial_m_s': _csv_float(
                    arrays['velocity_update_from_initial_m_s'][index]
                ),
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
        'median_velocity_m_s': _csv_float(row.median_velocity_m_s),
        'median_v2_m_s': _csv_float(row.median_v2_m_s),
        'min_velocity_m_s': _csv_float(row.min_velocity_m_s),
        'min_v2_m_s': _csv_float(row.min_v2_m_s),
        'max_velocity_m_s': _csv_float(row.max_velocity_m_s),
        'max_v2_m_s': _csv_float(row.max_v2_m_s),
        'max_abs_velocity_update_m_s': _csv_float(
            row.max_abs_velocity_update_m_s
        ),
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
    *,
    used_row_mask: np.ndarray,
) -> dict[str, float | None]:
    residual_ms = (
        np.asarray(result.residual_time_s, dtype=np.float64)[
            np.asarray(used_row_mask, dtype=bool)
        ]
        * 1000.0
    )
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


def _initial_cell_v2_m_s(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> float:
    layer = _request_cell_velocity_layer(req, layer_kind=layer_kind)
    if layer is not None:
        value = getattr(layer, 'initial_velocity_m_s', None)
        if value is not None:
            return float(value)
        if layer.kind == 'v2_t1' and req.model.initial_bedrock_velocity_m_s is not None:
            return float(req.model.initial_bedrock_velocity_m_s)
        min_velocity, max_velocity = _cell_velocity_layer_bounds(req, layer)
        if min_velocity is not None and max_velocity is not None:
            return 0.5 * (float(min_velocity) + float(max_velocity))
        raise RefractionStaticArtifactError(
            'cell velocity layer initial_velocity_m_s is required'
        )

    value = req.model.initial_bedrock_velocity_m_s
    if value is not None:
        return float(value)
    return 0.5 * (
        float(req.model.min_bedrock_velocity_m_s)
        + float(req.model.max_bedrock_velocity_m_s)
    )


def _history_smoothing_weight(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> float:
    layer = _request_cell_velocity_layer(req, layer_kind=layer_kind)
    if layer is not None and layer.smoothing_weight is not None:
        return float(layer.smoothing_weight)
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


def _cell_id_float_array(value: object) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).copy()
    out[out < 0] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


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


def _sum_correction_s(left: object, right: object) -> float:
    left_value = _float_or_nan(left)
    right_value = _float_or_nan(right)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float('nan')
    return float(left_value + right_value)


from app.services.refraction_static_artifacts.components import (  # noqa: E402
    write_refraction_static_component_qc_artifacts,  # noqa: F811
    write_refraction_static_components_csv,  # noqa: F811
)
from app.services.refraction_static_artifacts.static_tables import (  # noqa: E402
    _has_node_2layer_static_fields,  # noqa: F811
    _has_node_3layer_static_fields,  # noqa: F811
    _has_receiver_2layer_static_fields,  # noqa: F811
    _has_receiver_3layer_static_fields,  # noqa: F811
    _has_source_2layer_static_fields,  # noqa: F811
    _has_source_3layer_static_fields,  # noqa: F811
    _insert_after,  # noqa: F811
    _node_sh1_weathering_thickness_m,  # noqa: F811
    _receiver_sh1_weathering_thickness_m,  # noqa: F811
    _source_sh1_weathering_thickness_m,  # noqa: F811
    build_source_receiver_static_table_arrays,  # noqa: F811
    write_receiver_static_table_csv,  # noqa: F811
    write_refraction_time_term_spreadsheet_csv,  # noqa: F811
    write_refraction_time_term_spreadsheet_csv_from_static_tables,  # noqa: F811
    write_source_receiver_static_table_npz,  # noqa: F811
    write_source_static_table_csv,  # noqa: F811
)


__all__ = [
    'FIRST_BREAK_RESIDUALS_CSV_NAME',
    'FIRST_BREAK_FIT_QC_RESIDUAL_SIGN',
    'REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME',
    'REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME',
    'REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME',
    'REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME',
    'REFRACTION_GRID_MAP_QC_CSV_NAME',
    'REFRACTION_GRID_MAP_QC_JSON_NAME',
    'REFRACTION_GRID_MAP_QC_NPZ_NAME',
    'REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME',
    'REFRACTION_LINE_PROFILE_QC_JSON_NAME',
    'REFRACTION_LINE_PROFILE_QC_NPZ_NAME',
    'REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME',
    'REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME',
    'REFRACTION_REDUCED_TIME_QC_CSV_NAME',
    'REFRACTION_REDUCED_TIME_QC_JSON_NAME',
    'REFRACTION_REDUCED_TIME_QC_NPZ_NAME',
    'NEAR_SURFACE_MODEL_CSV_NAME',
    'REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
    'REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME',
    'REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME',
    'REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
    'REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
    'REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME',
    'REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME',
    'REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME',
    'REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME',
    'REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME',
    'REFRACTION_STATICS_CSV_NAME',
    'REFRACTION_STATIC_ARTIFACTS_JSON_NAME',
    'REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME',
    'REFRACTION_STATIC_COMPONENT_QC_JSON_NAME',
    'REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME',
    'REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME',
    'REFRACTION_STATIC_COMPONENTS_CSV_NAME',
    'REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME',
    'REFRACTION_STATIC_HISTORY_JSON_NAME',
    'REFRACTION_STATIC_REQUEST_JSON_NAME',
    'REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES',
    'REFRACTION_STATIC_QC_JSON_NAME',
    'REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME',
    'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
    'REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME',
    'REFRACTION_V1_ESTIMATES_CSV_NAME',
    'REFRACTION_V1_QC_JSON_NAME',
    'RECEIVER_STATIC_TABLE_CSV_NAME',
    'SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME',
    'SOURCE_STATIC_TABLE_CSV_NAME',
    'UPLOADED_REFRACTION_PICKS_NPZ_NAME',
    'RefractionCellSolverHistoryRow',
    'RefractionStaticArtifactError',
    'RefractionStaticArtifactSet',
    'build_refraction_cell_solver_history_rows',
    'build_refraction_first_break_fit_qc_arrays',
    'build_refraction_first_break_fit_qc_payload',
    'build_refraction_grid_map_qc_arrays',
    'build_refraction_grid_map_qc_payload',
    'build_refraction_line_profile_qc_arrays',
    'build_refraction_line_profile_qc_payload',
    'build_refraction_reduced_time_qc_arrays',
    'build_refraction_reduced_time_qc_payload',
    'build_refraction_static_component_qc_arrays',
    'build_refraction_static_component_qc_payload',
    'build_refraction_refractor_velocity_grid_arrays',
    'build_refraction_refractor_velocity_qc_payload',
    'build_refraction_static_history_payload',
    'build_refraction_static_qc_payload',
    'build_refraction_static_solution_arrays',
    'build_source_receiver_static_table_arrays',
    'refraction_static_double_application_qc',
    'refraction_static_trace_shift_component_names',
    'static_history_double_application_qc',
    'write_first_break_residuals_csv',
    'write_near_surface_model_csv',
    'write_refraction_first_break_fit_qc_csv',
    'write_refraction_first_break_fit_qc_json',
    'write_refraction_first_break_fit_qc_npz',
    'write_refraction_first_break_time_export_csv',
    'write_refraction_grid_map_qc_csv',
    'write_refraction_grid_map_qc_json',
    'write_refraction_grid_map_qc_npz',
    'write_refraction_line_profile_qc_artifacts',
    'write_refraction_reduced_time_qc_csv',
    'write_refraction_reduced_time_qc_json',
    'write_refraction_reduced_time_qc_npz',
    'write_refraction_cell_solver_history_csv',
    'write_refraction_refractor_velocity_cells_csv',
    'write_refraction_refractor_velocity_grid_npz',
    'write_refraction_refractor_velocity_qc_json',
    'write_refraction_static_artifacts',
    'write_refraction_static_component_qc_artifacts',
    'write_refraction_static_components_csv',
    'write_refraction_static_history_json',
    'write_refraction_static_qc_json',
    'write_refraction_static_solution_npz',
    'write_refraction_statics_csv',
    'write_refraction_time_term_spreadsheet_csv',
    'write_refraction_time_term_spreadsheet_csv_from_static_tables',
    'write_receiver_static_table_csv',
    'write_source_receiver_static_table_npz',
    'write_source_static_table_csv',
]
