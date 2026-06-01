"""Final artifact package writer for GLI refraction statics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.statics.refraction.contracts.apply import RefractionStaticApplyRequest
from app.statics.refraction.application.design_matrix import (
    all_refraction_design_matrix_layer_artifact_names,
)
from app.statics.refraction.artifacts.cell_velocity import (
    build_refraction_cell_solver_history_rows,
    build_refraction_refractor_velocity_grid_arrays,
    build_refraction_refractor_velocity_qc_payload,
    write_refraction_cell_solver_history_csv,
    write_refraction_refractor_velocity_cells_csv,
    write_refraction_refractor_velocity_grid_npz,
    write_refraction_refractor_velocity_qc_json,
)
from app.statics.refraction.artifacts.components import (
    build_refraction_static_component_qc_arrays,
    build_refraction_static_component_qc_payload,
    write_refraction_static_component_qc_artifacts,
    write_refraction_static_components_csv,
)
from app.statics.refraction.artifacts.contract import (
    FIRST_BREAK_FIT_QC_RESIDUAL_SIGN,
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
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
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    UPLOADED_REFRACTION_PICKS_NPZ_NAME,
    RefractionCellSolverHistoryRow,
    RefractionStaticArtifactError,
)
from app.statics.refraction.artifacts.final_tables import (
    write_near_surface_model_csv,
    write_refraction_statics_csv,
)
# Active writer path for first-break fit QC and reduced-time QC artifacts.
from app.statics.refraction.artifacts.first_break import (
    build_refraction_first_break_fit_qc_arrays,
    build_refraction_first_break_fit_qc_payload,
    build_refraction_reduced_time_qc_arrays,
    build_refraction_reduced_time_qc_payload,
    write_first_break_residuals_csv,
    write_refraction_first_break_fit_qc_csv,
    write_refraction_first_break_fit_qc_json,
    write_refraction_first_break_fit_qc_npz,
    write_refraction_first_break_time_export_csv,
    write_refraction_reduced_time_qc_csv,
    write_refraction_reduced_time_qc_json,
    write_refraction_reduced_time_qc_npz,
)
from app.statics.refraction.artifacts.grid_map import (
    build_refraction_grid_map_qc_arrays,
    build_refraction_grid_map_qc_payload,
    write_refraction_grid_map_qc_csv,
    write_refraction_grid_map_qc_json,
    write_refraction_grid_map_qc_npz,
)
from app.statics.refraction.artifacts.io import (
    _assert_strict_json,
    _write_json_atomic,
)
from app.statics.refraction.artifacts.line_profile import (
    build_refraction_line_profile_qc_arrays,
    build_refraction_line_profile_qc_payload,
    write_refraction_line_profile_qc_artifacts,
)
# Active writer path for static history and double-application QC artifacts.
from app.statics.refraction.artifacts.qc import (
    build_refraction_static_history_payload,
    build_refraction_static_qc_payload,
    refraction_static_double_application_qc,
    refraction_static_trace_shift_component_names,
    static_history_double_application_qc,
    write_refraction_static_history_json,
    write_refraction_static_qc_json,
)
from app.statics.refraction.artifacts.registry import (
    REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES,
    _artifact_entries_for_request,
    _artifact_list_for_qc,
    _build_manifest_payload,
    _cell_velocity_artifact_paths_for_request,
    _validate_declared_upstream_artifacts,
    _validate_upstream_artifact_names,
)
from app.statics.refraction.artifacts.solution import (
    build_refraction_static_solution_arrays,
    write_refraction_static_solution_npz,
)
from app.statics.refraction.artifacts.static_tables import (
    build_source_receiver_static_table_arrays,
    write_receiver_static_table_csv,
    write_refraction_time_term_spreadsheet_csv,
    write_refraction_time_term_spreadsheet_csv_from_static_tables,
    write_source_receiver_static_table_npz,
    write_source_static_table_csv,
)
from app.statics.refraction.artifacts.validation import (
    _validate_job_dir,
    _validate_resolved_first_layer,
    _validate_result,
)
from app.statics.refraction.domain.t1lsst import (
    REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
    write_refraction_t1lsst_1layer_components_csv,
)
from app.statics.refraction.domain.types import (
    RefractionDatumStaticsResult,
    RefractionStaticArtifactSet,
    ResolvedRefractionFirstLayer,
)
from app.statics.refraction.domain.v1 import (
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
)

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
    artifact_entries = _filter_present_design_matrix_diagnostics(
        root,
        artifact_entries,
    )
    qc = build_refraction_static_qc_payload(
        result=values.result,
        req=request,
        resolved_first_layer=first_layer,
        upstream_artifact_names=upstream_names,
    )
    qc['artifacts'] = _artifact_list_for_qc(artifact_entries)
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
    from app.statics.refraction.artifacts import (
        cell_velocity as _cell_velocity_artifacts,
    )

    for cell_artifacts in cell_velocity_artifact_paths:
        _cell_velocity_artifacts.write_refraction_refractor_velocity_cells_csv(
            result=values.result,
            req=request,
            path=cell_artifacts.cells_csv,
            layer_kind=cell_artifacts.layer_kind,
        )
        _cell_velocity_artifacts.write_refraction_refractor_velocity_grid_npz(
            result=values.result,
            req=request,
            path=cell_artifacts.grid_npz,
            layer_kind=cell_artifacts.layer_kind,
        )
        _cell_velocity_artifacts.write_refraction_refractor_velocity_qc_json(
            result=values.result,
            req=request,
            path=cell_artifacts.qc_json,
            layer_kind=cell_artifacts.layer_kind,
        )
        _cell_velocity_artifacts.write_refraction_cell_solver_history_csv(
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


def _filter_present_design_matrix_diagnostics(
    root: Path,
    artifact_entries: tuple[dict[str, str | bool], ...],
) -> tuple[dict[str, str | bool], ...]:
    design_matrix_names = set(all_refraction_design_matrix_layer_artifact_names())
    if not design_matrix_names:
        return artifact_entries
    return tuple(
        entry
        for entry in artifact_entries
        if str(entry['name']) not in design_matrix_names
        or (root / str(entry['name'])).is_file()
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
