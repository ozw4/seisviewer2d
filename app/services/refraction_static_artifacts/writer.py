"""Plan-based writer for final refraction static artifact packages."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_t1lsst import (
    REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
    write_refraction_t1lsst_1layer_components_csv,
)
from app.services.refraction_static_types import (
    RefractionDatumStaticsResult,
    RefractionStaticArtifactSet,
    ResolvedRefractionFirstLayer,
)
from app.services.refraction_static_artifacts import _legacy
from app.services.refraction_static_artifacts import history
from app.services.refraction_static_artifacts import qc as main_qc
from app.services.refraction_static_artifacts import solution
from app.services.refraction_static_artifacts.contract import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
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
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RefractionStaticArtifactError,
    _CellVelocityArtifactPaths,
)
from app.services.refraction_static_artifacts.io import (
    _assert_strict_json,
    _write_json_atomic,
)
from app.services.refraction_static_artifacts.registry import (
    _artifact_entries_for_request,
    _build_manifest_payload,
    _cell_velocity_artifact_paths_for_request,
    _validate_declared_upstream_artifacts,
    _validate_upstream_artifact_names,
)
from app.services.refraction_static_artifacts.validation import (
    _validate_job_dir,
    _validate_resolved_first_layer,
    _validate_result,
)


@dataclass(frozen=True)
class ArtifactWriteStep:
    name: str
    path: Path
    required: bool
    write_callable: Callable[[], None]
    post_check: bool = True


@dataclass(frozen=True)
class GridMapArtifactPaths:
    csv: Path
    npz: Path
    json: Path


@dataclass(frozen=True)
class ArtifactPathPlan:
    paths: RefractionStaticArtifactSet
    cell_velocity_artifact_paths: tuple[_CellVelocityArtifactPaths, ...]
    grid_map_artifact_paths: GridMapArtifactPaths | None


@dataclass(frozen=True)
class ArtifactWritePlan:
    paths: RefractionStaticArtifactSet
    manifest: dict[str, Any]
    steps: tuple[ArtifactWriteStep, ...]
    post_check_paths: tuple[Path, ...]
    upstream_artifact_names: tuple[str, ...]


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
    plan = _build_artifact_write_plan(
        result=result,
        req=req,
        job_dir=job_dir,
        resolved_first_layer=resolved_first_layer,
        upstream_artifact_names=upstream_artifact_names,
        source_job_id=source_job_id,
    )
    _execute_artifact_write_plan(plan)
    _validate_written_artifact_paths(plan.post_check_paths)
    _validate_declared_upstream_artifacts(
        plan.paths.job_dir,
        plan.upstream_artifact_names,
    )
    return plan.paths


def _build_artifact_write_plan(
    *,
    result: RefractionDatumStaticsResult,
    req: RefractionStaticApplyRequest,
    job_dir: Path,
    resolved_first_layer: ResolvedRefractionFirstLayer | None,
    upstream_artifact_names: Iterable[str],
    source_job_id: str | None,
) -> ArtifactWritePlan:
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
    qc = main_qc.build_refraction_static_qc_payload(
        result=values.result,
        req=request,
        resolved_first_layer=first_layer,
        upstream_artifact_names=upstream_names,
        artifact_entries=artifact_entries,
    )
    manifest = _build_manifest_payload(artifact_entries)
    _assert_strict_json(manifest, artifact_name=REFRACTION_STATIC_ARTIFACTS_JSON_NAME)

    path_plan = _build_artifact_path_plan(
        root=root,
        request=request,
        artifact_entries=artifact_entries,
        qc=qc,
    )
    return ArtifactWritePlan(
        paths=path_plan.paths,
        manifest=manifest,
        steps=_build_artifact_write_steps(
            result=values.result,
            request=request,
            first_layer=first_layer,
            qc=qc,
            paths=path_plan.paths,
            cell_velocity_artifact_paths=path_plan.cell_velocity_artifact_paths,
            grid_map_artifact_paths=path_plan.grid_map_artifact_paths,
            manifest=manifest,
            source_job_id=source_job_id,
        ),
        post_check_paths=_artifact_post_check_paths(
            paths=path_plan.paths,
            cell_velocity_artifact_paths=path_plan.cell_velocity_artifact_paths,
            grid_map_artifact_paths=path_plan.grid_map_artifact_paths,
        ),
        upstream_artifact_names=upstream_names,
    )


def _build_artifact_path_plan(
    *,
    root: Path,
    request: RefractionStaticApplyRequest,
    artifact_entries: tuple[dict[str, str | bool], ...],
    qc: dict[str, Any],
) -> ArtifactPathPlan:
    required_paths = _required_artifact_paths(root)
    t1lsst_components_path = _optional_t1lsst_artifact_path(root, request)
    cell_velocity_artifact_paths = _optional_cell_velocity_artifact_paths(
        root,
        request,
    )
    grid_map_paths = _optional_grid_map_artifact_paths(
        root,
        cell_velocity_artifact_paths,
    )
    first_cell_velocity_artifacts = (
        cell_velocity_artifact_paths[0]
        if cell_velocity_artifact_paths
        else None
    )

    return ArtifactPathPlan(
        paths=RefractionStaticArtifactSet(
            job_dir=root,
            **required_paths,
            artifact_names=_required_artifact_names(artifact_entries),
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
                grid_map_paths.csv if grid_map_paths is not None else None
            ),
            refraction_grid_map_qc_npz=(
                grid_map_paths.npz if grid_map_paths is not None else None
            ),
            refraction_grid_map_qc_json=(
                grid_map_paths.json if grid_map_paths is not None else None
            ),
        ),
        cell_velocity_artifact_paths=cell_velocity_artifact_paths,
        grid_map_artifact_paths=grid_map_paths,
    )


def _required_artifact_paths(root: Path) -> dict[str, Path]:
    return {
        'solution_npz': root / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        'qc_json': root / REFRACTION_STATIC_QC_JSON_NAME,
        'refraction_statics_csv': root / REFRACTION_STATICS_CSV_NAME,
        'near_surface_model_csv': root / NEAR_SURFACE_MODEL_CSV_NAME,
        'first_break_residuals_csv': root / FIRST_BREAK_RESIDUALS_CSV_NAME,
        'refraction_first_break_time_export_csv': (
            root / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME
        ),
        'refraction_first_break_fit_qc_csv': (
            root / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME
        ),
        'refraction_first_break_fit_qc_npz': (
            root / REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME
        ),
        'refraction_first_break_fit_qc_json': (
            root / REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME
        ),
        'refraction_reduced_time_qc_csv': root / REFRACTION_REDUCED_TIME_QC_CSV_NAME,
        'refraction_reduced_time_qc_npz': root / REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
        'refraction_reduced_time_qc_json': root / REFRACTION_REDUCED_TIME_QC_JSON_NAME,
        'refraction_static_components_csv': root / REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        'refraction_static_component_qc_trace_csv': (
            root / REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME
        ),
        'refraction_static_component_qc_endpoint_csv': (
            root / REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME
        ),
        'refraction_static_component_qc_npz': (
            root / REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME
        ),
        'refraction_static_component_qc_json': (
            root / REFRACTION_STATIC_COMPONENT_QC_JSON_NAME
        ),
        'source_static_table_csv': root / SOURCE_STATIC_TABLE_CSV_NAME,
        'receiver_static_table_csv': root / RECEIVER_STATIC_TABLE_CSV_NAME,
        'source_receiver_static_table_npz': root / SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
        'refraction_line_profile_qc_source_csv': (
            root / REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME
        ),
        'refraction_line_profile_qc_receiver_csv': (
            root / REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME
        ),
        'refraction_line_profile_qc_combined_csv': (
            root / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME
        ),
        'refraction_line_profile_qc_npz': root / REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
        'refraction_line_profile_qc_json': root / REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        'refraction_time_term_spreadsheet_csv': (
            root / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME
        ),
        'static_history_json': root / REFRACTION_STATIC_HISTORY_JSON_NAME,
        'manifest_json': root / REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    }


def _optional_t1lsst_artifact_path(
    root: Path,
    request: RefractionStaticApplyRequest,
) -> Path | None:
    if request.conversion.mode != 't1lsst_1layer':
        return None
    return root / REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME


def _optional_cell_velocity_artifact_paths(
    root: Path,
    request: RefractionStaticApplyRequest,
) -> tuple[_CellVelocityArtifactPaths, ...]:
    return _cell_velocity_artifact_paths_for_request(root, request)


def _optional_grid_map_artifact_paths(
    root: Path,
    cell_velocity_artifact_paths: tuple[_CellVelocityArtifactPaths, ...],
) -> GridMapArtifactPaths | None:
    if not cell_velocity_artifact_paths:
        return None
    return GridMapArtifactPaths(
        csv=root / REFRACTION_GRID_MAP_QC_CSV_NAME,
        npz=root / REFRACTION_GRID_MAP_QC_NPZ_NAME,
        json=root / REFRACTION_GRID_MAP_QC_JSON_NAME,
    )


def _required_artifact_names(
    artifact_entries: tuple[dict[str, str | bool], ...],
) -> tuple[str, ...]:
    return tuple(str(item['name']) for item in artifact_entries if bool(item['required']))


def _build_artifact_write_steps(
    *,
    result: RefractionDatumStaticsResult,
    request: RefractionStaticApplyRequest,
    first_layer: ResolvedRefractionFirstLayer | None,
    qc: dict[str, Any],
    paths: RefractionStaticArtifactSet,
    cell_velocity_artifact_paths: tuple[_CellVelocityArtifactPaths, ...],
    grid_map_artifact_paths: GridMapArtifactPaths | None,
    manifest: dict[str, Any],
    source_job_id: str | None,
) -> tuple[ArtifactWriteStep, ...]:
    return (
        _required_write_steps(
            result=result,
            request=request,
            first_layer=first_layer,
            qc=qc,
            paths=paths,
            source_job_id=source_job_id,
        )
        + _cell_velocity_write_steps(
            result=result,
            request=request,
            cell_velocity_artifact_paths=cell_velocity_artifact_paths,
        )
        + _grid_map_write_steps(
            result=result,
            request=request,
            grid_map_artifact_paths=grid_map_artifact_paths,
        )
        + _t1lsst_write_steps(
            result=result,
            paths=paths,
        )
        + (_manifest_write_step(paths=paths, manifest=manifest),)
    )


def _required_write_steps(
    *,
    result: RefractionDatumStaticsResult,
    request: RefractionStaticApplyRequest,
    first_layer: ResolvedRefractionFirstLayer | None,
    qc: dict[str, Any],
    paths: RefractionStaticArtifactSet,
    source_job_id: str | None,
) -> tuple[ArtifactWriteStep, ...]:
    return (
        ArtifactWriteStep(
            name=paths.solution_npz.name,
            path=paths.solution_npz,
            required=True,
            write_callable=lambda: solution.write_refraction_static_solution_npz(
                result=result,
                req=request,
                path=paths.solution_npz,
                resolved_first_layer=first_layer,
            ),
        ),
        ArtifactWriteStep(
            name=paths.qc_json.name,
            path=paths.qc_json,
            required=True,
            write_callable=lambda: main_qc.write_refraction_static_qc_json(
                result=result,
                req=request,
                path=paths.qc_json,
                qc=qc,
                resolved_first_layer=first_layer,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_statics_csv.name,
            path=paths.refraction_statics_csv,
            required=True,
            write_callable=lambda: solution.write_refraction_statics_csv(
                result=result,
                path=paths.refraction_statics_csv,
            ),
        ),
        ArtifactWriteStep(
            name=paths.near_surface_model_csv.name,
            path=paths.near_surface_model_csv,
            required=True,
            write_callable=lambda: solution.write_near_surface_model_csv(
                result=result,
                path=paths.near_surface_model_csv,
            ),
        ),
        ArtifactWriteStep(
            name=paths.first_break_residuals_csv.name,
            path=paths.first_break_residuals_csv,
            required=True,
            write_callable=lambda: _legacy.write_first_break_residuals_csv(
                result=result,
                path=paths.first_break_residuals_csv,
                req=request,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_first_break_time_export_csv.name,
            path=paths.refraction_first_break_time_export_csv,
            required=True,
            write_callable=lambda: _legacy.write_refraction_first_break_time_export_csv(
                result=result,
                path=paths.refraction_first_break_time_export_csv,
                req=request,
                source_job_id=source_job_id,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_first_break_fit_qc_csv.name,
            path=paths.refraction_first_break_fit_qc_csv,
            required=True,
            write_callable=lambda: _legacy.write_refraction_first_break_fit_qc_csv(
                result=result,
                req=request,
                path=paths.refraction_first_break_fit_qc_csv,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_first_break_fit_qc_npz.name,
            path=paths.refraction_first_break_fit_qc_npz,
            required=True,
            write_callable=lambda: _legacy.write_refraction_first_break_fit_qc_npz(
                result=result,
                req=request,
                path=paths.refraction_first_break_fit_qc_npz,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_first_break_fit_qc_json.name,
            path=paths.refraction_first_break_fit_qc_json,
            required=True,
            write_callable=lambda: _legacy.write_refraction_first_break_fit_qc_json(
                result=result,
                req=request,
                path=paths.refraction_first_break_fit_qc_json,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_reduced_time_qc_csv.name,
            path=paths.refraction_reduced_time_qc_csv,
            required=True,
            write_callable=lambda: _legacy.write_refraction_reduced_time_qc_csv(
                result=result,
                req=request,
                path=paths.refraction_reduced_time_qc_csv,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_reduced_time_qc_npz.name,
            path=paths.refraction_reduced_time_qc_npz,
            required=True,
            write_callable=lambda: _legacy.write_refraction_reduced_time_qc_npz(
                result=result,
                req=request,
                path=paths.refraction_reduced_time_qc_npz,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_reduced_time_qc_json.name,
            path=paths.refraction_reduced_time_qc_json,
            required=True,
            write_callable=lambda: _legacy.write_refraction_reduced_time_qc_json(
                result=result,
                req=request,
                path=paths.refraction_reduced_time_qc_json,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_static_components_csv.name,
            path=paths.refraction_static_components_csv,
            required=True,
            write_callable=lambda: _legacy.write_refraction_static_components_csv(
                result=result,
                path=paths.refraction_static_components_csv,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_static_component_qc_json.name,
            path=paths.refraction_static_component_qc_json,
            required=True,
            write_callable=lambda: _legacy.write_refraction_static_component_qc_artifacts(
                result=result,
                req=request,
                trace_csv_path=paths.refraction_static_component_qc_trace_csv,
                endpoint_csv_path=paths.refraction_static_component_qc_endpoint_csv,
                npz_path=paths.refraction_static_component_qc_npz,
                json_path=paths.refraction_static_component_qc_json,
            ),
        ),
        ArtifactWriteStep(
            name=paths.source_static_table_csv.name,
            path=paths.source_static_table_csv,
            required=True,
            write_callable=lambda: _legacy.write_source_static_table_csv(
                result=result,
                path=paths.source_static_table_csv,
            ),
        ),
        ArtifactWriteStep(
            name=paths.receiver_static_table_csv.name,
            path=paths.receiver_static_table_csv,
            required=True,
            write_callable=lambda: _legacy.write_receiver_static_table_csv(
                result=result,
                path=paths.receiver_static_table_csv,
            ),
        ),
        ArtifactWriteStep(
            name=paths.source_receiver_static_table_npz.name,
            path=paths.source_receiver_static_table_npz,
            required=True,
            write_callable=lambda: _legacy.write_source_receiver_static_table_npz(
                result=result,
                path=paths.source_receiver_static_table_npz,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_line_profile_qc_json.name,
            path=paths.refraction_line_profile_qc_json,
            required=True,
            write_callable=lambda: _legacy.write_refraction_line_profile_qc_artifacts(
                result=result,
                req=request,
                source_csv_path=paths.refraction_line_profile_qc_source_csv,
                receiver_csv_path=paths.refraction_line_profile_qc_receiver_csv,
                combined_csv_path=paths.refraction_line_profile_qc_combined_csv,
                npz_path=paths.refraction_line_profile_qc_npz,
                json_path=paths.refraction_line_profile_qc_json,
            ),
        ),
        ArtifactWriteStep(
            name=paths.refraction_time_term_spreadsheet_csv.name,
            path=paths.refraction_time_term_spreadsheet_csv,
            required=True,
            write_callable=lambda: _legacy.write_refraction_time_term_spreadsheet_csv(
                result=result,
                path=paths.refraction_time_term_spreadsheet_csv,
                source_job_id=source_job_id,
            ),
        ),
        ArtifactWriteStep(
            name=paths.static_history_json.name,
            path=paths.static_history_json,
            required=True,
            write_callable=lambda: history.write_refraction_static_history_json(
                result=result,
                req=request,
                path=paths.static_history_json,
            ),
        ),
    )


def _cell_velocity_write_steps(
    *,
    result: RefractionDatumStaticsResult,
    request: RefractionStaticApplyRequest,
    cell_velocity_artifact_paths: tuple[_CellVelocityArtifactPaths, ...],
) -> tuple[ArtifactWriteStep, ...]:
    steps: list[ArtifactWriteStep] = []
    for cell_artifacts in cell_velocity_artifact_paths:
        steps.extend(
            (
                ArtifactWriteStep(
                    name=cell_artifacts.cells_csv.name,
                    path=cell_artifacts.cells_csv,
                    required=True,
                    write_callable=lambda cell_artifacts=cell_artifacts: (
                        _legacy.write_refraction_refractor_velocity_cells_csv(
                            result=result,
                            req=request,
                            path=cell_artifacts.cells_csv,
                            layer_kind=cell_artifacts.layer_kind,
                        )
                    ),
                ),
                ArtifactWriteStep(
                    name=cell_artifacts.grid_npz.name,
                    path=cell_artifacts.grid_npz,
                    required=True,
                    write_callable=lambda cell_artifacts=cell_artifacts: (
                        _legacy.write_refraction_refractor_velocity_grid_npz(
                            result=result,
                            req=request,
                            path=cell_artifacts.grid_npz,
                            layer_kind=cell_artifacts.layer_kind,
                        )
                    ),
                ),
                ArtifactWriteStep(
                    name=cell_artifacts.qc_json.name,
                    path=cell_artifacts.qc_json,
                    required=True,
                    write_callable=lambda cell_artifacts=cell_artifacts: (
                        _legacy.write_refraction_refractor_velocity_qc_json(
                            result=result,
                            req=request,
                            path=cell_artifacts.qc_json,
                            layer_kind=cell_artifacts.layer_kind,
                        )
                    ),
                ),
                ArtifactWriteStep(
                    name=cell_artifacts.solver_history_csv.name,
                    path=cell_artifacts.solver_history_csv,
                    required=True,
                    write_callable=lambda cell_artifacts=cell_artifacts: (
                        _legacy.write_refraction_cell_solver_history_csv(
                            result=result,
                            req=request,
                            path=cell_artifacts.solver_history_csv,
                            layer_kind=cell_artifacts.layer_kind,
                        )
                    ),
                ),
            )
        )
    return tuple(steps)


def _grid_map_write_steps(
    *,
    result: RefractionDatumStaticsResult,
    request: RefractionStaticApplyRequest,
    grid_map_artifact_paths: GridMapArtifactPaths | None,
) -> tuple[ArtifactWriteStep, ...]:
    if grid_map_artifact_paths is None:
        return ()
    return (
        ArtifactWriteStep(
            name=grid_map_artifact_paths.csv.name,
            path=grid_map_artifact_paths.csv,
            required=True,
            write_callable=lambda: _legacy.write_refraction_grid_map_qc_csv(
                result=result,
                req=request,
                path=grid_map_artifact_paths.csv,
            ),
        ),
        ArtifactWriteStep(
            name=grid_map_artifact_paths.npz.name,
            path=grid_map_artifact_paths.npz,
            required=True,
            write_callable=lambda: _legacy.write_refraction_grid_map_qc_npz(
                result=result,
                req=request,
                path=grid_map_artifact_paths.npz,
            ),
        ),
        ArtifactWriteStep(
            name=grid_map_artifact_paths.json.name,
            path=grid_map_artifact_paths.json,
            required=True,
            write_callable=lambda: _legacy.write_refraction_grid_map_qc_json(
                result=result,
                req=request,
                path=grid_map_artifact_paths.json,
            ),
        ),
    )


def _t1lsst_write_steps(
    *,
    result: RefractionDatumStaticsResult,
    paths: RefractionStaticArtifactSet,
) -> tuple[ArtifactWriteStep, ...]:
    path = paths.refraction_t1lsst_1layer_components_csv
    if path is None:
        return ()
    return (
        ArtifactWriteStep(
            name=path.name,
            path=path,
            required=True,
            write_callable=lambda: write_refraction_t1lsst_1layer_components_csv(
                result=result,
                path=path,
            ),
        ),
    )


def _manifest_write_step(
    *,
    paths: RefractionStaticArtifactSet,
    manifest: dict[str, Any],
) -> ArtifactWriteStep:
    return ArtifactWriteStep(
        name=REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        path=paths.manifest_json,
        required=True,
        write_callable=lambda: _write_json_atomic(paths.manifest_json, manifest),
    )


def _artifact_post_check_paths(
    *,
    paths: RefractionStaticArtifactSet,
    cell_velocity_artifact_paths: tuple[_CellVelocityArtifactPaths, ...],
    grid_map_artifact_paths: GridMapArtifactPaths | None,
) -> tuple[Path, ...]:
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
    artifact_paths += _optional_artifact_post_check_paths(
        paths=paths,
        cell_velocity_artifact_paths=cell_velocity_artifact_paths,
        grid_map_artifact_paths=grid_map_artifact_paths,
    )
    return artifact_paths


def _optional_artifact_post_check_paths(
    *,
    paths: RefractionStaticArtifactSet,
    cell_velocity_artifact_paths: tuple[_CellVelocityArtifactPaths, ...],
    grid_map_artifact_paths: GridMapArtifactPaths | None,
) -> tuple[Path, ...]:
    artifact_paths: tuple[Path, ...] = ()
    if paths.refraction_t1lsst_1layer_components_csv is not None:
        artifact_paths += (paths.refraction_t1lsst_1layer_components_csv,)
    for cell_artifacts in cell_velocity_artifact_paths:
        artifact_paths += (
            cell_artifacts.cells_csv,
            cell_artifacts.grid_npz,
            cell_artifacts.qc_json,
            cell_artifacts.solver_history_csv,
        )
    if grid_map_artifact_paths is not None:
        artifact_paths += (
            grid_map_artifact_paths.csv,
            grid_map_artifact_paths.npz,
            grid_map_artifact_paths.json,
        )
    return artifact_paths


def _execute_artifact_write_plan(plan: ArtifactWritePlan) -> None:
    for step in plan.steps:
        step.write_callable()


def _validate_written_artifact_paths(artifact_paths: tuple[Path, ...]) -> None:
    for artifact_path in artifact_paths:
        if not artifact_path.is_file():
            raise RefractionStaticArtifactError(
                f'artifact file missing after write: {artifact_path.name}'
            )


__all__ = ['ArtifactWriteStep', 'write_refraction_static_artifacts']
