"""Artifact registry helpers for refraction static output packages."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.contracts.statics.refraction.apply import RefractionStaticApplyRequest
from app.services.refraction_static_layer_config import (
    normalize_refraction_static_layers,
)
from app.services.refraction_static_design_matrix import (
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    all_refraction_design_matrix_layer_artifact_names,
    refraction_design_matrix_layer_node_diagnostics_csv_name,
    refraction_design_matrix_layer_qc_json_name,
)
from app.services.refraction_static_preflight_diagnostics import (
    REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME,
    REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME,
)
from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
)
from app.services.refraction_static_t1lsst import (
    REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
)
from app.services.refraction_static_types import (
    RefractionLayerKind,
    ResolvedRefractionFirstLayer,
)
from app.services.refraction_static_uphole import (
    REFRACTION_UPHOLE_QC_JSON_NAME,
    REFRACTION_UPHOLE_SOURCES_CSV_NAME,
)
from app.services.refraction_static_v1 import (
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
)

from app.statics.refraction.artifacts.contract import (
    _CellVelocityArtifactNames,
    _CellVelocityArtifactPaths,
    ARTIFACT_VERSION,
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
    RefractionStaticArtifactError,
)


_CELL_VELOCITY_COMPONENT_BY_LAYER: dict[RefractionLayerKind, str] = {
    'v2_t1': 'v2',
    'v3_t2': 'v3',
    'vsub_t3': 'vsub',
}
_CELL_VELOCITY_LABEL_BY_LAYER: dict[RefractionLayerKind, str] = {
    'v2_t1': 'V2/T1',
    'v3_t2': 'V3/T2',
    'vsub_t3': 'Vsub/T3',
}
_CELL_VELOCITY_ARTIFACT_NAMES_BY_LAYER: dict[
    RefractionLayerKind,
    _CellVelocityArtifactNames,
] = {
    'v2_t1': _CellVelocityArtifactNames(
        cells_csv=REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        grid_npz=REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        qc_json=REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
        solver_history_csv=REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    ),
    'v3_t2': _CellVelocityArtifactNames(
        cells_csv=REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        grid_npz=REFRACTION_V3_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        qc_json=REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
        solver_history_csv=REFRACTION_V3_CELL_SOLVER_HISTORY_CSV_NAME,
    ),
    'vsub_t3': _CellVelocityArtifactNames(
        cells_csv=REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
        grid_npz=REFRACTION_VSUB_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
        qc_json=REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
        solver_history_csv=REFRACTION_VSUB_CELL_SOLVER_HISTORY_CSV_NAME,
    ),
}


def _cell_velocity_artifact_names(
    layer_kind: RefractionLayerKind,
) -> _CellVelocityArtifactNames:
    try:
        return _CELL_VELOCITY_ARTIFACT_NAMES_BY_LAYER[layer_kind]
    except KeyError as exc:
        raise RefractionStaticArtifactError(
            f'unsupported cell velocity layer kind: {layer_kind}'
        ) from exc


def _cell_velocity_artifact_entries_for_layer(
    layer_kind: RefractionLayerKind,
) -> tuple[dict[str, str | bool], ...]:
    names = _cell_velocity_artifact_names(layer_kind)
    label = _CELL_VELOCITY_LABEL_BY_LAYER[layer_kind]
    return (
        {
            'name': names.cells_csv,
            'kind': 'csv',
            'required': True,
            'description': f'Per-cell {label} refractor velocity grid and QC metrics',
        },
        {
            'name': names.grid_npz,
            'kind': 'npz',
            'required': True,
            'description': f'Machine-readable {label} refractor velocity cell grid',
        },
        {
            'name': names.qc_json,
            'kind': 'json',
            'required': True,
            'description': f'{label} refractor velocity cell QC summary',
        },
        {
            'name': names.solver_history_csv,
            'kind': 'csv',
            'required': True,
            'description': f'Cell {label} solver convergence and history summary',
        },
    )


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
        'name': REFRACTION_STATIC_HISTORY_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Static-component lineage and double-application audit history',
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
        'name': REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Observation-level first-break time QC export',
    },
    {
        'name': REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Viewer-ready observed-modeled first-break fit QC table',
    },
    {
        'name': REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
        'kind': 'npz',
        'required': True,
        'description': 'Machine-readable observed-modeled first-break fit QC arrays',
    },
    {
        'name': REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Observed-modeled first-break fit QC schema and summary',
    },
    {
        'name': REFRACTION_REDUCED_TIME_QC_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Reduced-time first-break QC table for LMO displays',
    },
    {
        'name': REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
        'kind': 'npz',
        'required': True,
        'description': 'Machine-readable reduced-time first-break QC arrays',
    },
    {
        'name': REFRACTION_REDUCED_TIME_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Reduced-time first-break QC schema and summary',
    },
    {
        'name': REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Source/receiver endpoint static component table',
    },
    {
        'name': REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Trace-level static component waterfall QC table',
    },
    {
        'name': REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Endpoint-level static component waterfall QC table',
    },
    {
        'name': REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
        'kind': 'npz',
        'required': True,
        'description': 'Machine-readable static component waterfall QC arrays',
    },
    {
        'name': REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Static component waterfall QC schema and summary',
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
    {
        'name': REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Source endpoint line-profile QC rows sorted by inline distance',
    },
    {
        'name': REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Receiver endpoint line-profile QC rows sorted by inline distance',
    },
    {
        'name': REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Combined source/receiver line-profile QC rows',
    },
    {
        'name': REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
        'kind': 'npz',
        'required': True,
        'description': 'Machine-readable source/receiver line-profile QC arrays',
    },
    {
        'name': REFRACTION_LINE_PROFILE_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Line-profile QC schema, availability, and summary',
    },
    {
        'name': REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Spreadsheet endpoint time terms, layers, statics, and statuses',
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

_SOURCE_DEPTH_FIELD_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'origin': 'upstream',
        'description': 'Source-depth field-correction QC summary',
    },
    {
        'name': REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'origin': 'upstream',
        'description': 'Resolved source-depth rows used by field corrections',
    },
)
_SOURCE_DEPTH_FIELD_ARTIFACT_NAMES = frozenset(
    str(item['name']) for item in _SOURCE_DEPTH_FIELD_ARTIFACTS
)

_UPHOLE_FIELD_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_UPHOLE_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'origin': 'upstream',
        'description': 'Uphole-time field-correction QC summary',
    },
    {
        'name': REFRACTION_UPHOLE_SOURCES_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'origin': 'upstream',
        'description': 'Resolved source uphole-time rows used by field corrections',
    },
)
_UPHOLE_FIELD_ARTIFACT_NAMES = frozenset(
    str(item['name']) for item in _UPHOLE_FIELD_ARTIFACTS
)

_UPSTREAM_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    _V1_ARTIFACTS + _SOURCE_DEPTH_FIELD_ARTIFACTS + _UPHOLE_FIELD_ARTIFACTS
)
_UPSTREAM_ARTIFACT_NAMES = frozenset(
    str(item['name']) for item in _UPSTREAM_ARTIFACTS
)

_T1LSST_1LAYER_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'T1LSST-compatible one-layer source/receiver components',
    },
)

_REFRACTOR_CELL_VELOCITY_ARTIFACTS = _cell_velocity_artifact_entries_for_layer(
    'v2_t1'
)
_ALL_REFRACTOR_CELL_VELOCITY_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    _REFRACTOR_CELL_VELOCITY_ARTIFACTS
    + _cell_velocity_artifact_entries_for_layer('v3_t2')
    + _cell_velocity_artifact_entries_for_layer('vsub_t3')
)
_GRID_MAP_QC_ARTIFACTS: tuple[dict[str, str | bool], ...] = (
    {
        'name': REFRACTION_GRID_MAP_QC_CSV_NAME,
        'kind': 'csv',
        'required': True,
        'description': 'Viewer-ready refraction cell velocity grid map QC rows',
    },
    {
        'name': REFRACTION_GRID_MAP_QC_NPZ_NAME,
        'kind': 'npz',
        'required': True,
        'description': 'Machine-readable refraction cell velocity grid map QC arrays',
    },
    {
        'name': REFRACTION_GRID_MAP_QC_JSON_NAME,
        'kind': 'json',
        'required': True,
        'description': 'Refraction cell velocity grid map QC summary',
    },
)

_ARTIFACT_CONTENT_TYPE_BY_KIND = {
    'csv': 'text/csv',
    'json': 'application/json',
    'npz': 'application/octet-stream',
}

REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES = frozenset(
    str(item['name'])
    for item in (
        _ARTIFACTS
        + _UPSTREAM_ARTIFACTS
        + _T1LSST_1LAYER_ARTIFACTS
        + _ALL_REFRACTOR_CELL_VELOCITY_ARTIFACTS
        + _GRID_MAP_QC_ARTIFACTS
    )
) | {
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_STATIC_FAILURE_DIAGNOSTICS_JSON_NAME,
    REFRACTION_STATIC_PREFLIGHT_QC_JSON_NAME,
    REFRACTION_STATIC_PREFLIGHT_OBSERVATIONS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    UPLOADED_REFRACTION_PICKS_NPZ_NAME,
} | frozenset(all_refraction_design_matrix_layer_artifact_names())


def registered_artifact_names() -> frozenset[str]:
    """Return all artifact names accepted by refraction artifact downloads."""
    return REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES


def _request_cell_velocity_layer_kinds(
    req: RefractionStaticApplyRequest,
) -> tuple[RefractionLayerKind, ...]:
    if req.model.method != 'multilayer_time_term':
        return ('v2_t1',) if req.model.bedrock_velocity_mode == 'solve_cell' else ()
    return tuple(
        config.kind
        for config in normalize_refraction_static_layers(req.model)
        if config.velocity_mode == 'solve_cell'
    )


def _cell_velocity_layer_kind(
    req: RefractionStaticApplyRequest,
    *,
    layer_kind: RefractionLayerKind | None = None,
) -> RefractionLayerKind:
    layer_kinds = _request_cell_velocity_layer_kinds(req)
    if layer_kind is not None:
        if layer_kind not in layer_kinds:
            raise RefractionStaticArtifactError(
                f'cell velocity layer {layer_kind} is not configured as solve_cell'
            )
        return layer_kind
    if len(layer_kinds) == 1:
        return layer_kinds[0]
    if not layer_kinds and req.model.bedrock_velocity_mode == 'solve_cell':
        return 'v2_t1'
    if layer_kinds:
        raise RefractionStaticArtifactError(
            'cell velocity layer kind is required when multiple solve_cell '
            'layers are configured'
        )
    raise RefractionStaticArtifactError(
        'cell velocity artifacts require a solve_cell velocity layer'
    )


def _cell_velocity_artifact_names_for_request(
    req: RefractionStaticApplyRequest,
) -> _CellVelocityArtifactNames:
    return _cell_velocity_artifact_names(_cell_velocity_layer_kind(req))


def cell_velocity_artifact_paths_for_request(
    root: Path,
    req: RefractionStaticApplyRequest,
) -> tuple[_CellVelocityArtifactPaths, ...]:
    paths: list[_CellVelocityArtifactPaths] = []
    for layer_kind in _request_cell_velocity_layer_kinds(req):
        names = _cell_velocity_artifact_names(layer_kind)
        paths.append(
            _CellVelocityArtifactPaths(
                layer_kind=layer_kind,
                cells_csv=root / names.cells_csv,
                grid_npz=root / names.grid_npz,
                qc_json=root / names.qc_json,
                solver_history_csv=root / names.solver_history_csv,
            )
        )
    return tuple(paths)


def artifact_entries_for_request(
    req: RefractionStaticApplyRequest,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    *,
    upstream_artifact_names: Iterable[str] = (),
) -> tuple[dict[str, str | bool], ...]:
    return (
        _ARTIFACTS
        + _grid_map_qc_artifact_entries(req)
        + _refractor_cell_velocity_artifact_entries(req)
        + _design_matrix_diagnostics_artifact_entries(req)
        + _t1lsst_artifact_entries(req)
        + _upstream_artifact_entries(
            _validate_upstream_artifact_names(
                upstream_artifact_names,
                resolved_first_layer=resolved_first_layer,
                req=req,
            )
        )
    )


def _grid_map_qc_artifact_entries(
    req: RefractionStaticApplyRequest,
) -> tuple[dict[str, str | bool], ...]:
    if _request_cell_velocity_layer_kinds(req):
        return _GRID_MAP_QC_ARTIFACTS
    return ()


def _refractor_cell_velocity_artifact_entries(
    req: RefractionStaticApplyRequest,
) -> tuple[dict[str, str | bool], ...]:
    entries: list[dict[str, str | bool]] = []
    for layer_kind in _request_cell_velocity_layer_kinds(req):
        entries.extend(_cell_velocity_artifact_entries_for_layer(layer_kind))
    return tuple(entries)


def _design_matrix_diagnostics_artifact_entries(
    req: RefractionStaticApplyRequest,
) -> tuple[dict[str, str | bool], ...]:
    if req.model.method != 'multilayer_time_term':
        return ()
    entries: list[dict[str, str | bool]] = []
    for config in normalize_refraction_static_layers(req.model):
        label = _CELL_VELOCITY_LABEL_BY_LAYER[config.kind]
        entries.extend(
            (
                {
                    'name': refraction_design_matrix_layer_qc_json_name(config.kind),
                    'kind': 'json',
                    'required': True,
                    'description': f'{label} design-matrix QC summary',
                },
                {
                    'name': (
                        refraction_design_matrix_layer_node_diagnostics_csv_name(
                            config.kind
                        )
                    ),
                    'kind': 'csv',
                    'required': True,
                    'description': f'{label} design-matrix node diagnostics',
                },
            )
        )
    return tuple(entries)


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
    req: RefractionStaticApplyRequest | None = None,
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
        if name not in _UPSTREAM_ARTIFACT_NAMES:
            raise RefractionStaticArtifactError(
                f'unsupported upstream artifact: {name}'
            )
        seen.add(name)
        values.append(name)

    value_set = set(values)
    if not value_set:
        return ()

    _validate_upstream_artifact_group(
        value_set=value_set,
        expected=_V1_ARTIFACT_NAMES,
        group_label='upstream V1 artifacts',
    )
    _validate_upstream_artifact_group(
        value_set=value_set,
        expected=_SOURCE_DEPTH_FIELD_ARTIFACT_NAMES,
        group_label='upstream source-depth field-correction artifacts',
    )
    _validate_upstream_artifact_group(
        value_set=value_set,
        expected=_UPHOLE_FIELD_ARTIFACT_NAMES,
        group_label='upstream uphole field-correction artifacts',
    )

    if value_set & _V1_ARTIFACT_NAMES:
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

    if req is not None:
        if (
            value_set & _SOURCE_DEPTH_FIELD_ARTIFACT_NAMES
            and req.field_corrections.source_depth.mode != 'weathering_velocity_time'
        ):
            raise RefractionStaticArtifactError(
                'source-depth field-correction artifacts are only valid when '
                'field_corrections.source_depth.mode is weathering_velocity_time'
            )
        if value_set & _UPHOLE_FIELD_ARTIFACT_NAMES and (
            req.field_corrections.uphole.mode != 'header_time'
        ):
            raise RefractionStaticArtifactError(
                'uphole field-correction artifacts are only valid when '
                'field_corrections.uphole.mode is header_time'
            )

    return tuple(
        str(item['name'])
        for item in _UPSTREAM_ARTIFACTS
        if str(item['name']) in value_set
    )


def _validate_upstream_artifact_group(
    *,
    value_set: set[str],
    expected: frozenset[str],
    group_label: str,
) -> None:
    present = value_set & expected
    if not present or present == expected:
        return
    expected_text = ', '.join(sorted(expected))
    raise RefractionStaticArtifactError(
        f'{group_label} must include all of: {expected_text}'
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
    return tuple(item for item in _UPSTREAM_ARTIFACTS if str(item['name']) in name_set)


def build_manifest_payload(
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
                'content_type': _artifact_content_type(str(item['kind'])),
                'required': bool(item['required']),
                'origin': str(item.get('origin', 'final')),
                'description': str(item['description']),
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


def _artifact_content_type(kind: str) -> str:
    try:
        return _ARTIFACT_CONTENT_TYPE_BY_KIND[kind]
    except KeyError as exc:
        raise RefractionStaticArtifactError(
            f'unsupported artifact kind for content type: {kind}'
        ) from exc


_artifact_entries_for_request = artifact_entries_for_request
_build_manifest_payload = build_manifest_payload
_cell_velocity_artifact_paths_for_request = cell_velocity_artifact_paths_for_request


__all__ = [
    'artifact_entries_for_request',
    'build_manifest_payload',
    'cell_velocity_artifact_paths_for_request',
    'registered_artifact_names',
    '_artifact_content_type',
    '_artifact_entries_for_request',
    '_artifact_list_for_qc',
    '_build_manifest_payload',
    '_cell_velocity_artifact_entries_for_layer',
    '_cell_velocity_artifact_names',
    '_cell_velocity_artifact_names_for_request',
    '_cell_velocity_artifact_paths_for_request',
    '_cell_velocity_layer_kind',
    '_design_matrix_diagnostics_artifact_entries',
    '_grid_map_qc_artifact_entries',
    '_refractor_cell_velocity_artifact_entries',
    '_request_cell_velocity_layer_kinds',
    '_t1lsst_artifact_entries',
    '_upstream_artifact_entries',
    '_validate_declared_upstream_artifacts',
    '_validate_upstream_artifact_names',
]
