"""Artifact registry helpers for refraction static output packages."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_layer_config import (
    normalize_refraction_static_layers,
)
from app.services.refraction_static_types import (
    RefractionLayerKind,
    ResolvedRefractionFirstLayer,
)

from app.services.refraction_static_artifacts.contract import (
    _ARTIFACTS,
    _ARTIFACT_CONTENT_TYPE_BY_KIND,
    _CellVelocityArtifactNames,
    _CellVelocityArtifactPaths,
    _GRID_MAP_QC_ARTIFACTS,
    _T1LSST_1LAYER_ARTIFACTS,
    _UPHOLE_FIELD_ARTIFACT_NAMES,
    _UPSTREAM_ARTIFACT_NAMES,
    _UPSTREAM_ARTIFACTS,
    _SOURCE_DEPTH_FIELD_ARTIFACT_NAMES,
    _V1_ARTIFACT_NAMES,
    _cell_velocity_artifact_entries_for_layer,
    _cell_velocity_artifact_names,
    ARTIFACT_VERSION,
    REFRACTION_STATIC_REGISTERED_ARTIFACT_NAMES,
    RefractionStaticArtifactError,
)


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
    '_grid_map_qc_artifact_entries',
    '_refractor_cell_velocity_artifact_entries',
    '_request_cell_velocity_layer_kinds',
    '_t1lsst_artifact_entries',
    '_upstream_artifact_entries',
    '_validate_declared_upstream_artifacts',
    '_validate_upstream_artifact_names',
]
