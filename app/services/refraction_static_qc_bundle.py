"""Compact QC bundle assembly for completed refraction static jobs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.api.schemas import RefractionStaticQcBundleRequest
from app.services.job_manager import JobManager
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_GRID_MAP_QC_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
)
from app.services.refraction_static_export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)

_DOWNSAMPLING_METHOD = 'even_index_floor_first_last'
_COORDINATE_MODES = {'line_2d_projected', 'grid_3d'}


class RefractionStaticQcBundleError(ValueError):
    """Raised when a QC bundle cannot be assembled from job artifacts."""


@dataclass(frozen=True)
class _TabularViewSpec:
    include: str
    view_name: str
    artifact_name: str


_TABULAR_VIEW_SPECS: tuple[_TabularViewSpec, ...] = (
    _TabularViewSpec(
        include='first_break',
        view_name='first_break_fit',
        artifact_name=REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    ),
    _TabularViewSpec(
        include='first_break',
        view_name='first_break_residual',
        artifact_name=FIRST_BREAK_RESIDUALS_CSV_NAME,
    ),
    _TabularViewSpec(
        include='reduced_time',
        view_name='reduced_time',
        artifact_name=REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    ),
    _TabularViewSpec(
        include='profiles',
        view_name='line_profiles',
        artifact_name=REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    ),
    _TabularViewSpec(
        include='static_components',
        view_name='static_component_qc_endpoint',
        artifact_name=REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    ),
    _TabularViewSpec(
        include='static_components',
        view_name='static_component_qc_trace',
        artifact_name=REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    ),
    _TabularViewSpec(
        include='static_components',
        view_name='static_components',
        artifact_name=REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    ),
    _TabularViewSpec(
        include='cells',
        view_name='refraction_grid_map_qc',
        artifact_name=REFRACTION_GRID_MAP_QC_CSV_NAME,
    ),
    _TabularViewSpec(
        include='cells',
        view_name='refractor_cells',
        artifact_name=REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    ),
    _TabularViewSpec(
        include='cells',
        view_name='v3_refractor_cells',
        artifact_name=REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    ),
    _TabularViewSpec(
        include='cells',
        view_name='vsub_refractor_cells',
        artifact_name=REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    ),
)


def build_refraction_static_qc_bundle(
    *,
    job_id: str,
    job: dict[str, object],
    req: RefractionStaticQcBundleRequest,
) -> dict[str, Any]:
    """Build a compact viewer-oriented QC bundle from existing artifacts."""
    if job.get('statics_kind') != 'refraction':
        raise RefractionStaticQcBundleError(
            f'Job {job_id} is not a refraction statics job'
        )
    if not JobManager.is_ready_status_value(job.get('status')):
        raise RefractionStaticQcBundleError(
            f'Job {job_id} is not complete; current state is '
            f'{JobManager.normalize_status_value(job.get("status"))}'
        )

    artifacts_dir = _job_artifacts_dir(job, job_id)
    manifest = _read_json_artifact(
        artifacts_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    )
    _validate_manifest(manifest, artifacts_dir)
    qc = _read_json_artifact(
        artifacts_dir / REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
    )
    sign_convention = _extract_sign_convention(qc)
    coordinate_mode = _resolve_coordinate_mode(qc, req.coordinate_mode)
    artifact_refs = _artifact_refs(manifest, artifacts_dir)

    available_views: list[str] = []
    unavailable_views: list[str] = []
    views: dict[str, dict[str, Any]] = {}
    downsampling: dict[str, dict[str, Any]] = {}

    includes = set(req.include)
    if 'summary' in includes:
        available_views.append('summary')

    for spec in _TABULAR_VIEW_SPECS:
        if spec.include not in includes:
            continue
        artifact_path = artifacts_dir / spec.artifact_name
        if not artifact_path.is_file():
            continue
        view_payload = _read_tabular_view(
            artifact_path,
            view_name=spec.view_name,
            max_points=req.max_points,
        )
        views[spec.view_name] = view_payload
        downsampling[spec.view_name] = {
            'total_points': view_payload['total_points'],
            'returned_points': view_payload['returned_points'],
            'downsampled': view_payload['downsampled'],
            'method': view_payload['downsampling_method'],
        }
        available_views.append(spec.view_name)

    for include in req.include:
        if include in {'summary', 'gather_preview'}:
            continue
        if any(spec.include == include and spec.view_name in views for spec in _TABULAR_VIEW_SPECS):
            continue
        unavailable_views.append(include)
    if 'gather_preview' in includes:
        unavailable_views.append('gather_preview')

    if not available_views:
        requested = ', '.join(req.include)
        raise RefractionStaticQcBundleError(
            f'Requested QC bundle views are not available from existing '
            f'refraction artifacts: {requested}'
        )

    return {
        'job_id': job_id,
        'statics_kind': 'refraction',
        'sign_convention': sign_convention,
        'coordinate_mode': coordinate_mode,
        'summary': _summary_from_qc(qc, job),
        'artifacts': artifact_refs,
        'available_views': available_views,
        'unavailable_views': unavailable_views,
        'views': views,
        'downsampling': downsampling,
    }


def _job_artifacts_dir(job: dict[str, object], job_id: str) -> Path:
    raw = job.get('artifacts_dir')
    if not isinstance(raw, str) or not raw:
        raise RefractionStaticQcBundleError(
            f'Job {job_id} metadata is missing artifacts_dir'
        )
    path = Path(raw)
    if not path.is_dir():
        raise RefractionStaticQcBundleError(
            f'Job {job_id} artifacts directory is not available'
        )
    return path


def _read_json_artifact(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RefractionStaticQcBundleError(
            f'Refraction QC bundle requires artifact {artifact_name}'
        )
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RefractionStaticQcBundleError(
            f'Refraction artifact {artifact_name} is not valid JSON'
        ) from exc
    if not isinstance(payload, dict):
        raise RefractionStaticQcBundleError(
            f'Refraction artifact {artifact_name} must contain a JSON object'
        )
    return payload


def _validate_manifest(manifest: dict[str, Any], artifacts_dir: Path) -> None:
    if manifest.get('job_kind') != 'statics' or manifest.get('statics_kind') != 'refraction':
        raise RefractionStaticQcBundleError(
            'Refraction artifact manifest is not for a refraction statics job'
        )
    raw_artifacts = manifest.get('artifacts')
    if not isinstance(raw_artifacts, list):
        raise RefractionStaticQcBundleError(
            'Refraction artifact manifest is missing artifacts list'
        )
    for raw_item in raw_artifacts:
        if not isinstance(raw_item, dict):
            raise RefractionStaticQcBundleError(
                'Refraction artifact manifest contains an invalid artifact entry'
            )
        raw_name = raw_item.get('name')
        if not isinstance(raw_name, str) or not raw_name:
            raise RefractionStaticQcBundleError(
                'Refraction artifact manifest contains an unnamed artifact'
            )
        if Path(raw_name).name != raw_name:
            raise RefractionStaticQcBundleError(
                f'Refraction artifact manifest contains a non-plain file name: {raw_name}'
            )
        if bool(raw_item.get('required', False)) and not (artifacts_dir / raw_name).is_file():
            raise RefractionStaticQcBundleError(
                f'Required refraction artifact is missing: {raw_name}'
            )


def _extract_sign_convention(qc: dict[str, Any]) -> str:
    raw = qc.get('sign_convention')
    if isinstance(raw, str):
        sign_convention = raw
    elif isinstance(raw, dict) and isinstance(raw.get('trace_shift_s'), str):
        sign_convention = raw['trace_shift_s']
    else:
        raise RefractionStaticQcBundleError(
            'Refraction QC artifact is missing sign_convention'
        )
    if sign_convention != REFRACTION_STATIC_REPO_SIGN_CONVENTION:
        raise RefractionStaticQcBundleError(
            'Refraction QC artifact has unsupported sign_convention: '
            f'{sign_convention!r}'
        )
    return sign_convention


def _resolve_coordinate_mode(qc: dict[str, Any], requested: str) -> str:
    artifact_mode = _coordinate_mode_from_qc(qc)
    if requested == 'auto':
        return artifact_mode or 'auto'
    if artifact_mode is not None and artifact_mode != requested:
        raise RefractionStaticQcBundleError(
            'Requested coordinate_mode does not match refraction QC artifact: '
            f'{requested!r} != {artifact_mode!r}'
        )
    return requested


def _coordinate_mode_from_qc(qc: dict[str, Any]) -> str | None:
    cells = qc.get('refractor_velocity_cells')
    if isinstance(cells, dict):
        raw_mode = cells.get('coordinate_mode')
        if isinstance(raw_mode, str) and raw_mode in _COORDINATE_MODES:
            return raw_mode
    by_layer = qc.get('refractor_velocity_cells_by_layer')
    if isinstance(by_layer, dict):
        for raw_layer in by_layer.values():
            if not isinstance(raw_layer, dict):
                continue
            raw_mode = raw_layer.get('coordinate_mode')
            if isinstance(raw_mode, str) and raw_mode in _COORDINATE_MODES:
                return raw_mode
    return None


def _artifact_refs(
    manifest: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, str]:
    artifact_names: list[str] = []
    raw_artifacts = manifest.get('artifacts')
    if isinstance(raw_artifacts, list):
        for raw_item in raw_artifacts:
            if not isinstance(raw_item, dict):
                continue
            raw_name = raw_item.get('name')
            if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
                continue
            if raw_name not in artifact_names:
                artifact_names.append(raw_name)
    for artifact_name in (
        REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
        REFRACTION_STATIC_REQUEST_JSON_NAME,
    ):
        if (
            (artifacts_dir / artifact_name).is_file()
            and artifact_name not in artifact_names
        ):
            artifact_names.append(artifact_name)

    key_counts: dict[str, int] = {}
    for artifact_name in artifact_names:
        key = _artifact_ref_base_key(artifact_name)
        key_counts[key] = key_counts.get(key, 0) + 1

    refs: dict[str, str] = {}
    for artifact_name in artifact_names:
        base_key = _artifact_ref_base_key(artifact_name)
        if key_counts[base_key] > 1:
            refs[_artifact_ref_suffix_key(artifact_name)] = artifact_name
        else:
            refs[base_key] = artifact_name
    return dict(sorted(refs.items()))


def _artifact_ref_base_key(name: str) -> str:
    path = Path(name)
    if path.suffix:
        return path.stem
    return path.name


def _artifact_ref_suffix_key(name: str) -> str:
    path = Path(name)
    if not path.suffix:
        return path.name
    return f'{path.stem}_{path.suffix.removeprefix(".")}'


def _summary_from_qc(
    qc: dict[str, Any],
    job: dict[str, object],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'status': 'ok',
        'job_state': JobManager.normalize_status_value(job.get('status')),
    }
    for key in (
        'artifact_version',
        'method',
        'workflow',
        'static_component',
        'conversion_mode',
        'layer_count',
        'request',
        'velocity',
        'datum',
        'observations',
        'nodes',
        'endpoints',
        'first_break_fit',
        'statics',
        'status_counts',
        'warnings',
        'enabled_layer_kinds',
        'observation_gates',
        'refractor_velocity_cells',
        'refractor_velocity_cells_by_layer',
        'layers',
        'field_corrections',
        'static_history',
    ):
        if key in qc:
            summary[key] = qc[key]
    return summary


def _read_tabular_view(
    path: Path,
    *,
    view_name: str,
    max_points: int,
) -> dict[str, Any]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    sampled_rows = _deterministic_sample(rows, max_points)
    return {
        'artifact': path.name,
        'columns': columns,
        'total_points': len(rows),
        'returned_points': len(sampled_rows),
        'downsampled': len(sampled_rows) < len(rows),
        'downsampling_method': _DOWNSAMPLING_METHOD,
        'records': sampled_rows,
    }


def _deterministic_sample(
    rows: list[dict[str, str | None]],
    max_points: int,
) -> list[dict[str, str | None]]:
    total = len(rows)
    if total <= max_points:
        return rows
    if max_points == 1:
        return [rows[0]]
    last = total - 1
    indices = [(index * last) // (max_points - 1) for index in range(max_points)]
    return [rows[index] for index in indices]


__all__ = [
    'RefractionStaticQcBundleError',
    'build_refraction_static_qc_bundle',
]
