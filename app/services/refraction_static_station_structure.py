"""Station-structure QC assembly for completed refraction static jobs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from app.api.schemas import RefractionStaticStationStructureRequest
from app.services.job_manager import JobManager
from app.services.refraction_static_artifacts import (
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)


class RefractionStaticStationStructureError(ValueError):
    """Raised when station-structure QC cannot be assembled."""


_NUMERIC_SUFFIX_RE = re.compile(r'([-+]?\d+(?:\.\d+)?)\s*$')

_VELOCITY_FIELDS = ('v2', 'v1', 'v3', 'vsub')
_DEPTH_FIELDS = (
    'sh1',
    'sh2',
    'sh3',
    'refractor_depth',
    'refractor_elevation',
    'layer1_base_elevation',
    'layer2_base_elevation',
)

_DEPTH_LABELS = {
    'sh1': 'Weathering thickness SH1',
    'sh2': 'Weathering thickness SH2',
    'sh3': 'Weathering thickness SH3',
    'refractor_depth': 'Refractor depth',
    'refractor_elevation': 'Refractor elevation',
    'layer1_base_elevation': 'Layer 1 base elevation',
    'layer2_base_elevation': 'Layer 2 base elevation',
}


def build_refraction_static_station_structure(
    *,
    job_id: str,
    job: dict[str, object],
    req: RefractionStaticStationStructureRequest,
) -> dict[str, Any]:
    """Build columnar station-structure QC data from completed artifacts."""
    if job.get('statics_kind') != 'refraction':
        raise RefractionStaticStationStructureError(
            f'Job {job_id} is not a refraction statics job'
        )
    if not JobManager.is_ready_status_value(job.get('status')):
        raise RefractionStaticStationStructureError(
            f'Job {job_id} is not complete; current state is '
            f'{JobManager.normalize_status_value(job.get("status"))}'
        )

    artifacts_dir = _job_artifacts_dir(job, job_id)
    rows = _read_endpoint_rows(artifacts_dir)
    if not rows:
        raise RefractionStaticStationStructureError(
            'Station-structure QC requires endpoint static artifacts'
        )

    warnings: list[str] = []
    source_rows, receiver_rows, filter_status = _apply_gather_filter(
        rows,
        artifacts_dir=artifacts_dir,
        req=req,
        warnings=warnings,
    )
    x_axis, x_axis_label = _resolve_x_axis(source_rows + receiver_rows, req.x_axis)
    velocity_field = _resolve_velocity_field(source_rows, receiver_rows, req.velocity_field)
    depth_field = _resolve_depth_field(source_rows, receiver_rows, req.depth_field)
    _append_linked_node_velocity_warning(
        source_rows,
        receiver_rows,
        x_axis,
        velocity_field,
        warnings,
    )

    return {
        'job_id': job_id,
        'statics_kind': 'refraction',
        'view_kind': 'station_structure',
        'x_axis': x_axis,
        'x_axis_label': x_axis_label,
        'filter_status': filter_status,
        'gather_range': {
            'start': req.gather_start,
            'end': req.gather_end,
        },
        'colors': {
            'source': 'cyan',
            'receiver': 'red',
        },
        'time_term': {
            'field': _time_term_field_for_velocity(velocity_field),
            'label': 'Time-term distribution',
            'unit': 'ms',
            'source': _series(
                source_rows,
                'source',
                x_axis,
                lambda kind, row: _time_term_candidates(kind, row, velocity_field),
            ),
            'receiver': _series(
                receiver_rows,
                'receiver',
                x_axis,
                lambda kind, row: _time_term_candidates(kind, row, velocity_field),
            ),
        },
        'velocity': {
            'field': velocity_field,
            'label': f'Velocity structure: {velocity_field.upper()}',
            'unit': 'm/s',
            'source': _series(
                source_rows,
                'source',
                x_axis,
                lambda kind, row: _velocity_candidates(kind, row, velocity_field),
            ),
            'receiver': _series(
                receiver_rows,
                'receiver',
                x_axis,
                lambda kind, row: _velocity_candidates(kind, row, velocity_field),
            ),
        },
        'depth': {
            'field': depth_field,
            'label': _DEPTH_LABELS[depth_field],
            'unit': 'm',
            'source': _series(
                source_rows,
                'source',
                x_axis,
                lambda kind, row: _depth_candidates(kind, row, depth_field),
            ),
            'receiver': _series(
                receiver_rows,
                'receiver',
                x_axis,
                lambda kind, row: _depth_candidates(kind, row, depth_field),
            ),
        },
        'warnings': warnings,
    }


def _job_artifacts_dir(job: dict[str, object], job_id: str) -> Path:
    raw = job.get('artifacts_dir')
    if not isinstance(raw, str) or not raw:
        raise RefractionStaticStationStructureError(
            f'Job {job_id} metadata is missing artifacts_dir'
        )
    path = Path(raw)
    if not path.is_dir():
        raise RefractionStaticStationStructureError(
            f'Job {job_id} artifacts directory is not available'
        )
    return path


def _read_endpoint_rows(artifacts_dir: Path) -> list[dict[str, Any]]:
    combined = artifacts_dir / REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME
    if combined.is_file():
        rows = _read_csv(combined)
        if rows:
            return rows

    rows: list[dict[str, Any]] = []
    source_path = artifacts_dir / SOURCE_STATIC_TABLE_CSV_NAME
    if source_path.is_file():
        for row in _read_csv(source_path):
            row.setdefault('endpoint_kind', 'source')
            if not row.get('endpoint_key') and row.get('source_endpoint_key'):
                row['endpoint_key'] = row['source_endpoint_key']
            rows.append(row)
    receiver_path = artifacts_dir / RECEIVER_STATIC_TABLE_CSV_NAME
    if receiver_path.is_file():
        for row in _read_csv(receiver_path):
            row.setdefault('endpoint_kind', 'receiver')
            if not row.get('endpoint_key') and row.get('receiver_endpoint_key'):
                row['endpoint_key'] = row['receiver_endpoint_key']
            rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _apply_gather_filter(
    rows: list[dict[str, Any]],
    *,
    artifacts_dir: Path,
    req: RefractionStaticStationStructureRequest,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    source_rows = [row for row in rows if _endpoint_kind(row) == 'source']
    receiver_rows = [row for row in rows if _endpoint_kind(row) == 'receiver']
    if req.gather_start is None and req.gather_end is None:
        return source_rows, receiver_rows, 'unfiltered'

    filtered_sources = [
        row for row in source_rows if _include_gather_number(_source_gather_number(row), req)
    ]
    observations = _read_observation_rows(artifacts_dir)
    if observations is None:
        warnings.append(
            'Receiver participation for the selected shot-gather range could not '
            'be determined from available first-break QC artifacts; receiver '
            'series is unfiltered.'
        )
        return filtered_sources, receiver_rows, 'receiver_participation_unavailable'

    receiver_keys: set[str] = set()
    receiver_numbers: set[float] = set()
    has_filterable_observation = False
    for row in observations:
        gather_number = _observation_source_gather_number(row)
        if gather_number is None:
            continue
        has_filterable_observation = True
        if not _include_gather_number(gather_number, req):
            continue
        receiver_key = _first_text(
            row,
            'receiver_endpoint_key',
            'receiver_key',
            'endpoint_key',
        )
        if receiver_key:
            receiver_keys.add(receiver_key)
        receiver_number = _numeric_or_none(
            _first_present(
                row,
                'receiver_id',
                'endpoint_id',
                'global_receiver_number',
                'station_number',
                'receiver_number',
            )
        )
        if receiver_number is not None:
            receiver_numbers.add(receiver_number)

    if not has_filterable_observation:
        warnings.append(
            'Receiver participation for the selected shot-gather range could not '
            'be determined because available first-break QC artifacts do not '
            'include usable source gather identifiers; receiver series is '
            'unfiltered.'
        )
        return filtered_sources, receiver_rows, 'receiver_participation_unavailable'

    filtered_receivers = [
        row
        for row in receiver_rows
        if _endpoint_identity_matches(row, receiver_keys, receiver_numbers)
    ]
    return filtered_sources, filtered_receivers, 'ok'


def _read_observation_rows(artifacts_dir: Path) -> list[dict[str, Any]] | None:
    npz_path = artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME
    if npz_path.is_file():
        return _read_observation_rows_from_npz(npz_path)
    csv_path = artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME
    if csv_path.is_file():
        return _read_csv(csv_path)
    return None


def _read_observation_rows_from_npz(path: Path) -> list[dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            n_rows = 0
            for key in data.files:
                value = np.asarray(data[key])
                if value.ndim > 0:
                    n_rows = int(value.shape[0])
                    break
            rows: list[dict[str, Any]] = []
            for index in range(n_rows):
                row: dict[str, Any] = {}
                for key in data.files:
                    value = np.asarray(data[key])
                    if value.ndim == 0 or value.shape[0] != n_rows:
                        continue
                    row[key] = _np_scalar_to_python(value[index])
                rows.append(row)
            return rows
    except Exception as exc:  # noqa: BLE001
        raise RefractionStaticStationStructureError(
            f'Could not read refraction observation artifact {path.name}'
        ) from exc


def _np_scalar_to_python(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def _resolve_x_axis(
    rows: list[dict[str, Any]],
    requested: str,
) -> tuple[str, str]:
    if requested != 'auto':
        if requested == 'inline_m':
            return 'inline_m', 'Inline distance (m)'
        return requested, _x_axis_label(requested)
    for field in ('global_receiver_number', 'station_number', 'receiver_number'):
        if any(_numeric_or_none(row.get(field)) is not None for row in rows):
            return field, _x_axis_label(field)
    for field in ('endpoint_id', 'source_id', 'receiver_id'):
        if any(_numeric_or_none(row.get(field)) is not None for row in rows):
            return field, _x_axis_label(field)
    if any(_numeric_suffix(_endpoint_key(row)) is not None for row in rows):
        return 'endpoint_key_numeric_suffix', 'Station number'
    if any(_numeric_or_none(row.get('inline_m')) is not None for row in rows):
        return 'inline_m', 'Inline distance (m)'
    raise RefractionStaticStationStructureError(
        'Station-structure QC requires a station-like x-axis field'
    )


def _x_axis_label(field: str) -> str:
    if field == 'global_receiver_number':
        return 'Global receiver number'
    if field in {'station_number', 'receiver_number', 'endpoint_id', 'source_id', 'receiver_id'}:
        return 'Station number'
    if field == 'inline_m':
        return 'Inline distance (m)'
    return field.replace('_', ' ').title()


def _resolve_velocity_field(
    source_rows: list[dict[str, Any]],
    receiver_rows: list[dict[str, Any]],
    requested: str,
) -> str:
    if requested != 'auto':
        return requested
    for field in _VELOCITY_FIELDS:
        if _has_series_values(source_rows, receiver_rows, lambda kind, row: _velocity_candidates(kind, row, field)):
            return field
    return 'v2'


def _resolve_depth_field(
    source_rows: list[dict[str, Any]],
    receiver_rows: list[dict[str, Any]],
    requested: str,
) -> str:
    if requested != 'auto':
        return requested
    for field in _DEPTH_FIELDS:
        if _has_series_values(source_rows, receiver_rows, lambda kind, row: _depth_candidates(kind, row, field)):
            return field
    return 'sh1'


def _has_series_values(
    source_rows: list[dict[str, Any]],
    receiver_rows: list[dict[str, Any]],
    candidates_for: Any,
) -> bool:
    for kind, rows in (('source', source_rows), ('receiver', receiver_rows)):
        for row in rows:
            if _candidate_value(row, candidates_for(kind, row)) is not None:
                return True
    return False


def _append_linked_node_velocity_warning(
    source_rows: list[dict[str, Any]],
    receiver_rows: list[dict[str, Any]],
    x_axis: str,
    velocity_field: str,
    warnings: list[str],
) -> None:
    for kind, rows in (('source', source_rows), ('receiver', receiver_rows)):
        for row in rows:
            if _x_value(row, kind, x_axis) is None:
                continue
            field, value = _candidate_value_with_field(
                row,
                _velocity_candidates(kind, row, velocity_field),
            )
            if field == 'linked_node_velocity_m_s' and value is not None:
                warnings.append(
                    'Velocity structure uses linked_node_velocity_m_s for one or '
                    'more endpoints because endpoint-side velocity fields are '
                    'unavailable; linked source and receiver endpoints may share '
                    'the same velocity value.'
                )
                return


def _series(
    rows: list[dict[str, Any]],
    endpoint_kind: str,
    x_axis: str,
    candidates_for: Any,
) -> dict[str, list[Any]]:
    points: list[tuple[float, float, str, str]] = []
    for row in rows:
        x_value = _x_value(row, endpoint_kind, x_axis)
        y_value = _candidate_value(row, candidates_for(endpoint_kind, row))
        if x_value is None or y_value is None:
            continue
        points.append((x_value, y_value, _endpoint_key(row), _status(row)))
    points.sort(key=lambda point: (point[0], point[2]))
    return {
        'x': [_json_number(point[0]) for point in points],
        'y': [float(point[1]) for point in points],
        'endpoint_key': [point[2] for point in points],
        'status': [point[3] for point in points],
    }


def _x_value(row: dict[str, Any], endpoint_kind: str, x_axis: str) -> float | None:
    if x_axis == 'endpoint_key_numeric_suffix':
        return _numeric_suffix(_endpoint_key(row))
    if x_axis == 'source_id' and endpoint_kind != 'source':
        return _numeric_or_none(row.get('receiver_id')) or _numeric_or_none(row.get('source_id'))
    if x_axis == 'receiver_id' and endpoint_kind != 'receiver':
        return _numeric_or_none(row.get('source_id')) or _numeric_or_none(row.get('receiver_id'))
    value = _numeric_or_none(row.get(x_axis))
    if value is not None:
        return value
    if x_axis in {'endpoint_id', 'source_id', 'receiver_id'}:
        return _numeric_suffix(_endpoint_key(row))
    return None


def _json_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return float(value)


def _time_term_field_for_velocity(velocity_field: str) -> str:
    if velocity_field == 'v3':
        return 't2'
    if velocity_field == 'vsub':
        return 't3'
    return 't1'


def _time_term_candidates(
    endpoint_kind: str,
    row: dict[str, Any],
    velocity_field: str,
) -> tuple[str, ...]:
    layer_field = _time_term_field_for_velocity(velocity_field)
    if endpoint_kind == 'source':
        return (
            'source_time_term_ms',
            'source_half_intercept_time_ms',
            f'source_{layer_field}_ms',
            'half_intercept_time_ms',
            f'{layer_field}_ms',
        )
    return (
        'receiver_time_term_ms',
        'receiver_half_intercept_time_ms',
        f'receiver_{layer_field}_ms',
        'half_intercept_time_ms',
        f'{layer_field}_ms',
    )


def _velocity_candidates(
    endpoint_kind: str,
    row: dict[str, Any],
    field: str,
) -> tuple[str, ...]:
    return (
        f'{endpoint_kind}_{field}_m_s',
        f'{endpoint_kind}_velocity_m_s',
        f'{field}_m_s',
        'linked_node_velocity_m_s',
    )


def _depth_candidates(
    endpoint_kind: str,
    row: dict[str, Any],
    field: str,
) -> tuple[str, ...]:
    if field in {'sh1', 'sh2', 'sh3'}:
        return (
            f'{endpoint_kind}_{field}_m',
            f'{endpoint_kind}_weathering_thickness_m',
            f'{field}_m',
            f'{field}_weathering_thickness_m',
            'weathering_thickness_m',
        )
    if field == 'refractor_depth':
        return (
            f'{endpoint_kind}_refractor_depth_m',
            'refractor_depth_m',
        )
    if field == 'refractor_elevation':
        return (
            f'{endpoint_kind}_refractor_elevation_m',
            'final_refractor_elevation_m',
            'refractor_elevation_m',
        )
    return (
        f'{endpoint_kind}_{field}_m',
        f'{field}_m',
    )


def _candidate_value(row: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    return _candidate_value_with_field(row, candidates)[1]


def _candidate_value_with_field(
    row: dict[str, Any],
    candidates: tuple[str, ...],
) -> tuple[str | None, float | None]:
    for field in candidates:
        value = _numeric_or_none(row.get(field))
        if value is not None:
            return field, value
    return None, None


def _endpoint_kind(row: dict[str, Any]) -> str:
    return str(row.get('endpoint_kind') or '').strip().lower()


def _endpoint_key(row: dict[str, Any]) -> str:
    return _first_text(
        row,
        'endpoint_key',
        'source_endpoint_key',
        'receiver_endpoint_key',
    ) or ''


def _status(row: dict[str, Any]) -> str:
    return _first_text(
        row,
        'static_status',
        'solution_status',
        'weathering_status',
        'v2_status',
        'status',
    ) or 'ok'


def _source_gather_number(row: dict[str, Any]) -> float | None:
    value = _numeric_or_none(
        _first_present(
            row,
            'source_id',
            'source_number',
            'shot_id',
            'gather_id',
            'source_endpoint_key',
        )
    )
    if value is not None:
        return value
    return _numeric_suffix(_first_text(row, 'source_endpoint_key', 'endpoint_key') or '')


def _observation_source_gather_number(row: dict[str, Any]) -> float | None:
    value = _numeric_or_none(
        _first_present(
            row,
            'source_id',
            'source_number',
            'shot_id',
            'gather_id',
            'source_endpoint_key',
        )
    )
    if value is not None:
        return value
    return _numeric_suffix(_first_text(row, 'source_endpoint_key') or '')


def _include_gather_number(
    gather_number: float | None,
    req: RefractionStaticStationStructureRequest,
) -> bool:
    if gather_number is None:
        return True
    if req.gather_start is not None and gather_number < req.gather_start:
        return False
    if req.gather_end is not None and gather_number > req.gather_end:
        return False
    return True


def _endpoint_identity_matches(
    row: dict[str, Any],
    keys: set[str],
    numbers: set[float],
) -> bool:
    endpoint_key = _endpoint_key(row)
    if endpoint_key and endpoint_key in keys:
        return True
    endpoint_number = _numeric_or_none(
        _first_present(
            row,
            'receiver_id',
            'endpoint_id',
            'global_receiver_number',
            'station_number',
            'receiver_number',
        )
    )
    if endpoint_number is None:
        endpoint_number = _numeric_suffix(endpoint_key)
    return endpoint_number in numbers if endpoint_number is not None else False


def _first_present(row: dict[str, Any], *keys: str) -> object:
    for key in keys:
        value = row.get(key)
        if value is not None and value != '':
            return value
    return None


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    value = _first_present(row, *keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _numeric_or_none(value: object) -> float | None:
    if value is None or value == '':
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _numeric_suffix(str(value))
    return number if np.isfinite(number) else None


def _numeric_suffix(value: str) -> float | None:
    match = _NUMERIC_SUFFIX_RE.search(str(value).strip())
    if not match:
        return None
    try:
        number = float(match.group(1))
    except ValueError:
        return None
    return number if np.isfinite(number) else None


__all__ = [
    'RefractionStaticStationStructureError',
    'build_refraction_static_station_structure',
]
