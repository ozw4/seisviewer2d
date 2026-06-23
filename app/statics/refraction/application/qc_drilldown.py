"""Detailed QC drilldown for completed refraction static jobs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from app.statics.refraction.contracts.qc import RefractionStaticQcDrilldownRequest
from app.statics.refraction.application.job_status import (
    is_ready_status_value,
    normalize_status_value,
)
from app.statics.refraction.artifacts.export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)
from app.statics.refraction.artifacts import (
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)

_CAP_METHOD = 'observation_index_ascending_first_n'

_CELL_ARTIFACT_BY_LAYER = {
    'v2_t1': REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    'v3_t2': REFRACTION_V3_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    'vsub_t3': REFRACTION_VSUB_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
}

_TEXT_FIELDS = {
    'cell_velocity_component',
    'cell_velocity_layer_kind',
    'coordinate_mode',
    'endpoint_kind',
    'layer_kind',
    'reject_reason',
    'rejection_reason',
    'sign_convention',
    'source_endpoint_key',
    'status',
    'status_reason',
    'receiver_endpoint_key',
    'velocity_status',
}


class RefractionStaticQcDrilldownError(ValueError):
    """Raised when drilldown artifacts cannot be assembled."""


class RefractionStaticQcDrilldownNotFound(LookupError):
    """Raised when the requested endpoint or cell is not present."""


def build_refraction_static_qc_drilldown(
    *,
    job_id: str,
    job: dict[str, object],
    req: RefractionStaticQcDrilldownRequest,
) -> dict[str, Any]:
    """Build a detailed endpoint or cell QC drilldown from existing artifacts."""
    if job.get('statics_kind') != 'refraction':
        raise RefractionStaticQcDrilldownError(
            f'Job {job_id} is not a refraction statics job'
        )
    if not is_ready_status_value(job.get('status')):
        raise RefractionStaticQcDrilldownError(
            f'Job {job_id} is not complete; current state is '
            f'{normalize_status_value(job.get("status"))}'
        )

    artifacts_dir = _job_artifacts_dir(job, job_id)
    qc = _read_json_artifact(
        artifacts_dir / REFRACTION_STATIC_QC_JSON_NAME,
        REFRACTION_STATIC_QC_JSON_NAME,
    )
    sign_convention = _extract_sign_convention(qc)

    if req.target.kind == 'endpoint':
        return _build_endpoint_drilldown(
            job_id=job_id,
            artifacts_dir=artifacts_dir,
            sign_convention=sign_convention,
            req=req,
        )
    return _build_cell_drilldown(
        job_id=job_id,
        artifacts_dir=artifacts_dir,
        sign_convention=sign_convention,
        req=req,
    )


def _build_endpoint_drilldown(
    *,
    job_id: str,
    artifacts_dir: Path,
    sign_convention: str,
    req: RefractionStaticQcDrilldownRequest,
) -> dict[str, Any]:
    target = req.target
    if target.kind != 'endpoint':
        raise RefractionStaticQcDrilldownError('endpoint drilldown target is required')
    endpoint_kind = target.endpoint_kind
    endpoint_key = target.endpoint_key
    table_artifact = (
        SOURCE_STATIC_TABLE_CSV_NAME
        if endpoint_kind == 'source'
        else RECEIVER_STATIC_TABLE_CSV_NAME
    )
    key_column = f'{endpoint_kind}_endpoint_key'
    endpoint_rows = _read_csv_rows(artifacts_dir / table_artifact, table_artifact)
    endpoint_row = _find_row_by_text(
        endpoint_rows,
        column=key_column,
        value=endpoint_key,
    )
    if endpoint_row is None:
        raise RefractionStaticQcDrilldownNotFound(
            f'Refraction {endpoint_kind} endpoint was not found: {endpoint_key}'
        )

    observation_rows = _matching_observation_rows(
        artifacts_dir=artifacts_dir,
        predicate=lambda row: str(row.get(key_column, '')) == endpoint_key,
    )
    observations = _capped_observations(observation_rows, req.max_observations)
    residual_summary = _residual_summary(observation_rows)
    residual_summary.update(
        {
            'static_table_residual_rms_ms': _float_from_row(
                endpoint_row,
                'residual_rms_ms',
            ),
            'static_table_residual_mad_ms': _float_from_row(
                endpoint_row,
                'residual_mad_ms',
            ),
        }
    )
    static_components = _static_component_fields(endpoint_row)
    time_terms = _time_term_fields(endpoint_row)
    thicknesses = _thickness_fields(endpoint_row)
    velocities = _velocity_fields(endpoint_row)
    pick_counts = _pick_count_fields(endpoint_row)
    statuses = _status_fields(endpoint_row)
    endpoint_payload = {
        'endpoint_kind': endpoint_kind,
        'endpoint_key': endpoint_key,
        'table_artifact': table_artifact,
        'key_column': key_column,
        'row': _normalize_record(endpoint_row),
        'static_components': static_components,
        'time_terms': time_terms,
        'thicknesses': thicknesses,
        'velocities': velocities,
        'pick_counts': pick_counts,
        'statuses': statuses,
        'residual_summary': residual_summary,
    }
    return {
        'job_id': job_id,
        'statics_kind': 'refraction',
        'sign_convention': sign_convention,
        'drilldown_kind': 'endpoint',
        'target': {
            'kind': 'endpoint',
            'endpoint_kind': endpoint_kind,
            'endpoint_key': endpoint_key,
        },
        'max_observations': req.max_observations,
        'artifacts': {
            'endpoint_table': table_artifact,
            'observations': REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
            'qc': REFRACTION_STATIC_QC_JSON_NAME,
        },
        'endpoint': endpoint_payload,
        'static_components': static_components,
        'time_terms': time_terms,
        'thicknesses': thicknesses,
        'velocities': velocities,
        'pick_counts': pick_counts,
        'statuses': statuses,
        'residual_summary': residual_summary,
        'observations': observations,
    }


def _build_cell_drilldown(
    *,
    job_id: str,
    artifacts_dir: Path,
    sign_convention: str,
    req: RefractionStaticQcDrilldownRequest,
) -> dict[str, Any]:
    target = req.target
    if target.kind != 'cell':
        raise RefractionStaticQcDrilldownError('cell drilldown target is required')
    layer_kind = target.layer_kind
    cell_ix = int(target.cell_ix)
    cell_iy = int(target.cell_iy)
    cell_artifact = _CELL_ARTIFACT_BY_LAYER[layer_kind]
    cell_rows = _read_csv_rows(
        artifacts_dir / cell_artifact,
        cell_artifact,
        missing_as_not_found=True,
    )
    cell_row = _find_cell_row(cell_rows, layer_kind=layer_kind, ix=cell_ix, iy=cell_iy)
    if cell_row is None:
        raise RefractionStaticQcDrilldownNotFound(
            f'Refraction {layer_kind} cell was not found: '
            f'cell_ix={cell_ix}, cell_iy={cell_iy}'
        )

    observation_rows = _matching_observation_rows(
        artifacts_dir=artifacts_dir,
        predicate=lambda row: (
            str(row.get('layer_kind', '')) == layer_kind
            and _int_from_row(row, 'cell_ix') == cell_ix
            and _int_from_row(row, 'cell_iy') == cell_iy
        ),
    )
    observations = _capped_observations(observation_rows, req.max_observations)
    residual_summary = _residual_summary(observation_rows)
    residual_summary.update(
        {
            'cell_residual_rms_ms': _float_from_row(cell_row, 'residual_rms_ms'),
            'cell_residual_mad_ms': _float_from_row(cell_row, 'residual_mad_ms'),
            'cell_residual_mean_ms': _float_from_row(cell_row, 'residual_mean_ms'),
            'cell_residual_p95_abs_ms': _float_from_row(
                cell_row,
                'residual_p95_abs_ms',
            ),
        }
    )
    velocity = _cell_velocity_fields(cell_row)
    fold = _cell_fold_fields(cell_row)
    endpoint_counts = {
        'source_count': _int_from_row(cell_row, 'n_sources'),
        'receiver_count': _int_from_row(cell_row, 'n_receivers'),
    }
    neighbor_summary = _neighbor_velocity_summary(
        cell_rows,
        layer_kind=layer_kind,
        ix=cell_ix,
        iy=cell_iy,
    )
    cell_payload = {
        'layer_kind': layer_kind,
        'cell_ix': cell_ix,
        'cell_iy': cell_iy,
        'cell_id': _int_from_row(cell_row, 'cell_id'),
        'cell_artifact': cell_artifact,
        'row': _normalize_record(cell_row),
        'velocity': velocity,
        'fold': fold,
        'endpoint_counts': endpoint_counts,
        'residual_summary': residual_summary,
        'neighbor_velocity_summary': neighbor_summary,
    }
    return {
        'job_id': job_id,
        'statics_kind': 'refraction',
        'sign_convention': sign_convention,
        'drilldown_kind': 'cell',
        'target': {
            'kind': 'cell',
            'layer_kind': layer_kind,
            'cell_ix': cell_ix,
            'cell_iy': cell_iy,
        },
        'max_observations': req.max_observations,
        'artifacts': {
            'cell_table': cell_artifact,
            'observations': REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
            'qc': REFRACTION_STATIC_QC_JSON_NAME,
        },
        'cell': cell_payload,
        'velocity': velocity,
        'fold': fold,
        'endpoint_counts': endpoint_counts,
        'neighbor_velocity_summary': neighbor_summary,
        'residual_summary': residual_summary,
        'observations': observations,
    }


def _job_artifacts_dir(job: dict[str, object], job_id: str) -> Path:
    raw = job.get('artifacts_dir')
    if not isinstance(raw, str) or not raw:
        raise RefractionStaticQcDrilldownError(
            f'Job {job_id} metadata is missing artifacts_dir'
        )
    path = Path(raw)
    if not path.is_dir():
        raise RefractionStaticQcDrilldownError(
            f'Job {job_id} artifacts directory is not available'
        )
    return path


def _read_json_artifact(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RefractionStaticQcDrilldownError(
            f'Refraction QC drilldown requires artifact {artifact_name}'
        )
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RefractionStaticQcDrilldownError(
            f'Refraction artifact {artifact_name} is not valid JSON'
        ) from exc
    if not isinstance(payload, dict):
        raise RefractionStaticQcDrilldownError(
            f'Refraction artifact {artifact_name} must contain a JSON object'
        )
    return payload


def _extract_sign_convention(qc: dict[str, Any]) -> str:
    raw = qc.get('sign_convention')
    if isinstance(raw, str):
        sign_convention = raw
    elif isinstance(raw, dict) and isinstance(raw.get('trace_shift_s'), str):
        sign_convention = raw['trace_shift_s']
    else:
        raise RefractionStaticQcDrilldownError(
            'Refraction QC artifact is missing sign_convention'
        )
    if sign_convention != REFRACTION_STATIC_REPO_SIGN_CONVENTION:
        raise RefractionStaticQcDrilldownError(
            'Refraction QC artifact has unsupported sign_convention: '
            f'{sign_convention!r}'
        )
    return sign_convention


def _read_csv_rows(
    path: Path,
    artifact_name: str,
    *,
    missing_as_not_found: bool = False,
) -> list[dict[str, str | None]]:
    if not path.is_file():
        message = f'Refraction QC drilldown requires artifact {artifact_name}'
        if missing_as_not_found:
            raise RefractionStaticQcDrilldownNotFound(message)
        raise RefractionStaticQcDrilldownError(message)
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _find_row_by_text(
    rows: list[dict[str, str | None]],
    *,
    column: str,
    value: str,
) -> dict[str, str | None] | None:
    for row in rows:
        if str(row.get(column, '')) == value:
            return row
    return None


def _find_cell_row(
    rows: list[dict[str, str | None]],
    *,
    layer_kind: str,
    ix: int,
    iy: int,
) -> dict[str, str | None] | None:
    for row in rows:
        row_layer = str(row.get('cell_velocity_layer_kind') or layer_kind)
        if row_layer != layer_kind:
            continue
        row_ix = _int_from_row(row, 'cell_ix')
        if row_ix is None:
            row_ix = _int_from_row(row, 'ix')
        row_iy = _int_from_row(row, 'cell_iy')
        if row_iy is None:
            row_iy = _int_from_row(row, 'iy')
        if row_ix == ix and row_iy == iy:
            return row
    return None


def _matching_observation_rows(
    *,
    artifacts_dir: Path,
    predicate: Any,
) -> list[dict[str, str | None]]:
    rows = _read_csv_rows(
        artifacts_dir / REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
        REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    )
    matches = [row for row in rows if predicate(row)]
    return sorted(
        matches,
        key=lambda row: (
            _sort_int(row, 'observation_index'),
            _sort_int(row, 'trace_index_sorted'),
            _sort_text(row, 'source_endpoint_key'),
            _sort_text(row, 'receiver_endpoint_key'),
        ),
    )


def _capped_observations(
    rows: list[dict[str, str | None]],
    max_observations: int,
) -> dict[str, Any]:
    capped_rows = rows[:max_observations]
    return {
        'total_count': len(rows),
        'returned_count': len(capped_rows),
        'capped': len(capped_rows) < len(rows),
        'cap_method': _CAP_METHOD,
        'records': [_normalize_record(row) for row in capped_rows],
    }


def _residual_summary(rows: list[dict[str, str | None]]) -> dict[str, Any]:
    residual_ms = [_residual_ms(row) for row in rows]
    residual_ms = [value for value in residual_ms if value is not None]
    used_rows = [row for row in rows if _used_observation(row)]
    used_residual_ms = [_residual_ms(row) for row in used_rows]
    used_residual_ms = [value for value in used_residual_ms if value is not None]
    return {
        'total_count': len(rows),
        'used_count': len(used_rows),
        'rejected_count': max(0, len(rows) - len(used_rows)),
        'all_rms_ms': _rms(residual_ms),
        'all_mad_ms': _mad(residual_ms),
        'all_mean_ms': _mean(residual_ms),
        'all_p95_abs_ms': _p95_abs(residual_ms),
        'used_rms_ms': _rms(used_residual_ms),
        'used_mad_ms': _mad(used_residual_ms),
        'used_mean_ms': _mean(used_residual_ms),
        'used_p95_abs_ms': _p95_abs(used_residual_ms),
    }


def _static_component_fields(row: dict[str, str | None]) -> dict[str, Any]:
    return _normalize_selected(
        row,
        lambda key: (
            key.endswith('_shift_ms')
            or key.endswith('_correction_ms')
            or key in {'total_static_ms', 'total_applied_shift_ms'}
        ),
    )


def _time_term_fields(row: dict[str, str | None]) -> dict[str, Any]:
    out = _normalize_selected(row, lambda key: key in {'t1_ms', 't2_ms', 't3_ms'})
    for key, value in list(out.items()):
        if isinstance(value, (int, float)):
            out[f'{key.removesuffix("_ms")}_s'] = float(value) / 1000.0
    return out


def _thickness_fields(row: dict[str, str | None]) -> dict[str, Any]:
    return _normalize_selected(
        row,
        lambda key: (
            key in {'total_weathering_thickness_m', 'refractor_elevation_m'}
            or key.startswith('sh')
            or key.endswith('_thickness_m')
            or key.endswith('_base_elevation_m')
            or key == 'final_refractor_elevation_m'
        ),
    )


def _velocity_fields(row: dict[str, str | None]) -> dict[str, Any]:
    return _normalize_selected(
        row,
        lambda key: (
            key.endswith('_m_s')
            or key.endswith('_status')
            or key.endswith('_cell_id')
        )
        and (
            key.startswith('v')
            or key.startswith('source_v')
            or key.startswith('receiver_v')
        ),
    )


def _pick_count_fields(row: dict[str, str | None]) -> dict[str, Any]:
    return _normalize_selected(row, lambda key: 'pick_count' in key)


def _status_fields(row: dict[str, str | None]) -> dict[str, Any]:
    return _normalize_selected(row, lambda key: key.endswith('_status'))


def _cell_velocity_fields(row: dict[str, str | None]) -> dict[str, Any]:
    return _normalize_selected(
        row,
        lambda key: key
        in {
            'cell_id',
            'cell_ix',
            'cell_iy',
            'velocity_m_s',
            'v2_m_s',
            'slowness_s_per_m',
            'initial_velocity_m_s',
            'initial_v2_m_s',
            'velocity_update_from_initial_m_s',
            'v2_update_from_initial_m_s',
            'velocity_status',
            'status_reason',
            'cell_velocity_layer_kind',
            'cell_velocity_component',
            'active',
        },
    )


def _cell_fold_fields(row: dict[str, str | None]) -> dict[str, Any]:
    return _normalize_selected(
        row,
        lambda key: key
        in {
            'n_observations',
            'n_used_observations',
            'n_rejected_observations',
            'n_sources',
            'n_receivers',
        },
    )


def _neighbor_velocity_summary(
    rows: list[dict[str, str | None]],
    *,
    layer_kind: str,
    ix: int,
    iy: int,
) -> dict[str, Any]:
    neighbors: list[dict[str, Any]] = []
    neighbor_positions = {(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)}
    for row in rows:
        row_layer = str(row.get('cell_velocity_layer_kind') or layer_kind)
        if row_layer != layer_kind:
            continue
        row_ix = _int_from_row(row, 'cell_ix')
        if row_ix is None:
            row_ix = _int_from_row(row, 'ix')
        row_iy = _int_from_row(row, 'cell_iy')
        if row_iy is None:
            row_iy = _int_from_row(row, 'iy')
        if row_ix is None or row_iy is None or (row_ix, row_iy) not in neighbor_positions:
            continue
        neighbors.append(
            {
                'cell_id': _int_from_row(row, 'cell_id'),
                'cell_ix': row_ix,
                'cell_iy': row_iy,
                'velocity_m_s': _float_from_row(row, 'velocity_m_s'),
                'velocity_status': _text_from_row(row, 'velocity_status'),
                'n_observations': _int_from_row(row, 'n_observations'),
                'residual_rms_ms': _float_from_row(row, 'residual_rms_ms'),
            }
        )
    velocities = [
        item['velocity_m_s']
        for item in neighbors
        if isinstance(item.get('velocity_m_s'), (int, float))
    ]
    return {
        'available': bool(neighbors),
        'neighbor_count': len(neighbors),
        'velocity_min_m_s': min(velocities) if velocities else None,
        'velocity_median_m_s': _median(velocities),
        'velocity_max_m_s': max(velocities) if velocities else None,
        'velocity_mean_m_s': _mean(velocities),
        'neighbors': neighbors,
    }


def _normalize_selected(
    row: dict[str, str | None],
    predicate: Any,
) -> dict[str, Any]:
    return {
        key: _parse_scalar(key, value)
        for key, value in row.items()
        if key is not None and predicate(key)
    }


def _normalize_record(row: dict[str, str | None]) -> dict[str, Any]:
    return {key: _parse_scalar(key, value) for key, value in row.items()}


def _parse_scalar(key: str, value: str | None) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == '':
        return None
    lowered = text.lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    if _is_text_field(key):
        return text
    try:
        if any(char in text for char in ('.', 'e', 'E')):
            number = float(text)
            if not math.isfinite(number):
                return None
            return number
        return int(text)
    except ValueError:
        return text


def _is_text_field(key: str) -> bool:
    if key in _TEXT_FIELDS:
        return True
    return (
        key.endswith('_key')
        or key.endswith('_kind')
        or key.endswith('_mode')
        or key.endswith('_reason')
        or key.endswith('_status')
        or key.endswith('_component')
        or key.endswith('_convention')
    )


def _text_from_row(row: dict[str, str | None], key: str) -> str | None:
    raw = row.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _float_from_row(row: dict[str, str | None], key: str) -> float | None:
    raw = row.get(key)
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _int_from_row(row: dict[str, str | None], key: str) -> int | None:
    raw = row.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    int_value = int(value)
    if float(int_value) != value:
        return None
    return int_value


def _sort_int(row: dict[str, str | None], key: str) -> int:
    value = _int_from_row(row, key)
    if value is None:
        return 2**63 - 1
    return value


def _sort_text(row: dict[str, str | None], key: str) -> str:
    raw = row.get(key)
    return '' if raw is None else str(raw)


def _used_observation(row: dict[str, str | None]) -> bool:
    for key in ('used_in_solve', 'used_for_inversion', 'used'):
        raw = row.get(key)
        if raw is None:
            continue
        lowered = str(raw).strip().lower()
        if lowered in {'true', '1', 'yes'}:
            return True
        if lowered in {'false', '0', 'no'}:
            return False
    return str(row.get('status', '')).strip().lower() == 'ok'


def _residual_ms(row: dict[str, str | None]) -> float | None:
    for key in ('residual_time_ms', 'residual_ms'):
        value = _float_from_row(row, key)
        if value is not None:
            return value
    for key in ('residual_time_s', 'residual_s'):
        value = _float_from_row(row, key)
        if value is not None:
            return value * 1000.0
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _rms(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mad(values: list[float]) -> float | None:
    if not values:
        return None
    median = _median(values)
    if median is None:
        return None
    return _median([abs(value - median) for value in values])


def _p95_abs(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(abs(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


__all__ = [
    'RefractionStaticQcDrilldownError',
    'RefractionStaticQcDrilldownNotFound',
    'build_refraction_static_qc_drilldown',
]
