"""Server-side endpoint search for completed refraction QC jobs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from app.statics.refraction.contracts.qc import (
    RefractionStaticQcEndpointSearchRequest,
)
from app.statics.refraction.application.job_status import (
    is_ready_status_value,
    normalize_status_value,
)
from app.statics.refraction.artifacts import (
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
)


class RefractionStaticQcEndpointSearchError(ValueError):
    """Raised when endpoint search artifacts cannot be assembled."""


def build_refraction_static_qc_endpoint_search(
    *,
    job_id: str,
    job: dict[str, object],
    req: RefractionStaticQcEndpointSearchRequest,
) -> dict[str, Any]:
    """Search endpoint metadata from completed refraction static artifacts."""
    if job.get('statics_kind') != 'refraction':
        raise RefractionStaticQcEndpointSearchError(
            f'Job {job_id} is not a refraction statics job'
        )
    if not is_ready_status_value(job.get('status')):
        raise RefractionStaticQcEndpointSearchError(
            f'Job {job_id} is not complete; current state is '
            f'{normalize_status_value(job.get("status"))}'
        )

    artifacts_dir = _job_artifacts_dir(job, job_id)
    component_rows = _read_csv_rows(
        artifacts_dir / REFRACTION_STATIC_COMPONENTS_CSV_NAME,
        REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    )
    qc_by_key = _optional_endpoint_qc_rows(artifacts_dir)

    records = []
    for row in component_rows:
        record = _endpoint_record(row, qc_by_key=qc_by_key)
        if record is None:
            continue
        if req.endpoint_kind != 'both' and record['endpoint_kind'] != req.endpoint_kind:
            continue
        if not _matches_status(record, req.status_filter):
            continue
        if not _matches_query(record, req.query):
            continue
        records.append(record)

    records = _sort_records(records, req.sort)
    total = len(records)
    paged = records[req.offset : req.offset + req.limit]

    return {
        'job_id': job_id,
        'statics_kind': 'refraction',
        'endpoint_kind': req.endpoint_kind,
        'query': req.query,
        'total': total,
        'limit': req.limit,
        'offset': req.offset,
        'records': paged,
    }


def _job_artifacts_dir(job: dict[str, object], job_id: str) -> Path:
    raw = job.get('artifacts_dir')
    if not isinstance(raw, str) or not raw:
        raise RefractionStaticQcEndpointSearchError(
            f'Job {job_id} metadata is missing artifacts_dir'
        )
    path = Path(raw)
    if not path.is_dir():
        raise RefractionStaticQcEndpointSearchError(
            f'Job {job_id} artifacts directory is not available'
        )
    return path


def _read_csv_rows(path: Path, artifact_name: str) -> list[dict[str, str | None]]:
    if not path.is_file():
        raise RefractionStaticQcEndpointSearchError(
            f'Refraction QC endpoint search requires artifact {artifact_name}'
        )
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _optional_endpoint_qc_rows(
    artifacts_dir: Path,
) -> dict[str, dict[str, str | None]]:
    path = artifacts_dir / REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME
    if not path.is_file():
        return {}
    rows = _read_csv_rows(path, REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME)
    out: dict[str, dict[str, str | None]] = {}
    for row in rows:
        endpoint_key = _text_from_row(row, 'endpoint_key')
        if endpoint_key is not None:
            out[endpoint_key] = row
    return out


def _endpoint_record(
    row: dict[str, str | None],
    *,
    qc_by_key: dict[str, dict[str, str | None]],
) -> dict[str, Any] | None:
    endpoint_kind = _text_from_row(row, 'kind') or _text_from_row(row, 'endpoint_kind')
    if endpoint_kind not in {'source', 'receiver'}:
        return None
    endpoint_key = _text_from_row(row, 'endpoint_key')
    if endpoint_key is None:
        raise RefractionStaticQcEndpointSearchError(
            f'Refraction endpoint row is missing endpoint_key in '
            f'{REFRACTION_STATIC_COMPONENTS_CSV_NAME}'
        )

    merged = dict(row)
    qc_row = qc_by_key.get(endpoint_key)
    if qc_row is not None:
        for key, value in qc_row.items():
            if key in {'endpoint_kind', 'endpoint_key'}:
                continue
            if _non_empty(value) and not _non_empty(merged.get(key)):
                merged[key] = value

    record = {
        'endpoint_kind': endpoint_kind,
        'endpoint_key': endpoint_key,
        'station_id': _int_from_row(merged, 'station_id'),
        'node_id': _int_from_row(merged, 'node_id'),
        'x_m': _float_from_row(merged, 'x_m'),
        'y_m': _float_from_row(merged, 'y_m'),
        'surface_elevation_m': _float_from_row(merged, 'surface_elevation_m'),
        'pick_count': _int_from_row(merged, 'pick_count'),
        'residual_rms_ms': _float_from_row(merged, 'residual_rms_ms'),
        'datum_status': _text_from_row(merged, 'datum_status'),
        'static_status': _text_from_row(merged, 'static_status'),
    }
    record['label'] = _label(record)
    return record


def _label(record: dict[str, Any]) -> str:
    pieces = [str(record['endpoint_key'])]
    station_id = record.get('station_id')
    if station_id is not None:
        pieces.append(f'station {station_id}')
    node_id = record.get('node_id')
    if node_id is not None:
        pieces.append(f'node {node_id}')
    pick_count = record.get('pick_count')
    if pick_count is not None:
        pieces.append(f'picks {pick_count}')
    residual_rms_ms = record.get('residual_rms_ms')
    if residual_rms_ms is not None:
        pieces.append(f'RMS {_format_number(float(residual_rms_ms))} ms')
    pieces.append(_record_status(record))
    return ' - '.join(pieces)


def _format_number(value: float) -> str:
    text = f'{value:.3f}'.rstrip('0').rstrip('.')
    return text or '0'


def _matches_query(record: dict[str, Any], query: str) -> bool:
    needle = query.strip().lower()
    if not needle:
        return True
    for key in ('endpoint_key', 'station_id', 'node_id', 'label', 'x_m', 'y_m'):
        value = record.get(key)
        if value is None:
            continue
        if needle in str(value).lower():
            return True
    return False


def _matches_status(record: dict[str, Any], status_filter: str) -> bool:
    if status_filter == 'all':
        return True
    is_ok = _record_status(record) == 'ok'
    if status_filter == 'ok':
        return is_ok
    return not is_ok


def _record_status(record: dict[str, Any]) -> str:
    static_status = _clean_status(record.get('static_status'))
    if static_status is not None:
        return static_status
    datum_status = _clean_status(record.get('datum_status'))
    if datum_status is not None:
        return datum_status
    return 'unknown'


def _clean_status(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _sort_records(
    records: list[dict[str, Any]],
    sort: str,
) -> list[dict[str, Any]]:
    if sort == 'station_id_asc':
        return sorted(records, key=lambda item: _numeric_key(item, 'station_id'))
    if sort == 'station_id_desc':
        return sorted(records, key=lambda item: _numeric_key(item, 'station_id', True))
    if sort == 'residual_rms_desc':
        return sorted(
            records,
            key=lambda item: _numeric_key(item, 'residual_rms_ms', True),
        )
    if sort == 'residual_rms_asc':
        return sorted(records, key=lambda item: _numeric_key(item, 'residual_rms_ms'))
    if sort == 'pick_count_desc':
        return sorted(records, key=lambda item: _numeric_key(item, 'pick_count', True))
    if sort == 'pick_count_asc':
        return sorted(records, key=lambda item: _numeric_key(item, 'pick_count'))
    return sorted(
        records,
        key=lambda item: (str(item['endpoint_key']), str(item['endpoint_kind'])),
    )


def _numeric_key(
    record: dict[str, Any],
    field: str,
    desc: bool = False,
) -> tuple[bool, float, str, str]:
    raw = record.get(field)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        sort_value = -value if desc else value
        return (False, sort_value, str(record['endpoint_key']), str(record['endpoint_kind']))
    return (True, 0.0, str(record['endpoint_key']), str(record['endpoint_kind']))


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


def _non_empty(value: object) -> bool:
    return value is not None and str(value).strip() != ''


__all__ = [
    'RefractionStaticQcEndpointSearchError',
    'build_refraction_static_qc_endpoint_search',
]
