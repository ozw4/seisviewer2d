"""Uphole-time artifact writers for refraction field corrections."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from app.statics.refraction.domain.types import RefractionUpholeResult

REFRACTION_UPHOLE_QC_JSON_NAME = 'refraction_uphole_qc.json'
REFRACTION_UPHOLE_SOURCES_CSV_NAME = 'refraction_uphole_sources.csv'

_UPHOLE_COLUMNS = (
    'source_endpoint_key',
    'source_endpoint_id',
    'source_node_id',
    'uphole_time_s',
    'uphole_status',
    'uphole_pick_count',
    'uphole_trace_count',
)


def write_refraction_uphole_artifacts(
    job_dir: Path,
    result: RefractionUpholeResult,
) -> dict[str, Path]:
    """Write uphole QC JSON and one-row-per-source CSV artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_UPHOLE_QC_JSON_NAME
    source_path = root / REFRACTION_UPHOLE_SOURCES_CSV_NAME
    write_json_atomic(
        qc_path,
        result.qc,
        allow_nan=True,
        ensure_ascii=True,
        sort_keys=True,
    )
    write_csv_atomic(
        source_path,
        columns=_UPHOLE_COLUMNS,
        rows=_uphole_rows(result),
        extrasaction='raise',
        lineterminator='\r\n',
    )
    return {
        'qc_json': qc_path,
        'sources_csv': source_path,
    }


def _uphole_rows(result: RefractionUpholeResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        uphole_time = float(result.uphole_time_s[index])
        rows.append(
            {
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_endpoint_id': int(result.source_endpoint_id[index]),
                'source_node_id': int(result.source_node_id[index]),
                'uphole_time_s': _csv_float(uphole_time),
                'uphole_status': str(result.uphole_status[index]),
                'uphole_pick_count': int(result.uphole_pick_count[index]),
                'uphole_trace_count': int(result.uphole_trace_count[index]),
            }
        )
    return rows


def _csv_float(value: float) -> str:
    if not np.isfinite(value):
        return ''
    return f'{float(value):.17g}'


__all__ = [
    'REFRACTION_UPHOLE_QC_JSON_NAME',
    'REFRACTION_UPHOLE_SOURCES_CSV_NAME',
    'write_refraction_uphole_artifacts',
]
