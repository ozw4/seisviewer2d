"""Source-depth artifact writers for refraction field corrections."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from app.statics.refraction.domain.types import RefractionSourceDepthResult

REFRACTION_SOURCE_DEPTH_QC_JSON_NAME = 'refraction_source_depth_qc.json'
REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME = 'refraction_source_depth_sources.csv'

_SOURCE_DEPTH_COLUMNS = (
    'source_endpoint_key',
    'source_endpoint_id',
    'source_node_id',
    'source_depth_m',
    'source_depth_status',
    'source_depth_pick_count',
    'source_depth_trace_count',
)


def write_refraction_source_depth_artifacts(
    job_dir: Path,
    result: RefractionSourceDepthResult,
) -> dict[str, Path]:
    """Write source-depth QC JSON and one-row-per-source CSV artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_SOURCE_DEPTH_QC_JSON_NAME
    source_path = root / REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME
    write_json_atomic(
        qc_path,
        result.qc,
        allow_nan=True,
        ensure_ascii=True,
        sort_keys=True,
    )
    write_csv_atomic(
        source_path,
        columns=_SOURCE_DEPTH_COLUMNS,
        rows=_source_depth_rows(result),
        extrasaction='raise',
        lineterminator='\r\n',
    )
    return {
        'qc_json': qc_path,
        'sources_csv': source_path,
    }


def _source_depth_rows(
    result: RefractionSourceDepthResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        depth = float(result.source_depth_m[index])
        rows.append(
            {
                'source_endpoint_key': str(result.source_endpoint_key[index]),
                'source_endpoint_id': int(result.source_endpoint_id[index]),
                'source_node_id': int(result.source_node_id[index]),
                'source_depth_m': _csv_float(depth),
                'source_depth_status': str(result.source_depth_status[index]),
                'source_depth_pick_count': int(result.source_depth_pick_count[index]),
                'source_depth_trace_count': int(result.source_depth_trace_count[index]),
            }
        )
    return rows


def _csv_float(value: float) -> str:
    if not np.isfinite(value):
        return ''
    return f'{float(value):.17g}'


__all__ = [
    'REFRACTION_SOURCE_DEPTH_QC_JSON_NAME',
    'REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME',
    'write_refraction_source_depth_artifacts',
]
