"""Direct-arrival V1 QC and estimate artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.common.artifact_io import write_csv_atomic, write_json_atomic
from app.statics.refraction.artifacts.io import _assert_strict_json
from seis_statics.refraction.types import RefractionV1EstimateResult

REFRACTION_V1_QC_JSON_NAME = 'refraction_v1_qc.json'
REFRACTION_V1_ESTIMATES_CSV_NAME = 'refraction_v1_estimates.csv'

_CSV_COLUMNS = (
    'group_kind',
    'group_key',
    'n_candidates',
    'n_used',
    'offset_min_m',
    'offset_max_m',
    'slope_s_per_m',
    'v1_m_s',
    'intercept_s',
    'residual_rms_ms',
    'residual_mad_ms',
    'status',
)


def write_refraction_v1_artifacts(
    job_dir: Path,
    result: RefractionV1EstimateResult,
) -> dict[str, Path]:
    """Write direct-arrival V1 QC JSON and per-source estimate CSV artifacts."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / REFRACTION_V1_QC_JSON_NAME
    csv_path = root / REFRACTION_V1_ESTIMATES_CSV_NAME
    _assert_strict_json(result.qc, artifact_name=qc_path.name)
    write_json_atomic(
        qc_path,
        result.qc,
        allow_nan=True,
        ensure_ascii=True,
        sort_keys=True,
    )
    write_csv_atomic(
        csv_path,
        columns=_CSV_COLUMNS,
        rows=_estimate_rows(result),
        extrasaction='raise',
        lineterminator='\r\n',
    )
    return {
        'qc_json': qc_path,
        'estimates_csv': csv_path,
    }


def _estimate_rows(result: RefractionV1EstimateResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, group_key in enumerate(result.group_key.tolist()):
        rows.append(
            {
                'group_kind': result.group_kind,
                'group_key': str(group_key),
                'n_candidates': int(result.group_n_candidates[index]),
                'n_used': int(result.group_n_used[index]),
                'offset_min_m': _csv_float(result.group_offset_min_m[index]),
                'offset_max_m': _csv_float(result.group_offset_max_m[index]),
                'slope_s_per_m': _csv_float(result.group_slope_s_per_m[index]),
                'v1_m_s': _csv_float(result.group_v1_m_s[index]),
                'intercept_s': _csv_float(result.group_intercept_s[index]),
                'residual_rms_ms': _csv_float(
                    result.group_residual_rms_s[index] * 1000.0
                ),
                'residual_mad_ms': _csv_float(
                    result.group_residual_mad_s[index] * 1000.0
                ),
                'status': str(result.group_status[index]),
            }
        )
    return rows


def _csv_float(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return ''
    return f'{number:.10g}'


__all__ = [
    'REFRACTION_V1_ESTIMATES_CSV_NAME',
    'REFRACTION_V1_QC_JSON_NAME',
    'write_refraction_v1_artifacts',
]
