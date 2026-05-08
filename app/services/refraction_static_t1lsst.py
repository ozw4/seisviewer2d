"""T1LSST-compatible one-layer conversion helpers for refraction statics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.services.refraction_static_types import RefractionDatumStaticsResult

REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME = (
    'refraction_t1lsst_1layer_components.csv'
)
T1LSST_SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'

_T1LSST_COMPONENT_COLUMNS = (
    'endpoint_kind',
    'endpoint_key',
    'node_id',
    'x_m',
    'y_m',
    'surface_elevation_m',
    'floating_datum_elevation_m',
    'flat_datum_elevation_m',
    't1_ms',
    'v1_m_s',
    'v2_m_s',
    'sh1_weathering_thickness_m',
    'refractor_elevation_m',
    'weathering_correction_ms',
    'floating_datum_correction_ms',
    'flat_datum_correction_ms',
    'elevation_correction_ms',
    'total_static_ms',
    'total_applied_shift_ms',
    'solution_status',
    'weathering_status',
    'datum_status',
    'static_status',
    'sign_convention',
)


class RefractionT1LSSTError(ValueError):
    """Raised when T1LSST one-layer values cannot be computed or written."""


def compute_t1lsst_1layer_thickness(
    t1_s: np.ndarray,
    v1_m_s: float,
    v2_m_s: float,
) -> np.ndarray:
    """Compute one-layer ``SH1`` weathering thickness from T1, V1, and V2."""
    t1 = _coerce_float_array(t1_s, name='t1_s', allow_nonfinite=True)
    v1 = _positive_finite(v1_m_s, name='v1_m_s')
    v2 = _positive_finite(v2_m_s, name='v2_m_s')
    if v2 <= v1:
        raise RefractionT1LSSTError('v2_m_s must be greater than v1_m_s')
    denom = np.sqrt(v2 * v2 - v1 * v1)
    return np.ascontiguousarray(t1 * v2 * v1 / denom, dtype=np.float64)


def compute_t1lsst_1layer_weathering_correction(
    sh1_m: np.ndarray,
    v1_m_s: float,
    v2_m_s: float,
) -> np.ndarray:
    """Compute one-layer ``WCOR`` from SH1, V1, and V2 in seconds."""
    sh1 = _coerce_float_array(sh1_m, name='sh1_m', allow_nonfinite=True)
    v1 = _positive_finite(v1_m_s, name='v1_m_s')
    v2 = _positive_finite(v2_m_s, name='v2_m_s')
    if v2 <= v1:
        raise RefractionT1LSSTError('v2_m_s must be greater than v1_m_s')
    return np.ascontiguousarray(sh1 * (1.0 / v2 - 1.0 / v1), dtype=np.float64)


def compose_t1lsst_1layer_static_table_components(
    result: RefractionDatumStaticsResult,
) -> list[dict[str, object]]:
    """Return source/receiver rows using IRAS-style one-layer component names."""
    if not isinstance(result, RefractionDatumStaticsResult):
        raise RefractionT1LSSTError(
            'result must be a RefractionDatumStaticsResult instance'
        )
    v1 = _positive_finite(result.weathering_velocity_m_s, name='result.weathering_velocity_m_s')
    v2 = _positive_finite(result.bedrock_velocity_m_s, name='result.bedrock_velocity_m_s')
    if v2 <= v1:
        raise RefractionT1LSSTError(
            'result.bedrock_velocity_m_s must be greater than result.weathering_velocity_m_s'
        )

    solution_status = _node_lookup(result.node_id, result.node_solution_status)
    weathering_status = _node_lookup(result.node_id, result.node_weathering_status)
    flat_datum = _nan_if_none(result.flat_datum_elevation_m)

    rows: list[dict[str, object]] = []
    for index in range(int(result.source_endpoint_key.shape[0])):
        node_id = int(result.source_node_id[index])
        rows.append(
            _endpoint_row(
                endpoint_kind='source',
                endpoint_key=result.source_endpoint_key[index],
                node_id=node_id,
                x_m=result.source_x_m[index],
                y_m=result.source_y_m[index],
                surface_elevation_m=result.source_surface_elevation_m[index],
                floating_datum_elevation_m=(
                    result.source_floating_datum_elevation_m[index]
                ),
                flat_datum_elevation_m=flat_datum,
                t1_s=result.source_half_intercept_time_s[index],
                v1_m_s=v1,
                v2_m_s=v2,
                sh1_m=result.source_weathering_thickness_m[index],
                refractor_elevation_m=result.source_refractor_elevation_m[index],
                weathering_correction_s=(
                    result.source_weathering_replacement_shift_s[index]
                ),
                floating_datum_correction_s=(
                    result.source_floating_datum_elevation_shift_s[index]
                ),
                flat_datum_correction_s=result.source_flat_datum_shift_s[index],
                total_static_s=result.source_refraction_shift_s[index],
                solution_status=solution_status.get(node_id, 'missing_node'),
                weathering_status=weathering_status.get(node_id, 'missing_node'),
                datum_status=result.source_datum_status[index],
                static_status=result.source_datum_status[index],
            )
        )

    for index in range(int(result.receiver_endpoint_key.shape[0])):
        node_id = int(result.receiver_node_id[index])
        rows.append(
            _endpoint_row(
                endpoint_kind='receiver',
                endpoint_key=result.receiver_endpoint_key[index],
                node_id=node_id,
                x_m=result.receiver_x_m[index],
                y_m=result.receiver_y_m[index],
                surface_elevation_m=result.receiver_surface_elevation_m[index],
                floating_datum_elevation_m=(
                    result.receiver_floating_datum_elevation_m[index]
                ),
                flat_datum_elevation_m=flat_datum,
                t1_s=result.receiver_half_intercept_time_s[index],
                v1_m_s=v1,
                v2_m_s=v2,
                sh1_m=result.receiver_weathering_thickness_m[index],
                refractor_elevation_m=result.receiver_refractor_elevation_m[index],
                weathering_correction_s=(
                    result.receiver_weathering_replacement_shift_s[index]
                ),
                floating_datum_correction_s=(
                    result.receiver_floating_datum_elevation_shift_s[index]
                ),
                flat_datum_correction_s=result.receiver_flat_datum_shift_s[index],
                total_static_s=result.receiver_refraction_shift_s[index],
                solution_status=solution_status.get(node_id, 'missing_node'),
                weathering_status=weathering_status.get(node_id, 'missing_node'),
                datum_status=result.receiver_datum_status[index],
                static_status=result.receiver_datum_status[index],
            )
        )
    return rows


def write_refraction_t1lsst_1layer_components_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    """Write the T1LSST-compatible one-layer component CSV artifact."""
    rows = compose_t1lsst_1layer_static_table_components(result)
    _write_csv_atomic(Path(path), rows, _T1LSST_COMPONENT_COLUMNS)


def _endpoint_row(
    *,
    endpoint_kind: str,
    endpoint_key: object,
    node_id: int,
    x_m: object,
    y_m: object,
    surface_elevation_m: object,
    floating_datum_elevation_m: object,
    flat_datum_elevation_m: object,
    t1_s: object,
    v1_m_s: float,
    v2_m_s: float,
    sh1_m: object,
    refractor_elevation_m: object,
    weathering_correction_s: object,
    floating_datum_correction_s: object,
    flat_datum_correction_s: object,
    total_static_s: object,
    solution_status: object,
    weathering_status: object,
    datum_status: object,
    static_status: object,
) -> dict[str, object]:
    elevation_correction_s = _sum_correction_s(
        floating_datum_correction_s,
        flat_datum_correction_s,
    )
    return {
        'endpoint_kind': endpoint_kind,
        'endpoint_key': str(endpoint_key),
        'node_id': int(node_id),
        'x_m': _csv_float(x_m),
        'y_m': _csv_float(y_m),
        'surface_elevation_m': _csv_float(surface_elevation_m),
        'floating_datum_elevation_m': _csv_float(floating_datum_elevation_m),
        'flat_datum_elevation_m': _csv_float(flat_datum_elevation_m),
        't1_ms': _csv_ms(t1_s),
        'v1_m_s': _csv_float(v1_m_s),
        'v2_m_s': _csv_float(v2_m_s),
        'sh1_weathering_thickness_m': _csv_float(sh1_m),
        'refractor_elevation_m': _csv_float(refractor_elevation_m),
        'weathering_correction_ms': _csv_ms(weathering_correction_s),
        'floating_datum_correction_ms': _csv_ms(floating_datum_correction_s),
        'flat_datum_correction_ms': _csv_ms(flat_datum_correction_s),
        'elevation_correction_ms': _csv_ms(elevation_correction_s),
        'total_static_ms': _csv_ms(total_static_s),
        'total_applied_shift_ms': _csv_ms(total_static_s),
        'solution_status': str(solution_status),
        'weathering_status': str(weathering_status),
        'datum_status': str(datum_status),
        'static_status': str(static_status),
        'sign_convention': T1LSST_SIGN_CONVENTION,
    }


def _sum_correction_s(left: object, right: object) -> float:
    left_value = _as_float_or_nan(left)
    right_value = _as_float_or_nan(right)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float('nan')
    return float(left_value + right_value)


def _node_lookup(node_id: np.ndarray, values: np.ndarray) -> dict[int, Any]:
    return {
        int(raw_node): values[index]
        for index, raw_node in enumerate(np.asarray(node_id).tolist())
    }


def _coerce_float_array(
    value: object,
    *,
    name: str,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RefractionT1LSSTError(f'{name} must be numeric') from exc
    if array.dtype == object:
        raise RefractionT1LSSTError(f'{name} must not have object dtype')
    if not allow_nonfinite and not np.all(np.isfinite(array)):
        raise RefractionT1LSSTError(f'{name} must contain finite values')
    return np.ascontiguousarray(array, dtype=np.float64)


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise RefractionT1LSSTError(f'{name} must be finite and positive')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionT1LSSTError(f'{name} must be finite and positive') from exc
    if not np.isfinite(out) or out <= 0.0:
        raise RefractionT1LSSTError(f'{name} must be finite and positive')
    return out


def _as_float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(out):
        return float('nan')
    return out


def _nan_if_none(value: object) -> float:
    return float('nan') if value is None else float(value)


def _csv_float(value: object) -> str | float:
    out = _as_float_or_nan(value)
    if not np.isfinite(out):
        return ''
    return float(out)


def _csv_ms(value_s: object) -> str | float:
    out = _as_float_or_nan(value_s)
    if not np.isfinite(out):
        return ''
    return float(out * 1000.0)


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


__all__ = [
    'REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME',
    'T1LSST_SIGN_CONVENTION',
    'RefractionT1LSSTError',
    'compute_t1lsst_1layer_thickness',
    'compute_t1lsst_1layer_weathering_correction',
    'compose_t1lsst_1layer_static_table_components',
    'write_refraction_t1lsst_1layer_components_csv',
]
