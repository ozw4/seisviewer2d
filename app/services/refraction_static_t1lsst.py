"""T1LSST-compatible conversion helpers for refraction statics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.services.refraction_static_status import (
    classify_refraction_endpoint_static_status,
)
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
    """Raised when T1LSST values cannot be computed or written."""


@dataclass(frozen=True)
class RefractionT1LSST2LayerThicknessResult:
    """Two-layer T1LSST thicknesses with endpoint conversion status."""

    sh1_m: np.ndarray
    sh2_m: np.ndarray
    status: np.ndarray
    weathering_correction_s: np.ndarray | None = None


@dataclass(frozen=True)
class RefractionT1LSST3LayerThicknessResult:
    """Three-layer T1LSST thicknesses with endpoint conversion status."""

    sh1_m: np.ndarray
    sh2_m: np.ndarray
    sh3_m: np.ndarray
    status: np.ndarray
    weathering_correction_s: np.ndarray | None = None


def compute_t1lsst_1layer_thickness(
    t1_s: np.ndarray,
    v1_m_s: float,
    v2_m_s: float | np.ndarray,
) -> np.ndarray:
    """Compute one-layer ``SH1`` weathering thickness from T1, V1, and V2."""
    t1 = _coerce_float_array(t1_s, name='t1_s', allow_nonfinite=True)
    v1 = _positive_finite(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    if np.any(v2 <= v1):
        raise RefractionT1LSSTError('v2_m_s must be greater than v1_m_s')
    denom = np.sqrt(v2 * v2 - v1 * v1)
    return np.ascontiguousarray(t1 * v2 * v1 / denom, dtype=np.float64)


def compute_t1lsst_1layer_weathering_correction(
    sh1_m: np.ndarray,
    v1_m_s: float,
    v2_m_s: float | np.ndarray,
) -> np.ndarray:
    """Compute one-layer ``WCOR`` from SH1, V1, and V2 in seconds."""
    sh1 = _coerce_float_array(sh1_m, name='sh1_m', allow_nonfinite=True)
    v1 = _positive_finite(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    if np.any(v2 <= v1):
        raise RefractionT1LSSTError('v2_m_s must be greater than v1_m_s')
    return np.ascontiguousarray(sh1 * (1.0 / v2 - 1.0 / v1), dtype=np.float64)


def compute_t1lsst_2layer_thicknesses(
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two-layer ``SH1`` and ``SH2`` thicknesses from T1/T2 terms.

    Negative thicknesses are status-coded by
    :func:`compute_t1lsst_2layer_thicknesses_with_status`; this convenience
    wrapper returns the thickness arrays with invalid components set to NaN.
    """
    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        strict_velocity_order=True,
    )
    return result.sh1_m, result.sh2_m


def compute_t1lsst_2layer_thicknesses_with_status(
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    *,
    strict_velocity_order: bool = False,
) -> RefractionT1LSST2LayerThicknessResult:
    """Compute two-layer ``SH1``/``SH2`` and status-code invalid endpoints."""
    if strict_velocity_order:
        t1, t2, v1, v2, v3 = _coerce_2layer_inputs(
            t1_s=t1_s,
            t2_s=t2_s,
            v1_m_s=v1_m_s,
            v2_m_s=v2_m_s,
            v3_m_s=v3_m_s,
        )
        invalid_nonfinite = np.zeros(t1.shape, dtype=bool)
        invalid_velocity_order = np.zeros(t1.shape, dtype=bool)
    else:
        t1, t2, v1, v2, v3 = _coerce_2layer_status_inputs(
            t1_s=t1_s,
            t2_s=t2_s,
            v1_m_s=v1_m_s,
            v2_m_s=v2_m_s,
            v3_m_s=v3_m_s,
        )
        invalid_nonfinite = ~(
            np.isfinite(t1)
            & np.isfinite(t2)
            & np.isfinite(v1)
            & np.isfinite(v2)
            & np.isfinite(v3)
        )
        invalid_velocity_order = (
            ~invalid_nonfinite
            & ((v1 <= 0.0) | (v2 <= v1) | (v3 <= v2))
        )

    status = np.full(t1.shape, 'ok', dtype='<U32')
    status[invalid_nonfinite] = 'invalid_nonfinite_input'
    status[invalid_velocity_order] = 'invalid_velocity_order'
    valid = status == 'ok'
    sh1 = np.full(t1.shape, np.nan, dtype=np.float64)
    sh2 = np.full(t1.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        valid_t1 = t1[valid]
        valid_t2 = t2[valid]
        valid_v1 = v1[valid]
        valid_v2 = v2[valid]
        valid_v3 = v3[valid]
        scos12 = np.sqrt(1.0 - (valid_v1 / valid_v2) ** 2)
        valid_sh1 = valid_t1 * valid_v1 / scos12
        scos13 = np.sqrt(1.0 - (valid_v1 / valid_v3) ** 2)
        scos23 = np.sqrt(1.0 - (valid_v2 / valid_v3) ** 2)
        valid_sh2 = (
            (valid_t2 - valid_sh1 * scos13 / valid_v1)
            * valid_v2
            / scos23
        )
        sh1[valid] = valid_sh1
        sh2[valid] = valid_sh2

    negative_sh1 = np.isfinite(sh1) & (sh1 < 0.0)
    negative_sh2 = np.isfinite(sh2) & (sh2 < 0.0)
    invalid_negative = negative_sh1 | negative_sh2
    status[invalid_negative] = 'invalid_negative_thickness'
    sh1[negative_sh1] = np.nan
    sh2[negative_sh1 | negative_sh2] = np.nan
    wcor = np.full(t1.shape, np.nan, dtype=np.float64)
    wcor_valid = status == 'ok'
    if np.any(wcor_valid):
        wcor[wcor_valid] = (
            sh1[wcor_valid] * (1.0 / v3[wcor_valid] - 1.0 / v1[wcor_valid])
            + sh2[wcor_valid] * (1.0 / v3[wcor_valid] - 1.0 / v2[wcor_valid])
        )
    return RefractionT1LSST2LayerThicknessResult(
        sh1_m=np.array(sh1, dtype=np.float64, copy=True, order='C'),
        sh2_m=np.array(sh2, dtype=np.float64, copy=True, order='C'),
        status=np.array(status, dtype='<U32', copy=True, order='C'),
        weathering_correction_s=np.array(wcor, dtype=np.float64, copy=True, order='C'),
    )


def compute_t1lsst_2layer_weathering_correction(
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
) -> np.ndarray:
    """Compute two-layer replacement ``WCOR`` to V3 in seconds."""
    sh1 = _coerce_float_array(sh1_m, name='sh1_m', allow_nonfinite=True)
    sh2 = _coerce_float_array(sh2_m, name='sh2_m', allow_nonfinite=True)
    v1 = _positive_finite_float_array(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    v3 = _positive_finite_float_array(v3_m_s, name='v3_m_s')
    sh1, sh2, v1, v2, v3 = _broadcast_t1lsst_arrays(
        (sh1, sh2, v1, v2, v3),
        names=('sh1_m', 'sh2_m', 'v1_m_s', 'v2_m_s', 'v3_m_s'),
    )
    _validate_2layer_velocity_order(v1=v1, v2=v2, v3=v3)
    return np.array(
        sh1 * (1.0 / v3 - 1.0 / v1)
        + sh2 * (1.0 / v3 - 1.0 / v2),
        dtype=np.float64,
        copy=True,
        order='C',
    )


def compute_t1lsst_3layer_thicknesses(
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    t3_s: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    vsub_m_s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute three-layer ``SH1``/``SH2``/``SH3`` thicknesses."""
    result = compute_t1lsst_3layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s,
        t3_s=t3_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
        strict_velocity_order=True,
    )
    return result.sh1_m, result.sh2_m, result.sh3_m


def compute_t1lsst_3layer_thicknesses_with_status(
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    t3_s: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    vsub_m_s: np.ndarray | float,
    *,
    strict_velocity_order: bool = False,
) -> RefractionT1LSST3LayerThicknessResult:
    """Compute three-layer thicknesses and status-code invalid endpoints."""
    if strict_velocity_order:
        t1, t2, t3, v1, v2, v3, vsub = _coerce_3layer_inputs(
            t1_s=t1_s,
            t2_s=t2_s,
            t3_s=t3_s,
            v1_m_s=v1_m_s,
            v2_m_s=v2_m_s,
            v3_m_s=v3_m_s,
            vsub_m_s=vsub_m_s,
        )
        invalid_nonfinite = np.zeros(t1.shape, dtype=bool)
        invalid_velocity_order = np.zeros(t1.shape, dtype=bool)
    else:
        t1, t2, t3, v1, v2, v3, vsub = _coerce_3layer_status_inputs(
            t1_s=t1_s,
            t2_s=t2_s,
            t3_s=t3_s,
            v1_m_s=v1_m_s,
            v2_m_s=v2_m_s,
            v3_m_s=v3_m_s,
            vsub_m_s=vsub_m_s,
        )
        invalid_nonfinite = ~(
            np.isfinite(t1)
            & np.isfinite(t2)
            & np.isfinite(t3)
            & np.isfinite(v1)
            & np.isfinite(v2)
            & np.isfinite(v3)
            & np.isfinite(vsub)
        )
        invalid_velocity_order = (
            ~invalid_nonfinite
            & ((v1 <= 0.0) | (v2 <= v1) | (v3 <= v2) | (vsub <= v3))
        )

    status = np.full(t1.shape, 'ok', dtype='<U32')
    status[invalid_nonfinite] = 'invalid_nonfinite_input'
    status[invalid_velocity_order] = 'invalid_velocity_order'
    valid = status == 'ok'
    sh1 = np.full(t1.shape, np.nan, dtype=np.float64)
    sh2 = np.full(t1.shape, np.nan, dtype=np.float64)
    sh3 = np.full(t1.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        valid_t1 = t1[valid]
        valid_t2 = t2[valid]
        valid_t3 = t3[valid]
        valid_v1 = v1[valid]
        valid_v2 = v2[valid]
        valid_v3 = v3[valid]
        valid_vsub = vsub[valid]
        c12 = np.sqrt(1.0 - (valid_v1 / valid_v2) ** 2)
        c13 = np.sqrt(1.0 - (valid_v1 / valid_v3) ** 2)
        c23 = np.sqrt(1.0 - (valid_v2 / valid_v3) ** 2)
        c1sub = np.sqrt(1.0 - (valid_v1 / valid_vsub) ** 2)
        c2sub = np.sqrt(1.0 - (valid_v2 / valid_vsub) ** 2)
        c3sub = np.sqrt(1.0 - (valid_v3 / valid_vsub) ** 2)
        valid_sh1 = valid_t1 * valid_v1 / c12
        valid_sh2 = (
            (valid_t2 - valid_sh1 * c13 / valid_v1)
            * valid_v2
            / c23
        )
        valid_sh3 = (
            valid_t3
            - valid_sh1 * c1sub / valid_v1
            - valid_sh2 * c2sub / valid_v2
        ) * valid_v3 / c3sub
        sh1[valid] = valid_sh1
        sh2[valid] = valid_sh2
        sh3[valid] = valid_sh3

    invalid_negative = (
        (np.isfinite(sh1) & (sh1 < 0.0))
        | (np.isfinite(sh2) & (sh2 < 0.0))
        | (np.isfinite(sh3) & (sh3 < 0.0))
    )
    status[invalid_negative] = 'invalid_negative_thickness'
    sh1[invalid_negative] = np.nan
    sh2[invalid_negative] = np.nan
    sh3[invalid_negative] = np.nan
    wcor = np.full(t1.shape, np.nan, dtype=np.float64)
    wcor_valid = status == 'ok'
    if np.any(wcor_valid):
        wcor[wcor_valid] = (
            sh1[wcor_valid] * (1.0 / vsub[wcor_valid] - 1.0 / v1[wcor_valid])
            + sh2[wcor_valid] * (1.0 / vsub[wcor_valid] - 1.0 / v2[wcor_valid])
            + sh3[wcor_valid] * (1.0 / vsub[wcor_valid] - 1.0 / v3[wcor_valid])
        )
    return RefractionT1LSST3LayerThicknessResult(
        sh1_m=np.array(sh1, dtype=np.float64, copy=True, order='C'),
        sh2_m=np.array(sh2, dtype=np.float64, copy=True, order='C'),
        sh3_m=np.array(sh3, dtype=np.float64, copy=True, order='C'),
        status=np.array(status, dtype='<U32', copy=True, order='C'),
        weathering_correction_s=np.array(wcor, dtype=np.float64, copy=True, order='C'),
    )


def compute_t1lsst_3layer_weathering_correction(
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    sh3_m: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    vsub_m_s: np.ndarray | float,
) -> np.ndarray:
    """Compute three-layer replacement ``WCOR`` to Vsub in seconds."""
    sh1 = _coerce_float_array(sh1_m, name='sh1_m', allow_nonfinite=True)
    sh2 = _coerce_float_array(sh2_m, name='sh2_m', allow_nonfinite=True)
    sh3 = _coerce_float_array(sh3_m, name='sh3_m', allow_nonfinite=True)
    v1 = _positive_finite_float_array(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    v3 = _positive_finite_float_array(v3_m_s, name='v3_m_s')
    vsub = _positive_finite_float_array(vsub_m_s, name='vsub_m_s')
    sh1, sh2, sh3, v1, v2, v3, vsub = _broadcast_t1lsst_arrays(
        (sh1, sh2, sh3, v1, v2, v3, vsub),
        names=(
            'sh1_m',
            'sh2_m',
            'sh3_m',
            'v1_m_s',
            'v2_m_s',
            'v3_m_s',
            'vsub_m_s',
        ),
    )
    _validate_3layer_velocity_order(v1=v1, v2=v2, v3=v3, vsub=vsub)
    return np.array(
        sh1 * (1.0 / vsub - 1.0 / v1)
        + sh2 * (1.0 / vsub - 1.0 / v2)
        + sh3 * (1.0 / vsub - 1.0 / v3),
        dtype=np.float64,
        copy=True,
        order='C',
    )


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
    source_v2 = _endpoint_v2_m_s(
        result.source_v2_m_s,
        shape=int(result.source_endpoint_key.shape[0]),
        scalar_v2_m_s=v2,
    )
    receiver_v2 = _endpoint_v2_m_s(
        result.receiver_v2_m_s,
        shape=int(result.receiver_endpoint_key.shape[0]),
        scalar_v2_m_s=v2,
    )

    solution_status = _node_lookup(result.node_id, result.node_solution_status)
    weathering_status = _node_lookup(result.node_id, result.node_weathering_status)
    flat_datum = _nan_if_none(result.flat_datum_elevation_m)
    source_static_status = _endpoint_static_status_array(
        node_id=result.source_node_id,
        x_m=result.source_x_m,
        y_m=result.source_y_m,
        surface_elevation_m=result.source_surface_elevation_m,
        t1_s=result.source_half_intercept_time_s,
        weathering_thickness_m=result.source_weathering_thickness_m,
        total_shift_s=result.source_refraction_shift_s,
        datum_status=result.source_datum_status,
        node_solution_status=solution_status,
        node_weathering_status=weathering_status,
    )
    receiver_static_status = _endpoint_static_status_array(
        node_id=result.receiver_node_id,
        x_m=result.receiver_x_m,
        y_m=result.receiver_y_m,
        surface_elevation_m=result.receiver_surface_elevation_m,
        t1_s=result.receiver_half_intercept_time_s,
        weathering_thickness_m=result.receiver_weathering_thickness_m,
        total_shift_s=result.receiver_refraction_shift_s,
        datum_status=result.receiver_datum_status,
        node_solution_status=solution_status,
        node_weathering_status=weathering_status,
    )

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
                v2_m_s=source_v2[index],
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
                static_status=source_static_status[index],
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
                v2_m_s=receiver_v2[index],
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
                static_status=receiver_static_status[index],
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


def _endpoint_static_status_array(
    *,
    node_id: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    surface_elevation_m: np.ndarray,
    t1_s: np.ndarray,
    weathering_thickness_m: np.ndarray,
    total_shift_s: np.ndarray,
    datum_status: np.ndarray,
    node_solution_status: dict[int, Any],
    node_weathering_status: dict[int, Any],
) -> np.ndarray:
    statuses: list[str] = []
    for index, raw_node_id in enumerate(np.asarray(node_id).tolist()):
        endpoint_node_id = int(raw_node_id)
        solution_status = node_solution_status.get(endpoint_node_id, 'missing_solution')
        weathering_status = node_weathering_status.get(endpoint_node_id, 'missing_node')
        statuses.append(
            classify_refraction_endpoint_static_status(
                node_missing=endpoint_node_id not in node_solution_status,
                x_m=x_m[index],
                y_m=y_m[index],
                surface_elevation_m=surface_elevation_m[index],
                t1_s=t1_s[index],
                weathering_thickness_m=weathering_thickness_m[index],
                total_shift_s=total_shift_s[index],
                solution_status=solution_status,
                weathering_status=weathering_status,
                datum_status=datum_status[index],
            )
        )
    return _string_array(statuses)


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


def _string_array(value: object) -> np.ndarray:
    raw = [str(item) for item in np.asarray(value).tolist()]
    max_len = max([1, *(len(item) for item in raw)])
    return np.ascontiguousarray(raw, dtype=f'<U{max_len}')


def _endpoint_v2_m_s(
    value: object,
    *,
    shape: int,
    scalar_v2_m_s: float,
) -> np.ndarray:
    if value is None:
        return np.full(int(shape), float(scalar_v2_m_s), dtype=np.float64)
    return np.ascontiguousarray(value, dtype=np.float64)


def _coerce_2layer_inputs(
    *,
    t1_s: object,
    t2_s: object,
    v1_m_s: object,
    v2_m_s: object,
    v3_m_s: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t1 = _coerce_float_array(t1_s, name='t1_s')
    t2 = _coerce_float_array(t2_s, name='t2_s')
    v1 = _positive_finite_float_array(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    v3 = _positive_finite_float_array(v3_m_s, name='v3_m_s')
    t1, t2, v1, v2, v3 = _broadcast_t1lsst_arrays(
        (t1, t2, v1, v2, v3),
        names=('t1_s', 't2_s', 'v1_m_s', 'v2_m_s', 'v3_m_s'),
    )
    _validate_2layer_velocity_order(v1=v1, v2=v2, v3=v3)
    return t1, t2, v1, v2, v3


def _coerce_2layer_status_inputs(
    *,
    t1_s: object,
    t2_s: object,
    v1_m_s: object,
    v2_m_s: object,
    v3_m_s: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t1 = _coerce_float_array(t1_s, name='t1_s', allow_nonfinite=True)
    t2 = _coerce_float_array(t2_s, name='t2_s', allow_nonfinite=True)
    v1 = _coerce_float_array(v1_m_s, name='v1_m_s', allow_nonfinite=True)
    v2 = _coerce_float_array(v2_m_s, name='v2_m_s', allow_nonfinite=True)
    v3 = _coerce_float_array(v3_m_s, name='v3_m_s', allow_nonfinite=True)
    return _broadcast_t1lsst_arrays(
        (t1, t2, v1, v2, v3),
        names=('t1_s', 't2_s', 'v1_m_s', 'v2_m_s', 'v3_m_s'),
    )


def _coerce_3layer_inputs(
    *,
    t1_s: object,
    t2_s: object,
    t3_s: object,
    v1_m_s: object,
    v2_m_s: object,
    v3_m_s: object,
    vsub_m_s: object,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    t1 = _coerce_float_array(t1_s, name='t1_s')
    t2 = _coerce_float_array(t2_s, name='t2_s')
    t3 = _coerce_float_array(t3_s, name='t3_s')
    v1 = _positive_finite_float_array(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    v3 = _positive_finite_float_array(v3_m_s, name='v3_m_s')
    vsub = _positive_finite_float_array(vsub_m_s, name='vsub_m_s')
    t1, t2, t3, v1, v2, v3, vsub = _broadcast_t1lsst_arrays(
        (t1, t2, t3, v1, v2, v3, vsub),
        names=(
            't1_s',
            't2_s',
            't3_s',
            'v1_m_s',
            'v2_m_s',
            'v3_m_s',
            'vsub_m_s',
        ),
    )
    _validate_3layer_velocity_order(v1=v1, v2=v2, v3=v3, vsub=vsub)
    return t1, t2, t3, v1, v2, v3, vsub


def _coerce_3layer_status_inputs(
    *,
    t1_s: object,
    t2_s: object,
    t3_s: object,
    v1_m_s: object,
    v2_m_s: object,
    v3_m_s: object,
    vsub_m_s: object,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    t1 = _coerce_float_array(t1_s, name='t1_s', allow_nonfinite=True)
    t2 = _coerce_float_array(t2_s, name='t2_s', allow_nonfinite=True)
    t3 = _coerce_float_array(t3_s, name='t3_s', allow_nonfinite=True)
    v1 = _coerce_float_array(v1_m_s, name='v1_m_s', allow_nonfinite=True)
    v2 = _coerce_float_array(v2_m_s, name='v2_m_s', allow_nonfinite=True)
    v3 = _coerce_float_array(v3_m_s, name='v3_m_s', allow_nonfinite=True)
    vsub = _coerce_float_array(vsub_m_s, name='vsub_m_s', allow_nonfinite=True)
    return _broadcast_t1lsst_arrays(
        (t1, t2, t3, v1, v2, v3, vsub),
        names=(
            't1_s',
            't2_s',
            't3_s',
            'v1_m_s',
            'v2_m_s',
            'v3_m_s',
            'vsub_m_s',
        ),
    )


def _broadcast_t1lsst_arrays(
    arrays: tuple[np.ndarray, ...],
    *,
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    try:
        broadcasted = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        joined = ', '.join(names)
        raise RefractionT1LSSTError(
            f'{joined} must be broadcastable to a common shape'
        ) from exc
    return tuple(
        np.array(array, dtype=np.float64, copy=True, order='C')
        for array in broadcasted
    )


def _validate_2layer_velocity_order(
    *,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
) -> None:
    if np.any(v2 <= v1):
        raise RefractionT1LSSTError('v2_m_s must be greater than v1_m_s')
    if np.any(v3 <= v2):
        raise RefractionT1LSSTError('v3_m_s must be greater than v2_m_s')


def _validate_3layer_velocity_order(
    *,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    vsub: np.ndarray,
) -> None:
    _validate_2layer_velocity_order(v1=v1, v2=v2, v3=v3)
    if np.any(vsub <= v3):
        raise RefractionT1LSSTError('vsub_m_s must be greater than v3_m_s')


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


def _positive_finite_float_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    out = _coerce_float_array(value, name=name)
    if raw.ndim == 0:
        return np.asarray(_positive_finite(value, name=name), dtype=np.float64)
    if np.any(out <= 0.0):
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
    'RefractionT1LSST2LayerThicknessResult',
    'RefractionT1LSST3LayerThicknessResult',
    'RefractionT1LSSTError',
    'compute_t1lsst_1layer_thickness',
    'compute_t1lsst_1layer_weathering_correction',
    'compute_t1lsst_2layer_thicknesses',
    'compute_t1lsst_2layer_thicknesses_with_status',
    'compute_t1lsst_2layer_weathering_correction',
    'compute_t1lsst_3layer_thicknesses',
    'compute_t1lsst_3layer_thicknesses_with_status',
    'compute_t1lsst_3layer_weathering_correction',
    'compose_t1lsst_1layer_static_table_components',
    'write_refraction_t1lsst_1layer_components_csv',
]
