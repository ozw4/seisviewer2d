"""T1LSST-compatible refraction static artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.services.common.artifact_io import write_csv_atomic
from seis_statics.refraction.status import (
    classify_refraction_endpoint_static_status,
)
from seis_statics.refraction.t1lsst import (
    RefractionT1LSST1LayerEndpointComponents,
    RefractionT1LSSTError,
    compose_t1lsst_1layer_endpoint_component_rows,
)
from app.statics.refraction.domain.types import RefractionDatumStaticsResult

REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME = (
    'refraction_t1lsst_1layer_components.csv'
)

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


def write_refraction_t1lsst_1layer_components_csv(
    *,
    result: RefractionDatumStaticsResult,
    path: Path,
) -> None:
    """Write the T1LSST-compatible one-layer component CSV artifact."""
    rows = _compose_t1lsst_1layer_static_table_components(result)
    write_csv_atomic(
        Path(path),
        columns=_T1LSST_COMPONENT_COLUMNS,
        rows=rows,
        lineterminator='\r\n',
    )


def _compose_t1lsst_1layer_static_table_components(
    result: RefractionDatumStaticsResult,
) -> list[dict[str, object]]:
    v1 = _positive_finite(
        result.weathering_velocity_m_s,
        name='result.weathering_velocity_m_s',
    )
    v2 = _positive_finite(
        result.bedrock_velocity_m_s,
        name='result.bedrock_velocity_m_s',
    )
    if v2 <= v1:
        raise RefractionT1LSSTError(
            'result.bedrock_velocity_m_s must be greater than '
            'result.weathering_velocity_m_s'
        )

    solution_status = _node_lookup(result.node_id, result.node_solution_status)
    weathering_status = _node_lookup(result.node_id, result.node_weathering_status)
    source_rows = compose_t1lsst_1layer_endpoint_component_rows(
        RefractionT1LSST1LayerEndpointComponents(
            endpoint_kind='source',
            endpoint_key=result.source_endpoint_key,
            node_id=result.source_node_id,
            x_m=result.source_x_m,
            y_m=result.source_y_m,
            surface_elevation_m=result.source_surface_elevation_m,
            floating_datum_elevation_m=result.source_floating_datum_elevation_m,
            flat_datum_elevation_m=_flat_datum_array(
                result.flat_datum_elevation_m,
                shape=int(result.source_endpoint_key.shape[0]),
            ),
            t1_s=result.source_half_intercept_time_s,
            v1_m_s=v1,
            v2_m_s=_endpoint_v2_m_s(
                result.source_v2_m_s,
                shape=int(result.source_endpoint_key.shape[0]),
                scalar_v2_m_s=v2,
            ),
            sh1_m=result.source_weathering_thickness_m,
            refractor_elevation_m=result.source_refractor_elevation_m,
            weathering_correction_s=result.source_weathering_replacement_shift_s,
            floating_datum_correction_s=(
                result.source_floating_datum_elevation_shift_s
            ),
            flat_datum_correction_s=result.source_flat_datum_shift_s,
            total_static_s=result.source_refraction_shift_s,
            solution_status=_endpoint_node_status(
                result.source_node_id,
                solution_status,
                missing='missing_node',
            ),
            weathering_status=_endpoint_node_status(
                result.source_node_id,
                weathering_status,
                missing='missing_node',
            ),
            datum_status=result.source_datum_status,
            static_status=_endpoint_static_status_array(
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
            ),
        )
    )
    receiver_rows = compose_t1lsst_1layer_endpoint_component_rows(
        RefractionT1LSST1LayerEndpointComponents(
            endpoint_kind='receiver',
            endpoint_key=result.receiver_endpoint_key,
            node_id=result.receiver_node_id,
            x_m=result.receiver_x_m,
            y_m=result.receiver_y_m,
            surface_elevation_m=result.receiver_surface_elevation_m,
            floating_datum_elevation_m=result.receiver_floating_datum_elevation_m,
            flat_datum_elevation_m=_flat_datum_array(
                result.flat_datum_elevation_m,
                shape=int(result.receiver_endpoint_key.shape[0]),
            ),
            t1_s=result.receiver_half_intercept_time_s,
            v1_m_s=v1,
            v2_m_s=_endpoint_v2_m_s(
                result.receiver_v2_m_s,
                shape=int(result.receiver_endpoint_key.shape[0]),
                scalar_v2_m_s=v2,
            ),
            sh1_m=result.receiver_weathering_thickness_m,
            refractor_elevation_m=result.receiver_refractor_elevation_m,
            weathering_correction_s=result.receiver_weathering_replacement_shift_s,
            floating_datum_correction_s=(
                result.receiver_floating_datum_elevation_shift_s
            ),
            flat_datum_correction_s=result.receiver_flat_datum_shift_s,
            total_static_s=result.receiver_refraction_shift_s,
            solution_status=_endpoint_node_status(
                result.receiver_node_id,
                solution_status,
                missing='missing_node',
            ),
            weathering_status=_endpoint_node_status(
                result.receiver_node_id,
                weathering_status,
                missing='missing_node',
            ),
            datum_status=result.receiver_datum_status,
            static_status=_endpoint_static_status_array(
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
            ),
        )
    )
    return [*source_rows, *receiver_rows]


def _node_lookup(node_id: np.ndarray, values: np.ndarray) -> dict[int, Any]:
    return {
        int(raw_node): values[index]
        for index, raw_node in enumerate(np.asarray(node_id).tolist())
    }


def _endpoint_node_status(
    node_id: np.ndarray,
    status_by_node: dict[int, Any],
    *,
    missing: str,
) -> np.ndarray:
    statuses = [
        status_by_node.get(int(raw_node_id), missing)
        for raw_node_id in np.asarray(node_id).tolist()
    ]
    return _string_array(statuses)


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


def _endpoint_v2_m_s(
    value: object,
    *,
    shape: int,
    scalar_v2_m_s: float,
) -> np.ndarray:
    if value is None:
        return np.full(int(shape), float(scalar_v2_m_s), dtype=np.float64)
    return np.ascontiguousarray(value, dtype=np.float64)


def _flat_datum_array(value: object, *, shape: int) -> np.ndarray:
    number = float('nan') if value is None else float(value)
    return np.full(int(shape), number, dtype=np.float64)


def _string_array(value: object) -> np.ndarray:
    raw = [str(item) for item in np.asarray(value).tolist()]
    max_len = max([1, *(len(item) for item in raw)])
    return np.ascontiguousarray(raw, dtype=f'<U{max_len}')


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RefractionT1LSSTError(f'{name} must be finite and positive')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionT1LSSTError(f'{name} must be finite and positive') from exc
    if not np.isfinite(number) or number <= 0.0:
        raise RefractionT1LSSTError(f'{name} must be finite and positive')
    return number


__all__ = [
    'REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME',
    'write_refraction_t1lsst_1layer_components_csv',
]
