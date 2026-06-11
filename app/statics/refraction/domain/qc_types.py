"""Dependency-light refraction static QC series types and serializers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from io import StringIO
import math
from typing import Any, Final

import numpy as np

from app.statics.refraction.domain.export_units import (
    REFRACTION_STATIC_REPO_SIGN_CONVENTION,
)

REFRACTION_STATIC_QC_SIGN_CONVENTION: Final = REFRACTION_STATIC_REPO_SIGN_CONVENTION


@dataclass(frozen=True)
class RefractionFirstBreakQcSeries:
    """Trace-order first-break observed/modelled/residual QC values."""

    trace_index_sorted: np.ndarray
    source_endpoint_key: np.ndarray
    receiver_endpoint_key: np.ndarray
    offset_m: np.ndarray
    observed_time_s: np.ndarray
    modeled_time_s: np.ndarray
    residual_time_s: np.ndarray
    layer_kind: np.ndarray
    status: np.ndarray


@dataclass(frozen=True)
class RefractionProfileQcSeries:
    """Endpoint profile values for line/profile QC plots."""

    endpoint_kind: str
    endpoint_key: np.ndarray
    inline_m: np.ndarray
    t1_s: np.ndarray | None
    t2_s: np.ndarray | None
    t3_s: np.ndarray | None
    velocity_m_s: dict[str, np.ndarray]
    static_components_s: dict[str, np.ndarray]
    status: np.ndarray


@dataclass(frozen=True)
class RefractionCellQcSeries:
    """Per-cell refractor velocity and residual QC values."""

    layer_kind: str
    cell_id: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    x_min_m: np.ndarray
    x_max_m: np.ndarray
    y_min_m: np.ndarray
    y_max_m: np.ndarray
    x_center_m: np.ndarray
    y_center_m: np.ndarray
    active: np.ndarray
    n_observations: np.ndarray
    n_used_observations: np.ndarray
    n_rejected_observations: np.ndarray
    velocity_m_s: np.ndarray
    slowness_s_per_m: np.ndarray
    residual_rms_s: np.ndarray
    residual_mad_s: np.ndarray
    residual_mean_s: np.ndarray
    residual_p95_abs_s: np.ndarray
    smoothing_neighbor_count: np.ndarray
    status: np.ndarray


@dataclass(frozen=True)
class RefractionStaticComponentQcSeries:
    """Endpoint static components under the repo static-shift convention."""

    endpoint_kind: np.ndarray
    endpoint_key: np.ndarray
    component_shift_s: dict[str, np.ndarray]
    component_status: dict[str, np.ndarray]
    total_static_s: np.ndarray
    total_applied_shift_s: np.ndarray
    status: np.ndarray
    sign_convention: str = REFRACTION_STATIC_QC_SIGN_CONVENTION


def refraction_qc_to_json_safe(
    value: object,
    *,
    nonfinite_value: object = None,
) -> Any:
    """Convert dataclasses, mappings, arrays, and NumPy scalars to JSON-safe data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: refraction_qc_to_json_safe(
                getattr(value, field.name),
                nonfinite_value=nonfinite_value,
            )
            for field in fields(value)
        }

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return refraction_qc_to_json_safe(
                value.item(),
                nonfinite_value=nonfinite_value,
            )
        return [
            refraction_qc_to_json_safe(item, nonfinite_value=nonfinite_value)
            for item in value.tolist()
        ]

    if isinstance(value, np.generic):
        return refraction_qc_to_json_safe(
            value.item(),
            nonfinite_value=nonfinite_value,
        )

    if isinstance(value, Mapping):
        return {
            str(key): refraction_qc_to_json_safe(
                item,
                nonfinite_value=nonfinite_value,
            )
            for key, item in value.items()
        }

    if isinstance(value, tuple | list):
        return [
            refraction_qc_to_json_safe(item, nonfinite_value=nonfinite_value)
            for item in value
        ]

    if isinstance(value, float):
        return value if math.isfinite(value) else nonfinite_value

    if value is None or isinstance(value, bool | int | str):
        return value

    raise TypeError(f'unsupported QC value type: {type(value).__name__}')


def refraction_qc_series_to_json_safe_dict(
    series: object,
    *,
    nonfinite_value: object = None,
) -> dict[str, Any]:
    """Serialize a QC series dataclass to a JSON-safe dictionary."""
    if not is_dataclass(series) or isinstance(series, type):
        raise TypeError('series must be a dataclass instance')
    payload = refraction_qc_to_json_safe(series, nonfinite_value=nonfinite_value)
    if not isinstance(payload, dict):
        raise TypeError('series serializer did not produce a dictionary')
    return payload


def refraction_qc_series_to_csv_rows(
    series: object,
    *,
    nonfinite_value: object = '',
) -> list[dict[str, object]]:
    """Flatten a QC series dataclass to CSV-ready row dictionaries."""
    columns = _series_columns(series)
    row_count = _series_row_count(columns)
    rows: list[dict[str, object]] = []
    for row_index in range(row_count):
        rows.append(
            {
                name: refraction_qc_to_json_safe(
                    _column_value_at(value, row_index),
                    nonfinite_value=nonfinite_value,
                )
                for name, value in columns
            }
        )
    return rows


def refraction_qc_series_to_csv_text(
    series: object,
    *,
    nonfinite_value: object = '',
    lineterminator: str = '\n',
) -> str:
    """Serialize a QC series dataclass to CSV text with a header row."""
    columns = _series_columns(series)
    fieldnames = [name for name, _value in columns]
    rows = refraction_qc_series_to_csv_rows(
        series,
        nonfinite_value=nonfinite_value,
    )
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator=lineterminator,
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _series_columns(series: object) -> list[tuple[str, object]]:
    if not is_dataclass(series) or isinstance(series, type):
        raise TypeError('series must be a dataclass instance')

    columns: list[tuple[str, object]] = []
    for field in fields(series):
        value = getattr(series, field.name)
        if isinstance(value, Mapping):
            for key, item in value.items():
                columns.append((f'{field.name}_{key}', item))
        else:
            columns.append((field.name, value))
    return columns


def _series_row_count(columns: list[tuple[str, object]]) -> int:
    lengths = {
        length
        for _name, value in columns
        if (length := _column_length(value)) is not None
    }
    if len(lengths) > 1:
        raise ValueError(f'QC series columns have inconsistent lengths: {lengths}')
    if lengths:
        return lengths.pop()
    return 1


def _column_length(value: object) -> int | None:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return None
        return int(value.shape[0])
    return None


def _column_value_at(value: object, row_index: int) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value[row_index]
    return value


__all__ = [
    'REFRACTION_STATIC_QC_SIGN_CONVENTION',
    'RefractionCellQcSeries',
    'RefractionFirstBreakQcSeries',
    'RefractionProfileQcSeries',
    'RefractionStaticComponentQcSeries',
    'refraction_qc_series_to_csv_rows',
    'refraction_qc_series_to_csv_text',
    'refraction_qc_series_to_json_safe_dict',
    'refraction_qc_to_json_safe',
]
