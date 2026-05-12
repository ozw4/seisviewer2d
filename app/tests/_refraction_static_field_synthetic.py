"""Dependency-light synthetic fixtures for M4 refraction field corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.tests._refraction_multilayer_synthetic import (
    SyntheticMultiLayerRefractionDataset,
    make_2d_straight_two_layer_refraction_dataset,
)

EndpointKind = Literal['source', 'receiver']
ManualStaticSignConvention = Literal['applied_shift_s', 'delay_positive_ms']

SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'

_STATUS_DTYPE = '<U64'
_OK_STATUS = 'ok'
_STATUS_PRIORITY = (
    'missing_manual_static',
    'missing_uphole_time',
    'missing_source_depth',
    'invalid_manual_static',
    'invalid_uphole_time',
    'invalid_source_depth',
    'inconsistent_uphole_time',
    'inconsistent_source_depth',
    'inactive_source_endpoint',
)
_STATUS_NORMALIZATION = {
    'invalid_manual_static_value': 'invalid_manual_static',
}


@dataclass(frozen=True)
class SyntheticFieldEndpointTable:
    """Endpoint table fields used by M4 field-correction unit tests."""

    endpoint_kind: EndpointKind
    endpoint_key: np.ndarray
    endpoint_id: np.ndarray
    node_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    elevation_m: np.ndarray
    pick_count: np.ndarray
    valid_endpoint_mask: np.ndarray
    refraction_shift_s: np.ndarray


@dataclass(frozen=True)
class SyntheticManualStaticRow:
    """Manual-static row shape matching artifact/inline table inputs."""

    endpoint_kind: EndpointKind
    endpoint_key: str | None
    endpoint_id: int | None
    manual_static_input_s: float
    status: str = _OK_STATUS
    comment: str | None = None


@dataclass(frozen=True)
class SyntheticRefractionFieldCorrectionDataset:
    """Known-truth source-depth, uphole, manual-static, and trace composition data."""

    name: str
    sign_convention: str
    manual_static_sign_convention: ManualStaticSignConvention
    base_dataset: SyntheticMultiLayerRefractionDataset

    source_endpoint_table: SyntheticFieldEndpointTable
    receiver_endpoint_table: SyntheticFieldEndpointTable

    sorted_trace_index: np.ndarray
    source_endpoint_index: np.ndarray
    receiver_endpoint_index: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_endpoint_id_sorted: np.ndarray
    receiver_endpoint_id_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    offset_m: np.ndarray
    first_break_time_s: np.ndarray
    noiseless_first_break_time_s: np.ndarray
    valid_pick_mask: np.ndarray
    layer_kind: np.ndarray

    true_v1_m_s: float
    true_v2_m_s: float
    true_v3_m_s: float

    source_depth_m: np.ndarray
    source_depth_status: np.ndarray
    expected_source_depth_shift_s: np.ndarray
    uphole_time_s: np.ndarray
    uphole_status: np.ndarray
    expected_uphole_shift_s: np.ndarray

    source_manual_static_input_s: np.ndarray
    receiver_manual_static_input_s: np.ndarray
    source_manual_static_status: np.ndarray
    receiver_manual_static_status: np.ndarray
    expected_source_manual_static_shift_s: np.ndarray
    expected_receiver_manual_static_shift_s: np.ndarray
    manual_static_rows_without_duplicates: tuple[SyntheticManualStaticRow, ...]
    manual_static_rows: tuple[SyntheticManualStaticRow, ...]
    duplicate_manual_static_endpoint_keys: tuple[str, ...]

    expected_source_field_shift_s: np.ndarray
    source_field_static_status: np.ndarray
    expected_receiver_field_shift_s: np.ndarray
    receiver_field_static_status: np.ndarray

    expected_source_field_shift_s_sorted: np.ndarray
    expected_receiver_field_shift_s_sorted: np.ndarray
    expected_trace_field_shift_s: np.ndarray
    trace_field_static_status: np.ndarray
    expected_refraction_trace_shift_s: np.ndarray
    expected_final_trace_shift_s: np.ndarray

    missing_source_depth_endpoint_index: int | None = None
    missing_uphole_endpoint_index: int | None = None
    invalid_source_endpoint_index: int | None = None

    @property
    def pick_time_s(self) -> np.ndarray:
        """Alias matching existing synthetic refraction fixture naming."""
        return self.first_break_time_s

    def as_sorted_trace_arrays(self) -> dict[str, np.ndarray]:
        """Return sorted trace arrays commonly needed by M4 unit tests."""
        return {
            'sorted_trace_index': self.sorted_trace_index,
            'pick_time_s_sorted': self.first_break_time_s,
            'valid_pick_mask_sorted': self.valid_pick_mask,
            'source_endpoint_key_sorted': self.source_endpoint_key_sorted,
            'receiver_endpoint_key_sorted': self.receiver_endpoint_key_sorted,
            'source_endpoint_id_sorted': self.source_endpoint_id_sorted,
            'receiver_endpoint_id_sorted': self.receiver_endpoint_id_sorted,
            'source_node_id_sorted': self.source_node_id_sorted,
            'receiver_node_id_sorted': self.receiver_node_id_sorted,
            'source_depth_m_sorted': self.source_depth_m[self.source_endpoint_index],
            'uphole_time_s_sorted': self.uphole_time_s[self.source_endpoint_index],
            'offset_m_sorted': self.offset_m,
            'layer_kind_sorted': self.layer_kind,
            'refraction_trace_shift_s_sorted': self.expected_refraction_trace_shift_s,
            'trace_field_shift_s_sorted': self.expected_trace_field_shift_s,
            'final_trace_shift_s_sorted': self.expected_final_trace_shift_s,
        }


def make_clean_2d_field_corrections(
    *,
    manual_static_sign_convention: ManualStaticSignConvention = 'delay_positive_ms',
) -> SyntheticRefractionFieldCorrectionDataset:
    """Build a regular 2D line where every M4 field component is valid."""
    base = make_2d_straight_two_layer_refraction_dataset()
    source_depth_m = 4.0 + 1.5 * np.arange(
        base.source_endpoint_id.shape[0],
        dtype=np.float64,
    )
    uphole_time_s = 0.002 + 0.0015 * np.arange(
        base.source_endpoint_id.shape[0],
        dtype=np.float64,
    )
    source_manual_static_input_s = np.asarray(
        [0.0010, -0.0005, 0.0020, 0.0000, -0.0015, 0.0015],
        dtype=np.float64,
    )
    receiver_manual_static_input_s = (
        (np.arange(base.receiver_endpoint_id.shape[0], dtype=np.float64) % 5.0) - 2.0
    ) * 0.00075

    return _build_field_dataset(
        name='clean_2d_field_corrections',
        base=base,
        manual_static_sign_convention=manual_static_sign_convention,
        source_depth_m=source_depth_m,
        source_depth_status=np.full(source_depth_m.shape, _OK_STATUS, dtype=_STATUS_DTYPE),
        uphole_time_s=uphole_time_s,
        uphole_status=np.full(uphole_time_s.shape, _OK_STATUS, dtype=_STATUS_DTYPE),
        source_manual_static_input_s=source_manual_static_input_s,
        receiver_manual_static_input_s=receiver_manual_static_input_s,
        source_manual_static_status=np.full(
            source_manual_static_input_s.shape,
            _OK_STATUS,
            dtype=_STATUS_DTYPE,
        ),
        receiver_manual_static_status=np.full(
            receiver_manual_static_input_s.shape,
            _OK_STATUS,
            dtype=_STATUS_DTYPE,
        ),
    )


def make_messy_2d_field_corrections(
    *,
    manual_static_sign_convention: ManualStaticSignConvention = 'delay_positive_ms',
) -> SyntheticRefractionFieldCorrectionDataset:
    """Build a 2D line with missing values, duplicate manual rows, and one inactive source."""
    base = make_2d_straight_two_layer_refraction_dataset()
    source_depth_m = 4.0 + 1.5 * np.arange(
        base.source_endpoint_id.shape[0],
        dtype=np.float64,
    )
    uphole_time_s = 0.002 + 0.0015 * np.arange(
        base.source_endpoint_id.shape[0],
        dtype=np.float64,
    )
    source_depth_status = np.full(source_depth_m.shape, _OK_STATUS, dtype=_STATUS_DTYPE)
    uphole_status = np.full(uphole_time_s.shape, _OK_STATUS, dtype=_STATUS_DTYPE)

    missing_depth_index = 1
    missing_uphole_index = 2
    inactive_source_index = 3
    source_depth_m[missing_depth_index] = np.nan
    source_depth_status[missing_depth_index] = 'missing_source_depth'
    uphole_time_s[missing_uphole_index] = np.nan
    uphole_status[missing_uphole_index] = 'missing_uphole_time'
    source_depth_status[inactive_source_index] = 'inactive_source_endpoint'
    uphole_status[inactive_source_index] = 'inactive_source_endpoint'

    source_manual_static_input_s = np.asarray(
        [0.0010, -0.0005, 0.0020, 0.0000, -0.0015, 0.0015],
        dtype=np.float64,
    )
    receiver_manual_static_input_s = (
        (np.arange(base.receiver_endpoint_id.shape[0], dtype=np.float64) % 5.0) - 2.0
    ) * 0.00075
    source_manual_static_status = np.full(
        source_manual_static_input_s.shape,
        _OK_STATUS,
        dtype=_STATUS_DTYPE,
    )
    receiver_manual_static_status = np.full(
        receiver_manual_static_input_s.shape,
        _OK_STATUS,
        dtype=_STATUS_DTYPE,
    )

    return _build_field_dataset(
        name='messy_2d_field_corrections',
        base=base,
        manual_static_sign_convention=manual_static_sign_convention,
        source_depth_m=source_depth_m,
        source_depth_status=source_depth_status,
        uphole_time_s=uphole_time_s,
        uphole_status=uphole_status,
        source_manual_static_input_s=source_manual_static_input_s,
        receiver_manual_static_input_s=receiver_manual_static_input_s,
        source_manual_static_status=source_manual_static_status,
        receiver_manual_static_status=receiver_manual_static_status,
        inactive_source_endpoint_index=inactive_source_index,
        duplicate_source_endpoint_index=0,
        missing_source_depth_endpoint_index=missing_depth_index,
        missing_uphole_endpoint_index=missing_uphole_index,
    )


def _build_field_dataset(
    *,
    name: str,
    base: SyntheticMultiLayerRefractionDataset,
    manual_static_sign_convention: ManualStaticSignConvention,
    source_depth_m: np.ndarray,
    source_depth_status: np.ndarray,
    uphole_time_s: np.ndarray,
    uphole_status: np.ndarray,
    source_manual_static_input_s: np.ndarray,
    receiver_manual_static_input_s: np.ndarray,
    source_manual_static_status: np.ndarray,
    receiver_manual_static_status: np.ndarray,
    inactive_source_endpoint_index: int | None = None,
    duplicate_source_endpoint_index: int | None = None,
    missing_source_depth_endpoint_index: int | None = None,
    missing_uphole_endpoint_index: int | None = None,
) -> SyntheticRefractionFieldCorrectionDataset:
    _validate_manual_static_sign_convention(manual_static_sign_convention)
    source_node_id = np.ascontiguousarray(base.source_endpoint_node_id.copy(), dtype=np.int64)
    if inactive_source_endpoint_index is not None:
        source_node_id[int(inactive_source_endpoint_index)] = -1
    receiver_node_id = np.ascontiguousarray(
        base.receiver_endpoint_node_id.copy(),
        dtype=np.int64,
    )

    source_table = _endpoint_table(
        endpoint_kind='source',
        endpoint_key=base.source_endpoint_id,
        endpoint_id=base.source_endpoint_id,
        node_id=source_node_id,
        x_m=base.source_endpoint_x_m,
        y_m=base.source_endpoint_y_m,
        elevation_m=base.source_endpoint_elevation_m,
        endpoint_index_sorted=base.source_endpoint_index,
        refraction_shift_s=base.true_source_endpoint_total_static_s,
    )
    receiver_table = _endpoint_table(
        endpoint_kind='receiver',
        endpoint_key=base.receiver_endpoint_id,
        endpoint_id=base.receiver_endpoint_id,
        node_id=receiver_node_id,
        x_m=base.receiver_endpoint_x_m,
        y_m=base.receiver_endpoint_y_m,
        elevation_m=base.receiver_endpoint_elevation_m,
        endpoint_index_sorted=base.receiver_endpoint_index,
        refraction_shift_s=base.true_receiver_endpoint_total_static_s,
    )

    expected_source_depth_shift_s = _component_shift(
        source_depth_m / float(base.true_v1_m_s),
        source_depth_status,
    )
    expected_uphole_shift_s = _component_shift(-uphole_time_s, uphole_status)
    expected_source_manual_static_shift_s = _component_shift(
        _manual_static_shift_s(
            source_manual_static_input_s,
            sign_convention=manual_static_sign_convention,
        ),
        source_manual_static_status,
    )
    expected_receiver_manual_static_shift_s = _component_shift(
        _manual_static_shift_s(
            receiver_manual_static_input_s,
            sign_convention=manual_static_sign_convention,
        ),
        receiver_manual_static_status,
    )

    expected_source_field_shift_s, source_field_static_status = _endpoint_field_shift(
        (
            expected_source_depth_shift_s,
            expected_uphole_shift_s,
            expected_source_manual_static_shift_s,
        ),
        (
            source_depth_status,
            uphole_status,
            source_manual_static_status,
        ),
    )
    (
        expected_receiver_field_shift_s,
        receiver_field_static_status,
    ) = _endpoint_field_shift(
        (expected_receiver_manual_static_shift_s,),
        (receiver_manual_static_status,),
    )

    expected_source_field_shift_s_sorted = expected_source_field_shift_s[
        base.source_endpoint_index
    ]
    expected_receiver_field_shift_s_sorted = expected_receiver_field_shift_s[
        base.receiver_endpoint_index
    ]
    trace_field_static_status = _trace_field_status(
        source_field_static_status[base.source_endpoint_index],
        receiver_field_static_status[base.receiver_endpoint_index],
    )
    expected_trace_field_shift_s = np.full(
        base.sorted_trace_index.shape,
        np.nan,
        dtype=np.float64,
    )
    ok_trace = trace_field_static_status == _OK_STATUS
    expected_trace_field_shift_s[ok_trace] = (
        expected_source_field_shift_s_sorted[ok_trace]
        + expected_receiver_field_shift_s_sorted[ok_trace]
    )
    expected_refraction_trace_shift_s = (
        base.true_source_total_static_s + base.true_receiver_total_static_s
    )
    expected_final_trace_shift_s = expected_refraction_trace_shift_s + expected_trace_field_shift_s

    manual_rows_without_duplicates = _manual_static_rows(
        source_endpoint_key=source_table.endpoint_key,
        source_endpoint_id=source_table.endpoint_id,
        source_input_s=source_manual_static_input_s,
        source_status=source_manual_static_status,
        receiver_endpoint_key=receiver_table.endpoint_key,
        receiver_endpoint_id=receiver_table.endpoint_id,
        receiver_input_s=receiver_manual_static_input_s,
        receiver_status=receiver_manual_static_status,
    )
    duplicate_keys: tuple[str, ...] = ()
    manual_rows = manual_rows_without_duplicates
    if duplicate_source_endpoint_index is not None:
        duplicate_row = _duplicate_manual_static_row(
            source_table.endpoint_key[int(duplicate_source_endpoint_index)],
            source_table.endpoint_id[int(duplicate_source_endpoint_index)],
        )
        manual_rows = manual_rows_without_duplicates + (duplicate_row,)
        duplicate_keys = (str(duplicate_row.endpoint_key),)

    return SyntheticRefractionFieldCorrectionDataset(
        name=name,
        sign_convention=SIGN_CONVENTION,
        manual_static_sign_convention=manual_static_sign_convention,
        base_dataset=base,
        source_endpoint_table=source_table,
        receiver_endpoint_table=receiver_table,
        sorted_trace_index=_i64(base.sorted_trace_index),
        source_endpoint_index=_i64(base.source_endpoint_index),
        receiver_endpoint_index=_i64(base.receiver_endpoint_index),
        source_endpoint_key_sorted=_key_array(source_table.endpoint_key[base.source_endpoint_index]),
        receiver_endpoint_key_sorted=_key_array(
            receiver_table.endpoint_key[base.receiver_endpoint_index],
        ),
        source_endpoint_id_sorted=_i64(source_table.endpoint_id[base.source_endpoint_index]),
        receiver_endpoint_id_sorted=_i64(
            receiver_table.endpoint_id[base.receiver_endpoint_index],
        ),
        source_node_id_sorted=_i64(source_table.node_id[base.source_endpoint_index]),
        receiver_node_id_sorted=_i64(
            receiver_table.node_id[base.receiver_endpoint_index],
        ),
        offset_m=_f64(base.offset_m),
        first_break_time_s=_f64(base.first_break_time_s),
        noiseless_first_break_time_s=_f64(base.noiseless_first_break_time_s),
        valid_pick_mask=np.ascontiguousarray(base.valid_mask, dtype=bool),
        layer_kind=np.ascontiguousarray(base.layer_kind, dtype='<U16'),
        true_v1_m_s=float(base.true_v1_m_s),
        true_v2_m_s=float(base.true_v2_m_s),
        true_v3_m_s=float(base.true_v3_m_s),
        source_depth_m=_f64(source_depth_m),
        source_depth_status=_status(source_depth_status),
        expected_source_depth_shift_s=_f64(expected_source_depth_shift_s),
        uphole_time_s=_f64(uphole_time_s),
        uphole_status=_status(uphole_status),
        expected_uphole_shift_s=_f64(expected_uphole_shift_s),
        source_manual_static_input_s=_f64(source_manual_static_input_s),
        receiver_manual_static_input_s=_f64(receiver_manual_static_input_s),
        source_manual_static_status=_status(source_manual_static_status),
        receiver_manual_static_status=_status(receiver_manual_static_status),
        expected_source_manual_static_shift_s=_f64(
            expected_source_manual_static_shift_s,
        ),
        expected_receiver_manual_static_shift_s=_f64(
            expected_receiver_manual_static_shift_s,
        ),
        manual_static_rows_without_duplicates=manual_rows_without_duplicates,
        manual_static_rows=manual_rows,
        duplicate_manual_static_endpoint_keys=duplicate_keys,
        expected_source_field_shift_s=_f64(expected_source_field_shift_s),
        source_field_static_status=_status(source_field_static_status),
        expected_receiver_field_shift_s=_f64(expected_receiver_field_shift_s),
        receiver_field_static_status=_status(receiver_field_static_status),
        expected_source_field_shift_s_sorted=_f64(
            expected_source_field_shift_s_sorted,
        ),
        expected_receiver_field_shift_s_sorted=_f64(
            expected_receiver_field_shift_s_sorted,
        ),
        expected_trace_field_shift_s=_f64(expected_trace_field_shift_s),
        trace_field_static_status=_status(trace_field_static_status),
        expected_refraction_trace_shift_s=_f64(expected_refraction_trace_shift_s),
        expected_final_trace_shift_s=_f64(expected_final_trace_shift_s),
        missing_source_depth_endpoint_index=missing_source_depth_endpoint_index,
        missing_uphole_endpoint_index=missing_uphole_endpoint_index,
        invalid_source_endpoint_index=inactive_source_endpoint_index,
    )


def _endpoint_table(
    *,
    endpoint_kind: EndpointKind,
    endpoint_key: np.ndarray,
    endpoint_id: np.ndarray,
    node_id: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    elevation_m: np.ndarray,
    endpoint_index_sorted: np.ndarray,
    refraction_shift_s: np.ndarray,
) -> SyntheticFieldEndpointTable:
    pick_count = np.bincount(
        np.asarray(endpoint_index_sorted, dtype=np.int64),
        minlength=int(endpoint_key.shape[0]),
    )
    return SyntheticFieldEndpointTable(
        endpoint_kind=endpoint_kind,
        endpoint_key=_key_array(endpoint_key),
        endpoint_id=_i64(endpoint_id),
        node_id=_i64(node_id),
        x_m=_f64(x_m),
        y_m=_f64(y_m),
        elevation_m=_f64(elevation_m),
        pick_count=_i64(pick_count),
        valid_endpoint_mask=np.ascontiguousarray(np.asarray(node_id) >= 0, dtype=bool),
        refraction_shift_s=_f64(refraction_shift_s),
    )


def _manual_static_rows(
    *,
    source_endpoint_key: np.ndarray,
    source_endpoint_id: np.ndarray,
    source_input_s: np.ndarray,
    source_status: np.ndarray,
    receiver_endpoint_key: np.ndarray,
    receiver_endpoint_id: np.ndarray,
    receiver_input_s: np.ndarray,
    receiver_status: np.ndarray,
) -> tuple[SyntheticManualStaticRow, ...]:
    rows: list[SyntheticManualStaticRow] = []
    for index, key in enumerate(source_endpoint_key.tolist()):
        rows.append(
            SyntheticManualStaticRow(
                endpoint_kind='source',
                endpoint_key=str(key),
                endpoint_id=int(source_endpoint_id[index]),
                manual_static_input_s=float(source_input_s[index]),
                status=str(source_status[index]),
            )
        )
    for index, key in enumerate(receiver_endpoint_key.tolist()):
        rows.append(
            SyntheticManualStaticRow(
                endpoint_kind='receiver',
                endpoint_key=str(key),
                endpoint_id=int(receiver_endpoint_id[index]),
                manual_static_input_s=float(receiver_input_s[index]),
                status=str(receiver_status[index]),
            )
        )
    return tuple(rows)


def _duplicate_manual_static_row(
    endpoint_key: object,
    endpoint_id: object,
) -> SyntheticManualStaticRow:
    return SyntheticManualStaticRow(
        endpoint_kind='source',
        endpoint_key=str(endpoint_key),
        endpoint_id=int(endpoint_id),
        manual_static_input_s=0.123,
        status=_OK_STATUS,
        comment='intentional duplicate row for messy fixture',
    )


def _component_shift(values: np.ndarray, status: np.ndarray) -> np.ndarray:
    shift = np.asarray(values, dtype=np.float64).copy()
    status_arr = np.asarray(status)
    shift[status_arr != _OK_STATUS] = np.nan
    return np.ascontiguousarray(shift, dtype=np.float64)


def _endpoint_field_shift(
    shifts: tuple[np.ndarray, ...],
    statuses: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    endpoint_count = int(shifts[0].shape[0])
    total = np.full(endpoint_count, np.nan, dtype=np.float64)
    field_status = np.full(endpoint_count, _OK_STATUS, dtype=_STATUS_DTYPE)
    for index in range(endpoint_count):
        status = _prioritized_status(tuple(status[index] for status in statuses))
        if status != _OK_STATUS:
            field_status[index] = status
            continue
        values = np.asarray([shift[index] for shift in shifts], dtype=np.float64)
        if not bool(np.all(np.isfinite(values))):
            field_status[index] = 'invalid_field_shift'
            continue
        total[index] = float(np.sum(values))
    return (
        np.ascontiguousarray(total, dtype=np.float64),
        np.ascontiguousarray(field_status, dtype=_STATUS_DTYPE),
    )


def _trace_field_status(
    source_status: np.ndarray,
    receiver_status: np.ndarray,
) -> np.ndarray:
    result = np.full(source_status.shape, _OK_STATUS, dtype=_STATUS_DTYPE)
    for index in range(int(result.shape[0])):
        result[index] = _prioritized_status((source_status[index], receiver_status[index]))
    return np.ascontiguousarray(result, dtype=_STATUS_DTYPE)


def _manual_static_shift_s(
    value_s: np.ndarray,
    *,
    sign_convention: ManualStaticSignConvention,
) -> np.ndarray:
    _validate_manual_static_sign_convention(sign_convention)
    values = np.asarray(value_s, dtype=np.float64)
    if sign_convention == 'applied_shift_s':
        return np.ascontiguousarray(values, dtype=np.float64)
    return np.ascontiguousarray(-values, dtype=np.float64)


def _validate_manual_static_sign_convention(
    value: ManualStaticSignConvention,
) -> None:
    if value not in {'applied_shift_s', 'delay_positive_ms'}:
        raise ValueError(f'invalid manual_static_sign_convention: {value!r}')


def _prioritized_status(statuses: tuple[object, ...]) -> str:
    normalized = [
        _normalize_status(status)
        for status in statuses
        if _normalize_status(status) not in {_OK_STATUS, 'not_applicable', 'none', ''}
    ]
    if not normalized:
        return _OK_STATUS
    for candidate in _STATUS_PRIORITY:
        if candidate in normalized:
            return candidate
    return normalized[0]


def _normalize_status(value: object) -> str:
    text = str(value)
    return _STATUS_NORMALIZATION.get(text, text)


def _key_array(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values).astype(object, copy=False), dtype=object)


def _status(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=_STATUS_DTYPE)


def _f64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _i64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)
