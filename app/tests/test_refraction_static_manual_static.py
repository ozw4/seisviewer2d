from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.refraction_static_manual_static import (
    RefractionManualStaticTableRow,
    load_refraction_manual_static_table_rows,
    resolve_refraction_manual_static,
)


def _keys(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _ids(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _row(
    *,
    kind: str,
    key: str | None = None,
    endpoint_id: int | None = None,
    value_s: float,
) -> RefractionManualStaticTableRow:
    return RefractionManualStaticTableRow(
        endpoint_kind=kind,
        endpoint_key=key,
        endpoint_id=endpoint_id,
        station_id=None,
        node_id=None,
        x_m=None,
        y_m=None,
        manual_static_input_s=float(value_s),
        status='ok' if np.isfinite(value_s) else 'invalid_manual_static_value',
        comment=None,
        source_name='test',
        row_number=1,
    )


def _resolve(
    rows: tuple[RefractionManualStaticTableRow, ...],
    *,
    sign_convention: str = 'applied_shift_s',
    allow_missing_endpoints: bool = True,
):
    return resolve_refraction_manual_static(
        source_endpoint_key=_keys(['source:a', 'source:b']),
        source_endpoint_id=_ids([10, 11]),
        source_node_id=_ids([0, 1]),
        receiver_endpoint_key=_keys(['receiver:a', 'receiver:b']),
        receiver_endpoint_id=_ids([20, 21]),
        receiver_node_id=_ids([2, 3]),
        rows=rows,
        mode='artifact_table',
        sign_convention=sign_convention,
        allow_missing_endpoints=allow_missing_endpoints,
    )


def test_manual_static_table_loads_source_and_receiver_rows(tmp_path: Path) -> None:
    path = tmp_path / 'manual_statics.csv'
    path.write_text(
        '\n'.join(
            [
                'endpoint_kind,endpoint_key,manual_static_ms,comment',
                'source,source:a,12.5,legacy',
                'receiver,receiver:a,-4.0,observer',
            ]
        ),
        encoding='utf-8',
    )

    rows = load_refraction_manual_static_table_rows(path)
    result = _resolve(tuple(rows))

    assert len(rows) == 2
    np.testing.assert_allclose(
        result.source_manual_static_shift_s,
        [0.0125, 0.0],
    )
    np.testing.assert_allclose(
        result.receiver_manual_static_shift_s,
        [-0.004, 0.0],
    )
    assert result.qc['n_manual_source_rows'] == 1
    assert result.qc['n_manual_receiver_rows'] == 1
    assert result.qc['n_matched_source_rows'] == 1
    assert result.qc['n_matched_receiver_rows'] == 1


def test_manual_static_table_converts_applied_shift_s() -> None:
    result = _resolve(
        (
            _row(kind='source', key='source:a', value_s=0.015),
            _row(kind='receiver', key='receiver:a', value_s=-0.006),
        ),
        sign_convention='applied_shift_s',
    )

    np.testing.assert_allclose(result.source_manual_static_shift_s[0], 0.015)
    np.testing.assert_allclose(result.receiver_manual_static_shift_s[0], -0.006)
    assert result.qc['manual_static_sign_convention'] == 'applied_shift_s'


def test_manual_static_table_converts_delay_positive_ms() -> None:
    result = _resolve(
        (
            _row(kind='source', key='source:a', value_s=0.015),
            _row(kind='receiver', key='receiver:a', value_s=-0.006),
        ),
        sign_convention='delay_positive_ms',
    )

    np.testing.assert_allclose(result.source_manual_static_shift_s[0], -0.015)
    np.testing.assert_allclose(result.receiver_manual_static_shift_s[0], 0.006)
    assert result.qc['manual_static_shift_formula'] == (
        'manual_static_shift_s = -manual_static_delay_s'
    )


def test_manual_static_table_rejects_missing_sign_convention() -> None:
    with pytest.raises(ValueError, match='invalid_manual_static_sign_convention'):
        _resolve((_row(kind='source', key='source:a', value_s=0.001),), sign_convention='')


def test_manual_static_table_marks_unmatched_rows() -> None:
    result = _resolve(
        (
            _row(kind='source', key='source:missing', value_s=0.001),
            _row(kind='receiver', endpoint_id=21, value_s=0.002),
        )
    )

    assert result.qc['n_unmatched_rows'] == 1
    assert result.qc['row_status_counts']['unmatched_manual_static_row'] == 1
    np.testing.assert_allclose(result.receiver_manual_static_shift_s, [0.0, 0.002])


def test_manual_static_table_does_not_synthesize_endpoint_ids() -> None:
    result = resolve_refraction_manual_static(
        source_endpoint_key=_keys(['source:a', 'source:b']),
        source_endpoint_id=None,
        source_node_id=_ids([0, 1]),
        receiver_endpoint_key=_keys(['receiver:a', 'receiver:b']),
        receiver_endpoint_id=None,
        receiver_node_id=_ids([2, 3]),
        rows=(_row(kind='source', endpoint_id=0, value_s=0.001),),
        mode='artifact_table',
        sign_convention='applied_shift_s',
    )

    assert result.qc['n_unmatched_rows'] == 1
    assert result.qc['row_status_counts']['unmatched_manual_static_row'] == 1
    np.testing.assert_allclose(result.source_manual_static_shift_s, [0.0, 0.0])


def test_manual_static_missing_allowed_becomes_noop_shift() -> None:
    result = _resolve((_row(kind='source', key='source:a', value_s=0.001),))

    np.testing.assert_allclose(result.source_manual_static_shift_s, [0.001, 0.0])
    np.testing.assert_allclose(result.receiver_manual_static_shift_s, [0.0, 0.0])
    np.testing.assert_array_equal(
        result.source_manual_static_status,
        ['ok', 'missing_manual_static'],
    )
    np.testing.assert_array_equal(
        result.receiver_manual_static_status,
        ['missing_manual_static', 'missing_manual_static'],
    )
    assert result.qc['n_missing_source_endpoints'] == 1
    assert result.qc['n_missing_receiver_endpoints'] == 2


def test_manual_static_table_rejects_duplicate_endpoint_rows() -> None:
    with pytest.raises(ValueError, match='duplicate_manual_static_row'):
        _resolve(
            (
                _row(kind='source', key='source:a', value_s=0.001),
                _row(kind='source', endpoint_id=10, value_s=0.002),
            )
        )


def test_manual_static_table_missing_values_can_be_rejected() -> None:
    with pytest.raises(ValueError, match='missing_manual_static'):
        _resolve(
            (
                _row(kind='source', key='source:a', value_s=0.001),
                _row(kind='source', key='source:b', value_s=0.002),
                _row(kind='receiver', key='receiver:a', value_s=0.003),
            ),
            allow_missing_endpoints=False,
        )


def test_manual_static_table_marks_invalid_value() -> None:
    result = _resolve((_row(kind='source', key='source:a', value_s=np.nan),))

    assert result.source_manual_static_status[0] == 'invalid_manual_static_value'
    assert np.isnan(result.source_manual_static_shift_s[0])
    assert result.qc['n_invalid_manual_static_values'] == 1
