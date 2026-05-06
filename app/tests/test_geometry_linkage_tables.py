from __future__ import annotations

import numpy as np
import pytest

from app.services.geometry_linkage_tables import build_endpoint_geometry_tables
from app.services.geometry_linkage_validation import GeometryLinkageHeaders


def _headers(
    *,
    source_x: np.ndarray | None = None,
    source_y: np.ndarray | None = None,
    receiver_x: np.ndarray | None = None,
    receiver_y: np.ndarray | None = None,
    coordinate_scalar: np.ndarray | None = None,
) -> GeometryLinkageHeaders:
    if source_x is None:
        source_x = np.array([0.0, 10.0, 0.0], dtype=np.float64)
    if source_y is None:
        source_y = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if receiver_x is None:
        receiver_x = np.array([100.0, 110.0, 100.0], dtype=np.float64)
    if receiver_y is None:
        receiver_y = np.array([200.0, 210.0, 200.0], dtype=np.float64)
    if coordinate_scalar is None:
        coordinate_scalar = np.ones(np.asarray(source_x).shape, dtype=np.int16)
    return GeometryLinkageHeaders(
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        coordinate_scalar=coordinate_scalar,
        checked_bytes=(71, 73, 77, 81, 85),
    )


def test_build_endpoint_geometry_tables_applies_positive_scalar() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([10.0, 20.0], dtype=np.float32),
            source_y=np.array([30.0, 40.0], dtype=np.float32),
            receiver_x=np.array([50.0, 60.0], dtype=np.float32),
            receiver_y=np.array([70.0, 80.0], dtype=np.float32),
            coordinate_scalar=np.array([2, 3], dtype=np.int16),
        )
    )

    np.testing.assert_allclose(tables.source_x_m_sorted, np.array([20.0, 60.0]))
    np.testing.assert_allclose(tables.source_y_m_sorted, np.array([60.0, 120.0]))
    np.testing.assert_allclose(tables.receiver_x_m_sorted, np.array([100.0, 180.0]))
    np.testing.assert_allclose(tables.receiver_y_m_sorted, np.array([140.0, 240.0]))


def test_build_endpoint_geometry_tables_applies_negative_scalar() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([10.0, 20.0], dtype=np.float64),
            source_y=np.array([30.0, 40.0], dtype=np.float64),
            receiver_x=np.array([50.0, 60.0], dtype=np.float64),
            receiver_y=np.array([70.0, 80.0], dtype=np.float64),
            coordinate_scalar=np.array([-2, -4], dtype=np.int16),
        )
    )

    np.testing.assert_allclose(tables.source_x_m_sorted, np.array([5.0, 5.0]))
    np.testing.assert_allclose(tables.source_y_m_sorted, np.array([15.0, 10.0]))
    np.testing.assert_allclose(tables.receiver_x_m_sorted, np.array([25.0, 15.0]))
    np.testing.assert_allclose(tables.receiver_y_m_sorted, np.array([35.0, 20.0]))


def test_build_endpoint_geometry_tables_treats_zero_scalar_as_one() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([10.0], dtype=np.float64),
            source_y=np.array([20.0], dtype=np.float64),
            receiver_x=np.array([30.0], dtype=np.float64),
            receiver_y=np.array([40.0], dtype=np.float64),
            coordinate_scalar=np.array([0], dtype=np.int16),
        )
    )

    np.testing.assert_allclose(tables.source_x_m_sorted, np.array([10.0]))
    np.testing.assert_allclose(tables.source_y_m_sorted, np.array([20.0]))
    np.testing.assert_allclose(tables.receiver_x_m_sorted, np.array([30.0]))
    np.testing.assert_allclose(tables.receiver_y_m_sorted, np.array([40.0]))


def test_build_endpoint_geometry_tables_counts_zero_scalars() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(coordinate_scalar=np.array([0, 1, 0], dtype=np.int16))
    )

    assert tables.scalar_zero_count == 2


def test_build_endpoint_geometry_tables_builds_unique_source_endpoints() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([2.0, 1.0, 2.0, 1.0], dtype=np.float64),
            source_y=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            receiver_x=np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
            receiver_y=np.array([20.0, 21.0, 22.0, 23.0], dtype=np.float64),
            coordinate_scalar=np.ones(4, dtype=np.int16),
        )
    )

    assert tables.source_endpoints.endpoint_kind == 'source'
    np.testing.assert_array_equal(tables.source_endpoints.endpoint_id, [0, 1, 2])
    np.testing.assert_allclose(tables.source_endpoints.x_m, [1.0, 1.0, 2.0])
    np.testing.assert_allclose(tables.source_endpoints.y_m, [0.0, 1.0, 0.0])


def test_build_endpoint_geometry_tables_builds_unique_receiver_endpoints() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
            source_y=np.array([20.0, 21.0, 22.0, 23.0], dtype=np.float64),
            receiver_x=np.array([3.0, 3.0, 2.0, 2.0], dtype=np.float64),
            receiver_y=np.array([5.0, 5.0, 4.0, 6.0], dtype=np.float64),
            coordinate_scalar=np.ones(4, dtype=np.int16),
        )
    )

    assert tables.receiver_endpoints.endpoint_kind == 'receiver'
    np.testing.assert_array_equal(tables.receiver_endpoints.endpoint_id, [0, 1, 2])
    np.testing.assert_allclose(tables.receiver_endpoints.x_m, [2.0, 2.0, 3.0])
    np.testing.assert_allclose(tables.receiver_endpoints.y_m, [4.0, 6.0, 5.0])


def test_build_endpoint_geometry_tables_keeps_source_receiver_tables_separate() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([10.0, 10.0], dtype=np.float64),
            source_y=np.array([20.0, 20.0], dtype=np.float64),
            receiver_x=np.array([10.0, 10.0], dtype=np.float64),
            receiver_y=np.array([20.0, 20.0], dtype=np.float64),
            coordinate_scalar=np.ones(2, dtype=np.int16),
        )
    )

    np.testing.assert_array_equal(tables.source_endpoints.endpoint_id, [0])
    np.testing.assert_array_equal(tables.receiver_endpoints.endpoint_id, [0])
    assert tables.source_endpoints.endpoint_kind == 'source'
    assert tables.receiver_endpoints.endpoint_kind == 'receiver'


def test_build_endpoint_geometry_tables_returns_sorted_trace_mappings() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([2.0, 1.0, 2.0, 1.0], dtype=np.float64),
            source_y=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            receiver_x=np.array([3.0, 3.0, 2.0, 2.0], dtype=np.float64),
            receiver_y=np.array([5.0, 5.0, 4.0, 6.0], dtype=np.float64),
            coordinate_scalar=np.ones(4, dtype=np.int16),
        )
    )

    assert tables.source_endpoint_id_sorted.shape == (4,)
    assert tables.receiver_endpoint_id_sorted.shape == (4,)
    np.testing.assert_array_equal(tables.source_endpoint_id_sorted, [2, 0, 2, 1])
    np.testing.assert_array_equal(tables.receiver_endpoint_id_sorted, [2, 2, 0, 1])


def test_build_endpoint_geometry_tables_endpoint_ids_are_xy_sorted() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([5.0, 1.0, 1.0, 3.0], dtype=np.float64),
            source_y=np.array([0.0, 9.0, 2.0, 4.0], dtype=np.float64),
            receiver_x=np.array([8.0, 7.0, 8.0, 6.0], dtype=np.float64),
            receiver_y=np.array([1.0, 0.0, 0.0, 9.0], dtype=np.float64),
            coordinate_scalar=np.ones(4, dtype=np.int16),
        )
    )

    np.testing.assert_array_equal(tables.source_endpoints.endpoint_id, [0, 1, 2, 3])
    np.testing.assert_allclose(tables.source_endpoints.x_m, [1.0, 1.0, 3.0, 5.0])
    np.testing.assert_allclose(tables.source_endpoints.y_m, [2.0, 9.0, 4.0, 0.0])
    np.testing.assert_array_equal(tables.receiver_endpoints.endpoint_id, [0, 1, 2, 3])
    np.testing.assert_allclose(tables.receiver_endpoints.x_m, [6.0, 7.0, 8.0, 8.0])
    np.testing.assert_allclose(tables.receiver_endpoints.y_m, [9.0, 0.0, 0.0, 1.0])


def test_build_endpoint_geometry_tables_first_sorted_trace_index_and_count() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(
            source_x=np.array([2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64),
            source_y=np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            receiver_x=np.array([3.0, 3.0, 2.0, 2.0, 3.0], dtype=np.float64),
            receiver_y=np.array([5.0, 5.0, 4.0, 6.0, 5.0], dtype=np.float64),
            coordinate_scalar=np.ones(5, dtype=np.int16),
        )
    )

    np.testing.assert_array_equal(
        tables.source_endpoints.first_sorted_trace_index,
        [1, 3, 0],
    )
    np.testing.assert_array_equal(tables.source_endpoints.trace_count, [1, 1, 3])
    np.testing.assert_array_equal(
        tables.receiver_endpoints.first_sorted_trace_index,
        [2, 3, 0],
    )
    np.testing.assert_array_equal(tables.receiver_endpoints.trace_count, [1, 1, 3])


def test_build_endpoint_geometry_tables_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='shape mismatch'):
        build_endpoint_geometry_tables(
            _headers(
                source_x=np.array([1.0, 2.0], dtype=np.float64),
                source_y=np.array([1.0], dtype=np.float64),
                receiver_x=np.array([1.0, 2.0], dtype=np.float64),
                receiver_y=np.array([1.0, 2.0], dtype=np.float64),
                coordinate_scalar=np.ones(2, dtype=np.int16),
            )
        )


def test_build_endpoint_geometry_tables_rejects_empty_input() -> None:
    empty = np.array([], dtype=np.float64)

    with pytest.raises(ValueError, match='at least one trace'):
        build_endpoint_geometry_tables(
            _headers(
                source_x=empty,
                source_y=empty,
                receiver_x=empty,
                receiver_y=empty,
                coordinate_scalar=np.array([], dtype=np.int16),
            )
        )


def test_build_endpoint_geometry_tables_rejects_non_finite_coordinate() -> None:
    with pytest.raises(ValueError, match='source_x.*finite values'):
        build_endpoint_geometry_tables(
            _headers(source_x=np.array([1.0, np.nan, 3.0], dtype=np.float64))
        )


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_build_endpoint_geometry_tables_rejects_non_finite_scalar(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match='coordinate_scalar.*finite values'):
        build_endpoint_geometry_tables(
            _headers(
                coordinate_scalar=np.array([1.0, bad_value, 3.0], dtype=np.float64)
            )
        )


def test_build_endpoint_geometry_tables_rejects_non_integer_scalar() -> None:
    with pytest.raises(ValueError, match='integer values'):
        build_endpoint_geometry_tables(
            _headers(coordinate_scalar=np.array([1.0, 2.5, 3.0], dtype=np.float64))
        )


def test_build_endpoint_geometry_tables_accepts_integer_valued_float_scalar() -> None:
    tables = build_endpoint_geometry_tables(
        _headers(coordinate_scalar=np.array([1.0, -2.0, 0.0], dtype=np.float64))
    )

    np.testing.assert_array_equal(tables.coordinate_scalar_sorted, [1, -2, 0])
    assert tables.coordinate_scalar_sorted.dtype == np.int64
    np.testing.assert_allclose(tables.source_x_m_sorted, [0.0, 5.0, 0.0])
