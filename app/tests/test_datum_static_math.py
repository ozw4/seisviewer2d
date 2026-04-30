from __future__ import annotations

import numpy as np
import pytest

from app.services.datum_static_math import compute_datum_static_shifts


def test_datum_static_math_without_source_depth() -> None:
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=np.array([100.0, 50.0], dtype=np.float64),
        receiver_elevation_m_sorted=np.array([80.0, -20.0], dtype=np.float64),
        datum_elevation_m=0.0,
        replacement_velocity_m_s=1000.0,
    )

    np.testing.assert_allclose(result.source_depth_m_sorted, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(result.source_depth_used_sorted, np.array([False, False]))
    np.testing.assert_allclose(result.source_elevation_m_sorted, np.array([100.0, 50.0]))
    np.testing.assert_allclose(result.source_shift_s_sorted, np.array([-0.1, -0.05]))
    np.testing.assert_allclose(result.receiver_shift_s_sorted, np.array([-0.08, 0.02]))
    np.testing.assert_allclose(result.trace_shift_s_sorted, np.array([-0.18, -0.03]))


def test_datum_static_math_with_source_depth() -> None:
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=np.array([100.0, 50.0], dtype=np.float64),
        source_depth_m_sorted=np.array([10.0, 5.0], dtype=np.float64),
        receiver_elevation_m_sorted=np.array([80.0, -20.0], dtype=np.float64),
        datum_elevation_m=0.0,
        replacement_velocity_m_s=1000.0,
    )

    np.testing.assert_allclose(result.source_depth_m_sorted, np.array([10.0, 5.0]))
    np.testing.assert_array_equal(result.source_depth_used_sorted, np.array([True, True]))
    np.testing.assert_allclose(result.source_elevation_m_sorted, np.array([90.0, 45.0]))
    np.testing.assert_allclose(result.source_shift_s_sorted, np.array([-0.09, -0.045]))
    np.testing.assert_allclose(result.receiver_shift_s_sorted, np.array([-0.08, 0.02]))
    np.testing.assert_allclose(result.trace_shift_s_sorted, np.array([-0.17, -0.025]))


def test_datum_static_sign_canonical_case() -> None:
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=np.array([100.0], dtype=np.float64),
        source_depth_m_sorted=np.array([0.0], dtype=np.float64),
        receiver_elevation_m_sorted=np.array([100.0], dtype=np.float64),
        datum_elevation_m=0.0,
        replacement_velocity_m_s=2000.0,
    )

    np.testing.assert_allclose(result.source_shift_s_sorted, np.array([-0.05]))
    np.testing.assert_allclose(result.receiver_shift_s_sorted, np.array([-0.05]))
    np.testing.assert_allclose(result.trace_shift_s_sorted, np.array([-0.10]))


@pytest.mark.parametrize(
    'array_name',
    [
        'source_shift_s_sorted',
        'receiver_shift_s_sorted',
        'trace_shift_s_sorted',
        'source_surface_elevation_m_sorted',
        'source_depth_m_sorted',
        'source_depth_used_sorted',
        'source_elevation_m_sorted',
        'receiver_elevation_m_sorted',
    ],
)
def test_datum_static_outputs_are_sorted_trace_order_1d_arrays(array_name: str) -> None:
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=np.array([1.0, 2.0], dtype=np.float64),
        receiver_elevation_m_sorted=np.array([3.0, 4.0], dtype=np.float64),
        datum_elevation_m=0.0,
        replacement_velocity_m_s=1000.0,
    )

    arr = getattr(result, array_name)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.shape == (2,)


def test_datum_static_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_datum_static_shifts(
            source_surface_elevation_m_sorted=np.array([1.0, 2.0], dtype=np.float64),
            receiver_elevation_m_sorted=np.array([1.0], dtype=np.float64),
            datum_elevation_m=0.0,
            replacement_velocity_m_s=1000.0,
        )


def test_datum_static_rejects_source_depth_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_datum_static_shifts(
            source_surface_elevation_m_sorted=np.array([1.0, 2.0], dtype=np.float64),
            source_depth_m_sorted=np.array([1.0], dtype=np.float64),
            receiver_elevation_m_sorted=np.array([1.0, 2.0], dtype=np.float64),
            datum_elevation_m=0.0,
            replacement_velocity_m_s=1000.0,
        )


def test_datum_static_rejects_non_1d_elevations() -> None:
    with pytest.raises(ValueError):
        compute_datum_static_shifts(
            source_surface_elevation_m_sorted=np.array([[1.0, 2.0]], dtype=np.float64),
            receiver_elevation_m_sorted=np.array([[1.0, 2.0]], dtype=np.float64),
            datum_elevation_m=0.0,
            replacement_velocity_m_s=1000.0,
        )


@pytest.mark.parametrize('replacement_velocity_m_s', [0.0, -1.0, np.nan, np.inf])
def test_datum_static_rejects_non_positive_replacement_velocity(
    replacement_velocity_m_s: float,
) -> None:
    with pytest.raises(ValueError):
        compute_datum_static_shifts(
            source_surface_elevation_m_sorted=np.array([1.0], dtype=np.float64),
            receiver_elevation_m_sorted=np.array([1.0], dtype=np.float64),
            datum_elevation_m=0.0,
            replacement_velocity_m_s=replacement_velocity_m_s,
        )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('source_surface_elevation_m_sorted', np.nan),
        ('source_surface_elevation_m_sorted', np.inf),
        ('receiver_elevation_m_sorted', np.nan),
        ('receiver_elevation_m_sorted', np.inf),
        ('source_depth_m_sorted', np.nan),
        ('source_depth_m_sorted', np.inf),
    ],
)
def test_datum_static_rejects_non_finite_elevation_or_depth(
    field: str,
    value: float,
) -> None:
    kwargs = {
        'source_surface_elevation_m_sorted': np.array([1.0], dtype=np.float64),
        'receiver_elevation_m_sorted': np.array([1.0], dtype=np.float64),
        'source_depth_m_sorted': np.array([1.0], dtype=np.float64),
        'datum_elevation_m': 0.0,
        'replacement_velocity_m_s': 1000.0,
    }
    kwargs[field] = np.array([value], dtype=np.float64)

    with pytest.raises(ValueError):
        compute_datum_static_shifts(**kwargs)


@pytest.mark.parametrize('datum_elevation_m', [np.nan, np.inf])
def test_datum_static_rejects_non_finite_datum(datum_elevation_m: float) -> None:
    with pytest.raises(ValueError):
        compute_datum_static_shifts(
            source_surface_elevation_m_sorted=np.array([1.0], dtype=np.float64),
            receiver_elevation_m_sorted=np.array([1.0], dtype=np.float64),
            datum_elevation_m=datum_elevation_m,
            replacement_velocity_m_s=1000.0,
        )
