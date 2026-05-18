from __future__ import annotations

import numpy as np
import pytest

from app.utils.segy_scalars import (
    apply_segy_scalar,
    count_zero_segy_scalars,
    normalize_elevation_unit,
)


def test_apply_segy_scalar_positive_negative_zero() -> None:
    values = np.array([10, 10, 10], dtype=np.int32)
    scalars = np.array([2, -4, 0], dtype=np.int16)

    scaled = apply_segy_scalar(values, scalars)

    np.testing.assert_allclose(scaled, np.array([20.0, 2.5, 10.0], dtype=np.float64))
    assert scaled.dtype == np.float64


def test_apply_segy_scalar_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        apply_segy_scalar(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1], dtype=np.int16),
        )


@pytest.mark.parametrize(
    'scalars',
    [
        np.array([1.0], dtype=np.float64),
        np.array([1.5], dtype=np.float64),
    ],
)
def test_apply_segy_scalar_rejects_non_integer_scalar(scalars: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_segy_scalar(np.array([1.0], dtype=np.float64), scalars)


def test_apply_segy_scalar_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError):
        apply_segy_scalar(np.array(['not numeric']), np.array([1], dtype=np.int16))


@pytest.mark.parametrize(
    'values',
    [
        np.array([np.nan], dtype=np.float64),
        np.array([np.inf], dtype=np.float64),
    ],
)
def test_apply_segy_scalar_rejects_non_finite_result(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_segy_scalar(values, np.array([1], dtype=np.int16))


def test_count_zero_segy_scalars() -> None:
    assert count_zero_segy_scalars(np.array([0, 1, -1, 0], dtype=np.int16)) == 2


def test_normalize_elevation_unit_m() -> None:
    elevations = normalize_elevation_unit(np.array([1.0, -2.5], dtype=np.float64), 'm')

    np.testing.assert_allclose(elevations, np.array([1.0, -2.5], dtype=np.float64))
    assert elevations.dtype == np.float64


def test_normalize_elevation_unit_ft() -> None:
    elevations = normalize_elevation_unit(np.array([10.0, -5.0], dtype=np.float64), 'ft')

    np.testing.assert_allclose(elevations, np.array([3.048, -1.524], dtype=np.float64))


def test_normalize_elevation_unit_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError):
        normalize_elevation_unit(np.array([1.0], dtype=np.float64), 'km')


@pytest.mark.parametrize(
    'values',
    [
        np.array([np.nan], dtype=np.float64),
        np.array([np.inf], dtype=np.float64),
    ],
)
def test_normalize_elevation_unit_rejects_non_finite_result(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_elevation_unit(values, 'm')
