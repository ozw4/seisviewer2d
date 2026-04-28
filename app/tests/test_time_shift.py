from __future__ import annotations

import numpy as np
import pytest

from app.utils.time_shift import shift_traces_linear


def test_shift_traces_linear_zero_shift_returns_same_values() -> None:
    traces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    shifted = shift_traces_linear(traces, np.array([0.0, 0.0]), dt=0.004)

    np.testing.assert_allclose(shifted, traces)


def test_shift_traces_linear_positive_shift_delays_impulse() -> None:
    traces = np.zeros((1, 110), dtype=np.float32)
    traces[0, 100] = 1.0

    shifted = shift_traces_linear(traces, np.array([0.008]), dt=0.004)

    assert shifted[0, 102] == pytest.approx(1.0)
    assert shifted[0, 100] == pytest.approx(0.0)


def test_shift_traces_linear_negative_shift_advances_impulse() -> None:
    traces = np.zeros((1, 110), dtype=np.float32)
    traces[0, 100] = 1.0

    shifted = shift_traces_linear(traces, np.array([-0.008]), dt=0.004)

    assert shifted[0, 98] == pytest.approx(1.0)
    assert shifted[0, 100] == pytest.approx(0.0)


def test_shift_traces_linear_fractional_shift_uses_linear_interpolation() -> None:
    traces = np.array([[0.0, 10.0, 20.0, 30.0]], dtype=np.float32)

    shifted = shift_traces_linear(
        traces,
        np.array([0.5], dtype=np.float64),
        dt=1.0,
        fill_value=-1.0,
    )

    np.testing.assert_allclose(
        shifted,
        np.array([[-1.0, 5.0, 15.0, 25.0]], dtype=np.float32),
    )


def test_shift_traces_linear_per_trace_shifts() -> None:
    traces = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    shifted = shift_traces_linear(
        traces,
        np.array([1.0, -1.0, 0.5], dtype=np.float64),
        dt=1.0,
    )

    np.testing.assert_allclose(
        shifted,
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )


def test_shift_traces_linear_out_of_range_samples_use_fill_value() -> None:
    traces = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

    shifted = shift_traces_linear(
        traces,
        np.array([1.0], dtype=np.float64),
        dt=1.0,
        fill_value=-5.0,
    )

    np.testing.assert_allclose(
        shifted,
        np.array([[-5.0, 10.0, 20.0]], dtype=np.float32),
    )


def test_shift_traces_linear_returns_float32_c_contiguous_array() -> None:
    traces = np.asfortranarray(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))

    shifted = shift_traces_linear(traces, np.array([0.0]), dt=1.0)

    assert shifted.dtype == np.float32
    assert shifted.flags.c_contiguous


def test_shift_traces_linear_does_not_mutate_input() -> None:
    traces = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    before = traces.copy()

    shift_traces_linear(traces, np.array([1.0]), dt=1.0)

    np.testing.assert_array_equal(traces, before)


def test_shift_traces_linear_rejects_non_2d_traces() -> None:
    with pytest.raises(ValueError):
        shift_traces_linear(np.array([1.0, 2.0], dtype=np.float32), np.array([0.0]), dt=1.0)


def test_shift_traces_linear_rejects_empty_trace_axis() -> None:
    with pytest.raises(ValueError):
        shift_traces_linear(np.empty((0, 2), dtype=np.float32), np.empty((0,)), dt=1.0)


def test_shift_traces_linear_rejects_empty_sample_axis() -> None:
    with pytest.raises(ValueError):
        shift_traces_linear(np.empty((2, 0), dtype=np.float32), np.zeros((2,)), dt=1.0)


def test_shift_traces_linear_rejects_shift_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        shift_traces_linear(np.zeros((2, 3), dtype=np.float32), np.zeros((3,)), dt=1.0)


@pytest.mark.parametrize('dt', [0.0, -1.0, np.nan, np.inf])
def test_shift_traces_linear_rejects_invalid_dt(dt: float) -> None:
    with pytest.raises(ValueError):
        shift_traces_linear(np.zeros((1, 3), dtype=np.float32), np.zeros((1,)), dt=dt)


@pytest.mark.parametrize('shifts_s', [np.array([np.nan]), np.array([np.inf])])
def test_shift_traces_linear_rejects_non_finite_shifts(shifts_s: np.ndarray) -> None:
    with pytest.raises(ValueError):
        shift_traces_linear(np.zeros((1, 3), dtype=np.float32), shifts_s, dt=1.0)


@pytest.mark.parametrize('fill_value', [np.nan, np.inf])
def test_shift_traces_linear_rejects_non_finite_fill_value(fill_value: float) -> None:
    with pytest.raises(ValueError):
        shift_traces_linear(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1,)),
            dt=1.0,
            fill_value=fill_value,
        )
