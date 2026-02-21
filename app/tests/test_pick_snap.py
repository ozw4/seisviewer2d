from __future__ import annotations

import numpy as np
import pytest

from app.utils.pick_snap import (
    parabolic_refine,
    snap_pick_index,
    snap_pick_time_s,
    zero_cross_refine,
)


def test_parabolic_refine_returns_expected_fractional_index() -> None:
    arr_sym = np.array([0.0, 1.0, 4.0, 1.0, 0.0], dtype=np.float64)
    assert parabolic_refine(arr_sym, 2) == pytest.approx(2.0)

    arr_asym = np.array([0.0, 1.0, 4.0, 3.0, 0.0], dtype=np.float64)
    assert parabolic_refine(arr_asym, 2) == pytest.approx(2.25)


def test_zero_cross_refine_returns_linear_interpolation() -> None:
    arr = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
    assert zero_cross_refine(arr, 1) == pytest.approx(1.5)


def test_snap_pick_index_peak_nearest_local_max_and_fallback() -> None:
    trace_local = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0], dtype=np.float64)
    got_local = snap_pick_index(
        trace_local,
        3.4,
        mode='peak',
        refine='none',
        window_samples=2,
    )
    assert got_local == pytest.approx(3.0)

    trace_fallback = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    got_fallback = snap_pick_index(
        trace_fallback,
        3.0,
        mode='peak',
        refine='none',
        window_samples=2,
    )
    assert got_fallback == pytest.approx(5.0)


def test_snap_pick_index_trough_nearest_local_min_and_fallback() -> None:
    trace_local = np.array([3.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0], dtype=np.float64)
    got_local = snap_pick_index(
        trace_local,
        3.2,
        mode='trough',
        refine='none',
        window_samples=2,
    )
    assert got_local == pytest.approx(3.0)

    trace_fallback = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    got_fallback = snap_pick_index(
        trace_fallback,
        3.0,
        mode='trough',
        refine='none',
        window_samples=2,
    )
    assert got_fallback == pytest.approx(1.0)


def test_snap_pick_index_rise_uses_upcrossing_then_gradient_fallback() -> None:
    trace_cross = np.array([-2.0, -1.0, 1.0, -1.0, 1.0, 2.0], dtype=np.float64)
    got_cross = snap_pick_index(
        trace_cross,
        4.0,
        mode='rise',
        refine='none',
        window_samples=3,
    )
    assert got_cross == pytest.approx(4.0)

    trace_fallback = np.array([0.1, 0.2, 0.4, 0.9, 1.0, 1.1], dtype=np.float64)
    got_fallback = snap_pick_index(
        trace_fallback,
        3.0,
        mode='rise',
        refine='none',
        window_samples=2,
    )
    assert got_fallback == pytest.approx(2.0)


def test_snap_pick_time_s_applies_dt_and_window_ms() -> None:
    trace = np.array([0.0, 1.0, 4.0, 3.0, 0.0], dtype=np.float64)
    got = snap_pick_time_s(
        trace,
        0.004,
        dt=0.002,
        mode='peak',
        refine='parabolic',
        window_ms=6.0,
    )
    assert got == pytest.approx(0.0045)


def test_snap_pick_input_validation_raises_value_error() -> None:
    trace = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError):
        snap_pick_time_s(
            trace,
            0.002,
            dt=0.0,
            mode='peak',
            refine='none',
            window_ms=4.0,
        )

    with pytest.raises(ValueError):
        snap_pick_index(
            trace,
            1.0,
            mode='unknown',
            refine='none',
            window_samples=2,
        )
