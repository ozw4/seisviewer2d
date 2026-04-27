from __future__ import annotations

import numpy as np
import pytest

from app.services.linear_moveout import compute_lmo_shift_seconds


def test_absolute_offset_mode_uses_scaled_absolute_offsets() -> None:
    offsets = np.array([-100.0, 50.0, -25.0], dtype=np.float32)

    shifts = compute_lmo_shift_seconds(
        offsets,
        velocity_mps=1000.0,
        offset_scale=2.0,
        offset_mode='absolute',
        ref_mode='zero',
        ref_trace=None,
        polarity=1,
    )

    np.testing.assert_allclose(shifts, np.array([0.2, 0.1, 0.05], dtype=np.float64))
    assert shifts.dtype == np.float64


def test_signed_offset_mode_keeps_signed_offsets() -> None:
    offsets = np.array([-100.0, 50.0], dtype=np.float64)

    shifts = compute_lmo_shift_seconds(
        offsets,
        velocity_mps=1000.0,
        offset_scale=1.0,
        offset_mode='signed',
        ref_mode='zero',
        ref_trace=None,
        polarity=1,
    )

    np.testing.assert_allclose(shifts, np.array([-0.1, 0.05], dtype=np.float64))


@pytest.mark.parametrize(
    ('ref_mode', 'ref_trace', 'expected'),
    [
        ('min', None, [0.0, 1.0, 2.0, 3.0]),
        ('first', None, [0.0, 1.0, 2.0, 3.0]),
        ('center', None, [-2.0, -1.0, 0.0, 1.0]),
        ('trace', 3, [-3.0, -2.0, -1.0, 0.0]),
        ('zero', None, [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_reference_modes(ref_mode: str, ref_trace: int | None, expected: list[float]) -> None:
    offsets = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

    shifts = compute_lmo_shift_seconds(
        offsets,
        velocity_mps=10.0,
        offset_scale=1.0,
        offset_mode='signed',
        ref_mode=ref_mode,
        ref_trace=ref_trace,
        polarity=1,
    )

    np.testing.assert_allclose(shifts, np.asarray(expected, dtype=np.float64))


def test_polarity_minus_one_reverses_shift_sign() -> None:
    offsets = np.array([0.0, 10.0, 20.0], dtype=np.float64)

    shifts = compute_lmo_shift_seconds(
        offsets,
        velocity_mps=10.0,
        offset_scale=1.0,
        offset_mode='signed',
        ref_mode='zero',
        ref_trace=None,
        polarity=-1,
    )

    np.testing.assert_allclose(shifts, np.array([0.0, -1.0, -2.0], dtype=np.float64))


def test_velocity_scales_shift_inversely() -> None:
    offsets = np.array([0.0, 10.0, 20.0], dtype=np.float64)

    slow = compute_lmo_shift_seconds(
        offsets,
        velocity_mps=10.0,
        offset_scale=1.0,
        offset_mode='signed',
        ref_mode='zero',
        ref_trace=None,
        polarity=1,
    )
    fast = compute_lmo_shift_seconds(
        offsets,
        velocity_mps=20.0,
        offset_scale=1.0,
        offset_mode='signed',
        ref_mode='zero',
        ref_trace=None,
        polarity=1,
    )

    np.testing.assert_allclose(fast, slow / 2.0)


@pytest.mark.parametrize(
    'offsets',
    [
        np.array([[1.0, 2.0]], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([1.0, np.nan], dtype=np.float64),
        np.array([1.0, np.inf], dtype=np.float64),
    ],
)
def test_invalid_offsets_raise_value_error(offsets: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compute_lmo_shift_seconds(
            offsets,
            velocity_mps=1000.0,
            offset_scale=1.0,
            offset_mode='signed',
            ref_mode='zero',
            ref_trace=None,
            polarity=1,
        )


@pytest.mark.parametrize('velocity_mps', [0.0, -1.0, np.nan, np.inf])
def test_invalid_velocity_raises_value_error(velocity_mps: float) -> None:
    with pytest.raises(ValueError):
        compute_lmo_shift_seconds(
            np.array([1.0], dtype=np.float64),
            velocity_mps=velocity_mps,
            offset_scale=1.0,
            offset_mode='signed',
            ref_mode='zero',
            ref_trace=None,
            polarity=1,
        )


@pytest.mark.parametrize('offset_scale', [0.0, np.nan, np.inf])
def test_invalid_offset_scale_raises_value_error(offset_scale: float) -> None:
    with pytest.raises(ValueError):
        compute_lmo_shift_seconds(
            np.array([1.0], dtype=np.float64),
            velocity_mps=1000.0,
            offset_scale=offset_scale,
            offset_mode='signed',
            ref_mode='zero',
            ref_trace=None,
            polarity=1,
        )


def test_invalid_offset_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        compute_lmo_shift_seconds(
            np.array([1.0], dtype=np.float64),
            velocity_mps=1000.0,
            offset_scale=1.0,
            offset_mode='invalid',
            ref_mode='zero',
            ref_trace=None,
            polarity=1,
        )


def test_invalid_ref_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        compute_lmo_shift_seconds(
            np.array([1.0], dtype=np.float64),
            velocity_mps=1000.0,
            offset_scale=1.0,
            offset_mode='signed',
            ref_mode='invalid',
            ref_trace=None,
            polarity=1,
        )


@pytest.mark.parametrize('ref_trace', [None, -1, 2])
def test_invalid_ref_trace_raises_value_error(ref_trace: int | None) -> None:
    with pytest.raises(ValueError):
        compute_lmo_shift_seconds(
            np.array([1.0, 2.0], dtype=np.float64),
            velocity_mps=1000.0,
            offset_scale=1.0,
            offset_mode='signed',
            ref_mode='trace',
            ref_trace=ref_trace,
            polarity=1,
        )


@pytest.mark.parametrize('polarity', [0, 2, -2, True])
def test_invalid_polarity_raises_value_error(polarity: int) -> None:
    with pytest.raises(ValueError):
        compute_lmo_shift_seconds(
            np.array([1.0], dtype=np.float64),
            velocity_mps=1000.0,
            offset_scale=1.0,
            offset_mode='signed',
            ref_mode='zero',
            ref_trace=None,
            polarity=polarity,
        )
