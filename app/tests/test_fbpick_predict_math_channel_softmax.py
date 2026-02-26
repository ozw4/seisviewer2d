from __future__ import annotations

import numpy as np

from app.api.schemas import PickOptions
from app.services import fbpick_predict_math
from app.services.batch_apply_service import _predict_section_picks_time_s


def test_normalize_prob_time_tracewise_and_marks_invalid() -> None:
    prob = np.array(
        [
            [1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float16,
    )

    prob_time, valid = fbpick_predict_math.normalize_prob_time(prob, chunk=2)

    assert prob_time.dtype == np.float32
    assert prob_time.flags.c_contiguous
    np.testing.assert_array_equal(valid, np.array([True, False, True], dtype=bool))
    np.testing.assert_allclose(
        np.sum(prob_time[valid], axis=1, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(prob_time[1], np.zeros((3,), dtype=np.float32))


def test_expectation_moments_and_expectation_idx_keep_invalid_as_nan() -> None:
    prob = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    prob_time, _ = fbpick_predict_math.normalize_prob_time(prob, chunk=1)

    mu, var = fbpick_predict_math._expectation_moments(prob_time, chunk=1)  # noqa: SLF001
    idx, sigma_ms = fbpick_predict_math.expectation_idx_and_sigma_ms(
        prob, dt=0.002, chunk=1
    )

    assert np.isnan(mu[0])
    assert np.isnan(var[0])
    assert np.isnan(idx[0])
    assert np.isnan(sigma_ms[0])
    assert float(mu[1]) == 1.0
    assert float(var[1]) == 0.0
    assert float(idx[1]) == 1.0
    assert float(sigma_ms[1]) == 0.0


def test_apply_sigma_gate_rejects_non_finite_sigma() -> None:
    idx = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    sigma_ms = np.array([np.nan, np.inf, 7.0, 50.0], dtype=np.float64)

    gated = fbpick_predict_math.apply_sigma_gate(idx, sigma_ms, sigma_ms_max=10.0)

    assert np.isnan(gated[0])
    assert np.isnan(gated[1])
    assert float(gated[2]) == 4.0
    assert np.isnan(gated[3])


def test_predict_section_picks_time_s_with_zero_mass_trace_continues() -> None:
    prob = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    raw_section = np.zeros_like(prob, dtype=np.float32)
    pick_options = PickOptions(method='argmax', subsample=True, sigma_ms_max=100.0)

    times_s = _predict_section_picks_time_s(
        prob=prob,
        raw_section=raw_section,
        dt=0.002,
        pick_options=pick_options,
        chunk=1,
    )

    assert times_s.shape == (2,)
    assert np.isnan(times_s[0])
    assert np.isfinite(times_s[1])
