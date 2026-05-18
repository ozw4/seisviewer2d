from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.services.first_break_qc_inputs import FirstBreakQcInputs
from app.services.first_break_qc_math import (
    compute_first_break_qc_metrics,
    compute_pick_time_after_datum,
    compute_residual_by_key1,
    fit_linear_offset_model,
)

DT = 0.004
N_SAMPLES = 512
OFFSET_BYTE = 37


def _base_inputs() -> FirstBreakQcInputs:
    offset = np.asarray([-1000.0, -500.0, 0.0, 500.0, 1000.0, 1500.0])
    datum_shift = np.asarray([0.010, -0.005, 0.0, 0.015, -0.010, 0.005])
    after_datum = 0.100 + 0.0001 * np.abs(offset)
    valid_mask = np.asarray([True, True, False, True, True, True], dtype=bool)
    picks = after_datum - datum_shift
    picks[~valid_mask] = np.nan
    return FirstBreakQcInputs(
        picks_time_s_sorted=np.asarray(picks, dtype=np.float64),
        valid_pick_mask_sorted=valid_mask,
        datum_trace_shift_s_sorted=np.asarray(datum_shift, dtype=np.float64),
        source_elevation_m_sorted=np.asarray(
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            dtype=np.float64,
        ),
        receiver_elevation_m_sorted=np.asarray(
            [150.0, 140.0, 130.0, 120.0, 110.0, 100.0],
            dtype=np.float64,
        ),
        offset_sorted=np.asarray(offset, dtype=np.float64),
        key1_sorted=np.asarray([10, 10, 10, 20, 20, 30], dtype=np.int64),
        key2_sorted=np.asarray([1, 2, 3, 1, 2, 1], dtype=np.int64),
        dt=DT,
        n_traces=6,
        n_samples=N_SAMPLES,
        offset_byte=OFFSET_BYTE,
        source_kind='batch_npz',
        metadata={'source': 'test'},
    )


def _correlation_inputs() -> FirstBreakQcInputs:
    return FirstBreakQcInputs(
        picks_time_s_sorted=np.asarray([1.0, 2.0, np.nan, 4.0], dtype=np.float64),
        valid_pick_mask_sorted=np.asarray([True, True, False, True], dtype=bool),
        datum_trace_shift_s_sorted=np.zeros(4, dtype=np.float64),
        source_elevation_m_sorted=np.asarray([10.0, 20.0, 30.0, 40.0]),
        receiver_elevation_m_sorted=np.asarray([40.0, 30.0, 20.0, 10.0]),
        offset_sorted=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        key1_sorted=np.asarray([1, 1, 2, 2], dtype=np.int64),
        key2_sorted=np.asarray([1, 2, 1, 2], dtype=np.int64),
        dt=DT,
        n_traces=4,
        n_samples=N_SAMPLES,
        offset_byte=OFFSET_BYTE,
        source_kind='manual',
        metadata={},
    )


def test_compute_pick_time_after_datum_adds_datum_shift() -> None:
    after_datum = compute_pick_time_after_datum(
        np.asarray([0.100, 0.200], dtype=np.float64),
        np.asarray([-0.010, 0.030], dtype=np.float64),
        np.asarray([True, True], dtype=bool),
    )

    np.testing.assert_allclose(after_datum, np.asarray([0.090, 0.230]))


def test_compute_pick_time_after_datum_preserves_nan_for_invalid_picks() -> None:
    after_datum = compute_pick_time_after_datum(
        np.asarray([0.100, np.nan], dtype=np.float64),
        np.asarray([0.010, 0.030], dtype=np.float64),
        np.asarray([True, False], dtype=bool),
    )

    np.testing.assert_allclose(after_datum, np.asarray([0.110, np.nan]), equal_nan=True)


def test_first_break_qc_stats_ignore_invalid_nan_picks() -> None:
    metrics = compute_first_break_qc_metrics(_base_inputs())

    assert metrics.raw_pick_stats.n_total == 6
    assert metrics.raw_pick_stats.n_valid == 5
    assert metrics.raw_pick_stats.n_nan == 1
    assert metrics.raw_pick_stats.median_s == pytest.approx(0.190)
    assert metrics.after_datum_pick_stats.n_valid == 5
    assert metrics.after_datum_pick_stats.n_nan == 1
    assert np.isnan(metrics.pick_time_after_datum_s_sorted[2])


def test_first_break_qc_elevation_correlations() -> None:
    metrics = compute_first_break_qc_metrics(_correlation_inputs())

    source_corr = metrics.correlations['raw_pick_vs_source_elevation']
    receiver_corr = metrics.correlations['raw_pick_vs_receiver_elevation']
    assert source_corr.status == 'ok'
    assert source_corr.r == pytest.approx(1.0)
    assert receiver_corr.status == 'ok'
    assert receiver_corr.r == pytest.approx(-1.0)


def test_first_break_qc_after_datum_vs_abs_offset_correlation() -> None:
    metrics = compute_first_break_qc_metrics(_base_inputs())

    corr = metrics.correlations['after_datum_pick_vs_abs_offset']
    assert corr.status == 'ok'
    assert corr.r == pytest.approx(1.0)


def test_first_break_qc_linear_offset_model_recovers_intercept_and_slowness() -> None:
    metrics = compute_first_break_qc_metrics(_base_inputs())

    assert metrics.linear_offset_fit.status == 'ok'
    assert metrics.linear_offset_fit.intercept_s == pytest.approx(0.100)
    assert metrics.linear_offset_fit.slowness_s_per_offset_unit == pytest.approx(0.0001)
    assert metrics.linear_offset_fit.r2 == pytest.approx(1.0)
    expected_model = 0.100 + 0.0001 * np.abs(_base_inputs().offset_sorted)
    expected_model[~_base_inputs().valid_pick_mask_sorted] = np.nan
    np.testing.assert_allclose(
        metrics.linear_moveout_model_s_sorted,
        expected_model,
        equal_nan=True,
    )


def test_first_break_qc_metrics_linear_model_nan_for_invalid_pick() -> None:
    metrics = compute_first_break_qc_metrics(_base_inputs())
    invalid_mask = ~_base_inputs().valid_pick_mask_sorted

    assert np.all(np.isnan(metrics.linear_moveout_model_s_sorted[invalid_mask]))
    assert np.all(np.isnan(metrics.residual_after_datum_s_sorted[invalid_mask]))
    assert np.all(np.isfinite(metrics.linear_moveout_model_s_sorted[~invalid_mask]))
    assert np.all(np.isfinite(metrics.residual_after_datum_s_sorted[~invalid_mask]))


def test_first_break_qc_linear_offset_model_uses_abs_offset() -> None:
    picks = np.asarray([0.7, 0.9, 1.1, 1.3], dtype=np.float64)
    offset = np.asarray([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)

    fit, model, residual = fit_linear_offset_model(
        picks,
        offset,
        np.ones(4, dtype=bool),
    )

    assert fit.status == 'ok'
    assert fit.intercept_s == pytest.approx(0.5)
    assert fit.slowness_s_per_offset_unit == pytest.approx(0.2)
    np.testing.assert_allclose(model, picks)
    np.testing.assert_allclose(residual, np.zeros(4), atol=1e-12)


def test_first_break_qc_residual_after_datum() -> None:
    metrics = compute_first_break_qc_metrics(_base_inputs())
    expected_residual = (
        metrics.pick_time_after_datum_s_sorted - metrics.linear_moveout_model_s_sorted
    )

    np.testing.assert_allclose(
        metrics.residual_after_datum_s_sorted,
        expected_residual,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        metrics.residual_valid_mask_sorted,
        np.asarray([True, True, False, True, True, True], dtype=bool),
    )


def test_first_break_qc_residual_by_key1_median_mad() -> None:
    by_key1 = compute_residual_by_key1(
        np.asarray([0.1, 0.3, np.nan, -0.2, 0.2, np.nan], dtype=np.float64),
        np.asarray([True, True, False, True, True, False], dtype=bool),
        np.asarray([10, 10, 10, 20, 20, 30], dtype=np.int64),
    )

    assert [row.key1 for row in by_key1] == [10, 20, 30]
    assert by_key1[0].n_used == 2
    assert by_key1[0].median_s == pytest.approx(0.2)
    assert by_key1[0].mad_s == pytest.approx(0.1)
    assert by_key1[0].mean_s == pytest.approx(0.2)
    assert by_key1[0].std_s == pytest.approx(0.1)
    assert by_key1[1].median_s == pytest.approx(0.0)
    assert by_key1[1].mad_s == pytest.approx(0.2)
    assert by_key1[2].n_used == 0
    assert by_key1[2].median_s is None
    assert by_key1[2].mad_s is None


def test_first_break_qc_constant_elevation_correlation_is_undefined() -> None:
    inputs = replace(
        _base_inputs(),
        source_elevation_m_sorted=np.ones(6, dtype=np.float64),
    )

    metrics = compute_first_break_qc_metrics(inputs)

    corr = metrics.correlations['raw_pick_vs_source_elevation']
    assert corr.status == 'constant_input'
    assert corr.r is None


def test_first_break_qc_insufficient_correlation_data_is_undefined() -> None:
    inputs = replace(
        _base_inputs(),
        picks_time_s_sorted=np.asarray([0.2, np.nan, np.nan, np.nan, np.nan, np.nan]),
        valid_pick_mask_sorted=np.asarray(
            [True, False, False, False, False, False],
            dtype=bool,
        ),
    )

    metrics = compute_first_break_qc_metrics(inputs)

    corr = metrics.correlations['raw_pick_vs_source_elevation']
    assert corr.status == 'insufficient_data'
    assert corr.r is None
    assert metrics.linear_offset_fit.status == 'insufficient_data'


def test_first_break_qc_constant_abs_offset_model_is_undefined() -> None:
    inputs = replace(
        _base_inputs(),
        offset_sorted=np.asarray([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0]),
    )

    metrics = compute_first_break_qc_metrics(inputs)

    assert metrics.linear_offset_fit.status == 'constant_abs_offset'
    assert metrics.linear_offset_fit.intercept_s is None
    assert np.all(np.isnan(metrics.linear_moveout_model_s_sorted))
    assert np.all(np.isnan(metrics.residual_after_datum_s_sorted))
    assert not np.any(metrics.residual_valid_mask_sorted)


def test_first_break_qc_require_linear_offset_model_rejects_undefined_model() -> None:
    inputs = replace(
        _base_inputs(),
        offset_sorted=np.asarray([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0]),
    )

    with pytest.raises(ValueError, match='constant_abs_offset'):
        compute_first_break_qc_metrics(inputs, require_linear_offset_model=True)


def test_first_break_qc_rejects_shape_mismatch() -> None:
    inputs = replace(
        _base_inputs(),
        source_elevation_m_sorted=np.asarray([1.0, 2.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='shape mismatch'):
        compute_first_break_qc_metrics(inputs)


def test_first_break_qc_rejects_no_valid_picks() -> None:
    inputs = replace(
        _base_inputs(),
        picks_time_s_sorted=np.full(6, np.nan, dtype=np.float64),
        valid_pick_mask_sorted=np.zeros(6, dtype=bool),
    )

    with pytest.raises(ValueError, match='valid pick'):
        compute_first_break_qc_metrics(inputs)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_first_break_qc_rejects_valid_pick_nan_or_inf(bad_value: float) -> None:
    picks = _base_inputs().picks_time_s_sorted.copy()
    picks[0] = bad_value
    inputs = replace(_base_inputs(), picks_time_s_sorted=picks)

    with pytest.raises(ValueError, match='valid picks must be finite|contains inf'):
        compute_first_break_qc_metrics(inputs)


def test_first_break_qc_rejects_invalid_pick_not_nan() -> None:
    picks = _base_inputs().picks_time_s_sorted.copy()
    picks[2] = 0.100
    inputs = replace(_base_inputs(), picks_time_s_sorted=picks)

    with pytest.raises(ValueError, match='invalid picks must be NaN'):
        compute_first_break_qc_metrics(inputs)


def test_first_break_qc_rejects_non_finite_datum_shift() -> None:
    datum_shift = _base_inputs().datum_trace_shift_s_sorted.copy()
    datum_shift[0] = np.inf
    inputs = replace(_base_inputs(), datum_trace_shift_s_sorted=datum_shift)

    with pytest.raises(ValueError, match='datum_trace_shift_s_sorted'):
        compute_first_break_qc_metrics(inputs)


@pytest.mark.parametrize('field', ['source_elevation_m_sorted', 'receiver_elevation_m_sorted'])
def test_first_break_qc_rejects_non_finite_elevation(field: str) -> None:
    values = getattr(_base_inputs(), field).copy()
    values[0] = np.nan
    inputs = replace(_base_inputs(), **{field: values})

    with pytest.raises(ValueError, match=field):
        compute_first_break_qc_metrics(inputs)


def test_first_break_qc_rejects_non_finite_offset() -> None:
    offset = _base_inputs().offset_sorted.copy()
    offset[0] = np.nan
    inputs = replace(_base_inputs(), offset_sorted=offset)

    with pytest.raises(ValueError, match='offset_sorted'):
        compute_first_break_qc_metrics(inputs)


@pytest.mark.parametrize('field', ['key1_sorted', 'key2_sorted'])
def test_first_break_qc_rejects_non_integer_key1_key2(field: str) -> None:
    values = getattr(_base_inputs(), field).astype(np.float64)
    values[0] += 0.5
    inputs = replace(_base_inputs(), **{field: values})

    with pytest.raises(ValueError, match='integer values'):
        compute_first_break_qc_metrics(inputs)
