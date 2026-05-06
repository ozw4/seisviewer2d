from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from app.services.residual_static_robust_solver import (
    ResidualStaticRobustOptions,
    build_residual_static_outlier_mask,
    compute_residual_static_robust_center_scale,
    robust_options_from_request_robust,
    solve_residual_static_robust_least_squares,
    validate_residual_static_robust_options,
)
from app.services.residual_static_sparse_solver import (
    ResidualStaticLsmrOptions,
    ResidualStaticStabilizationOptions,
)
from app.services.residual_static_types import ResidualStaticSolverInputs

SOURCE_DELAY_S = np.asarray([-0.012, -0.004, 0.006, 0.010], dtype=np.float64)
RECEIVER_DELAY_S = np.asarray(
    [-0.006, -0.002, 0.000, 0.003, 0.005],
    dtype=np.float64,
)
INTERCEPT_S = 0.080
SLOWNESS_S_PER_OFFSET_UNIT = 2.0e-5


def _synthetic_inputs(
    *,
    outlier_trace: int | None = None,
    outlier_s: float = 0.080,
    noise_s: np.ndarray | None = None,
    valid_pick_mask: np.ndarray | None = None,
    moveout_model: str = 'linear_abs_offset',
    **overrides: Any,
) -> ResidualStaticSolverInputs:
    source_unique_ids = np.asarray([101, 102, 103, 104], dtype=np.int32)
    receiver_unique_ids = np.asarray([201, 202, 203, 204, 205], dtype=np.int32)
    n_sources = int(source_unique_ids.shape[0])
    n_receivers = int(receiver_unique_ids.shape[0])
    source_index = np.repeat(np.arange(n_sources, dtype=np.int64), n_receivers)
    receiver_index = np.tile(np.arange(n_receivers, dtype=np.int64), n_sources)
    n_traces = int(source_index.shape[0])

    abs_offset = np.asarray(
        [
            [100.0, 185.0, 260.0, 415.0, 520.0],
            [150.0, 245.0, 395.0, 525.0, 610.0],
            [230.0, 315.0, 475.0, 655.0, 730.0],
            [280.0, 360.0, 540.0, 705.0, 820.0],
        ],
        dtype=np.float64,
    ).reshape(-1)
    offset_sign = np.where(np.arange(n_traces) % 2 == 0, -1.0, 1.0)
    offset = abs_offset * offset_sign

    if moveout_model == 'linear_abs_offset':
        moveout_s = INTERCEPT_S + SLOWNESS_S_PER_OFFSET_UNIT * abs_offset
        offset_sorted = offset
        abs_offset_sorted = abs_offset
        offset_byte = 37
    else:
        moveout_s = np.full(n_traces, INTERCEPT_S, dtype=np.float64)
        offset_sorted = None
        abs_offset_sorted = None
        offset_byte = None

    pick_time_after_datum = (
        moveout_s + SOURCE_DELAY_S[source_index] + RECEIVER_DELAY_S[receiver_index]
    )
    if noise_s is not None:
        pick_time_after_datum = pick_time_after_datum + noise_s
    if outlier_trace is not None:
        pick_time_after_datum = pick_time_after_datum.copy()
        pick_time_after_datum[outlier_trace] += outlier_s

    if valid_pick_mask is None:
        valid_mask = np.ones(n_traces, dtype=bool)
    else:
        valid_mask = np.ascontiguousarray(valid_pick_mask, dtype=bool)

    source_valid_pick_counts = np.bincount(
        source_index[valid_mask],
        minlength=n_sources,
    ).astype(np.int64)
    receiver_valid_pick_counts = np.bincount(
        receiver_index[valid_mask],
        minlength=n_receivers,
    ).astype(np.int64)

    payload: dict[str, Any] = {
        'picks_time_s_sorted': pick_time_after_datum.copy(),
        'valid_pick_mask_sorted': valid_mask,
        'pick_time_after_datum_s_sorted': pick_time_after_datum.copy(),
        'datum_trace_shift_s_sorted': np.zeros(n_traces, dtype=np.float64),
        'source_id_sorted': source_unique_ids[source_index],
        'receiver_id_sorted': receiver_unique_ids[receiver_index],
        'source_unique_ids': source_unique_ids,
        'receiver_unique_ids': receiver_unique_ids,
        'source_index_sorted': source_index,
        'receiver_index_sorted': receiver_index,
        'source_valid_pick_counts': source_valid_pick_counts,
        'receiver_valid_pick_counts': receiver_valid_pick_counts,
        'offset_sorted': offset_sorted,
        'abs_offset_sorted': abs_offset_sorted,
        'key1_sorted': np.repeat([10, 20, 30, 40], n_receivers).astype(np.int64),
        'key2_sorted': np.tile([1, 2, 3, 4, 5], n_sources).astype(np.int64),
        'source_elevation_m_sorted': np.zeros(n_traces, dtype=np.float64),
        'receiver_elevation_m_sorted': np.zeros(n_traces, dtype=np.float64),
        'dt': 0.004,
        'n_traces': n_traces,
        'n_samples': 64,
        'key1_byte': 189,
        'key2_byte': 193,
        'source_id_byte': 17,
        'receiver_id_byte': 13,
        'offset_byte': offset_byte,
        'moveout_model': moveout_model,
        'input_file_id': 'corrected-file-id',
        'datum_source_file_id': 'source-file-id',
        'datum_job_id': 'datum-job',
        'pick_source_kind': 'batch_npz',
        'metadata': {'source': 'robust-test'},
    }
    payload.update(overrides)
    return ResidualStaticSolverInputs(**payload)


def _stabilization_options(**overrides: Any) -> ResidualStaticStabilizationOptions:
    payload: dict[str, Any] = {
        'gauge': 'zero_mean_source_receiver',
        'damping_lambda': 0.0,
        'min_valid_picks': 10,
        'min_picks_per_source': 1,
        'min_picks_per_receiver': 1,
        'max_abs_estimated_delay_ms': 250.0,
    }
    payload.update(overrides)
    return ResidualStaticStabilizationOptions(**payload)


def _lsmr_options() -> ResidualStaticLsmrOptions:
    return ResidualStaticLsmrOptions(
        atol=1.0e-12,
        btol=1.0e-12,
        conlim=1.0e12,
        maxiter=10000,
    )


def _noise(n_traces: int) -> np.ndarray:
    return np.asarray(
        [
            -0.0010,
            0.0007,
            -0.0002,
            0.0011,
            -0.0006,
            0.0003,
            -0.0008,
            0.0005,
            0.0010,
            -0.0004,
            0.0009,
            -0.0012,
            0.0004,
            -0.0007,
            0.0002,
            -0.0003,
            0.0012,
            -0.0009,
            0.0006,
            -0.0005,
        ],
        dtype=np.float64,
    )[:n_traces]


def test_validate_robust_options_accepts_defaults() -> None:
    options = validate_residual_static_robust_options(ResidualStaticRobustOptions())

    assert options == ResidualStaticRobustOptions()


def test_validate_robust_options_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match='method'):
        validate_residual_static_robust_options(
            ResidualStaticRobustOptions(method='median')  # type: ignore[arg-type]
        )


def test_validate_robust_options_rejects_bool_max_iterations() -> None:
    with pytest.raises(ValueError, match='max_iterations'):
        validate_residual_static_robust_options(
            ResidualStaticRobustOptions(max_iterations=True)
        )


def test_validate_robust_options_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match='threshold'):
        validate_residual_static_robust_options(
            ResidualStaticRobustOptions(threshold=0.0)
        )


@pytest.mark.parametrize('min_used_fraction', [0.0, 1.1, float('nan')])
def test_validate_robust_options_rejects_invalid_min_used_fraction(
    min_used_fraction: float,
) -> None:
    with pytest.raises(ValueError, match='min_used_fraction'):
        validate_residual_static_robust_options(
            ResidualStaticRobustOptions(min_used_fraction=min_used_fraction)
        )


def test_robust_options_from_request_like_object_without_schema_import() -> None:
    @dataclass(frozen=True)
    class RequestRobust:
        enabled: bool = False
        method: str = 'sigma'
        max_iterations: int = 5
        threshold: float = 3.0
        min_used_fraction: float = 0.75

    options = robust_options_from_request_robust(RequestRobust())

    assert options == ResidualStaticRobustOptions(
        enabled=False,
        method='sigma',
        max_iterations=5,
        threshold=3.0,
        min_used_fraction=0.75,
    )


def test_compute_mad_center_scale() -> None:
    center_s, scale_s = compute_residual_static_robust_center_scale(
        np.asarray([1.0, 2.0, 100.0]),
        method='mad',
    )

    assert center_s == 2.0
    assert scale_s == pytest.approx(1.4826)


def test_compute_sigma_center_scale() -> None:
    residual = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)

    center_s, scale_s = compute_residual_static_robust_center_scale(
        residual,
        method='sigma',
    )

    assert center_s == pytest.approx(float(np.mean(residual)))
    assert scale_s == pytest.approx(float(np.std(residual, ddof=0)))


def test_outlier_mask_mad_rejects_large_centered_residual() -> None:
    outlier_mask, center_s, scale_s, cutoff_s = build_residual_static_outlier_mask(
        np.asarray([0.0, 0.001, -0.001, 0.002, -0.002, 0.100]),
        method='mad',
        threshold=4.0,
    )

    np.testing.assert_array_equal(
        outlier_mask, [False, False, False, False, False, True]
    )
    assert center_s == pytest.approx(0.0005)
    assert scale_s > 0.0
    assert cutoff_s == pytest.approx(4.0 * scale_s)


def test_outlier_mask_sigma_rejects_large_centered_residual() -> None:
    outlier_mask, center_s, scale_s, cutoff_s = build_residual_static_outlier_mask(
        np.asarray([0.0, 0.001, -0.001, 0.002, -0.002, 0.100]),
        method='sigma',
        threshold=2.0,
    )

    np.testing.assert_array_equal(
        outlier_mask, [False, False, False, False, False, True]
    )
    assert center_s == pytest.approx(1.0 / 60.0)
    assert scale_s > 0.0
    assert cutoff_s == pytest.approx(2.0 * scale_s)


def test_outlier_mask_zero_scale_returns_no_outliers() -> None:
    outlier_mask, center_s, scale_s, cutoff_s = build_residual_static_outlier_mask(
        np.asarray([0.0, 0.0, 0.0]),
        method='mad',
        threshold=4.0,
    )

    np.testing.assert_array_equal(outlier_mask, [False, False, False])
    assert center_s == 0.0
    assert scale_s == 0.0
    assert cutoff_s == 0.0


def test_robust_disabled_runs_single_stabilized_solve() -> None:
    inputs = _synthetic_inputs(outlier_trace=0)

    result = solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        robust_options=ResidualStaticRobustOptions(enabled=False),
        lsmr_options=_lsmr_options(),
    )

    assert result.stop_reason == 'disabled'
    assert result.initial_solver_result is result.final_solver_result
    assert result.iteration_summaries == ()
    assert result.n_rejected_total == 0
    np.testing.assert_array_equal(
        result.final_used_mask_sorted, np.ones(20, dtype=bool)
    )


def test_robust_mad_rejects_synthetic_pick_outlier() -> None:
    outlier_trace = 0
    inputs = _synthetic_inputs(outlier_trace=outlier_trace)

    result = solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        robust_options=ResidualStaticRobustOptions(method='mad', threshold=4.0),
        lsmr_options=_lsmr_options(),
    )

    assert result.n_rejected_total == 1
    assert not bool(result.final_used_mask_sorted[outlier_trace])
    assert bool(result.rejected_mask_sorted[outlier_trace])
    assert result.rejected_iteration_sorted[outlier_trace] == 0
    np.testing.assert_array_equal(
        result.rejected_mask_sorted,
        result.initial_used_mask_sorted & ~result.final_used_mask_sorted,
    )
    np.testing.assert_array_equal(
        result.final_solver_result.used_mask_sorted,
        result.final_used_mask_sorted,
    )
    np.testing.assert_allclose(
        result.final_solver_result.parameter_parts.source_delay_s,
        SOURCE_DELAY_S,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        result.final_solver_result.parameter_parts.receiver_delay_s,
        RECEIVER_DELAY_S,
        atol=1.0e-8,
    )


def test_robust_sigma_rejects_synthetic_pick_outlier() -> None:
    outlier_trace = 0
    inputs = _synthetic_inputs(outlier_trace=outlier_trace)

    result = solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        robust_options=ResidualStaticRobustOptions(method='sigma', threshold=2.5),
        lsmr_options=_lsmr_options(),
    )

    assert result.n_rejected_total == 1
    assert not bool(result.final_used_mask_sorted[outlier_trace])
    assert result.stop_reason in {'converged', 'zero_scale'}


def test_robust_no_outliers_converges_without_rejection() -> None:
    inputs = _synthetic_inputs(noise_s=_noise(20))

    result = solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        robust_options=ResidualStaticRobustOptions(threshold=20.0),
        lsmr_options=_lsmr_options(),
    )

    assert result.stop_reason == 'converged'
    assert result.n_rejected_total == 0
    assert result.iteration_summaries[-1].converged is True


def test_robust_honors_initial_used_mask() -> None:
    outlier_trace = 0
    inputs = _synthetic_inputs(outlier_trace=outlier_trace)
    used_mask = np.ones(inputs.n_traces, dtype=bool)
    used_mask[outlier_trace] = False

    result = solve_residual_static_robust_least_squares(
        inputs,
        used_mask_sorted=used_mask,
        stabilization_options=_stabilization_options(),
        robust_options=ResidualStaticRobustOptions(method='mad', threshold=4.0),
        lsmr_options=_lsmr_options(),
    )

    np.testing.assert_array_equal(result.initial_used_mask_sorted, used_mask)
    np.testing.assert_array_equal(result.final_used_mask_sorted, used_mask)
    assert not result.rejected_mask_sorted[outlier_trace]
    assert result.rejected_iteration_sorted[outlier_trace] == -1


def test_robust_rejected_iteration_sorted_records_iteration() -> None:
    outlier_trace = 0
    inputs = _synthetic_inputs(outlier_trace=outlier_trace)

    result = solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        robust_options=ResidualStaticRobustOptions(method='mad', threshold=4.0),
        lsmr_options=_lsmr_options(),
    )

    assert result.rejected_iteration_sorted[outlier_trace] == 0
    assert np.all(result.rejected_iteration_sorted[~result.rejected_mask_sorted] == -1)
    assert result.iteration_summaries[0].iteration_index == 0
    assert result.iteration_summaries[0].n_rejected_this_iteration == 1


def test_robust_min_used_fraction_rejects_over_aggressive_mask() -> None:
    inputs = _synthetic_inputs(outlier_trace=0)

    with pytest.raises(ValueError, match='min_used_fraction'):
        solve_residual_static_robust_least_squares(
            inputs,
            stabilization_options=_stabilization_options(),
            robust_options=ResidualStaticRobustOptions(
                method='mad',
                threshold=4.0,
                min_used_fraction=1.0,
            ),
            lsmr_options=_lsmr_options(),
        )


def test_robust_preserves_zero_mean_gauge_in_final_result() -> None:
    result = solve_residual_static_robust_least_squares(
        _synthetic_inputs(outlier_trace=0),
        stabilization_options=_stabilization_options(),
        robust_options=ResidualStaticRobustOptions(method='mad', threshold=4.0),
        lsmr_options=_lsmr_options(),
    )

    np.testing.assert_allclose(
        np.mean(result.final_solver_result.parameter_parts.source_delay_s),
        0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.mean(result.final_solver_result.parameter_parts.receiver_delay_s),
        0.0,
        atol=1.0e-10,
    )


def test_robust_final_result_honors_estimated_delay_limit() -> None:
    with pytest.raises(ValueError, match='max_abs_estimated_delay_ms'):
        solve_residual_static_robust_least_squares(
            _synthetic_inputs(),
            stabilization_options=_stabilization_options(
                max_abs_estimated_delay_ms=1.0
            ),
            robust_options=ResidualStaticRobustOptions(),
            lsmr_options=_lsmr_options(),
        )


def test_robust_final_used_mask_is_subset_of_valid_pick_mask() -> None:
    valid_mask = np.ones(20, dtype=bool)
    valid_mask[3] = False
    inputs = _synthetic_inputs(
        noise_s=_noise(20),
        valid_pick_mask=valid_mask,
    )
    pick_time = inputs.pick_time_after_datum_s_sorted.copy()
    pick_time[3] = np.nan
    inputs = _synthetic_inputs(
        noise_s=_noise(20),
        valid_pick_mask=valid_mask,
        pick_time_after_datum_s_sorted=pick_time,
        picks_time_s_sorted=pick_time.copy(),
    )

    result = solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        robust_options=ResidualStaticRobustOptions(threshold=20.0),
        lsmr_options=_lsmr_options(),
    )

    assert np.all(result.final_used_mask_sorted <= inputs.valid_pick_mask_sorted)
