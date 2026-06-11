from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy import sparse

import app.services.residual_static_design_matrix as design_matrix
import seis_statics.residual.solver as package_sparse_solver
from app.services.residual_static_design_matrix import (
    ResidualStaticModelEvaluation,
    build_residual_static_observation_matrix_triplets,
)
from app.services.residual_static_sparse_solver import (
    ResidualStaticLsmrDiagnostics,
    ResidualStaticLsmrOptions,
    build_csr_matrix_from_residual_static_triplets,
    run_sparse_lsmr,
    solve_residual_static_least_squares,
    validate_lsmr_options,
)
from app.services.residual_static_types import ResidualStaticSolverInputs

PICK_TIME_AFTER_DATUM = np.asarray([0.100, 0.110, 0.120, 0.130])
VALID_PICK_MASK = np.asarray([True, True, True, True])
SOURCE_UNIQUE_IDS = np.asarray([10, 20], dtype=np.int64)
RECEIVER_UNIQUE_IDS = np.asarray([1, 2], dtype=np.int64)
SOURCE_INDEX = np.asarray([0, 0, 1, 1], dtype=np.int64)
RECEIVER_INDEX = np.asarray([0, 1, 0, 1], dtype=np.int64)
ABS_OFFSET = np.asarray([100.0, 200.0, 100.0, 200.0])
OFFSET = np.asarray([-100.0, 200.0, -100.0, 200.0])
N_TRACES = 4


def _inputs(**overrides: Any) -> ResidualStaticSolverInputs:
    payload: dict[str, Any] = {
        'picks_time_s_sorted': PICK_TIME_AFTER_DATUM.copy(),
        'valid_pick_mask_sorted': VALID_PICK_MASK.copy(),
        'pick_time_after_datum_s_sorted': PICK_TIME_AFTER_DATUM.copy(),
        'datum_trace_shift_s_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'source_id_sorted': np.asarray([10, 10, 20, 20], dtype=np.int64),
        'receiver_id_sorted': np.asarray([1, 2, 1, 2], dtype=np.int64),
        'source_unique_ids': SOURCE_UNIQUE_IDS.copy(),
        'receiver_unique_ids': RECEIVER_UNIQUE_IDS.copy(),
        'source_index_sorted': SOURCE_INDEX.copy(),
        'receiver_index_sorted': RECEIVER_INDEX.copy(),
        'source_valid_pick_counts': np.asarray([2, 2], dtype=np.int64),
        'receiver_valid_pick_counts': np.asarray([2, 2], dtype=np.int64),
        'offset_sorted': OFFSET.copy(),
        'abs_offset_sorted': ABS_OFFSET.copy(),
        'key1_sorted': np.asarray([100, 100, 200, 200], dtype=np.int64),
        'key2_sorted': np.asarray([1, 2, 1, 2], dtype=np.int64),
        'source_elevation_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'receiver_elevation_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'dt': 0.004,
        'n_traces': N_TRACES,
        'n_samples': 64,
        'key1_byte': 189,
        'key2_byte': 193,
        'source_id_byte': 17,
        'receiver_id_byte': 13,
        'offset_byte': 37,
        'moveout_model': 'linear_abs_offset',
        'input_file_id': 'corrected-file-id',
        'datum_source_file_id': 'source-file-id',
        'datum_job_id': 'datum-job',
        'pick_source_kind': 'batch_npz',
        'metadata': {'source': 'test'},
    }
    payload.update(overrides)
    return ResidualStaticSolverInputs(**payload)


def _moveout_none_inputs(**overrides: Any) -> ResidualStaticSolverInputs:
    payload: dict[str, Any] = {
        'moveout_model': 'none',
        'offset_byte': None,
        'offset_sorted': None,
        'abs_offset_sorted': None,
    }
    payload.update(overrides)
    return _inputs(**payload)


def _dense_triplet_matrix(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    triplets = build_residual_static_observation_matrix_triplets(
        inputs,
        used_mask_sorted=used_mask_sorted,
    )
    matrix = build_csr_matrix_from_residual_static_triplets(triplets)
    return matrix.toarray(), triplets.rhs_s


def _assert_model_reconstructs_used_picks(
    result,
    inputs: ResidualStaticSolverInputs,
    *,
    atol: float = 1.0e-10,
) -> None:
    used_mask = result.used_mask_sorted
    np.testing.assert_allclose(
        result.model_evaluation.modeled_pick_time_s_sorted[used_mask],
        inputs.pick_time_after_datum_s_sorted[used_mask],
        atol=atol,
    )
    np.testing.assert_allclose(
        result.model_evaluation.residual_s_sorted[used_mask],
        np.zeros(int(np.count_nonzero(used_mask)), dtype=np.float64),
        atol=atol,
    )


def test_scipy_is_available() -> None:
    assert sparse.csr_matrix([[1.0]]).shape == (1, 1)


def test_design_matrix_module_still_does_not_import_scipy() -> None:
    assert 'scipy' not in design_matrix.__dict__
    assert 'scipy' not in design_matrix.__loader__.get_source(  # type: ignore[union-attr]
        design_matrix.__name__
    )


def test_package_sparse_solver_module_imports_scipy_only_in_solver_layer() -> None:
    solver_source = package_sparse_solver.__loader__.get_source(  # type: ignore[union-attr]
        package_sparse_solver.__name__
    )

    assert 'scipy' in solver_source
    assert 'scipy' not in design_matrix.__dict__


def test_validate_lsmr_options_accepts_defaults() -> None:
    options = validate_lsmr_options(ResidualStaticLsmrOptions())

    assert options == ResidualStaticLsmrOptions()


def test_validate_lsmr_options_rejects_negative_atol() -> None:
    with pytest.raises(ValueError, match='atol'):
        validate_lsmr_options(ResidualStaticLsmrOptions(atol=-1.0))


def test_validate_lsmr_options_rejects_negative_btol() -> None:
    with pytest.raises(ValueError, match='btol'):
        validate_lsmr_options(ResidualStaticLsmrOptions(btol=-1.0))


def test_validate_lsmr_options_rejects_non_positive_conlim() -> None:
    with pytest.raises(ValueError, match='conlim'):
        validate_lsmr_options(ResidualStaticLsmrOptions(conlim=0.0))


def test_validate_lsmr_options_rejects_non_positive_maxiter() -> None:
    with pytest.raises(ValueError, match='maxiter'):
        validate_lsmr_options(ResidualStaticLsmrOptions(maxiter=0))


def test_validate_lsmr_options_rejects_bool_maxiter() -> None:
    with pytest.raises(ValueError, match='maxiter'):
        validate_lsmr_options(ResidualStaticLsmrOptions(maxiter=True))


def test_build_csr_matrix_from_triplets_linear_abs_offset() -> None:
    triplets = build_residual_static_observation_matrix_triplets(_inputs())

    matrix = build_csr_matrix_from_residual_static_triplets(triplets)

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(
        matrix.toarray(),
        [
            [1.0, 100.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 200.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 100.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 200.0, 0.0, 1.0, 0.0, 1.0],
        ],
    )


def test_build_csr_matrix_from_triplets_moveout_none() -> None:
    triplets = build_residual_static_observation_matrix_triplets(
        _moveout_none_inputs()
    )

    matrix = build_csr_matrix_from_residual_static_triplets(triplets)

    np.testing.assert_allclose(
        matrix.toarray(),
        [
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
        ],
    )


def test_build_csr_matrix_rejects_out_of_range_row_index() -> None:
    triplets = build_residual_static_observation_matrix_triplets(_inputs())
    row_indices = triplets.row_indices.copy()
    row_indices[0] = triplets.n_rows

    with pytest.raises(ValueError, match='row_indices'):
        build_csr_matrix_from_residual_static_triplets(
            replace(triplets, row_indices=row_indices)
        )


def test_build_csr_matrix_rejects_out_of_range_col_index() -> None:
    triplets = build_residual_static_observation_matrix_triplets(_inputs())
    col_indices = triplets.col_indices.copy()
    col_indices[0] = triplets.n_cols

    with pytest.raises(ValueError, match='col_indices'):
        build_csr_matrix_from_residual_static_triplets(
            replace(triplets, col_indices=col_indices)
        )


def test_build_csr_matrix_rejects_nonfinite_data() -> None:
    triplets = build_residual_static_observation_matrix_triplets(_inputs())
    data = triplets.data.copy()
    data[0] = np.nan

    with pytest.raises(ValueError, match='data'):
        build_csr_matrix_from_residual_static_triplets(replace(triplets, data=data))


def test_build_csr_matrix_rejects_nonfinite_rhs() -> None:
    triplets = build_residual_static_observation_matrix_triplets(_inputs())
    rhs_s = triplets.rhs_s.copy()
    rhs_s[0] = np.inf

    with pytest.raises(ValueError, match='rhs_s'):
        build_csr_matrix_from_residual_static_triplets(replace(triplets, rhs_s=rhs_s))


def test_build_csr_matrix_rejects_bool_indices() -> None:
    triplets = build_residual_static_observation_matrix_triplets(_inputs())

    with pytest.raises(ValueError, match='row_indices'):
        build_csr_matrix_from_residual_static_triplets(
            replace(triplets, row_indices=triplets.row_indices.astype(bool))
        )


def test_run_sparse_lsmr_solves_simple_overdetermined_system() -> None:
    matrix = sparse.csr_matrix(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    rhs_s = np.asarray([1.0, 2.0, 3.0])

    result = run_sparse_lsmr(matrix, rhs_s)

    np.testing.assert_allclose(result.parameter_vector, [1.0, 2.0], atol=1.0e-10)


def test_run_sparse_lsmr_solves_underdetermined_system() -> None:
    matrix = sparse.csr_matrix([[1.0, 1.0]], dtype=np.float64)

    result = run_sparse_lsmr(matrix, np.asarray([3.0]))

    np.testing.assert_allclose(matrix @ result.parameter_vector, [3.0], atol=1.0e-10)
    np.testing.assert_allclose(result.parameter_vector, [1.5, 1.5], atol=1.0e-10)


def test_run_sparse_lsmr_matches_numpy_lstsq_on_full_rank_system() -> None:
    matrix_dense = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=np.float64,
    )
    rhs_s = np.asarray([1.0, 2.0, 3.0, 0.5])
    expected = np.linalg.lstsq(matrix_dense, rhs_s, rcond=None)[0]

    result = run_sparse_lsmr(sparse.csr_matrix(matrix_dense), rhs_s)

    np.testing.assert_allclose(result.parameter_vector, expected, atol=1.0e-10)


def test_run_sparse_lsmr_records_diagnostics() -> None:
    result = run_sparse_lsmr(
        sparse.eye(2, format='csr', dtype=np.float64),
        np.asarray([1.0, 2.0]),
    )

    assert isinstance(result.diagnostics, ResidualStaticLsmrDiagnostics)
    assert isinstance(result.diagnostics.istop, int)
    assert isinstance(result.diagnostics.itn, int)
    assert np.isfinite(result.diagnostics.normr)
    assert np.isfinite(result.diagnostics.normar)
    assert np.isfinite(result.diagnostics.norma)
    assert np.isfinite(result.diagnostics.conda)
    assert np.isfinite(result.diagnostics.normx)


def test_run_sparse_lsmr_rejects_rhs_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='rhs_s'):
        run_sparse_lsmr(sparse.eye(2, format='csr'), np.asarray([1.0]))


def test_run_sparse_lsmr_rejects_nonfinite_rhs() -> None:
    with pytest.raises(ValueError, match='rhs_s'):
        run_sparse_lsmr(sparse.eye(2, format='csr'), np.asarray([1.0, np.nan]))


def test_run_sparse_lsmr_rejects_nonfinite_matrix_data() -> None:
    matrix = sparse.csr_matrix(
        ([1.0, np.nan], ([0, 1], [0, 1])),
        shape=(2, 2),
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match='matrix.data'):
        run_sparse_lsmr(matrix, np.asarray([1.0, 2.0]))


def test_solve_residual_static_least_squares_linear_abs_offset_exact_synthetic() -> None:
    inputs = _inputs()

    result = solve_residual_static_least_squares(inputs)
    matrix_dense, rhs_s = _dense_triplet_matrix(inputs)

    np.testing.assert_allclose(
        matrix_dense @ result.parameter_vector,
        rhs_s,
        atol=1.0e-10,
    )
    _assert_model_reconstructs_used_picks(result, inputs)
    assert result.parameter_vector.dtype == np.float64


def test_solve_residual_static_least_squares_moveout_none_exact_synthetic() -> None:
    inputs = _moveout_none_inputs()

    result = solve_residual_static_least_squares(inputs)
    matrix_dense, rhs_s = _dense_triplet_matrix(inputs)

    np.testing.assert_allclose(
        matrix_dense @ result.parameter_vector,
        rhs_s,
        atol=1.0e-10,
    )
    _assert_model_reconstructs_used_picks(result, inputs)


def test_solve_residual_static_least_squares_matches_numpy_lstsq_residual_norm() -> None:
    inputs = _inputs()
    matrix_dense, rhs_s = _dense_triplet_matrix(inputs)
    expected_x = np.linalg.lstsq(matrix_dense, rhs_s, rcond=None)[0]
    expected_norm = np.linalg.norm(matrix_dense @ expected_x - rhs_s)

    result = solve_residual_static_least_squares(inputs)
    actual_norm = np.linalg.norm(matrix_dense @ result.parameter_vector - rhs_s)

    np.testing.assert_allclose(actual_norm, expected_norm, atol=1.0e-10)


def test_solve_residual_static_least_squares_respects_used_mask_subset() -> None:
    inputs = _inputs()
    used_mask = np.asarray([True, False, True, False])

    result = solve_residual_static_least_squares(
        inputs,
        used_mask_sorted=used_mask,
    )

    assert result.n_observations == 2
    np.testing.assert_array_equal(result.used_mask_sorted, used_mask)
    np.testing.assert_array_equal(result.row_to_sorted_trace_index, [0, 2])
    np.testing.assert_array_equal(
        result.model_evaluation.residual_valid_mask_sorted,
        used_mask,
    )
    assert np.isnan(result.model_evaluation.residual_s_sorted[1])
    assert np.isnan(result.model_evaluation.residual_s_sorted[3])
    _assert_model_reconstructs_used_picks(result, inputs)


def test_solve_residual_static_least_squares_keeps_nan_residual_for_invalid_trace() -> None:
    valid_mask = np.asarray([True, False, True, True])
    inputs = _inputs(
        valid_pick_mask_sorted=valid_mask,
        pick_time_after_datum_s_sorted=np.asarray([0.100, np.nan, 0.120, 0.130]),
    )

    result = solve_residual_static_least_squares(inputs)

    assert np.isnan(result.model_evaluation.residual_s_sorted[1])
    np.testing.assert_array_equal(result.used_mask_sorted, valid_mask)
    assert np.all(np.isfinite(result.model_evaluation.residual_s_sorted[valid_mask]))
    np.testing.assert_allclose(
        result.model_evaluation.residual_s_sorted[valid_mask],
        np.zeros(int(np.count_nonzero(valid_mask)), dtype=np.float64),
        atol=1.0e-7,
    )


def test_solve_residual_static_least_squares_returns_parameter_parts() -> None:
    result = solve_residual_static_least_squares(_inputs())

    assert result.parameter_parts.slowness_s_per_offset_unit is not None
    assert result.parameter_parts.source_delay_s.shape == (2,)
    assert result.parameter_parts.receiver_delay_s.shape == (2,)
    assert result.parameter_parts.source_delay_s.dtype == np.float64
    assert result.parameter_parts.receiver_delay_s.dtype == np.float64


def test_solve_residual_static_least_squares_returns_model_evaluation() -> None:
    result = solve_residual_static_least_squares(_inputs())

    assert isinstance(result.model_evaluation, ResidualStaticModelEvaluation)
    assert result.model_evaluation.moveout_model_time_s_sorted.dtype == np.float64
    assert result.model_evaluation.estimated_trace_delay_s_sorted.dtype == np.float64
    assert result.model_evaluation.modeled_pick_time_s_sorted.dtype == np.float64
    assert result.model_evaluation.residual_s_sorted.dtype == np.float64


def test_solve_residual_static_least_squares_does_not_create_applied_shift() -> None:
    result = solve_residual_static_least_squares(_inputs())

    assert not hasattr(result, 'applied_residual_shift_s_sorted')
    assert not hasattr(result.model_evaluation, 'applied_residual_shift_s_sorted')


def test_solver_result_is_preliminary_rank_deficient_without_gauge() -> None:
    result = solve_residual_static_least_squares(_inputs())

    assert result.rank_deficient_possible is True
    assert result.n_observations == 4
    assert result.n_model_parameters == 6


def test_source_receiver_delays_are_not_compared_to_absolute_synthetic_values_before_gauge() -> None:
    inputs = _inputs()
    matrix_dense, rhs_s = _dense_triplet_matrix(inputs)

    result = solve_residual_static_least_squares(inputs)

    assert result.rank_deficient_possible is True
    np.testing.assert_allclose(
        matrix_dense @ result.parameter_vector,
        rhs_s,
        atol=1.0e-10,
    )
