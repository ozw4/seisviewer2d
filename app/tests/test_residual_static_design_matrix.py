from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import app.services.residual_static_design_matrix as design_matrix
from app.services.residual_static_design_matrix import (
    build_residual_static_column_layout,
    build_residual_static_observation_matrix_triplets,
    compute_linear_abs_offset_moveout_s,
    evaluate_residual_static_model,
    pack_residual_static_parameters,
    unpack_residual_static_parameters,
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


def _linear_layout():
    return build_residual_static_column_layout(_inputs())


def _linear_parameter_vector() -> np.ndarray:
    return pack_residual_static_parameters(
        layout=_linear_layout(),
        intercept_s=0.050,
        slowness_s_per_offset_unit=0.0001,
        source_delay_s=np.asarray([0.001, -0.001]),
        receiver_delay_s=np.asarray([0.002, -0.002]),
    )


def test_build_column_layout_linear_abs_offset() -> None:
    layout = build_residual_static_column_layout(_inputs())

    assert layout.moveout_model == 'linear_abs_offset'
    assert layout.intercept_col == 0
    assert layout.slowness_col == 1
    assert layout.source_delay_start_col == 2
    assert layout.receiver_delay_start_col == 4
    assert layout.n_sources == 2
    assert layout.n_receivers == 2
    assert layout.n_model_parameters == 6
    np.testing.assert_array_equal(layout.source_delay_cols, [2, 3])
    np.testing.assert_array_equal(layout.receiver_delay_cols, [4, 5])
    assert layout.source_delay_cols.dtype == np.int64
    assert layout.receiver_delay_cols.dtype == np.int64


def test_build_column_layout_moveout_none_omits_slowness_column() -> None:
    layout = build_residual_static_column_layout(_moveout_none_inputs())

    assert layout.moveout_model == 'none'
    assert layout.intercept_col == 0
    assert layout.slowness_col is None
    assert layout.source_delay_start_col == 1
    assert layout.receiver_delay_start_col == 3
    assert layout.n_model_parameters == 5
    np.testing.assert_array_equal(layout.source_delay_cols, [1, 2])
    np.testing.assert_array_equal(layout.receiver_delay_cols, [3, 4])


def test_build_column_layout_rejects_empty_sources() -> None:
    with pytest.raises(ValueError, match='source_unique_ids'):
        build_residual_static_column_layout(
            _inputs(source_unique_ids=np.asarray([], dtype=np.int64))
        )


def test_build_column_layout_rejects_empty_receivers() -> None:
    with pytest.raises(ValueError, match='receiver_unique_ids'):
        build_residual_static_column_layout(
            _inputs(receiver_unique_ids=np.asarray([], dtype=np.int64))
        )


def test_build_column_layout_rejects_source_index_out_of_range() -> None:
    with pytest.raises(ValueError, match='source_index_sorted'):
        build_residual_static_column_layout(
            _inputs(source_index_sorted=np.asarray([0, 0, 1, 2], dtype=np.int64))
        )


def test_build_column_layout_rejects_receiver_index_out_of_range() -> None:
    with pytest.raises(ValueError, match='receiver_index_sorted'):
        build_residual_static_column_layout(
            _inputs(receiver_index_sorted=np.asarray([0, 1, 0, 2], dtype=np.int64))
        )


def test_build_column_layout_rejects_bool_indices() -> None:
    with pytest.raises(ValueError, match='source_index_sorted'):
        build_residual_static_column_layout(
            _inputs(source_index_sorted=np.asarray([True, False, True, False]))
        )


def test_build_observation_triplets_linear_abs_offset() -> None:
    triplets = build_residual_static_observation_matrix_triplets(_inputs())

    np.testing.assert_array_equal(
        triplets.row_indices,
        np.repeat(np.arange(4, dtype=np.int64), 4),
    )
    np.testing.assert_array_equal(
        triplets.col_indices,
        [0, 1, 2, 4, 0, 1, 2, 5, 0, 1, 3, 4, 0, 1, 3, 5],
    )
    np.testing.assert_allclose(
        triplets.data,
        [1.0, 100.0, 1.0, 1.0, 1.0, 200.0, 1.0, 1.0,
         1.0, 100.0, 1.0, 1.0, 1.0, 200.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(triplets.rhs_s, PICK_TIME_AFTER_DATUM)
    np.testing.assert_array_equal(triplets.row_to_sorted_trace_index, [0, 1, 2, 3])
    np.testing.assert_array_equal(triplets.used_mask_sorted, VALID_PICK_MASK)
    assert triplets.n_rows == 4
    assert triplets.n_cols == 6
    assert triplets.row_indices.dtype == np.int64
    assert triplets.col_indices.dtype == np.int64
    assert triplets.data.dtype == np.float64
    assert triplets.rhs_s.dtype == np.float64
    assert triplets.used_mask_sorted.dtype == bool


def test_build_observation_triplets_moveout_none() -> None:
    triplets = build_residual_static_observation_matrix_triplets(
        _moveout_none_inputs()
    )

    np.testing.assert_array_equal(
        triplets.col_indices,
        [0, 1, 3, 0, 1, 4, 0, 2, 3, 0, 2, 4],
    )
    np.testing.assert_allclose(triplets.data, np.ones(12, dtype=np.float64))
    np.testing.assert_allclose(triplets.rhs_s, PICK_TIME_AFTER_DATUM)
    assert triplets.n_rows == 4
    assert triplets.n_cols == 5
    assert triplets.layout.slowness_col is None


def test_build_observation_triplets_uses_valid_pick_mask_by_default() -> None:
    valid_mask = np.asarray([True, False, True, False])
    pick_time = np.asarray([0.100, np.nan, 0.120, np.nan])

    triplets = build_residual_static_observation_matrix_triplets(
        _inputs(
            valid_pick_mask_sorted=valid_mask,
            pick_time_after_datum_s_sorted=pick_time,
        )
    )

    np.testing.assert_array_equal(triplets.row_to_sorted_trace_index, [0, 2])
    np.testing.assert_array_equal(triplets.used_mask_sorted, valid_mask)
    np.testing.assert_allclose(triplets.rhs_s, [0.100, 0.120])


def test_build_observation_triplets_accepts_used_mask_subset() -> None:
    used_mask = np.asarray([True, False, True, False])

    triplets = build_residual_static_observation_matrix_triplets(
        _inputs(),
        used_mask_sorted=used_mask,
    )

    np.testing.assert_array_equal(triplets.row_to_sorted_trace_index, [0, 2])
    np.testing.assert_array_equal(triplets.used_mask_sorted, used_mask)
    np.testing.assert_allclose(triplets.rhs_s, [0.100, 0.120])


def test_build_observation_triplets_rejects_used_mask_that_includes_invalid_pick() -> None:
    valid_mask = np.asarray([True, False, True, True])
    used_mask = np.asarray([True, True, False, False])

    with pytest.raises(ValueError, match='subset'):
        build_residual_static_observation_matrix_triplets(
            _inputs(
                valid_pick_mask_sorted=valid_mask,
                pick_time_after_datum_s_sorted=np.asarray(
                    [0.100, np.nan, 0.120, 0.130]
                ),
            ),
            used_mask_sorted=used_mask,
        )


def test_build_observation_triplets_rejects_zero_used_rows() -> None:
    with pytest.raises(ValueError, match='at least one'):
        build_residual_static_observation_matrix_triplets(
            _inputs(),
            used_mask_sorted=np.asarray([False, False, False, False]),
        )


def test_build_observation_triplets_rejects_nonfinite_pick_for_used_trace() -> None:
    with pytest.raises(ValueError, match='pick_time_after_datum_s_sorted'):
        build_residual_static_observation_matrix_triplets(
            _inputs(
                pick_time_after_datum_s_sorted=np.asarray(
                    [0.100, np.nan, 0.120, 0.130]
                )
            )
        )


def test_build_observation_triplets_rejects_missing_abs_offset_for_linear_abs_offset() -> None:
    with pytest.raises(ValueError, match='abs_offset_sorted'):
        build_residual_static_observation_matrix_triplets(
            _inputs(abs_offset_sorted=None)
        )


def test_build_observation_triplets_rejects_negative_abs_offset_for_linear_abs_offset() -> None:
    with pytest.raises(ValueError, match='abs_offset_sorted'):
        build_residual_static_observation_matrix_triplets(
            _inputs(abs_offset_sorted=np.asarray([100.0, -1.0, 100.0, 200.0]))
        )


def test_build_observation_triplets_does_not_require_offset_for_moveout_none() -> None:
    triplets = build_residual_static_observation_matrix_triplets(
        _moveout_none_inputs(offset_sorted=None, abs_offset_sorted=None)
    )

    assert triplets.n_rows == 4
    assert triplets.layout.slowness_col is None


def test_build_observation_triplets_has_deterministic_entry_order() -> None:
    triplets = build_residual_static_observation_matrix_triplets(
        _inputs(),
        used_mask_sorted=np.asarray([False, True, False, False]),
    )

    np.testing.assert_array_equal(triplets.row_indices, [0, 0, 0, 0])
    np.testing.assert_array_equal(triplets.col_indices, [0, 1, 2, 5])
    np.testing.assert_allclose(triplets.data, [1.0, 200.0, 1.0, 1.0])


def test_compute_linear_abs_offset_moveout_s() -> None:
    moveout = compute_linear_abs_offset_moveout_s(
        ABS_OFFSET,
        intercept_s=0.050,
        slowness_s_per_offset_unit=0.0001,
    )

    np.testing.assert_allclose(moveout, [0.060, 0.070, 0.060, 0.070])
    assert moveout.dtype == np.float64


def test_compute_linear_abs_offset_moveout_rejects_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match='abs_offset_sorted'):
        compute_linear_abs_offset_moveout_s(
            np.asarray([100.0, np.nan]),
            intercept_s=0.050,
            slowness_s_per_offset_unit=0.0001,
        )
    with pytest.raises(ValueError, match='intercept_s'):
        compute_linear_abs_offset_moveout_s(
            ABS_OFFSET,
            intercept_s=np.nan,
            slowness_s_per_offset_unit=0.0001,
        )
    with pytest.raises(ValueError, match='slowness_s_per_offset_unit'):
        compute_linear_abs_offset_moveout_s(
            ABS_OFFSET,
            intercept_s=0.050,
            slowness_s_per_offset_unit=np.inf,
        )


def test_pack_unpack_residual_static_parameters_linear_abs_offset() -> None:
    layout = _linear_layout()
    vector = pack_residual_static_parameters(
        layout=layout,
        intercept_s=0.050,
        slowness_s_per_offset_unit=0.0001,
        source_delay_s=np.asarray([0.001, -0.001]),
        receiver_delay_s=np.asarray([0.002, -0.002]),
    )

    np.testing.assert_allclose(
        vector,
        [0.050, 0.0001, 0.001, -0.001, 0.002, -0.002],
    )
    parts = unpack_residual_static_parameters(layout, vector)
    assert parts.intercept_s == pytest.approx(0.050)
    assert parts.slowness_s_per_offset_unit == pytest.approx(0.0001)
    np.testing.assert_allclose(parts.source_delay_s, [0.001, -0.001])
    np.testing.assert_allclose(parts.receiver_delay_s, [0.002, -0.002])
    assert vector.dtype == np.float64


def test_pack_unpack_residual_static_parameters_moveout_none() -> None:
    layout = build_residual_static_column_layout(_moveout_none_inputs())
    vector = pack_residual_static_parameters(
        layout=layout,
        intercept_s=0.050,
        slowness_s_per_offset_unit=None,
        source_delay_s=np.asarray([0.001, -0.001]),
        receiver_delay_s=np.asarray([0.002, -0.002]),
    )

    np.testing.assert_allclose(vector, [0.050, 0.001, -0.001, 0.002, -0.002])
    parts = unpack_residual_static_parameters(layout, vector)
    assert parts.intercept_s == pytest.approx(0.050)
    assert parts.slowness_s_per_offset_unit is None
    np.testing.assert_allclose(parts.source_delay_s, [0.001, -0.001])
    np.testing.assert_allclose(parts.receiver_delay_s, [0.002, -0.002])


def test_pack_parameters_rejects_wrong_source_delay_shape() -> None:
    with pytest.raises(ValueError, match='source_delay_s'):
        pack_residual_static_parameters(
            layout=_linear_layout(),
            intercept_s=0.050,
            slowness_s_per_offset_unit=0.0001,
            source_delay_s=np.asarray([0.001]),
            receiver_delay_s=np.asarray([0.002, -0.002]),
        )


def test_unpack_parameters_rejects_wrong_vector_length() -> None:
    with pytest.raises(ValueError, match='parameter_vector'):
        unpack_residual_static_parameters(
            _linear_layout(),
            np.asarray([0.050, 0.0001]),
        )


def test_evaluate_residual_static_model_linear_abs_offset() -> None:
    evaluation = evaluate_residual_static_model(
        _inputs(),
        _linear_layout(),
        _linear_parameter_vector(),
    )

    np.testing.assert_allclose(
        evaluation.moveout_model_time_s_sorted,
        [0.060, 0.070, 0.060, 0.070],
    )
    np.testing.assert_allclose(
        evaluation.estimated_trace_delay_s_sorted,
        [0.003, -0.001, 0.001, -0.003],
    )
    np.testing.assert_allclose(
        evaluation.modeled_pick_time_s_sorted,
        [0.063, 0.069, 0.061, 0.067],
    )
    np.testing.assert_allclose(
        evaluation.residual_s_sorted,
        [0.037, 0.041, 0.059, 0.063],
    )
    np.testing.assert_array_equal(
        evaluation.residual_valid_mask_sorted,
        VALID_PICK_MASK,
    )
    assert evaluation.residual_s_sorted.dtype == np.float64


def test_evaluate_residual_static_model_moveout_none() -> None:
    inputs = _moveout_none_inputs()
    layout = build_residual_static_column_layout(inputs)
    vector = pack_residual_static_parameters(
        layout=layout,
        intercept_s=0.050,
        slowness_s_per_offset_unit=None,
        source_delay_s=np.asarray([0.001, -0.001]),
        receiver_delay_s=np.asarray([0.002, -0.002]),
    )

    evaluation = evaluate_residual_static_model(inputs, layout, vector)

    np.testing.assert_allclose(
        evaluation.moveout_model_time_s_sorted,
        [0.050, 0.050, 0.050, 0.050],
    )
    np.testing.assert_allclose(
        evaluation.estimated_trace_delay_s_sorted,
        [0.003, -0.001, 0.001, -0.003],
    )
    np.testing.assert_allclose(
        evaluation.modeled_pick_time_s_sorted,
        [0.053, 0.049, 0.051, 0.047],
    )
    np.testing.assert_allclose(
        evaluation.residual_s_sorted,
        [0.047, 0.061, 0.069, 0.083],
    )


def test_evaluate_residual_static_model_keeps_nan_for_invalid_pick() -> None:
    inputs = _inputs(
        valid_pick_mask_sorted=np.asarray([True, False, True, True]),
        pick_time_after_datum_s_sorted=np.asarray([0.100, np.nan, 0.120, 0.130]),
    )

    evaluation = evaluate_residual_static_model(
        inputs,
        build_residual_static_column_layout(inputs),
        _linear_parameter_vector(),
    )

    assert np.isnan(evaluation.residual_s_sorted[1])
    np.testing.assert_array_equal(
        evaluation.residual_valid_mask_sorted,
        [True, False, True, True],
    )


def test_evaluate_residual_static_model_accepts_residual_mask_subset() -> None:
    residual_mask = np.asarray([True, False, True, False])

    evaluation = evaluate_residual_static_model(
        _inputs(),
        _linear_layout(),
        _linear_parameter_vector(),
        residual_mask_sorted=residual_mask,
    )

    np.testing.assert_array_equal(evaluation.residual_valid_mask_sorted, residual_mask)
    assert np.isnan(evaluation.residual_s_sorted[1])
    assert np.isnan(evaluation.residual_s_sorted[3])
    np.testing.assert_allclose(evaluation.residual_s_sorted[[0, 2]], [0.037, 0.059])


def test_evaluate_residual_static_model_rejects_residual_mask_that_includes_invalid_pick() -> None:
    inputs = _inputs(
        valid_pick_mask_sorted=np.asarray([True, False, True, True]),
        pick_time_after_datum_s_sorted=np.asarray([0.100, np.nan, 0.120, 0.130]),
    )

    with pytest.raises(ValueError, match='subset'):
        evaluate_residual_static_model(
            inputs,
            build_residual_static_column_layout(inputs),
            _linear_parameter_vector(),
            residual_mask_sorted=np.asarray([True, True, False, False]),
        )


def test_residual_static_design_matrix_module_does_not_import_scipy() -> None:
    assert 'scipy' not in design_matrix.__dict__
    assert 'scipy' not in design_matrix.__loader__.get_source(  # type: ignore[union-attr]
        design_matrix.__name__
    )
