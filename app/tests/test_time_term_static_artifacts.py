from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from app.services.time_term_apply_shift import (
    DELAY_TO_SHIFT_CONVENTION,
    FINAL_SHIFT_CONVENTION,
    SIGN_CONVENTION,
    TimeTermAppliedShiftResult,
)
from app.services.time_term_design_matrix import TimeTermDesignMatrix
from app.services.time_term_moveout import TimeTermMoveoutResult
from app.services.time_term_robust_solver import (
    TimeTermRobustIteration,
    TimeTermRobustSolverOptions,
    TimeTermRobustSolverResult,
)
from app.services.time_term_sparse_solver import (
    TimeTermSolverSystem,
    TimeTermSparseSolverResult,
)
from app.services.time_term_static_artifacts import (
    TIME_TERM_STATICS_CSV_NAME,
    TIME_TERM_STATIC_QC_JSON_NAME,
    TIME_TERM_STATIC_SOLUTION_NPZ_NAME,
    TimeTermStaticArtifactMetadata,
    build_time_term_qc_payload,
    build_time_term_solution_arrays,
    build_time_term_statics_csv_rows,
    write_time_term_static_artifacts,
)
from app.services.time_term_types import ORDER, TimeTermInversionInputs

N_TRACES = 3
N_NODES = 3
N_OBSERVATIONS = 3
DT = 0.002

SOURCE_NODE_ID_SORTED = np.asarray([0, 1, 2], dtype=np.int64)
RECEIVER_NODE_ID_SORTED = np.asarray([1, 2, 2], dtype=np.int64)
NODE_TIME_TERM_S = np.asarray([0.010, 0.005, -0.002], dtype=np.float64)
ESTIMATED_DELAY_S = np.asarray([0.015, 0.003, -0.004], dtype=np.float64)
APPLIED_WEATHERING_SHIFT_S = np.asarray([-0.015, -0.003, 0.004], dtype=np.float64)
DATUM_SHIFT_S = np.asarray([-0.020, -0.020, -0.020], dtype=np.float64)
RESIDUAL_SHIFT_S = np.asarray([0.001, 0.002, 0.003], dtype=np.float64)
FINAL_SHIFT_S = np.asarray([-0.034, -0.021, -0.013], dtype=np.float64)
PICK_AFTER_STATIC_S = np.asarray([0.081, 0.102, 0.123], dtype=np.float64)
MOVEOUT_TIME_S = np.asarray([0.050, 0.060, 0.070], dtype=np.float64)
MOVEOUT_DISTANCE_M = np.asarray([1000.0, 1200.0, 1400.0], dtype=np.float64)
ROW_DATA_S = PICK_AFTER_STATIC_S - MOVEOUT_TIME_S
ROW_RESIDUAL_AFTER_S = ROW_DATA_S - ESTIMATED_DELAY_S
FINAL_USED_MASK = np.asarray([True, False, True], dtype=bool)
REJECTED_MASK = np.asarray([False, True, False], dtype=bool)
REJECTED_ITERATION = np.asarray([-1, 0, -1], dtype=np.int64)

CSV_COLUMNS = [
    'sorted_trace_index',
    'source_id',
    'receiver_id',
    'source_node_id',
    'receiver_node_id',
    'offset_m',
    'source_x_m',
    'source_y_m',
    'receiver_x_m',
    'receiver_y_m',
    'source_elevation_m',
    'receiver_elevation_m',
    'source_depth_m',
    'pick_time_raw_s',
    'valid_pick',
    'pick_time_after_static_s',
    'moveout_time_s',
    'moveout_distance_m',
    'source_node_time_term_ms',
    'receiver_node_time_term_ms',
    'estimated_trace_time_term_delay_ms',
    'applied_weathering_shift_ms',
    'datum_trace_shift_ms',
    'residual_applied_shift_ms',
    'final_trace_shift_ms',
    'final_used',
    'rejected',
    'rejected_iteration',
    'row_index',
    'row_residual_before_ms',
    'row_residual_after_ms',
]

REQUIRED_SCALARS = {
    'schema_version',
    'artifact_kind',
    'order',
    'job_id',
    'input_file_id',
    'key1_byte',
    'key2_byte',
    'n_traces',
    'n_samples',
    'dt',
    'n_nodes',
    'n_observations',
    'n_final_used_traces',
    'n_rejected_traces',
    'pick_source_description',
    'datum_solution_path',
    'residual_solution_path',
    'linkage_artifact_path',
    'header_source_segy_path',
    'moveout_model',
    'refractor_velocity_m_s',
    'moveout_distance_source',
    'solver_name',
    'solver_istop',
    'solver_iterations',
    'solver_stop_message',
    'gauge_mode',
    'damping_lambda',
    'robust_enabled',
    'robust_method',
    'robust_threshold',
    'robust_stop_reason',
    'robust_n_iterations',
    'sign_convention',
    'delay_to_shift_convention',
    'final_shift_convention',
    'rejected_trace_policy',
}

REQUIRED_TRACE_ARRAYS = {
    'sorted_trace_index',
    'pick_time_raw_s_sorted',
    'valid_pick_mask_sorted',
    'pick_time_after_static_s_sorted',
    'moveout_time_s_sorted',
    'moveout_distance_m_sorted',
    'valid_moveout_mask_sorted',
    'source_node_id_sorted',
    'receiver_node_id_sorted',
    'source_node_time_term_s_sorted',
    'receiver_node_time_term_s_sorted',
    'estimated_trace_time_term_delay_s_sorted',
    'applied_weathering_shift_s_sorted',
    'datum_trace_shift_s_sorted',
    'residual_applied_shift_s_sorted',
    'final_trace_shift_s_sorted',
    'final_used_trace_mask_sorted',
    'rejected_trace_mask_sorted',
    'rejected_iteration_sorted',
    'source_id_sorted',
    'receiver_id_sorted',
    'offset_sorted',
    'source_x_m_sorted',
    'source_y_m_sorted',
    'receiver_x_m_sorted',
    'receiver_y_m_sorted',
    'source_elevation_m_sorted',
    'receiver_elevation_m_sorted',
    'source_depth_m_sorted',
}

REQUIRED_NODE_ARRAYS = {
    'node_id',
    'node_time_term_s',
    'node_time_term_ms',
    'source_observation_count_by_node',
    'receiver_observation_count_by_node',
    'total_observation_count_by_node',
    'component_id_by_node',
}

REQUIRED_ROW_ARRAYS = {
    'row_trace_index_sorted',
    'row_source_node_id',
    'row_receiver_node_id',
    'row_pick_time_after_static_s',
    'row_moveout_time_s',
    'row_data_s',
    'row_estimated_time_term_delay_s',
    'row_residual_before_s',
    'row_residual_after_s',
    'row_residual_after_ms',
    'initial_row_used_mask',
    'final_row_used_mask',
    'final_row_rejected_mask',
    'row_rejected_iteration',
}

REQUIRED_ROBUST_ARRAYS = {
    'robust_iteration_index',
    'robust_iteration_n_used',
    'robust_iteration_n_rejected_total',
    'robust_iteration_n_rejected_this_iteration',
    'robust_iteration_center_s',
    'robust_iteration_scale_s',
    'robust_iteration_threshold_s',
    'robust_iteration_rms_residual_after_s',
}


def _inputs(**overrides: Any) -> TimeTermInversionInputs:
    payload: dict[str, Any] = {
        'n_traces': N_TRACES,
        'n_samples': 200,
        'dt': DT,
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_time_raw_s_sorted': np.asarray([0.10, 0.12, 0.14], dtype=np.float64),
        'valid_pick_mask_sorted': np.ones(N_TRACES, dtype=bool),
        'datum_trace_shift_s_sorted': DATUM_SHIFT_S.copy(),
        'residual_applied_shift_s_sorted': RESIDUAL_SHIFT_S.copy(),
        'pick_time_after_static_s_sorted': PICK_AFTER_STATIC_S.copy(),
        'source_node_id_sorted': SOURCE_NODE_ID_SORTED.copy(),
        'receiver_node_id_sorted': RECEIVER_NODE_ID_SORTED.copy(),
        'n_nodes': N_NODES,
        'source_id_sorted': np.asarray([100, 101, 102], dtype=np.int64),
        'receiver_id_sorted': np.asarray([200, 201, 202], dtype=np.int64),
        'offset_sorted': np.asarray([1000.0, np.nan, 1400.0], dtype=np.float64),
        'source_x_m_sorted': np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
        'source_y_m_sorted': np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        'receiver_x_m_sorted': np.asarray([5.0, 15.0, 25.0], dtype=np.float64),
        'receiver_y_m_sorted': np.asarray([3.0, 4.0, 5.0], dtype=np.float64),
        'source_elevation_m_sorted': np.asarray([100.0, 101.0, 102.0]),
        'receiver_elevation_m_sorted': np.asarray([90.0, 91.0, 92.0]),
        'source_depth_m_sorted': np.asarray([10.0, 11.0, 12.0]),
        'input_file_id': 'synthetic-file',
        'pick_source_description': 'synthetic picks',
        'datum_solution_path': Path('datum_static_solution.npz'),
        'residual_solution_path': Path('residual_static_solution.npz'),
        'linkage_artifact_path': Path('geometry_linkage.npz'),
        'metadata': {
            'job_id': 'time-term-job',
            'request': {'robust': {'enabled': True}, 'threshold': np.float64(3.5)},
        },
    }
    payload.update(overrides)
    return TimeTermInversionInputs(**payload)


def _moveout(**overrides: Any) -> TimeTermMoveoutResult:
    payload: dict[str, Any] = {
        'model': 'linear_offset',
        'refractor_velocity_m_s': 2000.0,
        'distance_source': 'geometry',
        'distance_m_sorted': MOVEOUT_DISTANCE_M.copy(),
        'moveout_time_s_sorted': MOVEOUT_TIME_S.copy(),
        'valid_moveout_mask_sorted': np.ones(N_TRACES, dtype=bool),
        'reciprocal_pair_index_sorted': np.full(N_TRACES, -1, dtype=np.int64),
        'has_reciprocal_pair_mask_sorted': np.zeros(N_TRACES, dtype=bool),
        'geometry_distance_m_sorted': MOVEOUT_DISTANCE_M.copy(),
        'offset_abs_m_sorted': MOVEOUT_DISTANCE_M.copy(),
        'geometry_offset_mismatch_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
    }
    payload.update(overrides)
    return TimeTermMoveoutResult(**payload)


def _design(**overrides: Any) -> TimeTermDesignMatrix:
    matrix = sparse.csr_matrix(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    payload: dict[str, Any] = {
        'matrix': matrix,
        'data_s': ROW_DATA_S.copy(),
        'n_traces': N_TRACES,
        'n_observations': N_OBSERVATIONS,
        'n_nodes': N_NODES,
        'used_trace_mask_sorted': np.ones(N_TRACES, dtype=bool),
        'row_trace_index_sorted': np.arange(N_OBSERVATIONS, dtype=np.int64),
        'trace_to_row_index_sorted': np.arange(N_TRACES, dtype=np.int64),
        'source_node_id_sorted': SOURCE_NODE_ID_SORTED.copy(),
        'receiver_node_id_sorted': RECEIVER_NODE_ID_SORTED.copy(),
        'row_source_node_id': SOURCE_NODE_ID_SORTED.copy(),
        'row_receiver_node_id': RECEIVER_NODE_ID_SORTED.copy(),
        'row_pick_time_after_static_s': PICK_AFTER_STATIC_S.copy(),
        'row_moveout_time_s': MOVEOUT_TIME_S.copy(),
        'row_data_s': ROW_DATA_S.copy(),
        'source_observation_count_by_node': np.asarray([1, 1, 1], dtype=np.int64),
        'receiver_observation_count_by_node': np.asarray([0, 1, 2], dtype=np.int64),
        'total_observation_count_by_node': np.asarray([1, 2, 3], dtype=np.int64),
    }
    payload.update(overrides)
    return TimeTermDesignMatrix(**payload)


def _system(
    *,
    n_nodes: int = N_NODES,
    n_observations: int = N_OBSERVATIONS,
) -> TimeTermSolverSystem:
    n_augmented = n_observations + n_nodes + 1
    return TimeTermSolverSystem(
        augmented_matrix=sparse.csr_matrix((n_augmented, n_nodes), dtype=np.float64),
        augmented_data_s=np.zeros(n_augmented, dtype=np.float64),
        n_observation_rows=n_observations,
        n_damping_rows=n_nodes,
        n_gauge_rows=1,
        n_augmented_rows=n_augmented,
        n_nodes=n_nodes,
        damping_prior_s=np.zeros(n_nodes, dtype=np.float64),
        gauge_mode='mean_zero',
        component_id_by_node=np.zeros(n_nodes, dtype=np.int64),
        n_components=1,
        damping_lambda=0.01,
        gauge_weight=1.0,
        reference_node_id=None,
        min_total_observations_per_node=1,
        total_observation_count_by_node=np.asarray([1, 2, 3], dtype=np.int64),
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def _sparse_result(**overrides: Any) -> TimeTermSparseSolverResult:
    payload: dict[str, Any] = {
        'node_time_term_s': NODE_TIME_TERM_S.copy(),
        'estimated_trace_time_term_delay_s_sorted': ESTIMATED_DELAY_S.copy(),
        'row_estimated_time_term_delay_s': ESTIMATED_DELAY_S.copy(),
        'row_residual_before_s': ROW_DATA_S.copy(),
        'row_residual_after_s': ROW_RESIDUAL_AFTER_S.copy(),
        'row_residual_after_ms': ROW_RESIDUAL_AFTER_S * 1000.0,
        'rms_residual_before_s': _rms(ROW_DATA_S),
        'rms_residual_after_s': _rms(ROW_RESIDUAL_AFTER_S),
        'used_trace_mask_sorted': np.ones(N_TRACES, dtype=bool),
        'row_trace_index_sorted': np.arange(N_OBSERVATIONS, dtype=np.int64),
        'solver_name': 'lsmr',
        'solver_istop': 1,
        'solver_iterations': 7,
        'solver_normr': 0.01,
        'solver_normar': 0.02,
        'solver_conda': 1.5,
        'solver_message': 'synthetic converged',
        'system': _system(),
    }
    payload.update(overrides)
    return TimeTermSparseSolverResult(**payload)


def _robust_iteration(
    solver_result: TimeTermSparseSolverResult | None = None,
) -> TimeTermRobustIteration:
    solver = _sparse_result() if solver_result is None else solver_result
    return TimeTermRobustIteration(
        iteration=0,
        solver_result=solver,
        row_used_mask=np.ones(N_OBSERVATIONS, dtype=bool),
        row_rejected_this_iteration_mask=REJECTED_MASK.copy(),
        row_residual_s=ROW_RESIDUAL_AFTER_S.copy(),
        row_score=np.asarray([0.2, 4.2, 0.3], dtype=np.float64),
        scale_s=0.001,
        center_s=0.0,
        threshold_s=0.0035,
        n_used=N_OBSERVATIONS,
        n_rejected_total=1,
        n_rejected_this_iteration=1,
    )


def _robust_result(**overrides: Any) -> TimeTermRobustSolverResult:
    final_solver_result = overrides.pop('final_solver_result', _sparse_result())
    payload: dict[str, Any] = {
        'final_solver_result': final_solver_result,
        'iterations': (_robust_iteration(final_solver_result),),
        'initial_used_trace_mask_sorted': np.ones(N_TRACES, dtype=bool),
        'final_used_trace_mask_sorted': FINAL_USED_MASK.copy(),
        'final_rejected_trace_mask_sorted': REJECTED_MASK.copy(),
        'rejected_iteration_sorted': REJECTED_ITERATION.copy(),
        'initial_row_used_mask': np.ones(N_OBSERVATIONS, dtype=bool),
        'final_row_used_mask': FINAL_USED_MASK.copy(),
        'final_row_rejected_mask': REJECTED_MASK.copy(),
        'row_rejected_iteration': REJECTED_ITERATION.copy(),
        'method': 'mad',
        'enabled': True,
        'stop_reason': 'converged',
        'robust_options': TimeTermRobustSolverOptions(threshold=3.5),
        'n_initial_used_traces': N_TRACES,
        'n_final_used_traces': 2,
        'n_rejected_traces': 1,
        'final_used_fraction': 2.0 / 3.0,
    }
    payload.update(overrides)
    return TimeTermRobustSolverResult(**payload)


def _applied_shift(**overrides: Any) -> TimeTermAppliedShiftResult:
    payload: dict[str, Any] = {
        'n_traces': N_TRACES,
        'dt': DT,
        'node_time_term_s': NODE_TIME_TERM_S.copy(),
        'source_node_time_term_s_sorted': np.asarray(
            [0.010, 0.005, -0.002],
            dtype=np.float64,
        ),
        'receiver_node_time_term_s_sorted': np.asarray(
            [0.005, -0.002, -0.002],
            dtype=np.float64,
        ),
        'estimated_trace_time_term_delay_s_sorted': ESTIMATED_DELAY_S.copy(),
        'applied_weathering_shift_s_sorted': APPLIED_WEATHERING_SHIFT_S.copy(),
        'datum_trace_shift_s_sorted': DATUM_SHIFT_S.copy(),
        'residual_applied_shift_s_sorted': RESIDUAL_SHIFT_S.copy(),
        'final_trace_shift_s_sorted': FINAL_SHIFT_S.copy(),
        'valid_pick_mask_sorted': np.ones(N_TRACES, dtype=bool),
        'final_used_trace_mask_sorted': FINAL_USED_MASK.copy(),
        'rejected_trace_mask_sorted': REJECTED_MASK.copy(),
        'rejected_iteration_sorted': REJECTED_ITERATION.copy(),
        'sign_convention': SIGN_CONVENTION,
        'order': ORDER,
        'metadata': {
            'delay_to_shift_convention': DELAY_TO_SHIFT_CONVENTION,
            'final_shift_convention': FINAL_SHIFT_CONVENTION,
            'rejected_trace_policy': 'use_final_model',
        },
    }
    payload.update(overrides)
    return TimeTermAppliedShiftResult(**payload)


def _build_arrays(
    *,
    inputs: TimeTermInversionInputs | None = None,
    moveout: TimeTermMoveoutResult | None = None,
    design: TimeTermDesignMatrix | None = None,
    solver_result: TimeTermSparseSolverResult | TimeTermRobustSolverResult | None = None,
    applied_shift: TimeTermAppliedShiftResult | None = None,
    metadata: TimeTermStaticArtifactMetadata | None = None,
) -> dict[str, np.ndarray]:
    return build_time_term_solution_arrays(
        inputs=_inputs() if inputs is None else inputs,
        moveout=_moveout() if moveout is None else moveout,
        design=_design() if design is None else design,
        solver_result=_robust_result() if solver_result is None else solver_result,
        applied_shift=_applied_shift() if applied_shift is None else applied_shift,
        metadata=metadata,
    )


def test_build_time_term_solution_arrays_contains_required_fields() -> None:
    arrays = _build_arrays()

    assert REQUIRED_SCALARS.issubset(arrays)
    assert REQUIRED_TRACE_ARRAYS.issubset(arrays)
    assert REQUIRED_NODE_ARRAYS.issubset(arrays)
    assert REQUIRED_ROW_ARRAYS.issubset(arrays)
    assert REQUIRED_ROBUST_ARRAYS.issubset(arrays)
    assert arrays['n_traces'].item() == N_TRACES
    assert arrays['n_nodes'].item() == N_NODES
    assert arrays['n_observations'].item() == N_OBSERVATIONS
    assert arrays['sorted_trace_index'].tolist() == [0, 1, 2]


def test_time_term_solution_arrays_have_no_object_dtype() -> None:
    arrays = _build_arrays()

    assert all(np.asarray(value).dtype != object for value in arrays.values())


def test_time_term_solution_arrays_store_components_and_conventions() -> None:
    arrays = _build_arrays()

    np.testing.assert_allclose(
        arrays['applied_weathering_shift_s_sorted'],
        -arrays['estimated_trace_time_term_delay_s_sorted'],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        arrays['final_trace_shift_s_sorted'],
        arrays['datum_trace_shift_s_sorted']
        + arrays['residual_applied_shift_s_sorted']
        + arrays['applied_weathering_shift_s_sorted'],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert arrays['delay_to_shift_convention'].item() == DELAY_TO_SHIFT_CONVENTION
    assert arrays['final_shift_convention'].item() == FINAL_SHIFT_CONVENTION
    assert arrays['order'].item() == ORDER


def test_time_term_solution_arrays_write_nan_offset_when_missing() -> None:
    arrays = _build_arrays(inputs=_inputs(offset_sorted=None))

    assert np.isnan(arrays['offset_sorted']).all()


def test_time_term_solution_arrays_reject_shape_mismatch() -> None:
    bad_inputs = _inputs(source_id_sorted=np.asarray([1, 2], dtype=np.int64))

    with pytest.raises(ValueError, match='source_id_sorted shape mismatch'):
        _build_arrays(inputs=bad_inputs)


def test_time_term_solution_arrays_reject_node_id_out_of_range() -> None:
    bad_inputs = _inputs(
        source_node_id_sorted=np.asarray([0, 3, 2], dtype=np.int64)
    )

    with pytest.raises(ValueError, match='source_node_id_sorted'):
        _build_arrays(inputs=bad_inputs)


def test_time_term_solution_arrays_reject_shift_sign_mismatch() -> None:
    bad_shift = replace(
        _applied_shift(),
        applied_weathering_shift_s_sorted=ESTIMATED_DELAY_S.copy(),
    )

    with pytest.raises(ValueError, match='applied_weathering_shift_s_sorted'):
        _build_arrays(applied_shift=bad_shift)


def test_time_term_solution_arrays_reject_final_shift_composition_mismatch() -> None:
    bad_shift = replace(
        _applied_shift(),
        final_trace_shift_s_sorted=FINAL_SHIFT_S + 0.001,
    )

    with pytest.raises(ValueError, match='final_trace_shift_s_sorted'):
        _build_arrays(applied_shift=bad_shift)


def test_time_term_qc_payload_is_strict_json_and_contains_summaries() -> None:
    payload = build_time_term_qc_payload(
        inputs=_inputs(),
        moveout=_moveout(),
        design=_design(),
        solver_result=_robust_result(),
        applied_shift=_applied_shift(),
        metadata=TimeTermStaticArtifactMetadata(job_id='override-job'),
    )

    json.dumps(payload, allow_nan=False)
    assert payload['artifact_kind'] == 'time_term_static_qc'
    assert payload['job']['job_id'] == 'override-job'
    assert payload['counts']['n_rejected_traces'] == 1
    assert payload['moveout']['distance_m']['count'] == N_TRACES
    assert payload['solver']['solver_name'] == 'lsmr'
    assert payload['robust']['iterations'][0]['n_rejected_this_iteration'] == 1
    assert payload['time_terms']['applied_weathering_shift_ms']['count'] == N_TRACES
    assert payload['components']['final_trace_shift_ms']['count'] == N_TRACES


def test_time_term_csv_rows_are_sorted_and_blank_missing_values() -> None:
    rows = build_time_term_statics_csv_rows(
        inputs=_inputs(),
        moveout=_moveout(),
        design=_design(),
        solver_result=_robust_result(),
        applied_shift=_applied_shift(),
    )

    assert len(rows) == N_TRACES
    assert [row['sorted_trace_index'] for row in rows] == [0, 1, 2]
    assert rows[1]['offset_m'] == ''
    assert rows[1]['rejected'] == 'true'
    assert rows[1]['rejected_iteration'] == 0
    assert rows[1]['row_index'] == 1
    assert rows[0]['final_used'] == 'true'


def test_write_time_term_static_artifacts_writes_npz_json_csv(tmp_path: Path) -> None:
    paths = write_time_term_static_artifacts(
        job_dir=tmp_path,
        inputs=_inputs(),
        moveout=_moveout(),
        design=_design(),
        solver_result=_robust_result(),
        applied_shift=_applied_shift(),
    )

    assert paths.solution_npz_path == tmp_path / TIME_TERM_STATIC_SOLUTION_NPZ_NAME
    assert paths.qc_json_path == tmp_path / TIME_TERM_STATIC_QC_JSON_NAME
    assert paths.statics_csv_path == tmp_path / TIME_TERM_STATICS_CSV_NAME

    with np.load(paths.solution_npz_path, allow_pickle=False) as data:
        assert REQUIRED_SCALARS.issubset(data.files)
        assert all(data[name].dtype != object for name in data.files)
        np.testing.assert_allclose(
            data['applied_weathering_shift_s_sorted'],
            -data['estimated_trace_time_term_delay_s_sorted'],
            rtol=0.0,
            atol=1.0e-12,
        )

    payload = json.loads(paths.qc_json_path.read_text(encoding='utf-8'))
    json.dumps(payload, allow_nan=False)

    with paths.statics_csv_path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == CSV_COLUMNS
    assert len(rows) == N_TRACES
    assert rows[0]['sorted_trace_index'] == '0'
    assert rows[1]['offset_m'] == ''


def test_time_term_artifact_writer_cleans_tmp_files_on_failure(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    (job_dir / TIME_TERM_STATICS_CSV_NAME).mkdir()

    with pytest.raises(OSError):
        write_time_term_static_artifacts(
            job_dir=job_dir,
            inputs=_inputs(),
            moveout=_moveout(),
            design=_design(),
            solver_result=_robust_result(),
            applied_shift=_applied_shift(),
        )

    assert list(job_dir.glob('*.tmp-*')) == []
