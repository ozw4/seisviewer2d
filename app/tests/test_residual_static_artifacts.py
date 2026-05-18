from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app.services.residual_static_artifacts as artifacts
from app.services.residual_static_artifacts import (
    ResidualStaticArtifactMetadata,
    build_residual_static_qc_payload,
    build_residual_static_solution_arrays,
    build_residual_statics_csv_rows,
    summarize_finite_values,
    write_residual_static_artifacts,
)
from app.services.residual_static_robust_solver import (
    ResidualStaticRobustOptions,
    ResidualStaticRobustSolveResult,
    solve_residual_static_robust_least_squares,
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

_REQUIRED_NPZ_FIELDS = {
    'schema_version',
    'artifact_kind',
    'order',
    'sign_convention',
    'estimated_trace_delay_s_sorted',
    'applied_residual_shift_s_sorted',
    'residual_before_s',
    'residual_after_s',
    'initial_used_mask_sorted',
    'used_mask_sorted',
    'final_used_mask_sorted',
    'rejected_mask_sorted',
    'rejected_iteration_sorted',
    'robust_iteration_index',
}


def _synthetic_inputs(
    *,
    outlier_trace: int | None = None,
    outlier_s: float = 0.080,
    valid_pick_mask: np.ndarray | None = None,
    moveout_model: str = 'linear_abs_offset',
    **overrides: Any,
) -> ResidualStaticSolverInputs:
    source_unique_ids = np.asarray([101, 102, 103, 104], dtype=np.int64)
    receiver_unique_ids = np.asarray([201, 202, 203, 204, 205], dtype=np.int64)
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
    offset = abs_offset * np.where(np.arange(n_traces) % 2 == 0, -1.0, 1.0)

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
        moveout_s
        + SOURCE_DELAY_S[source_index]
        + RECEIVER_DELAY_S[receiver_index]
    )
    if outlier_trace is not None:
        pick_time_after_datum = pick_time_after_datum.copy()
        pick_time_after_datum[outlier_trace] += outlier_s

    valid_mask = (
        np.ones(n_traces, dtype=bool)
        if valid_pick_mask is None
        else np.ascontiguousarray(valid_pick_mask, dtype=bool)
    )
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
        'metadata': {
            'job_id': 'residual-job',
            'datum_solution_artifact': 'datum_static_solution.npz',
            'pick_source_artifact': 'predicted_picks_time_s.npz',
        },
    }
    payload.update(overrides)
    return ResidualStaticSolverInputs(**payload)


def _solve(
    inputs: ResidualStaticSolverInputs,
    *,
    robust_options: ResidualStaticRobustOptions | None = None,
) -> ResidualStaticRobustSolveResult:
    return solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=ResidualStaticStabilizationOptions(
            min_valid_picks=10,
            min_picks_per_source=1,
            min_picks_per_receiver=1,
        ),
        robust_options=robust_options or ResidualStaticRobustOptions(threshold=4.0),
        lsmr_options=ResidualStaticLsmrOptions(
            atol=1.0e-12,
            btol=1.0e-12,
            conlim=1.0e12,
            maxiter=10000,
        ),
    )


@pytest.fixture(scope='module')
def solved_linear() -> tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult]:
    inputs = _synthetic_inputs(outlier_trace=0)
    return inputs, _solve(inputs)


def test_build_residual_static_solution_arrays_contains_required_fields(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    arrays = build_residual_static_solution_arrays(inputs, result)

    assert _REQUIRED_NPZ_FIELDS.issubset(arrays)
    assert arrays['n_traces'].item() == inputs.n_traces
    assert arrays['estimated_trace_delay_s_sorted'].shape == (inputs.n_traces,)
    assert arrays['source_delay_s_sorted'].shape == (inputs.n_traces,)
    assert arrays['receiver_delay_s_sorted'].shape == (inputs.n_traces,)


def test_solution_npz_has_no_object_dtype(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    arrays = build_residual_static_solution_arrays(inputs, result)

    assert all(np.asarray(value).dtype != object for value in arrays.values())


def test_solution_npz_saves_applied_shift_as_negative_estimated_delay(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    arrays = build_residual_static_solution_arrays(inputs, result)

    np.testing.assert_allclose(
        arrays['applied_residual_shift_s_sorted'],
        -arrays['estimated_trace_delay_s_sorted'],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_solution_npz_writes_offset_nan_arrays_for_moveout_none() -> None:
    inputs = _synthetic_inputs(moveout_model='none')
    result = _solve(inputs, robust_options=ResidualStaticRobustOptions(enabled=False))

    arrays = build_residual_static_solution_arrays(inputs, result)

    assert arrays['offset_byte'].item() == -1
    assert np.isnan(arrays['offset_sorted']).all()
    assert np.isnan(arrays['abs_offset_sorted']).all()


def test_solution_npz_writes_empty_iteration_arrays_when_robust_disabled() -> None:
    inputs = _synthetic_inputs(outlier_trace=0)
    result = _solve(inputs, robust_options=ResidualStaticRobustOptions(enabled=False))

    arrays = build_residual_static_solution_arrays(inputs, result)

    assert arrays['robust_enabled'].item() is False
    assert arrays['robust_iteration_index'].shape == (0,)
    assert arrays['robust_iteration_stop_reason'].shape == (0,)


def test_build_residual_static_qc_payload_is_strict_json(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    payload = build_residual_static_qc_payload(
        inputs,
        result,
        metadata=ResidualStaticArtifactMetadata(job_id='override-job'),
    )

    json.dumps(payload, allow_nan=False)
    assert payload['lineage']['job_id'] == 'override-job'
    assert payload['lineage']['input_file_id'] == inputs.input_file_id


def test_qc_json_counts_match_masks(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    payload = build_residual_static_qc_payload(inputs, result)

    assert payload['counts']['n_valid_picks'] == int(
        np.count_nonzero(inputs.valid_pick_mask_sorted)
    )
    assert payload['counts']['n_initial_used_picks'] == int(
        np.count_nonzero(result.initial_used_mask_sorted)
    )
    assert payload['counts']['n_final_used_picks'] == int(
        np.count_nonzero(result.final_used_mask_sorted)
    )
    assert payload['counts']['n_rejected_total'] == 1


def test_qc_json_stats_use_initial_and_final_masks(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    payload = build_residual_static_qc_payload(inputs, result)

    assert payload['stats']['residual_before_ms_initial_used']['count'] == int(
        np.count_nonzero(result.initial_used_mask_sorted)
    )
    assert payload['stats']['residual_after_ms_final_used']['count'] == int(
        np.count_nonzero(result.final_used_mask_sorted)
    )


def test_qc_json_uses_null_for_empty_stats() -> None:
    stats = summarize_finite_values(np.asarray([np.nan], dtype=np.float64))

    assert stats.count == 0
    assert stats.min is None
    assert stats.max is None
    assert stats.mean is None
    assert stats.median is None
    assert stats.mad is None
    assert stats.std is None


def test_build_residual_statics_csv_rows_one_row_per_trace_and_sorted_order(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    rows = build_residual_statics_csv_rows(inputs, result)

    assert len(rows) == inputs.n_traces
    assert [row['sorted_trace_index'] for row in rows] == list(range(inputs.n_traces))
    assert [row['key1'] for row in rows] == list(inputs.key1_sorted)
    assert rows[0]['rejected'] == 'true'
    assert isinstance(rows[0]['rejected_iteration'], int)


def test_csv_rows_blank_invalid_pick_residual_columns() -> None:
    valid_mask = np.ones(20, dtype=bool)
    valid_mask[3] = False
    inputs = _synthetic_inputs(valid_pick_mask=valid_mask)
    result = _solve(inputs, robust_options=ResidualStaticRobustOptions(enabled=False))

    rows = build_residual_statics_csv_rows(inputs, result)

    assert rows[3]['pick_time_after_datum_s'] == ''
    assert rows[3]['modeled_pick_time_s'] == ''
    assert rows[3]['residual_before_ms'] == ''
    assert rows[3]['residual_after_ms'] == ''


def test_csv_rows_boolean_values_are_lowercase_strings(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    row = build_residual_statics_csv_rows(inputs, result)[0]

    assert row['valid_pick'] == 'true'
    assert row['initial_used'] == 'true'
    assert row['final_used'] == 'false'
    assert row['rejected'] == 'true'


def test_csv_rows_blank_offset_for_moveout_none() -> None:
    inputs = _synthetic_inputs(moveout_model='none')
    result = _solve(inputs, robust_options=ResidualStaticRobustOptions(enabled=False))

    rows = build_residual_statics_csv_rows(inputs, result)

    assert rows[0]['offset'] == ''
    assert rows[0]['abs_offset'] == ''


def test_artifact_writer_rejects_shape_mismatch(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear
    bad_inputs = replace(inputs, key1_sorted=inputs.key1_sorted[:-1])

    with pytest.raises(ValueError, match='key1_sorted shape mismatch'):
        build_residual_static_solution_arrays(bad_inputs, result)


def test_artifact_writer_rejects_initial_used_not_subset_of_valid() -> None:
    valid_mask = np.ones(20, dtype=bool)
    valid_mask[0] = False
    inputs = _synthetic_inputs(valid_pick_mask=valid_mask)
    result = _solve(inputs, robust_options=ResidualStaticRobustOptions(enabled=False))
    initial_used = result.initial_used_mask_sorted.copy()
    initial_used[0] = True
    bad_result = replace(result, initial_used_mask_sorted=initial_used)

    with pytest.raises(ValueError, match='initial_used_mask_sorted'):
        build_residual_static_solution_arrays(inputs, bad_result)


def test_artifact_writer_rejects_final_used_not_subset_of_initial_used() -> None:
    inputs = _synthetic_inputs(outlier_trace=0)
    result = _solve(inputs)
    final_used = result.final_used_mask_sorted.copy()
    final_used[0] = True
    bad_result = replace(result, final_used_mask_sorted=final_used)

    with pytest.raises(ValueError, match='rejected_mask_sorted|final_used_mask_sorted'):
        build_residual_static_solution_arrays(inputs, bad_result)


def test_artifact_writer_rejects_inconsistent_rejected_mask(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear
    bad_result = replace(
        result,
        rejected_mask_sorted=np.zeros(inputs.n_traces, dtype=bool),
    )

    with pytest.raises(ValueError, match='rejected_mask_sorted'):
        build_residual_static_solution_arrays(inputs, bad_result)


def test_artifact_writer_rejects_wrong_applied_shift_sign(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear
    context = artifacts._build_artifact_context(inputs, result)
    arrays = build_residual_static_solution_arrays(inputs, result)
    arrays['applied_residual_shift_s_sorted'] = arrays[
        'estimated_trace_delay_s_sorted'
    ].copy()

    with pytest.raises(ValueError, match='negative estimated delay'):
        artifacts._validate_solution_arrays(context, arrays)


def test_artifact_writer_rejects_object_dtype_field(
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear
    context = artifacts._build_artifact_context(inputs, result)
    arrays = build_residual_static_solution_arrays(inputs, result)
    arrays['artifact_kind'] = np.asarray(['bad'], dtype=object)

    with pytest.raises(ValueError, match='object dtype'):
        artifacts._validate_solution_arrays(context, arrays)


def test_write_residual_static_artifacts_creates_three_files(
    tmp_path: Path,
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    paths = write_residual_static_artifacts(tmp_path, inputs, result)

    assert paths.solution_npz_path == tmp_path / 'residual_static_solution.npz'
    assert paths.qc_json_path == tmp_path / 'residual_static_qc.json'
    assert paths.statics_csv_path == tmp_path / 'residual_statics.csv'
    assert paths.solution_npz_path.is_file()
    assert paths.qc_json_path.is_file()
    assert paths.statics_csv_path.is_file()


def test_write_residual_static_artifacts_npz_can_be_loaded_with_allow_pickle_false(
    tmp_path: Path,
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    paths = write_residual_static_artifacts(tmp_path, inputs, result)

    with np.load(paths.solution_npz_path, allow_pickle=False) as data:
        assert _REQUIRED_NPZ_FIELDS.issubset(data.files)
        assert all(data[key].dtype != object for key in data.files)
        np.testing.assert_allclose(
            data['applied_residual_shift_s_sorted'],
            -data['estimated_trace_delay_s_sorted'],
            rtol=0.0,
            atol=1.0e-12,
        )


def test_write_residual_static_artifacts_json_dump_uses_allow_nan_false(
    tmp_path: Path,
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    paths = write_residual_static_artifacts(tmp_path, inputs, result)
    payload = json.loads(paths.qc_json_path.read_text(encoding='utf-8'))

    json.dumps(payload, allow_nan=False)
    assert payload['artifact_kind'] == 'residual_static_qc'


def test_write_residual_static_artifacts_writes_csv_rows(
    tmp_path: Path,
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear

    paths = write_residual_static_artifacts(tmp_path, inputs, result)

    with paths.statics_csv_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == inputs.n_traces
    assert rows[0]['sorted_trace_index'] == '0'
    assert rows[0]['rejected'] == 'true'


def test_write_residual_static_artifacts_cleans_tmp_files_on_failure(
    tmp_path: Path,
    solved_linear: tuple[ResidualStaticSolverInputs, ResidualStaticRobustSolveResult],
) -> None:
    inputs, result = solved_linear
    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    (job_dir / 'residual_statics.csv').mkdir()

    with pytest.raises(OSError):
        write_residual_static_artifacts(job_dir, inputs, result)

    assert list(job_dir.glob('*.tmp-*')) == []
