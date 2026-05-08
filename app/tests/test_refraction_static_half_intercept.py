from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.services.refraction_static_design_matrix import (
    build_refraction_static_design_matrix,
)
from app.services.refraction_static_bedrock import (
    estimate_global_bedrock_slowness_from_input_model,
)
from app.services.refraction_static_half_intercept import (
    REFRACTION_HALF_INTERCEPT_NODES_CSV_NAME,
    REFRACTION_HALF_INTERCEPT_QC_JSON_NAME,
    REFRACTION_HALF_INTERCEPT_RECEIVERS_CSV_NAME,
    REFRACTION_HALF_INTERCEPT_SOURCES_CSV_NAME,
    REFRACTION_HALF_INTERCEPT_TRACE_PREVIEW_CSV_NAME,
    RefractionHalfInterceptTimeError,
    build_refraction_half_intercept_time_model,
    build_refraction_half_intercept_time_model_from_bedrock_result,
    estimate_refraction_half_intercept_times_from_first_breaks,
)
from app.services.refraction_static_inputs import (
    RefractionEndpointTable,
    RefractionStaticInputModel,
)
from app.services.refraction_static_solver import solve_refraction_static_bounded_ls

NODE_ID = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
ACTIVE_NODE_ID = NODE_ID[:-1]
TRUE_HALF_INTERCEPT_S = np.asarray([0.010, 0.012, 0.015, 0.018, 0.020])
TRUE_BEDROCK_VELOCITY_M_S = 2500.0
TRUE_BEDROCK_SLOWNESS_S_PER_M = 1.0 / TRUE_BEDROCK_VELOCITY_M_S
WEATHERING_VELOCITY_M_S = 800.0
ROW_SOURCE_NODE_ID = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 0, 1, 2, 3, 4],
    dtype=np.int64,
)
ROW_RECEIVER_NODE_ID = np.asarray(
    [1, 2, 3, 4, 2, 3, 4, 3, 4, 4, 0, 1, 2, 3, 4],
    dtype=np.int64,
)
ROW_DISTANCE_M = np.asarray(
    [
        120.0,
        230.0,
        360.0,
        520.0,
        140.0,
        270.0,
        410.0,
        160.0,
        310.0,
        190.0,
        60.0,
        75.0,
        90.0,
        105.0,
        130.0,
    ],
    dtype=np.float64,
)


def _model(**overrides: Any) -> RefractionStaticModelRequest:
    payload = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': WEATHERING_VELOCITY_M_S,
        'bedrock_velocity_mode': 'solve_global',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': TRUE_BEDROCK_VELOCITY_M_S,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
    }
    payload.update(overrides)
    return RefractionStaticModelRequest.model_validate(payload)


def _solver(**overrides: Any) -> RefractionStaticSolverRequest:
    payload = {
        'damping': 0.0,
        'min_picks_per_node': 1,
        'max_abs_half_intercept_time_ms': 500.0,
        'robust': {
            'enabled': False,
            'method': 'mad',
            'threshold': 3.5,
            'max_iterations': 5,
            'min_used_fraction': 0.5,
            'min_used_observations': 1,
        },
    }
    robust_overrides = overrides.pop('robust', None)
    payload.update(overrides)
    if robust_overrides is not None:
        payload['robust'].update(robust_overrides)
    return RefractionStaticSolverRequest.model_validate(payload)


def _pick_times() -> np.ndarray:
    picks = (
        TRUE_HALF_INTERCEPT_S[ROW_SOURCE_NODE_ID]
        + TRUE_HALF_INTERCEPT_S[ROW_RECEIVER_NODE_ID]
        + ROW_DISTANCE_M / TRUE_BEDROCK_VELOCITY_M_S
    )
    return np.ascontiguousarray(picks, dtype=np.float64)


def _endpoint_table() -> RefractionEndpointTable:
    n_nodes = int(NODE_ID.shape[0])
    return RefractionEndpointTable(
        node_id=np.ascontiguousarray(NODE_ID, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(NODE_ID, dtype=np.int64),
        x_m=NODE_ID.astype(np.float64) * 100.0,
        y_m=np.zeros(n_nodes, dtype=np.float64),
        elevation_m=np.zeros(n_nodes, dtype=np.float64),
        kind=np.full(n_nodes, 'linked', dtype='<U16'),
        pick_count=np.zeros(n_nodes, dtype=np.int64),
    )


def _input_model(
    *,
    pick_time_s_sorted: np.ndarray | None = None,
    valid_observation_mask_sorted: np.ndarray | None = None,
    source_node_id_sorted: np.ndarray | None = None,
    receiver_node_id_sorted: np.ndarray | None = None,
    distance_m_sorted: np.ndarray | None = None,
) -> RefractionStaticInputModel:
    valid_picks = _pick_times() if pick_time_s_sorted is None else pick_time_s_sorted
    source = (
        ROW_SOURCE_NODE_ID
        if source_node_id_sorted is None
        else np.asarray(source_node_id_sorted, dtype=np.int64)
    )
    receiver = (
        ROW_RECEIVER_NODE_ID
        if receiver_node_id_sorted is None
        else np.asarray(receiver_node_id_sorted, dtype=np.int64)
    )
    distance = (
        ROW_DISTANCE_M
        if distance_m_sorted is None
        else np.asarray(distance_m_sorted, dtype=np.float64)
    )
    picks = np.concatenate((np.asarray(valid_picks, dtype=np.float64), [np.nan]))
    source = np.concatenate((source, [5])).astype(np.int64, copy=False)
    receiver = np.concatenate((receiver, [5])).astype(np.int64, copy=False)
    distance = np.concatenate((distance, [np.nan])).astype(np.float64, copy=False)
    n_traces = int(picks.shape[0])
    if valid_observation_mask_sorted is None:
        valid = np.concatenate((np.ones(n_traces - 1, dtype=bool), [False]))
    else:
        valid = np.asarray(valid_observation_mask_sorted, dtype=bool)
    trace_index = np.arange(n_traces, dtype=np.int64)
    node_x = NODE_ID.astype(np.float64) * 100.0
    source_x = node_x[source]
    receiver_x = node_x[receiver]
    zeros = np.zeros(n_traces, dtype=np.float64)
    return RefractionStaticInputModel(
        file_id='synthetic-file-id',
        n_traces=n_traces,
        sorted_trace_index=trace_index,
        pick_time_s_sorted=np.ascontiguousarray(picks, dtype=np.float64),
        valid_pick_mask_sorted=np.isfinite(picks),
        valid_observation_mask_sorted=np.ascontiguousarray(valid, dtype=bool),
        source_id_sorted=source.copy(),
        receiver_id_sorted=receiver.copy(),
        source_x_m_sorted=np.ascontiguousarray(source_x, dtype=np.float64),
        source_y_m_sorted=zeros.copy(),
        receiver_x_m_sorted=np.ascontiguousarray(receiver_x, dtype=np.float64),
        receiver_y_m_sorted=zeros.copy(),
        source_elevation_m_sorted=zeros.copy(),
        receiver_elevation_m_sorted=zeros.copy(),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.nan_to_num(distance, nan=0.0),
        offset_m_sorted=None,
        distance_m_sorted=np.ascontiguousarray(distance, dtype=np.float64),
        source_endpoint_key_sorted=np.asarray(
            [f'source:{int(value)}' for value in source],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{int(value)}' for value in receiver],
            dtype='<U32',
        ),
        source_node_id_sorted=np.ascontiguousarray(source, dtype=np.int64),
        receiver_node_id_sorted=np.ascontiguousarray(receiver, dtype=np.int64),
        node_x_m=np.ascontiguousarray(node_x, dtype=np.float64),
        node_y_m=np.zeros(int(NODE_ID.shape[0]), dtype=np.float64),
        node_elevation_m=np.zeros(int(NODE_ID.shape[0]), dtype=np.float64),
        node_kind=np.full(int(NODE_ID.shape[0]), 'linked', dtype='<U16'),
        rejection_reason_sorted=np.asarray(['ok'] * (n_traces - 1) + ['missing_pick']),
        qc={'linkage_used': True},
        endpoint_table=_endpoint_table(),
        metadata={},
    )


def _build_result(
    *,
    input_model: RefractionStaticInputModel | None = None,
    solver_request: RefractionStaticSolverRequest | None = None,
):
    model = _model()
    solver = _solver() if solver_request is None else solver_request
    inputs = _input_model() if input_model is None else input_model
    design = build_refraction_static_design_matrix(input_model=inputs, model=model)
    solved = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=model,
        solver=solver,
    )
    result = build_refraction_half_intercept_time_model(
        input_model=inputs,
        design_matrix=design,
        solver_result=solved,
        weathering_velocity_m_s=model.weathering_velocity_m_s,
        min_picks_per_node=solver.min_picks_per_node,
    )
    return inputs, design, solved, result


def test_public_apis_are_importable() -> None:
    assert callable(estimate_refraction_half_intercept_times_from_first_breaks)
    assert callable(build_refraction_half_intercept_time_model)
    assert callable(build_refraction_half_intercept_time_model_from_bedrock_result)


def test_build_from_bedrock_result_uses_debug_objects() -> None:
    inputs = _input_model()
    bedrock = estimate_global_bedrock_slowness_from_input_model(
        input_model=inputs,
        model=_model(),
        solver=_solver(),
        include_debug_objects=True,
    )

    result = build_refraction_half_intercept_time_model_from_bedrock_result(
        bedrock_result=bedrock
    )

    np.testing.assert_allclose(
        result.node_half_intercept_time_s[:5],
        TRUE_HALF_INTERCEPT_S,
        atol=1.0e-9,
    )
    assert result.bedrock_velocity_m_s == pytest.approx(
        TRUE_BEDROCK_VELOCITY_M_S,
        rel=1.0e-7,
    )


def test_node_endpoint_and_trace_tables_map_half_intercepts() -> None:
    inputs, _design, _solved, result = _build_result()

    np.testing.assert_array_equal(result.node_id, NODE_ID)
    np.testing.assert_allclose(
        result.node_half_intercept_time_s[:5],
        TRUE_HALF_INTERCEPT_S,
        atol=1.0e-9,
    )
    assert np.isnan(result.node_half_intercept_time_s[5])
    assert result.node_solution_status[5] == 'inactive'
    assert result.qc['n_inactive_nodes'] == 1

    source_zero = int(np.flatnonzero(result.source_node_id == 0)[0])
    receiver_zero = int(np.flatnonzero(result.receiver_node_id == 0)[0])
    assert result.source_half_intercept_time_s[source_zero] == pytest.approx(0.010)
    assert result.receiver_half_intercept_time_s[receiver_zero] == pytest.approx(0.010)
    assert result.source_endpoint_key.tolist() == [
        'source:0',
        'source:1',
        'source:2',
        'source:3',
        'source:4',
        'source:5',
    ]
    assert result.receiver_endpoint_key.tolist() == [
        'receiver:1',
        'receiver:2',
        'receiver:3',
        'receiver:4',
        'receiver:0',
        'receiver:5',
    ]
    assert np.isnan(result.source_half_intercept_time_s[-1])
    assert np.isnan(result.receiver_half_intercept_time_s[-1])

    assert result.sorted_trace_index.shape == (inputs.n_traces,)
    valid = result.valid_observation_mask_sorted
    np.testing.assert_allclose(
        result.estimated_intercept_time_sum_s_sorted[valid],
        TRUE_HALF_INTERCEPT_S[result.source_node_id_sorted[valid]]
        + TRUE_HALF_INTERCEPT_S[result.receiver_node_id_sorted[valid]],
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.estimated_bedrock_moveout_time_s_sorted[valid],
        ROW_DISTANCE_M / TRUE_BEDROCK_VELOCITY_M_S,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.estimated_first_break_time_s_sorted[valid],
        inputs.pick_time_s_sorted[valid],
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.first_break_residual_s_sorted[valid],
        0.0,
        atol=1.0e-9,
    )
    assert np.isnan(result.estimated_first_break_time_s_sorted[-1])
    assert np.isnan(result.first_break_residual_s_sorted[-1])
    assert not result.used_observation_mask_sorted[-1]


def test_endpoint_keys_preserve_long_distinct_strings_without_truncation() -> None:
    inputs = _input_model()
    common_prefix = 'k' * 192
    source_keys_by_node = {
        int(node): f'{common_prefix}source:{int(node)}' for node in NODE_ID.tolist()
    }
    receiver_keys_by_node = {
        int(node): f'{common_prefix}receiver:{int(node)}' for node in NODE_ID.tolist()
    }
    source_key_sorted = np.asarray(
        [source_keys_by_node[int(node)] for node in inputs.source_node_id_sorted],
        dtype=object,
    )
    receiver_key_sorted = np.asarray(
        [receiver_keys_by_node[int(node)] for node in inputs.receiver_node_id_sorted],
        dtype=object,
    )
    assert len({key[:192] for key in source_key_sorted.tolist()}) == 1
    assert len({key[:192] for key in receiver_key_sorted.tolist()}) == 1

    inputs = replace(
        inputs,
        source_endpoint_key_sorted=source_key_sorted,
        receiver_endpoint_key_sorted=receiver_key_sorted,
    )
    _inputs, _design, _solved, result = _build_result(input_model=inputs)

    assert result.source_endpoint_key.tolist() == [
        source_keys_by_node[node] for node in [0, 1, 2, 3, 4, 5]
    ]
    assert result.receiver_endpoint_key.tolist() == [
        receiver_keys_by_node[node] for node in [1, 2, 3, 4, 0, 5]
    ]
    assert result.source_endpoint_key.dtype == object
    assert result.receiver_endpoint_key.dtype == object


def test_node_statuses_report_bounds_and_low_fold() -> None:
    inputs, design, solved, _result = _build_result()
    active_half = solved.active_node_half_intercept_time_s.copy()
    active_half[0] = 0.0
    active_half[1] = 0.012
    lower = solved.lower_bounds.copy()
    upper = solved.upper_bounds.copy()
    lower[0] = 0.0
    upper[1] = 0.012

    clipped = build_refraction_half_intercept_time_model(
        input_model=inputs,
        design_matrix=design,
        solver_result=replace(
            solved,
            active_node_half_intercept_time_s=active_half,
            lower_bounds=lower,
            upper_bounds=upper,
        ),
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        min_picks_per_node=1,
    )
    assert clipped.node_solution_status[0] == 'clipped_lower'
    assert clipped.node_solution_status[1] == 'clipped_upper'

    low_fold = build_refraction_half_intercept_time_model(
        input_model=inputs,
        design_matrix=design,
        solver_result=solved,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        min_picks_per_node=100,
    )
    assert 'low_fold' in low_fold.node_solution_status.tolist()
    assert low_fold.qc['low_fold_node_count'] == 5


def test_residual_aggregation_uses_used_rows_without_same_node_double_count() -> None:
    inputs, design, solved, _result = _build_result()
    residual = np.arange(1, solved.residual_time_s.shape[0] + 1, dtype=np.float64)
    residual *= 0.001
    used = np.ones_like(solved.used_row_mask, dtype=bool)
    rejected = np.zeros_like(solved.rejected_by_robust_mask, dtype=bool)
    used[0] = False
    rejected[0] = True

    result = build_refraction_half_intercept_time_model(
        input_model=inputs,
        design_matrix=design,
        solver_result=replace(
            solved,
            residual_time_s=residual,
            used_row_mask=used,
            rejected_by_robust_mask=rejected,
        ),
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        min_picks_per_node=1,
    )

    assert result.node_pick_count[0] == 5
    assert result.node_used_pick_count[0] == 4
    assert result.node_rejected_pick_count[0] == 1
    expected_node0 = residual[[1, 2, 3, 10]]
    assert result.node_residual_rms_s[0] == pytest.approx(
        float(np.sqrt(np.mean(expected_node0 * expected_node0)))
    )
    source_zero = int(np.flatnonzero(result.source_node_id == 0)[0])
    receiver_zero = int(np.flatnonzero(result.receiver_node_id == 0)[0])
    assert result.source_residual_rms_s[source_zero] == pytest.approx(
        float(np.sqrt(np.mean(expected_node0 * expected_node0)))
    )
    assert result.receiver_residual_rms_s[receiver_zero] == pytest.approx(
        float(abs(residual[10]))
    )
    assert not result.used_observation_mask_sorted[0]


@pytest.mark.parametrize(
    ('patch_name', 'patch_value', 'match'),
    [
        ('bedrock_velocity_m_s', np.inf, 'bedrock_velocity'),
        ('bedrock_velocity_m_s', 0.0, 'bedrock_velocity'),
        ('active_node_id', np.asarray([999, 1, 2, 3, 4]), 'unknown active'),
        (
            'active_node_half_intercept_time_s',
            np.asarray([np.nan, 0.012, 0.015, 0.018, 0.020]),
            'finite',
        ),
        (
            'active_node_half_intercept_time_s',
            np.asarray([-0.001, 0.012, 0.015, 0.018, 0.020]),
            'non-negative',
        ),
        ('residual_time_s', np.zeros(14, dtype=np.float64), 'shape mismatch'),
        ('modeled_pick_time_s', None, 'required'),
    ],
)
def test_validation_rejects_solver_result_errors(
    patch_name: str,
    patch_value: object,
    match: str,
) -> None:
    inputs, design, solved, _result = _build_result()
    patched = replace(solved, **{patch_name: patch_value})

    with pytest.raises(RefractionHalfInterceptTimeError, match=match):
        build_refraction_half_intercept_time_model(
            input_model=inputs,
            design_matrix=design,
            solver_result=patched,
            weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        )


def test_validation_rejects_velocity_not_greater_than_weathering() -> None:
    inputs, design, solved, _result = _build_result()

    with pytest.raises(RefractionHalfInterceptTimeError, match='weathering'):
        build_refraction_half_intercept_time_model(
            input_model=inputs,
            design_matrix=design,
            solver_result=solved,
            weathering_velocity_m_s=3000.0,
        )


def test_validation_rejects_unknown_source_and_receiver_nodes() -> None:
    inputs, design, solved, _result = _build_result()
    bad_source = inputs.source_node_id_sorted.copy()
    bad_source[0] = 999
    with pytest.raises(RefractionHalfInterceptTimeError, match='source'):
        build_refraction_half_intercept_time_model(
            input_model=replace(inputs, source_node_id_sorted=bad_source),
            design_matrix=design,
            solver_result=solved,
            weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        )

    bad_receiver = inputs.receiver_node_id_sorted.copy()
    bad_receiver[0] = 999
    with pytest.raises(RefractionHalfInterceptTimeError, match='receiver'):
        build_refraction_half_intercept_time_model(
            input_model=replace(inputs, receiver_node_id_sorted=bad_receiver),
            design_matrix=design,
            solver_result=solved,
            weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        )


def test_validation_rejects_source_receiver_sorted_length_mismatch() -> None:
    inputs, design, solved, _result = _build_result()

    with pytest.raises(RefractionHalfInterceptTimeError, match='shape mismatch'):
        build_refraction_half_intercept_time_model(
            input_model=replace(
                inputs,
                source_node_id_sorted=inputs.source_node_id_sorted[:-1],
            ),
            design_matrix=design,
            solver_result=solved,
            weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        )


def test_artifacts_write_qc_and_csv_tables(tmp_path: Path) -> None:
    inputs = _input_model()
    model = _model()
    solver = _solver()
    design = build_refraction_static_design_matrix(input_model=inputs, model=model)
    solved = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=model,
        solver=solver,
    )
    result = build_refraction_half_intercept_time_model(
        input_model=inputs,
        design_matrix=design,
        solver_result=solved,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        min_picks_per_node=1,
        job_dir=tmp_path,
    )

    qc_path = tmp_path / REFRACTION_HALF_INTERCEPT_QC_JSON_NAME
    nodes_path = tmp_path / REFRACTION_HALF_INTERCEPT_NODES_CSV_NAME
    sources_path = tmp_path / REFRACTION_HALF_INTERCEPT_SOURCES_CSV_NAME
    receivers_path = tmp_path / REFRACTION_HALF_INTERCEPT_RECEIVERS_CSV_NAME
    trace_path = tmp_path / REFRACTION_HALF_INTERCEPT_TRACE_PREVIEW_CSV_NAME
    for path in (qc_path, nodes_path, sources_path, receivers_path, trace_path):
        assert path.is_file()

    qc = json.loads(qc_path.read_text(encoding='utf-8'))
    assert qc['n_nodes'] == 6
    assert qc['n_source_endpoints'] == 6
    assert qc['bedrock_velocity_m_s'] == pytest.approx(result.bedrock_velocity_m_s)
    assert qc['residual_rms_ms'] == pytest.approx(0.0, abs=1.0e-6)

    with nodes_path.open(encoding='utf-8', newline='') as handle:
        node_rows = list(csv.DictReader(handle))
    assert 'half_intercept_time_ms' in node_rows[0]
    assert node_rows[-1]['solution_status'] == 'inactive'

    with trace_path.open(encoding='utf-8', newline='') as handle:
        trace_rows = list(csv.DictReader(handle))
    assert 'estimated_first_break_time_ms' in trace_rows[0]
    assert trace_rows[-1]['valid_observation'] == 'False'
