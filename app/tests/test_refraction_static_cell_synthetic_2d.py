from __future__ import annotations

import numpy as np
import pytest

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.services.refraction_static_design_matrix import (
    build_refraction_static_design_matrix,
)
from app.services.refraction_static_solver import solve_refraction_static_bounded_ls
from app.services.refraction_static_types import (
    RefractionEndpointTable,
    RefractionStaticDesignMatrix,
    RefractionStaticInputModel,
    RefractionStaticSolverResult,
)
from app.tests.fixtures.refraction_synthetic import (
    SyntheticRefractionCellDataset,
    make_clean_2d_cell_refraction_dataset,
)

TRUE_CELL_V2_M_S = np.asarray([2200.0, 2400.0, 2600.0, 2800.0], dtype=np.float64)
MIN_OBSERVATIONS_PER_CELL = 5
SLOWNESS_ATOL_S_PER_M = 1.0e-12
VELOCITY_ATOL_M_S = 1.0e-5
TIME_ATOL_S = 1.0e-8


def test_cell_v2_t1_clean_2d_recovers_known_cell_velocities() -> None:
    dataset, design, result = _solve_clean_2d_cell_problem()

    expected_cell_id = np.arange(TRUE_CELL_V2_M_S.shape[0], dtype=np.int64)
    np.testing.assert_array_equal(design.active_cell_id, expected_cell_id)
    np.testing.assert_array_equal(result.active_cell_id, expected_cell_id)
    np.testing.assert_array_equal(design.inactive_cell_id, [])
    np.testing.assert_array_equal(
        result.row_midpoint_cell_id,
        dataset.true_cell_id_for_pick[result.row_trace_index_sorted],
    )
    assert np.all(dataset.cell_observation_count >= MIN_OBSERVATIONS_PER_CELL)

    np.testing.assert_allclose(
        result.cell_bedrock_slowness_s_per_m,
        1.0 / TRUE_CELL_V2_M_S,
        atol=SLOWNESS_ATOL_S_PER_M,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        TRUE_CELL_V2_M_S,
        atol=VELOCITY_ATOL_M_S,
        rtol=0.0,
    )


def test_cell_v2_t1_clean_2d_predicts_pick_times_with_near_zero_residual() -> None:
    _, _, result = _solve_clean_2d_cell_problem()

    np.testing.assert_allclose(
        result.modeled_pick_time_s,
        result.observed_pick_time_s,
        atol=TIME_ATOL_S,
        rtol=0.0,
    )
    assert _rms(result.residual_time_s) <= TIME_ATOL_S
    np.testing.assert_allclose(result.residual_time_s, 0.0, atol=TIME_ATOL_S)


def test_cell_v2_t1_clean_2d_summary_velocity_is_reciprocal_consistent() -> None:
    _, _, result = _solve_clean_2d_cell_problem()

    expected_summary_slowness = float(np.median(1.0 / TRUE_CELL_V2_M_S))

    assert result.bedrock_slowness_s_per_m == pytest.approx(
        expected_summary_slowness,
        abs=SLOWNESS_ATOL_S_PER_M,
    )
    assert result.bedrock_velocity_m_s == pytest.approx(
        1.0 / result.bedrock_slowness_s_per_m,
        abs=VELOCITY_ATOL_M_S,
    )
    assert result.qc['bedrock_velocity_solution_kind'] == 'per_cell'


def test_cell_v2_t1_clean_2d_t1_is_correct_up_to_gauge() -> None:
    dataset, design, result = _solve_clean_2d_cell_problem()

    node_t1_by_id = {
        int(node_id): float(result.node_half_intercept_time_s[index])
        for index, node_id in enumerate(result.node_id.tolist())
    }
    solved_source_t1 = np.asarray(
        [node_t1_by_id[int(node_id)] for node_id in design.row_source_node_id],
        dtype=np.float64,
    )
    solved_receiver_t1 = np.asarray(
        [node_t1_by_id[int(node_id)] for node_id in design.row_receiver_node_id],
        dtype=np.float64,
    )
    row_index = result.row_trace_index_sorted

    np.testing.assert_allclose(
        solved_source_t1 + solved_receiver_t1,
        dataset.true_source_t1_s[row_index] + dataset.true_receiver_t1_s[row_index],
        atol=TIME_ATOL_S,
        rtol=0.0,
    )


def _solve_clean_2d_cell_problem() -> tuple[
    SyntheticRefractionCellDataset,
    RefractionStaticDesignMatrix,
    RefractionStaticSolverResult,
]:
    dataset = make_clean_2d_cell_refraction_dataset(
        seed=431,
        cell_v2_m_s=TRUE_CELL_V2_M_S,
        n_sources=12,
        n_receivers=12,
        noise_std_s=0.0,
        outlier_fraction=0.0,
    )
    model = _cell_model(dataset)
    design = build_refraction_static_design_matrix(
        input_model=_input_model_from_dataset(dataset),
        model=model,
    )
    result = solve_refraction_static_bounded_ls(
        design_matrix=design,
        model=model,
        solver=_solver(),
    )
    return dataset, design, result


def _cell_model(dataset: SyntheticRefractionCellDataset) -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'gli_variable_thickness',
            'weathering_velocity_m_s': dataset.true_v1_m_s,
            'bedrock_velocity_mode': 'solve_cell',
            'bedrock_velocity_m_s': None,
            'initial_bedrock_velocity_m_s': 2600.0,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
            'max_weathering_thickness_m': None,
            'refractor_cell': {
                'number_of_cell_x': int(dataset.true_cell_v2_m_s.shape[1]),
                'size_of_cell_x_m': dataset.cell_size_x_m,
                'x_coordinate_origin_m': dataset.x_coordinate_origin_m,
                'number_of_cell_y': 1,
                'size_of_cell_y_m': None,
                'y_coordinate_origin_m': dataset.y_coordinate_origin_m,
                'assignment_mode': 'midpoint',
                'outside_grid_policy': 'reject',
                'coordinate_mode': 'grid_3d',
                'min_observations_per_cell': MIN_OBSERVATIONS_PER_CELL,
                'velocity_smoothing_weight': 0.0,
                'smoothing_reference_distance_m': None,
            },
        }
    )


def _solver() -> RefractionStaticSolverRequest:
    return RefractionStaticSolverRequest.model_validate(
        {
            'damping': 0.0,
            'min_picks_per_node': 1,
            'max_abs_half_intercept_time_ms': 200.0,
            'robust': {
                'enabled': False,
                'method': 'mad',
                'threshold': 3.5,
                'max_iterations': 5,
                'min_used_fraction': 0.5,
                'min_used_observations': 1,
            },
        }
    )


def _input_model_from_dataset(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionStaticInputModel:
    n_traces = int(dataset.pick_time_s.shape[0])
    endpoint_table = _endpoint_table_from_dataset(dataset)
    node_x_m = np.concatenate(
        (dataset.source_endpoint_x_m, dataset.receiver_endpoint_x_m)
    )
    node_y_m = np.concatenate(
        (dataset.source_endpoint_y_m, dataset.receiver_endpoint_y_m)
    )
    node_elevation_m = np.zeros(endpoint_table.node_id.shape, dtype=np.float64)
    node_kind = np.concatenate(
        (
            np.full(dataset.source_endpoint_id.shape, 'source', dtype='<U16'),
            np.full(dataset.receiver_endpoint_id.shape, 'receiver', dtype='<U16'),
        )
    )
    return RefractionStaticInputModel(
        file_id='clean-2d-cell-synthetic',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=dataset.pick_time_s,
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=dataset.valid_mask,
        source_id_sorted=dataset.source_id,
        receiver_id_sorted=dataset.receiver_id,
        source_x_m_sorted=dataset.source_x_m,
        source_y_m_sorted=dataset.source_y_m,
        receiver_x_m_sorted=dataset.receiver_x_m,
        receiver_y_m_sorted=dataset.receiver_y_m,
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=dataset.offset_m,
        offset_m_sorted=None,
        distance_m_sorted=dataset.offset_m,
        source_endpoint_key_sorted=np.asarray(
            [f'source:{int(value)}' for value in dataset.source_id],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{int(value)}' for value in dataset.receiver_id],
            dtype='<U32',
        ),
        source_node_id_sorted=dataset.source_node_id,
        receiver_node_id_sorted=dataset.receiver_node_id,
        node_x_m=np.ascontiguousarray(node_x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(node_y_m, dtype=np.float64),
        node_elevation_m=node_elevation_m,
        node_kind=node_kind,
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={},
        endpoint_table=endpoint_table,
        metadata={'synthetic_model': 'clean_2d_cell_refraction'},
    )


def _endpoint_table_from_dataset(
    dataset: SyntheticRefractionCellDataset,
) -> RefractionEndpointTable:
    node_id = np.concatenate(
        (dataset.source_endpoint_node_id, dataset.receiver_endpoint_node_id)
    )
    endpoint_id = np.concatenate(
        (dataset.source_endpoint_id, dataset.receiver_endpoint_id)
    )
    endpoint_x_m = np.concatenate(
        (dataset.source_endpoint_x_m, dataset.receiver_endpoint_x_m)
    )
    endpoint_y_m = np.concatenate(
        (dataset.source_endpoint_y_m, dataset.receiver_endpoint_y_m)
    )
    kind = np.concatenate(
        (
            np.full(dataset.source_endpoint_id.shape, 'source', dtype='<U16'),
            np.full(dataset.receiver_endpoint_id.shape, 'receiver', dtype='<U16'),
        )
    )
    return RefractionEndpointTable(
        node_id=np.ascontiguousarray(node_id, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(endpoint_id, dtype=np.int64),
        x_m=np.ascontiguousarray(endpoint_x_m, dtype=np.float64),
        y_m=np.ascontiguousarray(endpoint_y_m, dtype=np.float64),
        elevation_m=np.zeros(node_id.shape, dtype=np.float64),
        kind=kind,
        pick_count=np.zeros(node_id.shape, dtype=np.int64),
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))
