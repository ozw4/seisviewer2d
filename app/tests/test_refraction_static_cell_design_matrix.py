from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from app.api.schemas import RefractionStaticModelRequest
from app.statics.refraction.application.design_matrix import (
    build_refraction_static_design_matrix,
)
from app.statics.refraction.contracts.result_types import (
    RefractionEndpointTable,
    RefractionStaticInputModel,
)


def _model(
    *,
    number_of_cell_x: int = 4,
    size_of_cell_x_m: float = 10.0,
    min_observations_per_cell: int = 1,
    **overrides: Any,
) -> RefractionStaticModelRequest:
    payload: dict[str, Any] = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': 800.0,
        'bedrock_velocity_mode': 'solve_cell',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': 2500.0,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
        'refractor_cell': {
            'number_of_cell_x': number_of_cell_x,
            'size_of_cell_x_m': size_of_cell_x_m,
            'x_coordinate_origin_m': 0.0,
            'number_of_cell_y': 1,
            'size_of_cell_y_m': None,
            'y_coordinate_origin_m': 0.0,
            'assignment_mode': 'midpoint',
            'outside_grid_policy': 'reject',
            'min_observations_per_cell': min_observations_per_cell,
            'velocity_smoothing_weight': 0.0,
            'smoothing_reference_distance_m': None,
        },
    }
    payload.update(overrides)
    return RefractionStaticModelRequest.model_validate(payload)


def _global_model() -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'gli_variable_thickness',
            'weathering_velocity_m_s': 800.0,
            'bedrock_velocity_mode': 'solve_global',
            'bedrock_velocity_m_s': None,
            'initial_bedrock_velocity_m_s': 2500.0,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
            'max_weathering_thickness_m': None,
        }
    )


def _endpoint_table(node_id: np.ndarray) -> RefractionEndpointTable:
    n_nodes = int(node_id.shape[0])
    return RefractionEndpointTable(
        node_id=np.ascontiguousarray(node_id, dtype=np.int64),
        endpoint_id=np.ascontiguousarray(node_id, dtype=np.int64),
        x_m=np.arange(n_nodes, dtype=np.float64),
        y_m=np.zeros(n_nodes, dtype=np.float64),
        elevation_m=np.zeros(n_nodes, dtype=np.float64),
        kind=np.full(n_nodes, 'linked', dtype='<U16'),
        pick_count=np.zeros(n_nodes, dtype=np.int64),
    )


def _input_model(
    *,
    midpoint_x_m: list[float],
    source_node_id: list[int],
    receiver_node_id: list[int],
    distance_m: list[float],
    valid_observation: list[bool] | None = None,
    node_id: list[int] | None = None,
) -> RefractionStaticInputModel:
    n_traces = len(midpoint_x_m)
    if valid_observation is None:
        valid_observation = [True] * n_traces
    if node_id is None:
        node_id = sorted(set(source_node_id) | set(receiver_node_id))
    midpoint = np.asarray(midpoint_x_m, dtype=np.float64)
    source_x = midpoint - 1.0
    receiver_x = midpoint + 1.0
    source_node = np.asarray(source_node_id, dtype=np.int64)
    receiver_node = np.asarray(receiver_node_id, dtype=np.int64)
    distance = np.asarray(distance_m, dtype=np.float64)
    valid = np.asarray(valid_observation, dtype=bool)
    trace_index = np.arange(n_traces, dtype=np.int64)
    zeros = np.zeros(n_traces, dtype=np.float64)
    return RefractionStaticInputModel(
        file_id='file-id',
        n_traces=n_traces,
        sorted_trace_index=trace_index,
        pick_time_s_sorted=np.linspace(0.1, 0.1 * n_traces, n_traces),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=valid,
        source_id_sorted=np.arange(100, 100 + n_traces, dtype=np.int64),
        receiver_id_sorted=np.arange(200, 200 + n_traces, dtype=np.int64),
        source_x_m_sorted=source_x,
        source_y_m_sorted=zeros.copy(),
        receiver_x_m_sorted=receiver_x,
        receiver_y_m_sorted=zeros.copy(),
        source_elevation_m_sorted=zeros.copy(),
        receiver_elevation_m_sorted=zeros.copy(),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance.copy(),
        offset_m_sorted=None,
        distance_m_sorted=distance,
        source_endpoint_key_sorted=np.asarray(
            [f's:{value}' for value in source_node],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'r:{value}' for value in receiver_node],
            dtype='<U32',
        ),
        source_node_id_sorted=source_node,
        receiver_node_id_sorted=receiver_node,
        node_x_m=np.arange(len(node_id), dtype=np.float64),
        node_y_m=np.zeros(len(node_id), dtype=np.float64),
        node_elevation_m=np.zeros(len(node_id), dtype=np.float64),
        node_kind=np.full(len(node_id), 'linked', dtype='<U16'),
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={},
        endpoint_table=_endpoint_table(np.asarray(node_id, dtype=np.int64)),
        metadata={},
    )


def test_cell_design_matrix_has_node_and_cell_slowness_columns() -> None:
    design = build_refraction_static_design_matrix(
        input_model=_input_model(
            midpoint_x_m=[5.0, 15.0, 25.0],
            source_node_id=[10, 20, 30],
            receiver_node_id=[20, 30, 40],
            distance_m=[100.0, 200.0, 300.0],
            node_id=[10, 20, 30, 40, 50],
        ),
        model=_model(number_of_cell_x=4),
    )

    assert design.bedrock_velocity_mode == 'solve_cell'
    assert design.bedrock_slowness_col is None
    assert design.bedrock_slowness_cell_col_start == 4
    assert design.matrix.shape == (3, 7)
    assert design.n_parameters == 7
    np.testing.assert_array_equal(design.active_node_id, [10, 20, 30, 40])
    np.testing.assert_array_equal(design.active_cell_id, [0, 1, 2])
    np.testing.assert_array_equal(design.inactive_cell_id, [3])
    assert design.cell_id_to_col == {0: 4, 1: 5, 2: 6}
    np.testing.assert_array_equal(np.diff(design.matrix.indptr), [3, 3, 3])


def test_cell_design_matrix_uses_midpoint_cell_coefficient_offset() -> None:
    design = build_refraction_static_design_matrix(
        input_model=_input_model(
            midpoint_x_m=[5.0, 15.0, 15.0],
            source_node_id=[10, 20, 30],
            receiver_node_id=[20, 30, 40],
            distance_m=[120.0, 240.0, 360.0],
        ),
        model=_model(number_of_cell_x=3),
    )

    assert design.row_midpoint_cell_col is not None
    dense = design.matrix.toarray()
    for row_index, cell_col in enumerate(design.row_midpoint_cell_col.tolist()):
        assert dense[row_index, cell_col] == pytest.approx(
            design.row_distance_m[row_index]
        )
    np.testing.assert_array_equal(design.row_midpoint_cell_id, [0, 1, 1])


def test_cell_design_matrix_keeps_global_path_unchanged() -> None:
    design = build_refraction_static_design_matrix(
        input_model=_input_model(
            midpoint_x_m=[5.0, 15.0, 25.0],
            source_node_id=[10, 10, 30],
            receiver_node_id=[20, 30, 30],
            distance_m=[100.0, 200.0, 400.0],
            node_id=[10, 20, 30, 40],
        ),
        model=_global_model(),
    )

    assert design.bedrock_velocity_mode == 'solve_global'
    assert design.bedrock_slowness_col == 3
    assert design.bedrock_slowness_cell_col_start is None
    assert design.row_midpoint_cell_id is None
    np.testing.assert_allclose(
        design.matrix.toarray(),
        [
            [1.0, 1.0, 0.0, 100.0],
            [1.0, 0.0, 1.0, 200.0],
            [0.0, 0.0, 2.0, 400.0],
        ],
    )


def test_cell_design_matrix_rejects_outside_grid_rows() -> None:
    design = build_refraction_static_design_matrix(
        input_model=_input_model(
            midpoint_x_m=[5.0, 25.0, 15.0],
            source_node_id=[10, 20, 30],
            receiver_node_id=[20, 30, 40],
            distance_m=[100.0, 200.0, 300.0],
        ),
        model=_model(number_of_cell_x=2),
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [0, 2])
    np.testing.assert_array_equal(design.row_midpoint_cell_id, [0, 1])
    assert design.rejection_reason_sorted is not None
    assert (
        design.rejection_reason_sorted[1]
        == 'outside_refractor_cell_grid'
    )
    assert design.qc['n_observations_outside_grid'] == 1
    assert design.qc['n_observations_used'] == 2


def test_line_2d_projected_assigns_diagonal_line_by_inline_distance() -> None:
    input_model = _input_model(
        midpoint_x_m=[5.0, 25.0],
        source_node_id=[10, 20],
        receiver_node_id=[20, 30],
        distance_m=[100.0, 200.0],
    )
    azimuth_rad = np.deg2rad(45.0)

    def to_map(inline_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            inline_m * np.sin(azimuth_rad),
            inline_m * np.cos(azimuth_rad),
        )

    source_x, source_y = to_map(input_model.source_x_m_sorted)
    receiver_x, receiver_y = to_map(input_model.receiver_x_m_sorted)
    diagonal_input = replace(
        input_model,
        source_x_m_sorted=source_x,
        source_y_m_sorted=source_y,
        receiver_x_m_sorted=receiver_x,
        receiver_y_m_sorted=receiver_y,
    )
    payload = _model(number_of_cell_x=3).model_dump(mode='json')
    payload['refractor_cell'].update(
        {
            'coordinate_mode': 'line_2d_projected',
            'line_origin_x_m': 0.0,
            'line_origin_y_m': 0.0,
            'line_azimuth_deg': 45.0,
        }
    )
    design = build_refraction_static_design_matrix(
        input_model=diagonal_input,
        model=RefractionStaticModelRequest.model_validate(payload),
    )

    np.testing.assert_array_equal(design.row_midpoint_cell_id, [0, 2])
    assert design.qc['coordinate_mode'] == 'line_2d_projected'
    assert design.qc['line_azimuth_deg'] == pytest.approx(45.0)


def test_cell_design_matrix_marks_empty_cells_inactive() -> None:
    design = build_refraction_static_design_matrix(
        input_model=_input_model(
            midpoint_x_m=[5.0, 35.0],
            source_node_id=[10, 20],
            receiver_node_id=[20, 30],
            distance_m=[100.0, 200.0],
        ),
        model=_model(number_of_cell_x=4),
    )

    np.testing.assert_array_equal(design.active_cell_id, [0, 3])
    np.testing.assert_array_equal(design.inactive_cell_id, [1, 2])
    assert design.n_active_cells == 2
    assert design.n_inactive_cells == 2
    assert design.matrix.shape[1] == design.n_active_nodes + 2


def test_cell_design_matrix_rejects_cells_below_min_observations_per_cell() -> None:
    design = build_refraction_static_design_matrix(
        input_model=_input_model(
            midpoint_x_m=[5.0, 5.0, 15.0, 25.0, 25.0],
            source_node_id=[10, 20, 30, 40, 50],
            receiver_node_id=[20, 30, 40, 50, 60],
            distance_m=[100.0, 110.0, 120.0, 130.0, 140.0],
        ),
        model=_model(number_of_cell_x=4, min_observations_per_cell=2),
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [0, 1, 3, 4])
    np.testing.assert_array_equal(design.row_midpoint_cell_id, [0, 0, 2, 2])
    np.testing.assert_array_equal(design.active_cell_id, [0, 2])
    np.testing.assert_array_equal(design.inactive_cell_id, [1, 3])
    assert design.rejection_reason_sorted is not None
    assert design.rejection_reason_sorted[2] == 'below_min_observations_per_cell'
    assert design.qc['min_observations_per_cell'] == 2
    assert design.qc['n_low_fold_cells'] == 1
    assert design.qc['low_fold_cell_id'] == [1]
    assert design.qc['n_observations_rejected_by_low_fold_cell'] == 1
    assert design.qc['low_fold_cell_rejection_reason'] == (
        'below_min_observations_per_cell'
    )
    assert design.qc['cell_observation_count'] == [2, 1, 2, 0]
    assert design.qc['n_observations_used'] == 4


def test_cell_design_matrix_qc_reports_observation_counts_per_cell() -> None:
    design = build_refraction_static_design_matrix(
        input_model=_input_model(
            midpoint_x_m=[5.0, 5.0, 15.0, 25.0, 25.0, 25.0],
            source_node_id=[10, 20, 30, 40, 50, 60],
            receiver_node_id=[20, 30, 40, 50, 60, 70],
            distance_m=[100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        ),
        model=_model(number_of_cell_x=4),
    )

    assert design.qc['bedrock_velocity_mode'] == 'solve_cell'
    assert design.qc['cell_assignment_mode'] == 'midpoint'
    assert design.qc['coordinate_mode'] == 'grid_3d'
    assert design.qc['n_total_cells'] == 4
    assert design.qc['n_active_cells'] == 3
    assert design.qc['n_inactive_cells'] == 1
    assert design.qc['n_observations_outside_grid'] == 0
    assert design.qc['n_observations_used'] == 6
    assert design.qc['min_observations_per_active_cell'] == 1
    assert design.qc['median_observations_per_active_cell'] == pytest.approx(2.0)
    assert design.qc['max_observations_per_active_cell'] == 3
    assert design.qc['matrix_nnz'] == 18
    assert design.qc['matrix_shape'] == [6, design.n_parameters]
