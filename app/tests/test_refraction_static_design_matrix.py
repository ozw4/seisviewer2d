from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

from app.api.schemas import RefractionStaticModelRequest
from app.statics.refraction.application.design_matrix import (
    RefractionStaticDesignMatrixError,
    build_refraction_static_design_matrix,
    build_refraction_static_design_matrix_from_arrays,
)
from app.statics.refraction.domain.types import (
    RefractionEndpointTable,
    RefractionStaticDesignMatrix,
    RefractionStaticInputModel,
)


PICK_TIME = np.asarray([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
VALID_OBSERVATION = np.asarray([True, True, False, True], dtype=bool)
SOURCE_NODE_ID = np.asarray([10, 10, 20, 30], dtype=np.int64)
RECEIVER_NODE_ID = np.asarray([20, 30, 30, 30], dtype=np.int64)
DISTANCE_M = np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
NODE_ID = np.asarray([10, 20, 30, 40], dtype=np.int64)


def _model(**overrides: Any) -> RefractionStaticModelRequest:
    payload = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': 800.0,
        'bedrock_velocity_mode': 'solve_global',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': 2500.0,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
    }
    payload.update(overrides)
    return RefractionStaticModelRequest.model_validate(payload)


def _invalid_model(**overrides: Any) -> RefractionStaticModelRequest:
    payload = {
        'method': 'gli_variable_thickness',
        'weathering_velocity_m_s': 800.0,
        'bedrock_velocity_mode': 'solve_global',
        'bedrock_velocity_m_s': None,
        'initial_bedrock_velocity_m_s': 2500.0,
        'min_bedrock_velocity_m_s': 1200.0,
        'max_bedrock_velocity_m_s': 6000.0,
        'max_weathering_thickness_m': None,
    }
    payload.update(overrides)
    return RefractionStaticModelRequest.model_construct(**payload)


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
    pick_time_s_sorted: np.ndarray = PICK_TIME,
    valid_observation_mask_sorted: np.ndarray = VALID_OBSERVATION,
    source_node_id_sorted: np.ndarray = SOURCE_NODE_ID,
    receiver_node_id_sorted: np.ndarray = RECEIVER_NODE_ID,
    distance_m_sorted: np.ndarray = DISTANCE_M,
    node_id: np.ndarray = NODE_ID,
) -> RefractionStaticInputModel:
    pick_time = np.asarray(pick_time_s_sorted, dtype=np.float64)
    valid_observation = np.asarray(valid_observation_mask_sorted, dtype=bool)
    n_traces = int(valid_observation.shape[0])
    source_node_id = np.asarray(source_node_id_sorted, dtype=np.int64)
    receiver_node_id = np.asarray(receiver_node_id_sorted, dtype=np.int64)
    distance = np.asarray(distance_m_sorted, dtype=np.float64)
    node = np.asarray(node_id, dtype=np.int64)
    trace_index = np.arange(n_traces, dtype=np.int64)
    zeros = np.zeros(n_traces, dtype=np.float64)
    return RefractionStaticInputModel(
        file_id='file-id',
        n_traces=n_traces,
        sorted_trace_index=trace_index,
        pick_time_s_sorted=pick_time,
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=valid_observation,
        source_id_sorted=np.arange(100, 100 + n_traces, dtype=np.int64),
        receiver_id_sorted=np.arange(200, 200 + n_traces, dtype=np.int64),
        source_x_m_sorted=zeros.copy(),
        source_y_m_sorted=zeros.copy(),
        receiver_x_m_sorted=distance.copy(),
        receiver_y_m_sorted=zeros.copy(),
        source_elevation_m_sorted=zeros.copy(),
        receiver_elevation_m_sorted=zeros.copy(),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance.copy(),
        offset_m_sorted=None,
        distance_m_sorted=distance,
        source_endpoint_key_sorted=np.asarray(
            [f's:{value}' for value in source_node_id],
            dtype='<U32',
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'r:{value}' for value in receiver_node_id],
            dtype='<U32',
        ),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        node_x_m=np.arange(node.shape[0], dtype=np.float64),
        node_y_m=np.zeros(node.shape[0], dtype=np.float64),
        node_elevation_m=np.zeros(node.shape[0], dtype=np.float64),
        node_kind=np.full(node.shape[0], 'linked', dtype='<U16'),
        rejection_reason_sorted=np.full(n_traces, '', dtype='<U32'),
        qc={},
        endpoint_table=_endpoint_table(node),
        metadata={},
    )


def _build(
    input_model: RefractionStaticInputModel | None = None,
    model: RefractionStaticModelRequest | None = None,
) -> RefractionStaticDesignMatrix:
    return build_refraction_static_design_matrix(
        input_model=_input_model() if input_model is None else input_model,
        model=_model() if model is None else model,
    )


def test_solve_global_builds_csr_matrix_for_minimal_model() -> None:
    design = _build()

    assert sparse.isspmatrix_csr(design.matrix)
    assert design.matrix.shape == (3, 4)
    assert design.matrix.dtype == np.float64
    np.testing.assert_allclose(
        design.matrix.toarray(),
        [
            [1.0, 1.0, 0.0, 100.0],
            [1.0, 0.0, 1.0, 200.0],
            [0.0, 0.0, 2.0, 400.0],
        ],
    )


def test_solve_global_rhs_and_slowness_column_use_selected_rows() -> None:
    design = _build()

    np.testing.assert_allclose(design.rhs_s, [0.10, 0.20, 0.40])
    np.testing.assert_allclose(design.observed_pick_time_s, [0.10, 0.20, 0.40])
    np.testing.assert_allclose(design.matrix.toarray()[:, 3], [100.0, 200.0, 400.0])
    assert design.rhs_s.dtype == np.float64
    assert design.bedrock_slowness_col == 3


def test_solve_global_preserves_sorted_trace_order_and_row_metadata() -> None:
    design = _build()

    np.testing.assert_array_equal(design.row_trace_index_sorted, [0, 1, 3])
    np.testing.assert_array_equal(design.row_source_node_id, [10, 10, 30])
    np.testing.assert_array_equal(design.row_receiver_node_id, [20, 30, 30])
    np.testing.assert_allclose(design.row_distance_m, [100.0, 200.0, 400.0])


def test_solve_global_excludes_inactive_nodes_and_maps_columns_stably() -> None:
    design = _build()

    np.testing.assert_array_equal(design.active_node_id, [10, 20, 30])
    np.testing.assert_array_equal(design.inactive_node_id, [40])
    assert design.node_id_to_col == {10: 0, 20: 1, 30: 2}
    np.testing.assert_array_equal(design.source_node_col, [0, 0, 2])
    np.testing.assert_array_equal(design.receiver_node_col, [1, 2, 2])
    assert design.n_total_nodes == 4
    assert design.n_active_nodes == 3
    assert design.n_parameters == 4


def test_solve_global_preserves_input_node_order_for_active_mapping() -> None:
    design = _build(
        _input_model(node_id=np.asarray([30, 10, 20, 40], dtype=np.int64))
    )

    np.testing.assert_array_equal(design.active_node_id, [30, 10, 20])
    assert design.node_id_to_col == {30: 0, 10: 1, 20: 2}
    np.testing.assert_allclose(
        design.matrix.toarray(),
        [
            [0.0, 1.0, 1.0, 100.0],
            [1.0, 1.0, 0.0, 200.0],
            [2.0, 0.0, 0.0, 400.0],
        ],
    )


def test_solve_global_same_source_receiver_node_sums_to_two() -> None:
    design = _build()

    assert design.matrix[2, design.node_id_to_col[30]] == pytest.approx(2.0)
    assert design.matrix.nnz == 8


def test_solve_global_qc_contains_shape_sparsity_stats_and_counts() -> None:
    design = _build()

    assert design.qc['method'] == 'gli_variable_thickness'
    assert design.qc['bedrock_velocity_mode'] == 'solve_global'
    assert design.qc['matrix_shape'] == [3, 4]
    assert design.qc['matrix_nnz'] == 8
    assert design.qc['matrix_density'] == pytest.approx(8.0 / 12.0)
    assert design.qc['nnz_per_row_min'] == 2
    assert design.qc['nnz_per_row_max'] == 3
    assert design.qc['nnz_per_row_median'] == 3.0
    assert design.qc['distance_m_min'] == 100.0
    assert design.qc['distance_m_max'] == 400.0
    assert design.qc['distance_m_median'] == 200.0
    assert design.qc['pick_time_s_min'] == 0.10
    assert design.qc['pick_time_s_max'] == 0.40
    assert design.qc['pick_time_s_median'] == 0.20
    assert design.qc['source_receiver_same_node_count'] == 1
    assert design.qc['inactive_node_count'] == 1
    assert design.qc['n_inactive_nodes'] == 1
    assert design.qc['slowness_column_present'] is True
    assert design.qc['n_source_only_nodes'] == 1
    assert design.qc['n_receiver_only_nodes'] == 1
    assert design.qc['n_source_and_receiver_nodes'] == 1
    assert design.qc['n_connected_components'] == 1


def test_fixed_global_builds_matrix_without_slowness_column() -> None:
    design = _build(
        model=_model(
            bedrock_velocity_mode='fixed_global',
            bedrock_velocity_m_s=2500.0,
        )
    )

    assert sparse.isspmatrix_csr(design.matrix)
    assert design.matrix.shape == (3, 3)
    assert design.bedrock_slowness_col is None
    np.testing.assert_allclose(
        design.matrix.toarray(),
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
    )
    np.testing.assert_allclose(design.rhs_s, [0.06, 0.12, 0.24])


def test_fixed_global_stores_velocity_metadata_and_qc() -> None:
    design = _build(
        model=_model(
            bedrock_velocity_mode='fixed_global',
            bedrock_velocity_m_s=2500.0,
        )
    )

    assert design.bedrock_velocity_mode == 'fixed_global'
    assert design.fixed_bedrock_velocity_m_s == 2500.0
    assert design.fixed_bedrock_slowness_s_per_m == pytest.approx(0.0004)
    assert design.qc['fixed_bedrock_velocity_m_s'] == 2500.0
    assert design.qc['fixed_bedrock_slowness_s_per_m'] == pytest.approx(0.0004)
    assert design.qc['slowness_column_present'] is False


def test_fixed_global_rejects_missing_fixed_bedrock_velocity() -> None:
    with pytest.raises(ValueError, match='model.bedrock_velocity_m_s'):
        _build(
            model=_invalid_model(
                bedrock_velocity_mode='fixed_global',
                bedrock_velocity_m_s=None,
            )
        )


def test_fixed_global_rejects_velocity_not_greater_than_weathering_velocity() -> None:
    with pytest.raises(ValueError, match='greater than model.weathering_velocity_m_s'):
        _build(
            model=_invalid_model(
                bedrock_velocity_mode='fixed_global',
                weathering_velocity_m_s=2500.0,
                bedrock_velocity_m_s=2500.0,
            )
        )


def test_solve_global_matrix_multiplication_smoke() -> None:
    true_t = {10: 0.010, 20: 0.020, 30: 0.015}
    true_sb = 1.0 / 2500.0
    source = np.asarray([10, 10, 20], dtype=np.int64)
    receiver = np.asarray([20, 30, 30], dtype=np.int64)
    distance = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    picks = np.asarray(
        [
            true_t[int(src)] + true_t[int(rec)] + true_sb * dist
            for src, rec, dist in zip(source, receiver, distance, strict=True)
        ],
        dtype=np.float64,
    )
    design = _build(
        _input_model(
            pick_time_s_sorted=picks,
            valid_observation_mask_sorted=np.ones(3, dtype=bool),
            source_node_id_sorted=source,
            receiver_node_id_sorted=receiver,
            distance_m_sorted=distance,
            node_id=np.asarray([10, 20, 30], dtype=np.int64),
        )
    )

    p = np.r_[[true_t[int(node)] for node in design.active_node_id], true_sb]
    np.testing.assert_allclose(design.matrix @ p, design.rhs_s)


def test_fixed_global_matrix_multiplication_smoke() -> None:
    true_t = {10: 0.010, 20: 0.020, 30: 0.015}
    true_sb = 1.0 / 2500.0
    source = np.asarray([10, 10, 20], dtype=np.int64)
    receiver = np.asarray([20, 30, 30], dtype=np.int64)
    distance = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    picks = np.asarray(
        [
            true_t[int(src)] + true_t[int(rec)] + true_sb * dist
            for src, rec, dist in zip(source, receiver, distance, strict=True)
        ],
        dtype=np.float64,
    )
    design = _build(
        _input_model(
            pick_time_s_sorted=picks,
            valid_observation_mask_sorted=np.ones(3, dtype=bool),
            source_node_id_sorted=source,
            receiver_node_id_sorted=receiver,
            distance_m_sorted=distance,
            node_id=np.asarray([10, 20, 30], dtype=np.int64),
        ),
        model=_model(
            bedrock_velocity_mode='fixed_global',
            bedrock_velocity_m_s=2500.0,
        ),
    )

    p = np.asarray([true_t[int(node)] for node in design.active_node_id])
    np.testing.assert_allclose(design.matrix @ p, design.rhs_s)


def test_lower_level_array_builder_is_importable_and_builds_design() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=PICK_TIME,
        valid_observation_mask_sorted=VALID_OBSERVATION,
        source_node_id_sorted=SOURCE_NODE_ID,
        receiver_node_id_sorted=RECEIVER_NODE_ID,
        distance_m_sorted=DISTANCE_M,
        node_id=NODE_ID,
        bedrock_velocity_mode='solve_global',
    )

    assert isinstance(design, RefractionStaticDesignMatrix)
    assert design.matrix.shape == (3, 4)


def test_lower_level_array_builder_rejects_non_real_numeric_node_ids() -> None:
    with pytest.raises(
        RefractionStaticDesignMatrixError,
        match='source_node_id_sorted.*real numeric dtype',
    ):
        build_refraction_static_design_matrix_from_arrays(
            pick_time_s_sorted=PICK_TIME,
            valid_observation_mask_sorted=VALID_OBSERVATION,
            source_node_id_sorted=SOURCE_NODE_ID.astype('<U8'),
            receiver_node_id_sorted=RECEIVER_NODE_ID,
            distance_m_sorted=DISTANCE_M,
            node_id=NODE_ID,
            bedrock_velocity_mode='solve_global',
        )


@pytest.mark.parametrize(
    ('field', 'value', 'match'),
    [
        (
            'valid_observation_mask_sorted',
            np.asarray([False, False, False, False], dtype=bool),
            'at least one valid refraction observation',
        ),
        ('pick_time_s_sorted', np.asarray([0.1, 0.2]), 'pick_time_s_sorted shape'),
        (
            'source_node_id_sorted',
            np.asarray([10, 20], dtype=np.int64),
            'source_node_id_sorted shape',
        ),
        (
            'receiver_node_id_sorted',
            np.asarray([10, 20], dtype=np.int64),
            'receiver_node_id_sorted shape',
        ),
        (
            'distance_m_sorted',
            np.asarray([100.0, 200.0]),
            'distance_m_sorted shape',
        ),
        (
            'node_id',
            np.asarray([10, 20, 20, 40], dtype=np.int64),
            'node_id values must be unique',
        ),
    ],
)
def test_validation_rejects_shape_and_node_errors(
    field: str,
    value: np.ndarray,
    match: str,
) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError, match=match):
        _build(_input_model(**kwargs))


@pytest.mark.parametrize(
    ('field', 'value', 'match'),
    [
        (
            'pick_time_s_sorted',
            np.asarray([np.nan, 0.2, 0.3, 0.4], dtype=np.float64),
            'pick_time_s_sorted must be finite',
        ),
        (
            'pick_time_s_sorted',
            np.asarray([-0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            'pick_time_s_sorted must be non-negative',
        ),
        (
            'distance_m_sorted',
            np.asarray([np.inf, 200.0, 300.0, 400.0], dtype=np.float64),
            'distance_m_sorted must be finite',
        ),
        (
            'distance_m_sorted',
            np.asarray([0.0, 200.0, 300.0, 400.0], dtype=np.float64),
            'distance_m_sorted must be greater than 0',
        ),
        (
            'distance_m_sorted',
            np.asarray([-100.0, 200.0, 300.0, 400.0], dtype=np.float64),
            'distance_m_sorted must be greater than 0',
        ),
    ],
)
def test_validation_rejects_selected_bad_pick_and_distance_values(
    field: str,
    value: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _build(_input_model(**{field: value}))


def test_validation_allows_bad_values_on_unselected_rows() -> None:
    pick_time = PICK_TIME.copy()
    pick_time[2] = np.nan
    distance = DISTANCE_M.copy()
    distance[2] = -1.0

    design = _build(
        _input_model(pick_time_s_sorted=pick_time, distance_m_sorted=distance)
    )

    np.testing.assert_allclose(design.rhs_s, [0.10, 0.20, 0.40])


def test_validation_rejects_source_node_id_not_present_in_node_id() -> None:
    source = SOURCE_NODE_ID.copy()
    source[0] = 99

    with pytest.raises(ValueError, match='source_node_id_sorted.*99'):
        _build(_input_model(source_node_id_sorted=source))


def test_validation_rejects_receiver_node_id_not_present_in_node_id() -> None:
    receiver = RECEIVER_NODE_ID.copy()
    receiver[0] = 99

    with pytest.raises(ValueError, match='receiver_node_id_sorted.*99'):
        _build(_input_model(receiver_node_id_sorted=receiver))


def test_validation_rejects_unsupported_model_method() -> None:
    with pytest.raises(ValueError, match='model.method'):
        _build(model=_invalid_model(method='plus_minus'))


def test_validation_rejects_unsupported_bedrock_velocity_mode() -> None:
    with pytest.raises(ValueError, match='bedrock_velocity_mode'):
        _build(model=_invalid_model(bedrock_velocity_mode='per_node'))
