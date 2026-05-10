from __future__ import annotations

import numpy as np
import pytest

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.services.refraction_static_layer_config import (
    normalize_refraction_static_layers,
)
from app.services.refraction_static_layer_observations import (
    build_refraction_layer_observation_masks,
)
from app.services.refraction_static_multilayer_service import (
    RefractionLayerSolverContext,
    RefractionMultiLayerSolveError,
    solve_refraction_multilayer_time_terms,
)
from app.services.refraction_static_types import (
    RefractionEndpointTable,
    RefractionLayerSolveResult,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)
from app.tests._refraction_multilayer_synthetic import (
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
    SYNTHETIC_MULTILAYER_VSUB_M_S,
    SyntheticMultiLayerRefractionDataset,
    make_2d_straight_three_layer_refraction_dataset,
    make_2d_straight_two_layer_refraction_dataset,
)


def test_multilayer_orchestrator_runs_normalized_one_layer_v2_solve() -> None:
    dataset = make_2d_straight_two_layer_refraction_dataset()
    model = _model(
        [
            _layer(
                'v2_t1',
                min_offset_m=300.0,
                max_offset_m=800.0,
                velocity_mode='solve_global',
                initial_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
                min_velocity_m_s=1600.0,
                max_velocity_m_s=3200.0,
            ),
        ]
    )
    input_model = _input_model(dataset)
    masks = build_refraction_layer_observation_masks(
        input_model=input_model,
        model=model,
    )

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        resolved_first_layer=_resolved_first_layer(),
        normalized_layers=normalize_refraction_static_layers(model),
        layer_masks=masks,
        model=model,
        solver=RefractionStaticSolverRequest(robust={'enabled': False}),
    )

    assert result.enabled_layer_kinds == ('v2_t1',)
    assert len(result.layer_results) == 1
    layer = result.layer_results[0]
    assert layer.layer_kind == 'v2_t1'
    assert layer.layer_index == 1
    assert layer.velocity_mode == 'solve_global'
    assert layer.global_velocity_m_s is not None
    assert layer.global_slowness_s_per_m == pytest.approx(
        1.0 / layer.global_velocity_m_s
    )
    assert layer.trace_predicted_time_s_sorted.shape == (dataset.sorted_trace_index.size,)
    np.testing.assert_array_equal(
        layer.used_observation_mask_sorted,
        masks.layer_used_mask_sorted['v2_t1'],
    )
    assert result.qc['enabled_layer_kinds'] == ['v2_t1']
    assert result.qc['observation_gates']['v2_t1']['n_used_observations'] == (
        int(np.count_nonzero(masks.layer_used_mask_sorted['v2_t1']))
    )


def test_multilayer_orchestrator_dispatches_enabled_layers_in_order() -> None:
    dataset = make_2d_straight_three_layer_refraction_dataset()
    model = _model(
        [
            _layer(
                'v2_t1',
                min_offset_m=300.0,
                max_offset_m=800.0,
                velocity_mode='solve_global',
                initial_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
                min_velocity_m_s=1600.0,
                max_velocity_m_s=3200.0,
            ),
            _layer(
                'v3_t2',
                min_offset_m=1000.0,
                max_offset_m=1900.0,
                velocity_mode='solve_global',
                initial_velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
                min_velocity_m_s=2600.0,
                max_velocity_m_s=4800.0,
            ),
            _layer(
                'vsub_t3',
                min_offset_m=2200.0,
                max_offset_m=None,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=SYNTHETIC_MULTILAYER_VSUB_M_S,
                min_velocity_m_s=3800.0,
                max_velocity_m_s=6200.0,
            ),
        ]
    )
    input_model = _input_model(dataset)
    masks = build_refraction_layer_observation_masks(
        input_model=input_model,
        model=model,
    )
    calls: list[str] = []

    def fake_solver(context: RefractionLayerSolverContext) -> RefractionLayerSolveResult:
        calls.append(context.layer_config.kind)
        np.testing.assert_array_equal(
            context.input_model.valid_observation_mask_sorted,
            masks.layer_used_mask_sorted[context.layer_config.kind],
        )
        return _fake_layer_result(context)

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        resolved_first_layer=_resolved_first_layer(),
        normalized_layers=normalize_refraction_static_layers(model),
        layer_masks=masks,
        model=model,
        solver=RefractionStaticSolverRequest(),
        solver_dispatch={
            ('v2_t1', 'solve_global'): fake_solver,
            ('v3_t2', 'solve_global'): fake_solver,
            ('vsub_t3', 'fixed_global'): fake_solver,
        },
    )

    assert calls == ['v2_t1', 'v3_t2', 'vsub_t3']
    assert result.enabled_layer_kinds == ('v2_t1', 'v3_t2', 'vsub_t3')
    assert [layer.layer_index for layer in result.layer_results] == [1, 2, 3]
    assert result.qc['layers']['v3_t2']['layer_kind'] == 'v3_t2'
    assert result.qc['observation_gates']['vsub_t3']['max_offset_m'] is None


def test_multilayer_orchestrator_applies_layer_cell_settings_to_layer_model() -> None:
    dataset = make_2d_straight_two_layer_refraction_dataset()
    model = RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S,
            },
            'refractor_cell': {
                'number_of_cell_x': 6,
                'size_of_cell_x_m': 300.0,
                'x_coordinate_origin_m': 0.0,
                'number_of_cell_y': 1,
                'size_of_cell_y_m': None,
                'y_coordinate_origin_m': 0.0,
                'assignment_mode': 'midpoint',
                'outside_grid_policy': 'reject',
                'coordinate_mode': 'grid_3d',
                'min_observations_per_cell': 9,
                'velocity_smoothing_weight': 0.25,
                'smoothing_reference_distance_m': None,
            },
            'layers': [
                _layer(
                    'v2_t1',
                    min_offset_m=300.0,
                    max_offset_m=800.0,
                    velocity_mode='solve_cell',
                    initial_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
                    min_velocity_m_s=1600.0,
                    max_velocity_m_s=3200.0,
                    min_observations_per_cell=2,
                    smoothing_weight=3.0,
                )
            ],
        }
    )
    input_model = _input_model(dataset)
    masks = build_refraction_layer_observation_masks(
        input_model=input_model,
        model=model,
    )

    def fake_solver(context: RefractionLayerSolverContext) -> RefractionLayerSolveResult:
        assert context.model.refractor_cell is not None
        assert context.model.refractor_cell.min_observations_per_cell == 2
        assert context.model.refractor_cell.velocity_smoothing_weight == pytest.approx(
            3.0
        )
        return _fake_layer_result(context)

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        resolved_first_layer=_resolved_first_layer(),
        normalized_layers=normalize_refraction_static_layers(model),
        layer_masks=masks,
        model=model,
        solver=RefractionStaticSolverRequest(),
        solver_dispatch={('v2_t1', 'solve_cell'): fake_solver},
    )

    assert result.layer_results[0].velocity_mode == 'solve_cell'


def test_multilayer_orchestrator_rejects_unimplemented_vsub_layer() -> None:
    dataset = make_2d_straight_three_layer_refraction_dataset()
    model = _model(
        [
            _layer(
                'v2_t1',
                min_offset_m=300.0,
                max_offset_m=800.0,
                velocity_mode='solve_global',
                initial_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
                min_velocity_m_s=1600.0,
                max_velocity_m_s=3200.0,
            ),
            _layer(
                'v3_t2',
                min_offset_m=1000.0,
                max_offset_m=1900.0,
                velocity_mode='solve_global',
                initial_velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
                min_velocity_m_s=2600.0,
                max_velocity_m_s=4800.0,
            ),
            _layer(
                'vsub_t3',
                min_offset_m=2200.0,
                max_offset_m=None,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=SYNTHETIC_MULTILAYER_VSUB_M_S,
                min_velocity_m_s=3800.0,
                max_velocity_m_s=6200.0,
            ),
        ]
    )
    input_model = _input_model(dataset)
    masks = build_refraction_layer_observation_masks(
        input_model=input_model,
        model=model,
    )

    with pytest.raises(
        RefractionMultiLayerSolveError,
        match='vsub_t3.*fixed_global.*not implemented',
    ):
        solve_refraction_multilayer_time_terms(
            input_model=input_model,
            resolved_first_layer=_resolved_first_layer(),
            normalized_layers=normalize_refraction_static_layers(model),
            layer_masks=masks,
            model=model,
            solver=RefractionStaticSolverRequest(),
        )


def test_multilayer_orchestrator_rejects_empty_layer_mask() -> None:
    dataset = make_2d_straight_two_layer_refraction_dataset()
    model = _model(
        [
            _layer(
                'v2_t1',
                min_offset_m=9000.0,
                max_offset_m=None,
                velocity_mode='solve_global',
                initial_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
                min_velocity_m_s=1600.0,
                max_velocity_m_s=3200.0,
            ),
        ]
    )
    input_model = _input_model(dataset)
    masks = build_refraction_layer_observation_masks(
        input_model=input_model,
        model=model,
    )

    with pytest.raises(
        RefractionMultiLayerSolveError,
        match='v2_t1 has no valid observations',
    ):
        solve_refraction_multilayer_time_terms(
            input_model=input_model,
            resolved_first_layer=_resolved_first_layer(),
            normalized_layers=normalize_refraction_static_layers(model),
            layer_masks=masks,
            model=model,
            solver=RefractionStaticSolverRequest(),
        )


def _model(layers: list[dict[str, object]]) -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S,
            },
            'layers': layers,
        }
    )


def _layer(
    kind: str,
    *,
    min_offset_m: float,
    max_offset_m: float | None,
    velocity_mode: str,
    initial_velocity_m_s: float | None = None,
    fixed_velocity_m_s: float | None = None,
    min_velocity_m_s: float,
    max_velocity_m_s: float,
    min_observations_per_cell: int | None = None,
    smoothing_weight: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        'kind': kind,
        'enabled': True,
        'min_offset_m': min_offset_m,
        'max_offset_m': max_offset_m,
        'velocity_mode': velocity_mode,
        'min_velocity_m_s': min_velocity_m_s,
        'max_velocity_m_s': max_velocity_m_s,
    }
    if velocity_mode == 'fixed_global':
        payload['fixed_velocity_m_s'] = fixed_velocity_m_s
    else:
        payload['initial_velocity_m_s'] = initial_velocity_m_s
    if min_observations_per_cell is not None:
        payload['min_observations_per_cell'] = min_observations_per_cell
    if smoothing_weight is not None:
        payload['smoothing_weight'] = smoothing_weight
    return payload


def _input_model(
    dataset: SyntheticMultiLayerRefractionDataset,
) -> RefractionStaticInputModel:
    endpoint_table = _endpoint_table(dataset)
    return RefractionStaticInputModel(
        file_id=dataset.name,
        n_traces=int(dataset.sorted_trace_index.size),
        sorted_trace_index=dataset.sorted_trace_index,
        pick_time_s_sorted=dataset.first_break_time_s,
        valid_pick_mask_sorted=dataset.valid_mask,
        valid_observation_mask_sorted=dataset.valid_mask,
        source_id_sorted=dataset.source_id,
        receiver_id_sorted=dataset.receiver_id,
        source_x_m_sorted=dataset.source_x_m,
        source_y_m_sorted=dataset.source_y_m,
        receiver_x_m_sorted=dataset.receiver_x_m,
        receiver_y_m_sorted=dataset.receiver_y_m,
        source_elevation_m_sorted=dataset.source_elevation_m,
        receiver_elevation_m_sorted=dataset.receiver_elevation_m,
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=dataset.offset_m,
        offset_m_sorted=dataset.offset_m,
        distance_m_sorted=dataset.offset_m,
        source_endpoint_key_sorted=dataset.source_endpoint_key.astype(object),
        receiver_endpoint_key_sorted=dataset.receiver_endpoint_key.astype(object),
        source_node_id_sorted=dataset.source_node_id,
        receiver_node_id_sorted=dataset.receiver_node_id,
        node_x_m=endpoint_table.x_m,
        node_y_m=endpoint_table.y_m,
        node_elevation_m=endpoint_table.elevation_m,
        node_kind=endpoint_table.kind,
        rejection_reason_sorted=dataset.rejection_reason.astype('<U32'),
        qc={'n_valid_observations': int(np.count_nonzero(dataset.valid_mask))},
        endpoint_table=endpoint_table,
        metadata={'fixture': dataset.name},
    )


def _endpoint_table(
    dataset: SyntheticMultiLayerRefractionDataset,
) -> RefractionEndpointTable:
    node_id = np.concatenate(
        (dataset.source_endpoint_node_id, dataset.receiver_endpoint_node_id)
    ).astype(np.int64)
    pick_count = np.zeros(node_id.shape, dtype=np.int64)
    node_pos = {int(node): index for index, node in enumerate(node_id.tolist())}
    for source_node, receiver_node in zip(
        dataset.source_node_id.tolist(),
        dataset.receiver_node_id.tolist(),
        strict=True,
    ):
        pick_count[node_pos[int(source_node)]] += 1
        pick_count[node_pos[int(receiver_node)]] += 1
    return RefractionEndpointTable(
        node_id=node_id,
        endpoint_id=np.concatenate(
            (dataset.source_endpoint_id, dataset.receiver_endpoint_id)
        ).astype(np.int64),
        x_m=np.concatenate(
            (dataset.source_endpoint_x_m, dataset.receiver_endpoint_x_m)
        ).astype(np.float64),
        y_m=np.concatenate(
            (dataset.source_endpoint_y_m, dataset.receiver_endpoint_y_m)
        ).astype(np.float64),
        elevation_m=np.concatenate(
            (
                dataset.source_endpoint_elevation_m,
                dataset.receiver_endpoint_elevation_m,
            )
        ).astype(np.float64),
        kind=np.asarray(
            ['source'] * dataset.source_endpoint_id.size
            + ['receiver'] * dataset.receiver_endpoint_id.size,
            dtype='<U16',
        ),
        pick_count=pick_count,
    )


def _resolved_first_layer() -> ResolvedRefractionFirstLayer:
    return ResolvedRefractionFirstLayer(
        mode='constant',
        weathering_velocity_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        status='constant',
        qc={'weathering_velocity_m_s': SYNTHETIC_MULTILAYER_V1_M_S},
    )


def _fake_layer_result(
    context: RefractionLayerSolverContext,
) -> RefractionLayerSolveResult:
    n_traces = int(context.input_model.n_traces)
    source_count = len({str(key) for key in context.input_model.source_endpoint_key_sorted})
    receiver_count = len(
        {str(key) for key in context.input_model.receiver_endpoint_key_sorted}
    )
    node_count = int(context.input_model.endpoint_table.node_id.size)
    velocity = _configured_velocity(context)
    return RefractionLayerSolveResult(
        layer_kind=context.layer_config.kind,
        layer_index=context.layer_index,
        velocity_mode=context.layer_config.velocity_mode,
        source_time_term_s=np.zeros(source_count, dtype=np.float64),
        receiver_time_term_s=np.zeros(receiver_count, dtype=np.float64),
        node_time_term_s=np.zeros(node_count, dtype=np.float64),
        global_velocity_m_s=(
            None
            if context.layer_config.velocity_mode == 'solve_cell'
            else float(velocity)
        ),
        global_slowness_s_per_m=(
            None
            if context.layer_config.velocity_mode == 'solve_cell'
            else float(1.0 / velocity)
        ),
        cell_velocity_m_s=None,
        cell_slowness_s_per_m=None,
        trace_predicted_time_s_sorted=np.zeros(n_traces, dtype=np.float64),
        trace_residual_s_sorted=np.zeros(n_traces, dtype=np.float64),
        used_observation_mask_sorted=np.ascontiguousarray(
            context.input_model.valid_observation_mask_sorted,
            dtype=bool,
        ),
        layer_status='solved',
        qc={
            'layer_kind': context.layer_config.kind,
            'n_used': int(
                np.count_nonzero(context.input_model.valid_observation_mask_sorted)
            ),
        },
    )


def _configured_velocity(context: RefractionLayerSolverContext) -> float:
    if context.layer_config.fixed_velocity_m_s is not None:
        return float(context.layer_config.fixed_velocity_m_s)
    if context.layer_config.initial_velocity_m_s is not None:
        return float(context.layer_config.initial_velocity_m_s)
    raise AssertionError('test layer requires configured velocity')
