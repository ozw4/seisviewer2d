from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.statics.refraction.application.design_matrix import (
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    refraction_design_matrix_layer_node_diagnostics_csv_name,
    refraction_design_matrix_layer_qc_json_name,
)
from app.statics.refraction.domain.layer_config import (
    normalize_refraction_static_layers,
)
from app.statics.refraction.domain.layer_observations import (
    build_refraction_layer_observation_masks,
)
from app.statics.refraction.application.multilayer_service import (
    RefractionLayerSolverContext,
    RefractionMultiLayerSolveError,
    solve_refraction_multilayer_time_terms,
)
from app.statics.refraction.domain.types import (
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


def test_multilayer_orchestrator_runs_normalized_one_layer_v2_solve(
    tmp_path: Path,
) -> None:
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
        job_dir=tmp_path,
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
    layer_artifact_dir = tmp_path / 'refraction_design_matrix_v2_t1'
    assert (layer_artifact_dir / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).is_file()
    assert (
        layer_artifact_dir / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
    ).is_file()
    assert not (tmp_path / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).exists()
    assert not (
        tmp_path / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
    ).exists()
    root_qc_path = tmp_path / refraction_design_matrix_layer_qc_json_name('v2_t1')
    root_csv_path = (
        tmp_path / refraction_design_matrix_layer_node_diagnostics_csv_name('v2_t1')
    )
    assert root_qc_path.is_file()
    assert root_csv_path.is_file()
    root_qc = json.loads(root_qc_path.read_text(encoding='utf-8'))
    assert root_qc['layer_kind'] == 'v2_t1'
    assert root_qc['layer_index'] == 1
    assert root_qc['source_artifact_dir'] == 'refraction_design_matrix_v2_t1'


def test_multilayer_orchestrator_exposes_failed_layer_design_matrix_artifacts(
    tmp_path: Path,
) -> None:
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

    def _failing_solver(context: RefractionLayerSolverContext) -> RefractionLayerSolveResult:
        assert context.job_dir is not None
        context.job_dir.mkdir(parents=True, exist_ok=True)
        (context.job_dir / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).write_text(
            '{"n_all_zero_active_node_columns": 1}',
            encoding='utf-8',
        )
        (
            context.job_dir / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
        ).write_text('node_id,status\n1,all_zero_active_column\n', encoding='utf-8')
        raise ValueError(
            'refraction design matrix contains an all-zero active-node column'
        )

    with pytest.raises(ValueError, match='all-zero active-node column'):
        solve_refraction_multilayer_time_terms(
            input_model=input_model,
            resolved_first_layer=_resolved_first_layer(),
            normalized_layers=normalize_refraction_static_layers(model),
            layer_masks=masks,
            model=model,
            solver=RefractionStaticSolverRequest(robust={'enabled': False}),
            job_dir=tmp_path,
            solver_dispatch={('v2_t1', 'solve_global'): _failing_solver},
        )

    assert not (tmp_path / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).exists()
    assert not (
        tmp_path / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
    ).exists()
    root_qc_path = tmp_path / refraction_design_matrix_layer_qc_json_name('v2_t1')
    assert root_qc_path.is_file()
    root_qc = json.loads(root_qc_path.read_text(encoding='utf-8'))
    assert root_qc['layer_kind'] == 'v2_t1'
    assert root_qc['layer_index'] == 1
    assert root_qc['source_artifact_dir'] == 'refraction_design_matrix_v2_t1'
    assert (
        tmp_path / refraction_design_matrix_layer_node_diagnostics_csv_name('v2_t1')
    ).is_file()


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


def test_multilayer_design_matrix_diagnostics_are_layer_disambiguated(
    tmp_path: Path,
) -> None:
    dataset = make_2d_straight_three_layer_refraction_dataset()
    model = _model(
        [
            _layer(
                'v2_t1',
                min_offset_m=300.0,
                max_offset_m=800.0,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
                min_velocity_m_s=1600.0,
                max_velocity_m_s=3200.0,
            ),
            _layer(
                'v3_t2',
                min_offset_m=1000.0,
                max_offset_m=1900.0,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
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

    def fake_solver(context: RefractionLayerSolverContext) -> RefractionLayerSolveResult:
        assert context.job_dir is not None
        context.job_dir.mkdir(parents=True, exist_ok=True)
        (context.job_dir / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).write_text(
            '{"n_active_nodes": 2}',
            encoding='utf-8',
        )
        (
            context.job_dir / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
        ).write_text('node_id,status\n1,ok\n', encoding='utf-8')
        return _fake_layer_result(context)

    solve_refraction_multilayer_time_terms(
        input_model=input_model,
        resolved_first_layer=_resolved_first_layer(),
        normalized_layers=normalize_refraction_static_layers(model),
        layer_masks=masks,
        model=model,
        solver=RefractionStaticSolverRequest(),
        job_dir=tmp_path,
        solver_dispatch={
            ('v2_t1', 'fixed_global'): fake_solver,
            ('v3_t2', 'fixed_global'): fake_solver,
            ('vsub_t3', 'fixed_global'): fake_solver,
        },
    )

    assert not (tmp_path / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).exists()
    assert not (
        tmp_path / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME
    ).exists()
    for layer_kind, layer_index in (
        ('v2_t1', 1),
        ('v3_t2', 2),
        ('vsub_t3', 3),
    ):
        qc_path = tmp_path / refraction_design_matrix_layer_qc_json_name(layer_kind)
        csv_path = (
            tmp_path
            / refraction_design_matrix_layer_node_diagnostics_csv_name(layer_kind)
        )
        qc = json.loads(qc_path.read_text(encoding='utf-8'))
        assert qc['layer_kind'] == layer_kind
        assert qc['layer_index'] == layer_index
        assert qc['source_artifact_dir'] == f'refraction_design_matrix_{layer_kind}'
        assert csv_path.is_file()


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


def test_three_enabled_layers_are_solved_in_order_without_custom_dispatch() -> None:
    dataset = make_2d_straight_three_layer_refraction_dataset()
    model = _model(
        [
            _layer(
                'v2_t1',
                min_offset_m=300.0,
                max_offset_m=800.0,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
                min_velocity_m_s=1600.0,
                max_velocity_m_s=3200.0,
            ),
            _layer(
                'v3_t2',
                min_offset_m=1000.0,
                max_offset_m=1900.0,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
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

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        resolved_first_layer=_resolved_first_layer(),
        normalized_layers=normalize_refraction_static_layers(model),
        layer_masks=masks,
        model=model,
        solver=RefractionStaticSolverRequest(robust={'enabled': False}),
    )

    assert result.enabled_layer_kinds == ('v2_t1', 'v3_t2', 'vsub_t3')
    assert [layer.layer_kind for layer in result.layer_results] == [
        'v2_t1',
        'v3_t2',
        'vsub_t3',
    ]
    assert [layer.layer_index for layer in result.layer_results] == [1, 2, 3]
    assert result.layer_results[2].global_velocity_m_s == pytest.approx(
        SYNTHETIC_MULTILAYER_VSUB_M_S
    )
    assert 'vsub_t3' in result.qc['layers']
    assert result.qc['observation_gates']['vsub_t3']['n_used_observations'] == int(
        np.count_nonzero(masks.layer_used_mask_sorted['vsub_t3'])
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
