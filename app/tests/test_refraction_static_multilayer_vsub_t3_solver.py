from __future__ import annotations

import numpy as np
import pytest

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.statics.refraction.application.core_options import (
    normalized_layers_from_model_request as normalize_refraction_static_layers,
)
from app.statics.refraction.application.core_options import (
    layer_observation_masks_from_input_model as build_refraction_layer_observation_masks,
)
from app.statics.refraction.application.multilayer_service import (
    RefractionMultiLayerSolveError,
    solve_refraction_multilayer_time_terms,
)
from app.statics.refraction.contracts.result_types import (
    RefractionEndpointTable,
    RefractionLayerKind,
    RefractionLayerSolveResult,
    RefractionMultiLayerSolveResult,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)

V1_M_S = 800.0
V2_M_S = 2400.0
V3_M_S = 3600.0
VSUB_M_S = 4800.0
T1_S = np.asarray([0.008, 0.010, 0.012, 0.014, 0.016], dtype=np.float64)
T2_S = np.asarray([0.020, 0.024, 0.022, 0.026, 0.028], dtype=np.float64)
T3_S = np.asarray([0.036, 0.040, 0.038, 0.043, 0.046], dtype=np.float64)
SOURCE_NODE = np.asarray([0, 0, 0, 1, 1, 2, 0, 1, 2, 3], dtype=np.int64)
RECEIVER_NODE = np.asarray([1, 2, 3, 2, 4, 4, 0, 1, 2, 3], dtype=np.int64)
V2_OFFSET_M = np.asarray(
    [320.0, 450.0, 600.0, 380.0, 700.0, 520.0, 260.0, 300.0, 340.0, 480.0],
    dtype=np.float64,
)
V3_OFFSET_M = np.asarray(
    [1050.0, 1240.0, 1390.0, 1160.0, 1550.0, 1320.0, 1100.0, 1450.0, 1700.0, 1280.0],
    dtype=np.float64,
)
VSUB_OFFSET_M = np.asarray(
    [
        2400.0,
        2700.0,
        3100.0,
        2500.0,
        3300.0,
        2900.0,
        2450.0,
        2800.0,
        3250.0,
        3000.0,
    ],
    dtype=np.float64,
)


def test_vsub_t3_fixed_global_layer_solves_after_v3_t2() -> None:
    result = _run_multilayer(
        model=_model(vsub_velocity_mode='fixed_global'),
        input_model=_input_model(),
    )

    assert result.enabled_layer_kinds == ('v2_t1', 'v3_t2', 'vsub_t3')
    assert [layer.layer_kind for layer in result.layer_results] == [
        'v2_t1',
        'v3_t2',
        'vsub_t3',
    ]
    layer = _layer(result, 'vsub_t3')
    assert layer.velocity_mode == 'fixed_global'
    assert layer.layer_index == 3
    assert layer.global_velocity_m_s == pytest.approx(VSUB_M_S)
    assert layer.global_slowness_s_per_m == pytest.approx(1.0 / VSUB_M_S)
    assert layer.qc['velocity_sequence_reference_layer_kind'] == 'v3_t2'
    assert layer.qc['velocity_sequence_reference_m_s'] == pytest.approx(V3_M_S)
    assert result.qc['layers']['vsub_t3']['layer_kind'] == 'vsub_t3'
    assert result.qc['observation_gates']['vsub_t3']['n_used_observations'] == (
        VSUB_OFFSET_M.size
    )


def test_vsub_t3_solve_global_recovers_synthetic_vsub() -> None:
    result = _run_multilayer(
        model=_model(vsub_velocity_mode='solve_global'),
        input_model=_input_model(),
    )

    layer = _layer(result, 'vsub_t3')

    assert layer.velocity_mode == 'solve_global'
    assert layer.global_velocity_m_s == pytest.approx(VSUB_M_S, rel=1.0e-8)
    assert layer.global_slowness_s_per_m == pytest.approx(1.0 / VSUB_M_S, rel=1.0e-8)
    assert layer.qc['vsub_velocity_m_s'] == pytest.approx(VSUB_M_S, rel=1.0e-8)
    assert layer.qc['vsub_slowness_s_per_m'] == pytest.approx(
        1.0 / VSUB_M_S,
        rel=1.0e-8,
    )
    assert layer.qc['n_observations'] == VSUB_OFFSET_M.size
    assert layer.qc['residual_rms_ms'] == pytest.approx(0.0, abs=1.0e-5)


def test_vsub_t3_result_contains_t3_terms_and_layer_index_3() -> None:
    result = _run_multilayer(
        model=_model(vsub_velocity_mode='fixed_global'),
        input_model=_input_model(),
    )

    layer = _layer(result, 'vsub_t3')

    assert layer.layer_index == 3
    np.testing.assert_allclose(layer.node_time_term_s, T3_S, atol=1.0e-9)
    np.testing.assert_allclose(
        layer.source_time_term_s,
        _endpoint_terms(result.source_endpoint_key, T3_S),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        layer.receiver_time_term_s,
        _endpoint_terms(result.receiver_endpoint_key, T3_S),
        atol=1.0e-9,
    )


def test_vsub_t3_rejects_velocity_not_greater_than_v3() -> None:
    with pytest.raises(
        RefractionMultiLayerSolveError,
        match='vsub_t3 velocity must be greater than v3_t2 velocity',
    ):
        _run_multilayer(
            model=_model(
                vsub_velocity_mode='fixed_global',
                vsub_fixed_velocity_m_s=V3_M_S,
            ),
            input_model=_input_model(),
        )


def _run_multilayer(
    *,
    model: RefractionStaticModelRequest,
    input_model: RefractionStaticInputModel,
) -> RefractionMultiLayerSolveResult:
    masks = build_refraction_layer_observation_masks(
        input_model=input_model,
        model=model,
    )
    return solve_refraction_multilayer_time_terms(
        input_model=input_model,
        resolved_first_layer=_resolved_first_layer(),
        normalized_layers=normalize_refraction_static_layers(model),
        layer_masks=masks,
        model=model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
    )


def _resolved_first_layer() -> ResolvedRefractionFirstLayer:
    return ResolvedRefractionFirstLayer(
        mode='constant',
        weathering_velocity_m_s=V1_M_S,
        status='constant',
        qc={},
    )


def _layer(
    result: RefractionMultiLayerSolveResult,
    kind: RefractionLayerKind,
) -> RefractionLayerSolveResult:
    for layer in result.layer_results:
        if layer.layer_kind == kind:
            return layer
    raise AssertionError(f'{kind} layer result was not returned')


def _model(
    *,
    vsub_velocity_mode: str,
    vsub_fixed_velocity_m_s: float = VSUB_M_S,
) -> RefractionStaticModelRequest:
    vsub_layer: dict[str, object] = {
        'kind': 'vsub_t3',
        'enabled': True,
        'min_offset_m': 2200.0,
        'max_offset_m': None,
        'velocity_mode': vsub_velocity_mode,
        'min_velocity_m_s': 3000.0,
        'max_velocity_m_s': 6200.0,
    }
    if vsub_velocity_mode == 'fixed_global':
        vsub_layer['fixed_velocity_m_s'] = vsub_fixed_velocity_m_s
    else:
        vsub_layer['initial_velocity_m_s'] = VSUB_M_S

    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': V1_M_S,
            },
            'layers': [
                {
                    'kind': 'v2_t1',
                    'enabled': True,
                    'min_offset_m': 250.0,
                    'max_offset_m': 800.0,
                    'velocity_mode': 'fixed_global',
                    'fixed_velocity_m_s': V2_M_S,
                    'min_velocity_m_s': 1600.0,
                    'max_velocity_m_s': 3200.0,
                },
                {
                    'kind': 'v3_t2',
                    'enabled': True,
                    'min_offset_m': 1000.0,
                    'max_offset_m': 1900.0,
                    'velocity_mode': 'fixed_global',
                    'fixed_velocity_m_s': V3_M_S,
                    'min_velocity_m_s': 2600.0,
                    'max_velocity_m_s': 4800.0,
                },
                vsub_layer,
            ],
        }
    )


def _input_model() -> RefractionStaticInputModel:
    pick_time = np.concatenate(
        (
            T1_S[SOURCE_NODE] + T1_S[RECEIVER_NODE] + V2_OFFSET_M / V2_M_S,
            T2_S[SOURCE_NODE] + T2_S[RECEIVER_NODE] + V3_OFFSET_M / V3_M_S,
            T3_S[SOURCE_NODE] + T3_S[RECEIVER_NODE] + VSUB_OFFSET_M / VSUB_M_S,
        )
    )
    source_node = np.concatenate((SOURCE_NODE, SOURCE_NODE, SOURCE_NODE)).astype(
        np.int64
    )
    receiver_node = np.concatenate((RECEIVER_NODE, RECEIVER_NODE, RECEIVER_NODE)).astype(
        np.int64
    )
    distance_m = np.concatenate((V2_OFFSET_M, V3_OFFSET_M, VSUB_OFFSET_M)).astype(
        np.float64
    )
    n_traces = int(pick_time.shape[0])
    trace_index = np.arange(n_traces, dtype=np.int64)
    zeros = np.zeros(n_traces, dtype=np.float64)
    endpoint_table = _endpoint_table(source_node, receiver_node)

    return RefractionStaticInputModel(
        file_id='synthetic-vsub-t3',
        n_traces=n_traces,
        sorted_trace_index=trace_index,
        pick_time_s_sorted=np.ascontiguousarray(pick_time, dtype=np.float64),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=source_node.copy(),
        receiver_id_sorted=receiver_node.copy(),
        source_x_m_sorted=zeros.copy(),
        source_y_m_sorted=zeros.copy(),
        receiver_x_m_sorted=distance_m.copy(),
        receiver_y_m_sorted=zeros.copy(),
        source_elevation_m_sorted=zeros.copy(),
        receiver_elevation_m_sorted=zeros.copy(),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance_m.copy(),
        offset_m_sorted=distance_m.copy(),
        distance_m_sorted=np.ascontiguousarray(distance_m, dtype=np.float64),
        source_endpoint_key_sorted=np.asarray(
            [f's:{node}' for node in source_node.tolist()],
            dtype=object,
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'r:{node}' for node in receiver_node.tolist()],
            dtype=object,
        ),
        source_node_id_sorted=source_node.copy(),
        receiver_node_id_sorted=receiver_node.copy(),
        node_x_m=endpoint_table.x_m,
        node_y_m=endpoint_table.y_m,
        node_elevation_m=endpoint_table.elevation_m,
        node_kind=endpoint_table.kind,
        rejection_reason_sorted=np.full(n_traces, 'ok', dtype='<U32'),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
    )


def _endpoint_table(
    source_node: np.ndarray,
    receiver_node: np.ndarray,
) -> RefractionEndpointTable:
    node_id = np.arange(T1_S.size, dtype=np.int64)
    pick_count = np.zeros(node_id.shape, dtype=np.int64)
    for node in np.concatenate((source_node, receiver_node)).tolist():
        pick_count[int(node)] += 1
    return RefractionEndpointTable(
        node_id=node_id,
        endpoint_id=node_id.copy(),
        x_m=node_id.astype(np.float64),
        y_m=np.zeros(node_id.shape, dtype=np.float64),
        elevation_m=np.zeros(node_id.shape, dtype=np.float64),
        kind=np.full(node_id.shape, 'linked', dtype='<U16'),
        pick_count=pick_count,
    )


def _endpoint_terms(
    endpoint_key: np.ndarray,
    terms: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [float(terms[int(str(key).split(':', maxsplit=1)[1])]) for key in endpoint_key],
        dtype=np.float64,
    )
