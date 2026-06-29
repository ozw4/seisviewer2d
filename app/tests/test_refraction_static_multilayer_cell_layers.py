from __future__ import annotations

from dataclasses import replace

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
    solve_refraction_multilayer_time_terms,
)
from app.statics.refraction.contracts.result_types import (
    RefractionLayerKind,
    RefractionLayerSolveResult,
    RefractionMultiLayerSolveResult,
    RefractionStaticInputModel,
)
from app.tests.test_refraction_static_multilayer_v3_t2_solver import (
    RECEIVER_NODE as V3_RECEIVER_NODE,
    SOURCE_NODE as V3_SOURCE_NODE,
    T1_S as V3_T1_S,
    T2_S as V3_T2_S,
    V1_M_S,
    V2_M_S,
    V2_OFFSET_M,
    V3_OFFSET_M,
    _input_model as _v3_input_model,
    _resolved_first_layer,
)
from app.tests.test_refraction_static_multilayer_vsub_t3_solver import (
    RECEIVER_NODE as VSUB_RECEIVER_NODE,
    SOURCE_NODE as VSUB_SOURCE_NODE,
    T1_S as VSUB_T1_S,
    T2_S as VSUB_T2_S,
    T3_S as VSUB_T3_S,
    V2_M_S as VSUB_V2_M_S,
    V2_OFFSET_M as VSUB_V2_OFFSET_M,
    V3_M_S,
    V3_OFFSET_M as VSUB_V3_OFFSET_M,
    VSUB_OFFSET_M,
    _input_model as _vsub_input_model,
)

V3_CELL_ORIGIN_M = 500.0
V3_CELL_SIZE_M = 150.0
V3_CELL_VELOCITY_M_S = np.asarray([3300.0, 3700.0, 4100.0], dtype=np.float64)

VSUB_CELL_ORIGIN_M = 1200.0
VSUB_CELL_SIZE_M = 175.0
VSUB_CELL_VELOCITY_M_S = np.asarray([4500.0, 5000.0, 5500.0], dtype=np.float64)


def test_v3_t2_solve_cell_recovers_synthetic_cell_velocity() -> None:
    result = _run_multilayer(
        input_model=_v3_cell_input_model(),
        model=_v3_solve_cell_model(),
    )

    layer = _layer(result, 'v3_t2')

    assert layer.velocity_mode == 'solve_cell'
    np.testing.assert_array_equal(layer.active_cell_id, [0, 1, 2])
    np.testing.assert_allclose(
        layer.cell_velocity_m_s,
        V3_CELL_VELOCITY_M_S,
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        layer.cell_slowness_s_per_m,
        1.0 / V3_CELL_VELOCITY_M_S,
        rtol=1.0e-7,
    )
    assert layer.qc['layer_kind'] == 'v3_t2'
    assert layer.qc['velocity_mode'] == 'solve_cell'


def test_vsub_t3_solve_cell_recovers_synthetic_cell_velocity() -> None:
    result = _run_multilayer(
        input_model=_vsub_cell_input_model(),
        model=_vsub_solve_cell_model(),
    )

    layer = _layer(result, 'vsub_t3')

    assert layer.velocity_mode == 'solve_cell'
    np.testing.assert_array_equal(layer.active_cell_id, [0, 1, 2])
    np.testing.assert_allclose(
        layer.cell_velocity_m_s,
        VSUB_CELL_VELOCITY_M_S,
        rtol=1.0e-7,
    )


def test_layer_specific_min_observations_per_cell_is_applied_to_v3() -> None:
    result = _run_multilayer(
        input_model=_v3_cell_input_model(),
        model=_v3_solve_cell_model(min_observations_per_cell=2),
    )

    layer = _layer(result, 'v3_t2')

    assert layer.qc['min_observations_per_cell'] == 2
    assert layer.qc['n_low_fold_cells'] == 1
    assert layer.qc['n_observations_rejected_by_low_fold_cell'] == 1
    np.testing.assert_array_equal(layer.active_cell_id, [0, 1])
    np.testing.assert_array_equal(layer.inactive_cell_id, [2])
    assert layer.cell_velocity_status is not None
    assert layer.cell_velocity_status.tolist() == ['solved', 'solved', 'low_fold']


def test_layer_specific_smoothing_weight_is_applied_to_v3() -> None:
    result = _run_multilayer(
        input_model=_v3_cell_input_model(),
        model=_v3_solve_cell_model(smoothing_weight=2.5),
    )

    layer = _layer(result, 'v3_t2')

    assert layer.cell_velocity_m_s is not None
    assert np.ptp(layer.cell_velocity_m_s) < np.ptp(V3_CELL_VELOCITY_M_S)
    assert layer.qc['cell_bedrock_velocity_median_m_s'] != pytest.approx(
        np.median(V3_CELL_VELOCITY_M_S)
    )


def _run_multilayer(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelRequest,
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
            min_picks_per_node=1,
            robust={'enabled': False},
        ),
    )


def _v3_cell_input_model() -> RefractionStaticInputModel:
    v3_cell_id = _midpoint_cell_id(
        V3_OFFSET_M,
        origin_m=V3_CELL_ORIGIN_M,
        size_m=V3_CELL_SIZE_M,
    )
    pick_time = np.concatenate(
        (
            V3_T1_S[V3_SOURCE_NODE]
            + V3_T1_S[V3_RECEIVER_NODE]
            + V2_OFFSET_M / V2_M_S,
            V3_T2_S[V3_SOURCE_NODE]
            + V3_T2_S[V3_RECEIVER_NODE]
            + V3_OFFSET_M / V3_CELL_VELOCITY_M_S[v3_cell_id],
        )
    )
    return replace(
        _v3_input_model(),
        pick_time_s_sorted=np.ascontiguousarray(pick_time, dtype=np.float64),
    )


def _vsub_cell_input_model() -> RefractionStaticInputModel:
    vsub_cell_id = _midpoint_cell_id(
        VSUB_OFFSET_M,
        origin_m=VSUB_CELL_ORIGIN_M,
        size_m=VSUB_CELL_SIZE_M,
    )
    pick_time = np.concatenate(
        (
            VSUB_T1_S[VSUB_SOURCE_NODE]
            + VSUB_T1_S[VSUB_RECEIVER_NODE]
            + VSUB_V2_OFFSET_M / VSUB_V2_M_S,
            VSUB_T2_S[VSUB_SOURCE_NODE]
            + VSUB_T2_S[VSUB_RECEIVER_NODE]
            + VSUB_V3_OFFSET_M / V3_M_S,
            VSUB_T3_S[VSUB_SOURCE_NODE]
            + VSUB_T3_S[VSUB_RECEIVER_NODE]
            + VSUB_OFFSET_M / VSUB_CELL_VELOCITY_M_S[vsub_cell_id],
        )
    )
    return replace(
        _vsub_input_model(),
        pick_time_s_sorted=np.ascontiguousarray(pick_time, dtype=np.float64),
    )


def _v3_solve_cell_model(
    *,
    min_observations_per_cell: int = 1,
    smoothing_weight: float = 0.0,
) -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': V1_M_S,
            },
            'refractor_cell': _refractor_cell_payload(
                origin_m=V3_CELL_ORIGIN_M,
                size_m=V3_CELL_SIZE_M,
                min_observations_per_cell=9,
                smoothing_weight=0.0,
            ),
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
                    'max_offset_m': None,
                    'velocity_mode': 'solve_cell',
                    'initial_velocity_m_s': 3700.0,
                    'min_velocity_m_s': 3000.0,
                    'max_velocity_m_s': 4500.0,
                    'min_observations_per_cell': min_observations_per_cell,
                    'smoothing_weight': smoothing_weight,
                },
            ],
        }
    )


def _vsub_solve_cell_model() -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': V1_M_S,
            },
            'refractor_cell': _refractor_cell_payload(
                origin_m=VSUB_CELL_ORIGIN_M,
                size_m=VSUB_CELL_SIZE_M,
                min_observations_per_cell=1,
                smoothing_weight=0.0,
            ),
            'layers': [
                {
                    'kind': 'v2_t1',
                    'enabled': True,
                    'min_offset_m': 250.0,
                    'max_offset_m': 800.0,
                    'velocity_mode': 'fixed_global',
                    'fixed_velocity_m_s': VSUB_V2_M_S,
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
                {
                    'kind': 'vsub_t3',
                    'enabled': True,
                    'min_offset_m': 2200.0,
                    'max_offset_m': None,
                    'velocity_mode': 'solve_cell',
                    'initial_velocity_m_s': 5000.0,
                    'min_velocity_m_s': 4000.0,
                    'max_velocity_m_s': 6000.0,
                },
            ],
        }
    )


def _refractor_cell_payload(
    *,
    origin_m: float,
    size_m: float,
    min_observations_per_cell: int,
    smoothing_weight: float,
) -> dict[str, object]:
    return {
        'number_of_cell_x': 3,
        'size_of_cell_x_m': size_m,
        'x_coordinate_origin_m': origin_m,
        'number_of_cell_y': 1,
        'size_of_cell_y_m': None,
        'y_coordinate_origin_m': 0.0,
        'assignment_mode': 'midpoint',
        'outside_grid_policy': 'reject',
        'coordinate_mode': 'grid_3d',
        'min_observations_per_cell': min_observations_per_cell,
        'velocity_smoothing_weight': smoothing_weight,
        'smoothing_reference_distance_m': None,
    }


def _midpoint_cell_id(
    offset_m: np.ndarray,
    *,
    origin_m: float,
    size_m: float,
) -> np.ndarray:
    midpoint = np.asarray(offset_m, dtype=np.float64) * 0.5
    return np.floor((midpoint - float(origin_m)) / float(size_m)).astype(np.int64)


def _layer(
    result: RefractionMultiLayerSolveResult,
    kind: RefractionLayerKind,
) -> RefractionLayerSolveResult:
    for layer in result.layer_results:
        if layer.layer_kind == kind:
            return layer
    raise AssertionError(f'{kind} layer result was not returned')
