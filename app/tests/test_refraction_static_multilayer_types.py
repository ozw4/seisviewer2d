from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, get_type_hints

import numpy as np
import pytest

from app.services.refraction_static_types import (
    RefractionLayerKind,
    RefractionLayerSolveResult,
    RefractionLayerVelocityMode,
    RefractionMultiLayerSolveResult,
    RefractionMultiLayerStaticComponents,
)


def _float_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _int_array(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _bool_array(values: list[bool]) -> np.ndarray:
    return np.asarray(values, dtype=np.bool_)


def _layer_result(**overrides: object) -> RefractionLayerSolveResult:
    payload: dict[str, object] = {
        'layer_kind': 'v3_t2',
        'layer_index': 2,
        'velocity_mode': 'solve_cell',
        'source_time_term_s': _float_array([0.010, 0.012]),
        'receiver_time_term_s': _float_array([0.011, 0.013]),
        'node_time_term_s': _float_array([0.010, 0.011, 0.012, 0.013]),
        'global_velocity_m_s': None,
        'global_slowness_s_per_m': None,
        'cell_velocity_m_s': _float_array([3200.0, 3400.0]),
        'cell_slowness_s_per_m': _float_array([1.0 / 3200.0, 1.0 / 3400.0]),
        'trace_predicted_time_s_sorted': _float_array([0.22, 0.31]),
        'trace_residual_s_sorted': _float_array([0.001, -0.002]),
        'used_observation_mask_sorted': _bool_array([True, False]),
        'layer_status': 'solved',
        'qc': {'n_used': 1},
    }
    payload.update(overrides)
    return RefractionLayerSolveResult(**payload)


def test_layer_solve_result_is_frozen_and_typed() -> None:
    result = _layer_result()

    assert result.layer_kind == 'v3_t2'
    assert result.layer_index == 2
    assert result.velocity_mode == 'solve_cell'
    assert result.cell_velocity_m_s is not None
    assert result.cell_velocity_m_s.dtype == np.float64
    assert result.used_observation_mask_sorted.dtype == np.bool_
    assert result.qc == {'n_used': 1}

    hints = get_type_hints(RefractionLayerSolveResult)
    assert hints['layer_kind'] == RefractionLayerKind
    assert hints['velocity_mode'] == RefractionLayerVelocityMode
    assert hints['source_time_term_s'] is np.ndarray
    assert hints['used_observation_mask_sorted'] is np.ndarray
    assert hints['qc'] == dict[str, Any]

    with pytest.raises(FrozenInstanceError):
        result.layer_status = 'changed'


def test_multilayer_solve_result_collects_layer_outputs() -> None:
    v2 = _layer_result(
        layer_kind='v2_t1',
        layer_index=1,
        velocity_mode='solve_global',
        global_velocity_m_s=2400.0,
        global_slowness_s_per_m=1.0 / 2400.0,
        cell_velocity_m_s=None,
        cell_slowness_s_per_m=None,
    )
    v3 = _layer_result()

    result = RefractionMultiLayerSolveResult(
        enabled_layer_kinds=('v2_t1', 'v3_t2'),
        layer_results=(v2, v3),
        source_endpoint_key=_int_array([101, 102]),
        receiver_endpoint_key=_int_array([201, 202]),
        source_node_id=_int_array([0, 1]),
        receiver_node_id=_int_array([2, 3]),
        qc={'enabled_layer_count': 2},
    )

    assert result.enabled_layer_kinds == ('v2_t1', 'v3_t2')
    assert [layer.layer_kind for layer in result.layer_results] == ['v2_t1', 'v3_t2']
    assert result.source_endpoint_key.tolist() == [101, 102]
    assert result.receiver_node_id.tolist() == [2, 3]

    with pytest.raises(FrozenInstanceError):
        result.qc = {}


def test_multilayer_static_components_support_optional_deeper_layers() -> None:
    components = RefractionMultiLayerStaticComponents(
        source_t1_s=_float_array([0.010, 0.012]),
        source_t2_s=_float_array([0.020, 0.022]),
        source_t3_s=None,
        receiver_t1_s=_float_array([0.011, 0.013]),
        receiver_t2_s=_float_array([0.021, 0.023]),
        receiver_t3_s=None,
        source_sh1_m=_float_array([8.0, 9.0]),
        source_sh2_m=_float_array([18.0, 19.0]),
        source_sh3_m=None,
        receiver_sh1_m=_float_array([7.0, 8.0]),
        receiver_sh2_m=_float_array([17.0, 18.0]),
        receiver_sh3_m=None,
        source_weathering_correction_s=_float_array([-0.006, -0.007]),
        receiver_weathering_correction_s=_float_array([-0.005, -0.006]),
        qc={'sign_convention': 'corrected(t) = raw(t - shift_s)'},
    )

    assert components.source_t2_s is not None
    assert components.source_t2_s.tolist() == [0.020, 0.022]
    assert components.source_t3_s is None
    assert components.receiver_sh3_m is None
    assert components.source_weathering_correction_s[0] < 0.0

    with pytest.raises(FrozenInstanceError):
        components.qc = {}
