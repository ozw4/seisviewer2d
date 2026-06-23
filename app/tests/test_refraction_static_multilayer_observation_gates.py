from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from seis_statics.refraction.layer_observations import (
    INVALID_OFFSET_REJECTION_REASON,
    OUTSIDE_LAYER_GATE_REJECTION_REASON,
    refraction_layer_observation_qc,
)

from app.statics.refraction.core_options import (
    layer_observation_masks_from_arrays,
    layer_observation_qc_for_viewer,
)
from app.statics.refraction.contracts.model import RefractionStaticModelRequest


def _v2_layer(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'kind': 'v2_t1',
        'enabled': True,
        'min_offset_m': 0.0,
        'max_offset_m': 100.0,
        'velocity_mode': 'solve_global',
        'initial_velocity_m_s': 2400.0,
        'min_velocity_m_s': 1200.0,
        'max_velocity_m_s': 3500.0,
    }
    payload.update(overrides)
    return payload


def _v3_layer(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'kind': 'v3_t2',
        'enabled': True,
        'min_offset_m': 101.0,
        'max_offset_m': 250.0,
        'velocity_mode': 'solve_global',
        'initial_velocity_m_s': 3600.0,
        'min_velocity_m_s': 2600.0,
        'max_velocity_m_s': 6500.0,
    }
    payload.update(overrides)
    return payload


def _vsub_layer(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'kind': 'vsub_t3',
        'enabled': True,
        'min_offset_m': 251.0,
        'max_offset_m': None,
        'velocity_mode': 'fixed_global',
        'fixed_velocity_m_s': 5200.0,
        'min_velocity_m_s': 3800.0,
        'max_velocity_m_s': 9000.0,
    }
    payload.update(overrides)
    return payload


def _model(
    layers: list[dict[str, object]],
    *,
    allow_overlapping_layer_gates: bool = False,
) -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': 800.0,
            },
            'allow_overlapping_layer_gates': allow_overlapping_layer_gates,
            'layers': layers,
        }
    )


def _build_masks(
    model: Any,
    *,
    offset_m: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    reasons: np.ndarray | None = None,
) -> Any:
    if offset_m is None:
        offset_m = np.asarray([50.0, 100.0, 101.0, 250.0, 251.0, 500.0, np.nan, 20.0])
        valid = np.asarray([True, True, True, True, True, True, True, False])
        reasons = np.asarray(['ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'missing_pick'])
    if valid is None:
        valid = np.ones(offset_m.shape, dtype=bool)
    if reasons is None:
        reasons = np.full(offset_m.shape, 'ok')
    return layer_observation_masks_from_arrays(
        base_valid_mask_sorted=valid,
        offset_m_sorted=offset_m,
        rejection_reason_sorted=reasons,
        model=model,
    )


def test_non_overlapping_layer_gates_build_external_masks_and_qc() -> None:
    masks = _build_masks(_model([_v2_layer(), _v3_layer(), _vsub_layer()]))

    assert masks.layer_kind.tolist() == ['v2_t1', 'v3_t2', 'vsub_t3']
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['v2_t1'],
        np.asarray([True, False, False, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['v3_t2'],
        np.asarray([False, False, True, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['vsub_t3'],
        np.asarray([False, False, False, False, True, True, False, False]),
    )
    assert masks.layer_rejection_reason_sorted['v2_t1'][2] == (
        OUTSIDE_LAYER_GATE_REJECTION_REASON
    )
    assert masks.layer_rejection_reason_sorted['v2_t1'][6] == (
        INVALID_OFFSET_REJECTION_REASON
    )
    assert masks.layer_rejection_reason_sorted['v2_t1'][7] == 'missing_pick'

    qc = refraction_layer_observation_qc(masks)
    assert qc['assignment_policy'] == 'reject_overlap'
    assert qc['layer_candidate_count'] == {
        'v2_t1': 2,
        'v3_t2': 1,
        'vsub_t3': 2,
    }
    assert qc['layer_observation_count'] == {
        'v2_t1': 1,
        'v3_t2': 1,
        'vsub_t3': 2,
    }
    assert qc['unassigned_valid_observation_count'] == 3


def test_overlapping_layer_gates_are_rejected_by_default() -> None:
    with pytest.raises(ValidationError, match='offset gates must not overlap'):
        _model(
            [
                _v2_layer(max_offset_m=150.0),
                _v3_layer(min_offset_m=100.0),
            ]
        )


def test_boundary_touching_layer_gates_are_half_open() -> None:
    model = _model(
        [
            _v2_layer(max_offset_m=100.0),
            _v3_layer(min_offset_m=100.0),
        ]
    )

    masks = _build_masks(model, offset_m=np.asarray([99.0, 100.0, 101.0]))

    assert masks.layer_used_mask_sorted['v2_t1'].tolist() == [True, False, False]
    assert masks.layer_used_mask_sorted['v3_t2'].tolist() == [False, True, True]


def test_observation_overlap_is_allowed_when_explicitly_configured() -> None:
    allowed = _model(
        [
            _v2_layer(max_offset_m=110.0),
            _v3_layer(min_offset_m=100.0),
        ],
        allow_overlapping_layer_gates=True,
    )
    masks = _build_masks(allowed, offset_m=np.asarray([99.0, 100.0, 101.0]))

    assert masks.assignment_policy == 'independent'
    assert masks.layer_used_mask_sorted['v2_t1'].tolist() == [True, True, True]
    assert masks.layer_used_mask_sorted['v3_t2'].tolist() == [False, True, True]
    assert masks.overlapping_valid_observation_count == 2


def test_layer_candidate_count_includes_in_gate_base_rejections() -> None:
    masks = _build_masks(
        _model([_v2_layer()]),
        offset_m=np.asarray([10.0, 20.0, 90.0, 150.0, np.nan]),
        valid=np.asarray([True, False, False, True, True]),
        reasons=np.asarray(
            [
                '',
                'missing_pick',
                'bad_geometry',
                '',
                '',
            ]
        ),
    )

    qc = refraction_layer_observation_qc(masks)

    assert qc['layer_candidate_count']['v2_t1'] == 3
    assert qc['layer_observation_count']['v2_t1'] == 1
    np.testing.assert_array_equal(
        masks.layer_rejection_reason_sorted['v2_t1'][:3],
        np.asarray(['ok', 'missing_pick', 'bad_geometry']),
    )


def test_disabled_deeper_layers_are_omitted_from_external_config() -> None:
    model = _model(
        [
            _v2_layer(),
            _v3_layer(enabled=False, min_offset_m=None, max_offset_m=None),
            _vsub_layer(enabled=False, min_offset_m=None, max_offset_m=None),
        ]
    )

    masks = _build_masks(model)

    assert masks.layer_kind.tolist() == ['v2_t1']
    assert set(masks.layer_used_mask_sorted) == {'v2_t1'}
    assert refraction_layer_observation_qc(masks)['layer_count'] == 1


def test_disabled_deeper_layers_remain_in_viewer_qc_shape() -> None:
    model = _model(
        [
            _v2_layer(),
            _v3_layer(enabled=False, min_offset_m=None, max_offset_m=None),
            _vsub_layer(enabled=False, min_offset_m=None, max_offset_m=None),
        ]
    )

    masks = _build_masks(model)
    qc = layer_observation_qc_for_viewer(masks, model=model)

    assert list(qc) == ['v2_t1', 'v3_t2', 'vsub_t3']
    assert qc['v2_t1']['enabled'] is True
    assert qc['v3_t2'] == {
        'enabled': False,
        'n_candidate_observations': 0,
        'n_used_observations': 0,
        'min_offset_m': None,
        'max_offset_m': None,
        'rejection_counts': {'layer_disabled': 8},
    }
    assert qc['vsub_t3']['enabled'] is False
    assert qc['vsub_t3']['rejection_counts'] == {'layer_disabled': 8}
