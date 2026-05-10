from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from app.api.schemas import RefractionStaticLayerRequest, RefractionStaticModelRequest
from app.services.refraction_static_layer_observations import (
    LAYER_REJECTION_DISABLED,
    LAYER_REJECTION_MISSING_OFFSET,
    LAYER_REJECTION_OUTSIDE_GATE,
    RefractionLayerObservationMaskError,
    build_refraction_layer_observation_masks_from_arrays,
    refraction_layer_observation_qc,
)


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
    return build_refraction_layer_observation_masks_from_arrays(
        base_valid_mask_sorted=valid,
        offset_m_sorted=offset_m,
        rejection_reason_sorted=reasons,
        model=model,
    )


def test_non_overlapping_layer_gates_build_deterministic_masks_and_qc() -> None:
    masks = _build_masks(_model([_v2_layer(), _v3_layer(), _vsub_layer()]))

    assert masks.layer_kind.tolist() == ['v2_t1', 'v3_t2', 'vsub_t3']
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['v2_t1'],
        np.asarray([True, True, False, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['v3_t2'],
        np.asarray([False, False, True, True, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['vsub_t3'],
        np.asarray([False, False, False, False, True, True, False, False]),
    )
    assert masks.layer_rejection_reason_sorted['v2_t1'][2] == LAYER_REJECTION_OUTSIDE_GATE
    assert masks.layer_rejection_reason_sorted['v2_t1'][6] == (
        LAYER_REJECTION_MISSING_OFFSET
    )
    assert masks.layer_rejection_reason_sorted['v2_t1'][7] == 'missing_pick'

    qc = refraction_layer_observation_qc(masks)
    assert qc['v2_t1']['enabled'] is True
    assert qc['v2_t1']['n_candidate_observations'] == 2
    assert qc['v2_t1']['n_used_observations'] == 2
    assert qc['v3_t2']['n_candidate_observations'] == 2
    assert qc['v3_t2']['n_used_observations'] == 2
    assert qc['vsub_t3']['n_candidate_observations'] == 2
    assert qc['vsub_t3']['n_used_observations'] == 2


def test_overlapping_layer_gates_are_rejected_by_default() -> None:
    with pytest.raises(ValidationError, match='offset gates must not overlap'):
        _model(
            [
                _v2_layer(max_offset_m=150.0),
                _v3_layer(min_offset_m=100.0),
            ]
        )


def test_boundary_touching_layer_gates_are_rejected_by_default() -> None:
    with pytest.raises(ValidationError, match='offset gates must not overlap'):
        _model(
            [
                _v2_layer(max_offset_m=100.0),
                _v3_layer(min_offset_m=100.0),
            ]
        )


def test_boundary_touching_layer_gates_are_rejected_by_mask_validation() -> None:
    layer_model = SimpleNamespace(
        method='multilayer_time_term',
        allow_overlapping_layer_gates=False,
        layers=[
            RefractionStaticLayerRequest.model_validate(
                _v2_layer(max_offset_m=100.0)
            ),
            RefractionStaticLayerRequest.model_validate(
                _v3_layer(min_offset_m=100.0)
            ),
        ],
    )

    with pytest.raises(RefractionLayerObservationMaskError, match='overlaps'):
        _build_masks(layer_model, offset_m=np.asarray([99.0, 100.0, 101.0]))


def test_observation_overlap_is_allowed_when_explicitly_configured() -> None:
    allowed = _model(
        [
            _v2_layer(max_offset_m=100.0),
            _v3_layer(min_offset_m=100.0),
        ],
        allow_overlapping_layer_gates=True,
    )
    masks = _build_masks(allowed, offset_m=np.asarray([99.0, 100.0, 101.0]))
    assert masks.layer_used_mask_sorted['v2_t1'].tolist() == [True, True, False]
    assert masks.layer_used_mask_sorted['v3_t2'].tolist() == [False, True, True]


def test_disabled_deeper_layers_have_empty_masks_and_clear_qc() -> None:
    model = _model(
        [
            _v2_layer(),
            _v3_layer(enabled=False, min_offset_m=None, max_offset_m=None),
            _vsub_layer(enabled=False, min_offset_m=None, max_offset_m=None),
        ]
    )

    masks = _build_masks(model)

    assert not np.any(masks.layer_used_mask_sorted['v3_t2'])
    assert not np.any(masks.layer_used_mask_sorted['vsub_t3'])
    assert set(masks.layer_rejection_reason_sorted['v3_t2'].tolist()) == {
        LAYER_REJECTION_DISABLED
    }
    qc = refraction_layer_observation_qc(masks)
    assert qc['v3_t2']['enabled'] is False
    assert qc['v3_t2']['n_candidate_observations'] == 0
    assert qc['v3_t2']['n_used_observations'] == 0
    assert qc['vsub_t3']['enabled'] is False
    assert qc['vsub_t3']['n_used_observations'] == 0


def test_open_ended_deepest_gate_uses_infinite_internal_max_and_null_qc() -> None:
    masks = _build_masks(
        _model([_v2_layer(), _v3_layer(), _vsub_layer()]),
        offset_m=np.asarray([250.0, 251.0, 350.0, 1000.0]),
        valid=np.asarray([True, True, True, True]),
        reasons=np.asarray(['ok', 'ok', 'ok', 'ok']),
    )

    vsub_index = masks.layer_kind.tolist().index('vsub_t3')
    assert np.isposinf(masks.layer_max_offset_m[vsub_index])
    assert masks.layer_used_mask_sorted['vsub_t3'].tolist() == [
        False,
        True,
        True,
        True,
    ]
    qc = refraction_layer_observation_qc(masks)
    assert qc['vsub_t3']['max_offset_m'] is None
    assert qc['vsub_t3']['n_used_observations'] == 3
