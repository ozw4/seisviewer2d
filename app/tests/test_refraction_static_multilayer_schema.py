from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticModelRequest,
)
from app.statics.refraction.core_options import (
    normalized_layers_from_model_request as normalize_refraction_static_layers,
)


def _v2_layer(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'kind': 'v2_t1',
        'enabled': True,
        'min_offset_m': 300.0,
        'max_offset_m': 1800.0,
        'velocity_mode': 'solve_cell',
        'initial_velocity_m_s': 2400.0,
        'min_velocity_m_s': 1200.0,
        'max_velocity_m_s': 5000.0,
        'min_observations_per_cell': 5,
        'smoothing_weight': 0.0,
    }
    payload.update(overrides)
    return payload


def _v3_layer(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'kind': 'v3_t2',
        'enabled': True,
        'min_offset_m': 1800.0,
        'max_offset_m': 3500.0,
        'velocity_mode': 'solve_global',
        'initial_velocity_m_s': 3500.0,
        'min_velocity_m_s': 1800.0,
        'max_velocity_m_s': 7000.0,
    }
    payload.update(overrides)
    return payload


def _vsub_layer(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'kind': 'vsub_t3',
        'enabled': True,
        'min_offset_m': 3500.0,
        'max_offset_m': None,
        'velocity_mode': 'fixed_global',
        'fixed_velocity_m_s': 4500.0,
        'min_velocity_m_s': 2500.0,
        'max_velocity_m_s': 9000.0,
    }
    payload.update(overrides)
    return payload


def _refractor_cell() -> dict[str, object]:
    return {
        'number_of_cell_x': 4,
        'size_of_cell_x_m': 500.0,
        'x_coordinate_origin_m': 0.0,
        'number_of_cell_y': 1,
        'size_of_cell_y_m': None,
        'y_coordinate_origin_m': 0.0,
        'assignment_mode': 'midpoint',
        'outside_grid_policy': 'reject',
        'coordinate_mode': 'grid_3d',
        'min_observations_per_cell': 5,
        'velocity_smoothing_weight': 0.0,
        'smoothing_reference_distance_m': None,
    }


def _multilayer_model(
    layers: list[dict[str, object]],
    *,
    include_refractor_cell: bool = True,
) -> dict[str, object]:
    model: dict[str, object] = {
        'method': 'multilayer_time_term',
        'first_layer': {
            'mode': 'constant',
            'weathering_velocity_m_s': 800.0,
        },
        'layers': layers,
    }
    if include_refractor_cell:
        model['refractor_cell'] = _refractor_cell()
    return model


def _apply_payload(
    *,
    model: dict[str, object],
    conversion: dict[str, object],
) -> dict[str, Any]:
    return {
        'file_id': 'raw-file-id',
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'first-break-job-id',
        },
        'linkage': {
            'mode': 'required',
            'job_id': 'linkage-job-id',
        },
        'model': model,
        'conversion': conversion,
    }


def _validate_model(layers: list[dict[str, object]]) -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        deepcopy(_multilayer_model(layers))
    )


def _validate_model_without_refractor_cell(
    layers: list[dict[str, object]],
) -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest.model_validate(
        deepcopy(_multilayer_model(layers, include_refractor_cell=False))
    )


def test_multilayer_schema_accepts_one_layer_request() -> None:
    model = _validate_model([_v2_layer()])

    layers = normalize_refraction_static_layers(model)

    assert model.method == 'multilayer_time_term'
    assert model.enabled_refraction_layer_count == 1
    assert [layer.kind for layer in layers] == ['v2_t1']
    assert layers[0].velocity_mode == 'solve_cell'
    assert layers[0].initial_velocity_m_s == pytest.approx(2400.0)


def test_multilayer_schema_accepts_two_layer_request() -> None:
    model = _validate_model([_v2_layer(), _v3_layer()])

    layers = normalize_refraction_static_layers(model)

    assert model.enabled_refraction_layer_count == 2
    assert [layer.kind for layer in layers] == ['v2_t1', 'v3_t2']
    assert layers[1].initial_velocity_m_s == pytest.approx(3500.0)


def test_multilayer_schema_accepts_three_layer_request() -> None:
    model = _validate_model([_v2_layer(), _v3_layer(), _vsub_layer()])

    layers = normalize_refraction_static_layers(model)

    assert model.enabled_refraction_layer_count == 3
    assert [layer.kind for layer in layers] == ['v2_t1', 'v3_t2', 'vsub_t3']
    assert layers[2].velocity_mode == 'fixed_global'
    assert layers[2].fixed_velocity_m_s == pytest.approx(4500.0)


def test_multilayer_schema_preserves_disabled_layer_configs() -> None:
    model = _validate_model(
        [
            _v2_layer(),
            _v3_layer(enabled=False),
            _vsub_layer(enabled=False),
        ]
    )

    enabled = normalize_refraction_static_layers(model)

    assert [layer.kind for layer in enabled] == ['v2_t1']
    assert [layer.kind for layer in model.layers or ()] == [
        'v2_t1',
        'v3_t2',
        'vsub_t3',
    ]
    assert [layer.enabled for layer in model.layers or ()] == [True, False, False]


def test_legacy_refraction_schema_normalizes_to_single_v2_layer() -> None:
    model = RefractionStaticModelRequest.model_validate(
        {
            'method': 'gli_variable_thickness',
            'weathering_velocity_m_s': 800.0,
            'bedrock_velocity_mode': 'solve_global',
            'initial_bedrock_velocity_m_s': 2500.0,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
        }
    )

    layers = normalize_refraction_static_layers(model)

    assert [layer.kind for layer in layers] == ['v2_t1']
    assert layers[0].initial_velocity_m_s == pytest.approx(2500.0)


def test_multilayer_schema_rejects_invalid_layer_order() -> None:
    with pytest.raises(ValidationError, match='ordered v2_t1, v3_t2, vsub_t3'):
        _validate_model([_v3_layer(), _v2_layer()])


def test_multilayer_schema_rejects_duplicate_layer_kind() -> None:
    with pytest.raises(ValidationError, match='duplicate layer kinds'):
        _validate_model(
            [_v2_layer(), _v2_layer(min_offset_m=1900.0, max_offset_m=2600.0)]
        )


def test_multilayer_schema_rejects_missing_fixed_velocity() -> None:
    with pytest.raises(ValidationError, match='fixed_velocity_m_s is required'):
        _validate_model(
            [
                _v2_layer(),
                _v3_layer(),
                _vsub_layer(fixed_velocity_m_s=None),
            ]
        )


def test_multilayer_schema_rejects_solve_layer_without_initial_velocity() -> None:
    with pytest.raises(
        ValidationError,
        match='initial_velocity_m_s or model.initial_bedrock_velocity_m_s',
    ):
        _validate_model([_v2_layer(), _v3_layer(initial_velocity_m_s=None)])


def test_multilayer_schema_rejects_solve_cell_without_refractor_cell() -> None:
    with pytest.raises(ValidationError, match='model.refractor_cell is required'):
        _validate_model_without_refractor_cell([_v2_layer()])


def test_multilayer_schema_rejects_inconsistent_velocity_bounds() -> None:
    with pytest.raises(ValidationError, match='velocity bounds must be greater'):
        _validate_model([_v2_layer(), _v3_layer(min_velocity_m_s=1000.0)])


def test_multilayer_schema_rejects_non_deepest_null_max_offset() -> None:
    with pytest.raises(ValidationError, match='max_offset_m may be null only'):
        _validate_model([_v2_layer(max_offset_m=None), _v3_layer()])


def test_multilayer_conversion_requires_matching_enabled_layer_count() -> None:
    payload = _apply_payload(
        model=_multilayer_model([_v2_layer(), _v3_layer()]),
        conversion={'mode': 't1lsst_multilayer', 'layer_count': 2},
    )
    req = RefractionStaticApplyRequest.model_validate(payload)

    assert req.conversion.mode == 't1lsst_multilayer'
    assert req.conversion.layer_count == 2

    bad_payload = deepcopy(payload)
    bad_payload['conversion']['layer_count'] = 3
    with pytest.raises(
        ValidationError,
        match=(
            'conversion.layer_count=3.*enabled layer kinds=v2_t1, v3_t2'
        ),
    ):
        RefractionStaticApplyRequest.model_validate(bad_payload)


def test_multilayer_conversion_accepts_three_layer_contract() -> None:
    payload = _apply_payload(
        model=_multilayer_model([_v2_layer(), _v3_layer(), _vsub_layer()]),
        conversion={'mode': 't1lsst_multilayer', 'layer_count': 3},
    )
    req = RefractionStaticApplyRequest.model_validate(payload)

    assert req.conversion.mode == 't1lsst_multilayer'
    assert req.conversion.layer_count == 3
    assert [
        layer.kind
        for layer in normalize_refraction_static_layers(req.model)
    ] == ['v2_t1', 'v3_t2', 'vsub_t3']


def test_multilayer_conversion_rejects_layer_count_two_with_vsub_enabled() -> None:
    payload = _apply_payload(
        model=_multilayer_model([_v2_layer(), _v3_layer(), _vsub_layer()]),
        conversion={'mode': 't1lsst_multilayer', 'layer_count': 2},
    )

    with pytest.raises(
        ValidationError,
        match=(
            'conversion.layer_count=2.*enabled layer kinds=v2_t1, v3_t2, vsub_t3'
        ),
    ):
        RefractionStaticApplyRequest.model_validate(payload)
