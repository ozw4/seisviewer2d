from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import RefractionStaticModelRequest
from app.services.refraction_static_service import normalize_refraction_first_layer_request


def _model_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
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
    return payload


def test_refraction_static_legacy_weathering_velocity_request_still_valid() -> None:
    model = RefractionStaticModelRequest.model_validate(_model_payload())

    assert model.first_layer is None
    assert model.first_layer_mode == 'constant'
    assert model.resolved_weathering_velocity_m_s == pytest.approx(800.0)

    resolved = normalize_refraction_first_layer_request(model)
    assert resolved.mode == 'constant'
    assert resolved.weathering_velocity_m_s == pytest.approx(800.0)
    assert resolved.qc['resolved_weathering_velocity_m_s'] == pytest.approx(800.0)


def test_refraction_static_first_layer_constant_request_valid() -> None:
    model = RefractionStaticModelRequest.model_validate(
        _model_payload(
            weathering_velocity_m_s=None,
            first_layer={
                'mode': 'constant',
                'weathering_velocity_m_s': 800.0,
            },
        )
    )

    assert model.weathering_velocity_m_s is None
    assert model.first_layer_mode == 'constant'
    assert model.resolved_weathering_velocity_m_s == pytest.approx(800.0)

    resolved = normalize_refraction_first_layer_request(model)
    assert resolved.weathering_velocity_m_s == pytest.approx(800.0)


def test_refraction_static_rejects_v1_greater_than_bedrock_min() -> None:
    with pytest.raises(
        ValidationError,
        match='model.min_bedrock_velocity_m_s must be greater than '
        'model.resolved_weathering_velocity_m_s',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                weathering_velocity_m_s=None,
                first_layer={
                    'mode': 'constant',
                    'weathering_velocity_m_s': 1300.0,
                },
            )
        )


def test_refraction_static_rejects_conflicting_legacy_and_first_layer_v1() -> None:
    with pytest.raises(
        ValidationError,
        match='model.weathering_velocity_m_s and '
        'model.first_layer.weathering_velocity_m_s must match',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                weathering_velocity_m_s=800.0,
                first_layer={
                    'mode': 'constant',
                    'weathering_velocity_m_s': 900.0,
                },
            )
        )


def test_refraction_static_estimate_direct_arrival_public_request_valid_without_v1() -> None:
    model = RefractionStaticModelRequest.model_validate(
        _model_payload(
            weathering_velocity_m_s=None,
            first_layer={
                'mode': 'estimate_direct_arrival',
                'min_direct_offset_m': 20.0,
                'max_direct_offset_m': 140.0,
            },
        )
    )

    assert model.first_layer is not None
    assert model.first_layer.mode == 'estimate_direct_arrival'

    with pytest.raises(
        ValueError,
        match='requires a resolved weathering velocity',
    ):
        normalize_refraction_first_layer_request(model)


def test_refraction_static_model_rejects_estimated_first_layer_v1_value() -> None:
    with pytest.raises(
        ValidationError,
        match='model.first_layer.weathering_velocity_m_s must be omitted when '
        'model.first_layer.mode is estimate_direct_arrival',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                weathering_velocity_m_s=None,
                first_layer={
                    'mode': 'estimate_direct_arrival',
                    'weathering_velocity_m_s': 812.5,
                    'min_direct_offset_m': 20.0,
                    'max_direct_offset_m': 140.0,
                },
            )
        )


def test_refraction_static_model_rejects_estimated_legacy_v1_value() -> None:
    with pytest.raises(
        ValidationError,
        match='model.weathering_velocity_m_s must be omitted when '
        'model.first_layer.mode is estimate_direct_arrival',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                weathering_velocity_m_s=812.5,
                first_layer={
                    'mode': 'estimate_direct_arrival',
                    'min_direct_offset_m': 20.0,
                    'max_direct_offset_m': 140.0,
                },
            )
        )


def test_refraction_static_first_layer_estimate_requires_direct_offset_gate() -> None:
    with pytest.raises(
        ValidationError,
        match='min_direct_offset_m and .*max_direct_offset_m are required',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                weathering_velocity_m_s=None,
                first_layer={'mode': 'estimate_direct_arrival'},
            )
        )
