from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import RefractionStaticModelRequest
from app.statics.refraction.core_options import (
    normalize_first_layer_from_model_request as normalize_refraction_first_layer_request,
)


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


def _refractor_cell_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'number_of_cell_x': 20,
        'size_of_cell_x_m': 500.0,
        'x_coordinate_origin_m': 0.0,
        'number_of_cell_y': 1,
        'size_of_cell_y_m': 1000.0,
        'y_coordinate_origin_m': 0.0,
        'assignment_mode': 'midpoint',
        'outside_grid_policy': 'reject',
        'min_observations_per_cell': 5,
        'velocity_smoothing_weight': 0.1,
        'smoothing_reference_distance_m': 500.0,
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


def test_refraction_static_solve_cell_request_valid() -> None:
    model = RefractionStaticModelRequest.model_validate(
        _model_payload(
            weathering_velocity_m_s=None,
            first_layer={
                'mode': 'constant',
                'weathering_velocity_m_s': 800.0,
            },
            bedrock_velocity_mode='solve_cell',
            initial_bedrock_velocity_m_s=2400.0,
            refractor_cell=_refractor_cell_payload(
                number_of_cell_x=20,
                number_of_cell_y=2,
                size_of_cell_y_m=1000.0,
            ),
        )
    )

    assert model.bedrock_velocity_mode == 'solve_cell'
    assert model.refractor_cell is not None
    assert model.refractor_cell.assignment_mode == 'midpoint'
    assert model.refractor_cell.outside_grid_policy == 'reject'
    assert model.refractor_cell.number_of_cell_x == 20
    assert model.refractor_cell.number_of_cell_y == 2


def test_refraction_static_solve_cell_requires_refractor_cell() -> None:
    with pytest.raises(
        ValidationError,
        match='model.refractor_cell is required',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(bedrock_velocity_mode='solve_cell')
        )


@pytest.mark.parametrize(
    ('bedrock_velocity_mode', 'bedrock_velocity_m_s'),
    [
        ('solve_global', None),
        ('fixed_global', 2500.0),
    ],
)
def test_refraction_static_global_modes_reject_refractor_cell(
    bedrock_velocity_mode: str,
    bedrock_velocity_m_s: float | None,
) -> None:
    with pytest.raises(
        ValidationError,
        match='model.refractor_cell is only allowed',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                bedrock_velocity_mode=bedrock_velocity_mode,
                bedrock_velocity_m_s=bedrock_velocity_m_s,
                refractor_cell=_refractor_cell_payload(),
            )
        )


@pytest.mark.parametrize(
    ('field_name', 'value'),
    [
        ('number_of_cell_x', 0),
        ('number_of_cell_y', 0),
        ('min_observations_per_cell', 0),
    ],
)
def test_refraction_static_refractor_cell_rejects_invalid_counts(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(
        ValidationError,
        match=f'model.refractor_cell.{field_name}',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                bedrock_velocity_mode='solve_cell',
                refractor_cell=_refractor_cell_payload(**{field_name: value}),
            )
        )


@pytest.mark.parametrize(
    ('cell_overrides', 'error_field'),
    [
        ({'size_of_cell_x_m': 0.0}, 'size_of_cell_x_m'),
        ({'size_of_cell_y_m': 0.0}, 'size_of_cell_y_m'),
        (
            {'number_of_cell_y': 2, 'size_of_cell_y_m': None},
            'size_of_cell_y_m',
        ),
        ({'velocity_smoothing_weight': -0.1}, 'velocity_smoothing_weight'),
        (
            {'smoothing_reference_distance_m': 0.0},
            'smoothing_reference_distance_m',
        ),
    ],
)
def test_refraction_static_refractor_cell_rejects_invalid_sizes(
    cell_overrides: dict[str, object],
    error_field: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match=f'model.refractor_cell.{error_field}',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                bedrock_velocity_mode='solve_cell',
                refractor_cell=_refractor_cell_payload(**cell_overrides),
            )
        )


def test_refraction_static_refractor_cell_rejects_unknown_assignment_mode() -> None:
    with pytest.raises(
        ValidationError,
        match='model.refractor_cell.assignment_mode',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                bedrock_velocity_mode='solve_cell',
                refractor_cell=_refractor_cell_payload(
                    assignment_mode='nearest',
                ),
            )
        )


def test_refraction_static_refractor_cell_rejects_unknown_outside_grid_policy() -> None:
    with pytest.raises(
        ValidationError,
        match='model.refractor_cell.outside_grid_policy',
    ):
        RefractionStaticModelRequest.model_validate(
            _model_payload(
                bedrock_velocity_mode='solve_cell',
                refractor_cell=_refractor_cell_payload(
                    outside_grid_policy='clip',
                ),
            )
        )


def test_refraction_static_phase1_legacy_request_still_valid() -> None:
    solve_global = RefractionStaticModelRequest.model_validate(_model_payload())
    fixed_global = RefractionStaticModelRequest.model_validate(
        _model_payload(
            bedrock_velocity_mode='fixed_global',
            bedrock_velocity_m_s=2500.0,
            initial_bedrock_velocity_m_s=None,
        )
    )

    assert solve_global.bedrock_velocity_mode == 'solve_global'
    assert solve_global.refractor_cell is None
    assert fixed_global.bedrock_velocity_mode == 'fixed_global'
    assert fixed_global.refractor_cell is None
