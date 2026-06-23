from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticApplyRequest,
    RefractionStaticApplyResponse,
    RefractionStaticDatumRequest,
    RefractionStaticGeometryRequest,
    RefractionStaticLinkageRequest,
    RefractionStaticModelRequest,
    RefractionStaticMoveoutRequest,
    RefractionStaticPickSourceRequest,
    RefractionStaticRefractorCellRequest,
    RefractionStaticReducedTimeQcRequest,
    RefractionStaticRobustRequest,
    RefractionStaticSolverRequest,
)


def _payload() -> dict[str, Any]:
    return {
        'file_id': 'raw-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'first-break-job-id',
            'artifact_name': 'predicted_picks_time_s.npz',
        },
        'geometry': {
            'source_id_byte': 9,
            'receiver_id_byte': 13,
            'source_x_byte': 73,
            'source_y_byte': 77,
            'receiver_x_byte': 81,
            'receiver_y_byte': 85,
            'source_elevation_byte': 45,
            'receiver_elevation_byte': 41,
            'source_depth_byte': None,
            'coordinate_scalar_byte': 71,
            'elevation_scalar_byte': 69,
            'coordinate_unit': 'm',
            'elevation_unit': 'm',
        },
        'linkage': {
            'mode': 'required',
            'job_id': 'linkage-job-id',
            'artifact_name': 'geometry_linkage.npz',
        },
        'model': {
            'method': 'gli_variable_thickness',
            'weathering_velocity_m_s': 800.0,
            'bedrock_velocity_mode': 'solve_global',
            'bedrock_velocity_m_s': None,
            'initial_bedrock_velocity_m_s': 2500.0,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
            'max_weathering_thickness_m': None,
        },
        'moveout': {
            'model': 'head_wave_linear_offset',
            'distance_source': 'geometry',
            'offset_byte': 37,
            'min_offset_m': None,
            'max_offset_m': None,
            'allow_missing_offset': False,
            'max_geometry_offset_mismatch_m': None,
        },
        'solver': {
            'damping': 0.01,
            'min_picks_per_node': 1,
            'max_abs_half_intercept_time_ms': 500.0,
            'robust': {
                'enabled': True,
                'method': 'mad',
                'threshold': 3.5,
                'max_iterations': 5,
                'min_used_fraction': 0.5,
                'min_used_observations': 1,
            },
        },
        'datum': {
            'mode': 'floating_and_flat',
            'floating_datum_mode': 'constant',
            'flat_datum_elevation_m': 200.0,
            'floating_datum_elevation_m': 100.0,
            'smoothing_radius_m': None,
            'smoothing_window_nodes': 11,
            'smoothing_method': 'moving_average',
            'floating_datum_job_id': None,
            'floating_datum_artifact_name': None,
            'allow_flat_datum_above_topography': True,
            'allow_flat_datum_below_refractor': False,
        },
        'apply': {
            'mode': 'refraction_from_raw',
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
            'register_corrected_file': False,
        },
    }


def _minimal_payload() -> dict[str, Any]:
    return {
        'file_id': 'raw-file-id',
        'pick_source': {
            'kind': 'batch_predicted_npz',
            'job_id': 'first-break-job-id',
        },
        'model': {
            'weathering_velocity_m_s': 800.0,
        },
    }


def _validate(payload: dict[str, Any]) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(deepcopy(payload))


def test_refraction_static_schema_models_forbid_extra_fields() -> None:
    for model_cls in (
        RefractionStaticPickSourceRequest,
        RefractionStaticGeometryRequest,
        RefractionStaticLinkageRequest,
        RefractionStaticModelRequest,
        RefractionStaticMoveoutRequest,
        RefractionStaticRobustRequest,
        RefractionStaticSolverRequest,
        RefractionStaticRefractorCellRequest,
        RefractionStaticDatumRequest,
        RefractionStaticReducedTimeQcRequest,
        RefractionStaticApplyOptions,
        RefractionStaticApplyRequest,
        RefractionStaticApplyResponse,
    ):
        assert model_cls.model_config.get('extra') == 'forbid'


def test_refraction_static_request_accepts_fully_populated_valid_request() -> None:
    req = _validate(_payload())

    assert req.file_id == 'raw-file-id'
    assert req.key1_byte == 189
    assert req.key2_byte == 193
    assert req.pick_source.artifact_name == 'predicted_picks_time_s.npz'
    assert req.geometry.source_depth_byte is None
    assert req.linkage.mode == 'required'
    assert req.linkage.job_id == 'linkage-job-id'
    assert req.linkage.artifact_name == 'geometry_linkage.npz'
    assert req.model.method == 'gli_variable_thickness'
    assert req.model.bedrock_velocity_m_s is None
    assert req.moveout.model == 'head_wave_linear_offset'
    assert req.solver.robust.enabled is True
    assert req.datum.mode == 'floating_and_flat'
    assert req.apply.register_corrected_file is False


def test_refraction_static_request_accepts_minimal_valid_request_and_defaults() -> None:
    req = _validate(_minimal_payload())

    assert req.key1_byte == 189
    assert req.key2_byte == 193
    assert req.pick_source.artifact_name == 'predicted_picks_time_s.npz'
    assert req.geometry.source_id_byte == 9
    assert req.geometry.receiver_id_byte == 13
    assert req.geometry.source_depth_byte is None
    assert req.geometry.coordinate_unit == 'm'
    assert req.geometry.elevation_unit == 'm'
    assert req.linkage.mode == 'none'
    assert req.linkage.job_id is None
    assert req.linkage.artifact_name == 'geometry_linkage.npz'
    assert req.model.method == 'gli_variable_thickness'
    assert req.model.bedrock_velocity_mode == 'solve_global'
    assert req.model.bedrock_velocity_m_s is None
    assert req.model.initial_bedrock_velocity_m_s is None
    assert req.model.min_bedrock_velocity_m_s == 1200.0
    assert req.model.max_bedrock_velocity_m_s == 6000.0
    assert req.model.max_weathering_thickness_m is None
    assert req.moveout.model == 'head_wave_linear_offset'
    assert req.moveout.distance_source == 'geometry'
    assert req.moveout.offset_byte == 37
    assert req.moveout.min_offset_m is None
    assert req.moveout.max_offset_m is None
    assert req.moveout.allow_missing_offset is False
    assert req.moveout.max_geometry_offset_mismatch_m is None
    assert req.solver.damping == 0.01
    assert req.solver.min_picks_per_node == 1
    assert req.solver.max_abs_half_intercept_time_ms == 500.0
    assert req.solver.robust.method == 'mad'
    assert req.solver.robust.threshold == 3.5
    assert req.solver.robust.max_iterations == 5
    assert req.solver.robust.min_used_fraction == 0.5
    assert req.solver.robust.min_used_observations == 1
    assert req.datum.mode == 'none'
    assert req.datum.floating_datum_mode == 'smoothed_topography'
    assert req.datum.flat_datum_elevation_m is None
    assert req.datum.smoothing_window_nodes == 11
    assert req.datum.allow_flat_datum_above_topography is True
    assert req.datum.allow_flat_datum_below_refractor is False
    assert req.reduced_time_qc.reduction_velocity_mode == 'layer_velocity'
    assert req.reduced_time_qc.fixed_velocity_m_s is None
    assert req.apply.mode == 'refraction_from_raw'
    assert req.apply.interpolation == 'linear'
    assert req.apply.fill_value == 0.0
    assert req.apply.max_abs_shift_ms == 250.0
    assert req.apply.output_dtype == 'float32'
    assert req.apply.register_corrected_file is False


def _refractor_cell_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'number_of_cell_x': 3,
        'size_of_cell_x_m': 10.0,
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
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    'missing_field',
    ['line_origin_x_m', 'line_origin_y_m', 'line_azimuth_deg'],
)
def test_refraction_static_refractor_cell_line_2d_projected_requires_origin_and_azimuth(
    missing_field: str,
) -> None:
    payload = _refractor_cell_payload(
        coordinate_mode='line_2d_projected',
        line_origin_x_m=1000.0,
        line_origin_y_m=2000.0,
        line_azimuth_deg=45.0,
    )
    payload[missing_field] = None

    with pytest.raises(ValidationError, match=missing_field):
        RefractionStaticRefractorCellRequest.model_validate(payload)


def test_refraction_static_refractor_cell_line_2d_projected_requires_single_y_cell() -> None:
    payload = _refractor_cell_payload(
        coordinate_mode='line_2d_projected',
        line_origin_x_m=1000.0,
        line_origin_y_m=2000.0,
        line_azimuth_deg=45.0,
        number_of_cell_y=2,
        size_of_cell_y_m=25.0,
    )

    with pytest.raises(ValidationError, match='number_of_cell_y must be 1'):
        RefractionStaticRefractorCellRequest.model_validate(payload)


def test_refraction_datum_request_defaults_to_inactive_mode() -> None:
    datum = RefractionStaticDatumRequest()

    assert datum.mode == 'none'
    assert datum.flat_datum_elevation_m is None


@pytest.mark.parametrize(
    'mutator',
    [
        lambda payload: payload.pop('datum'),
        lambda payload: payload.update({'datum': {}}),
    ],
)
def test_refraction_static_request_defaults_missing_or_empty_datum(
    mutator: Any,
) -> None:
    payload = _payload()
    mutator(payload)

    req = _validate(payload)

    assert req.datum.mode == 'none'
    assert req.datum.flat_datum_elevation_m is None


@pytest.mark.parametrize(
    ('mutator', 'match'),
    [
        (lambda payload: payload.update({'file_id': ''}), 'file_id'),
        (lambda payload: payload.update({'key1_byte': 0}), 'key1_byte'),
        (lambda payload: payload.update({'key2_byte': 241}), 'key2_byte'),
        (lambda payload: payload.update({'unexpected': True}), 'unexpected'),
        (
            lambda payload: payload['geometry'].update({'unexpected': True}),
            'unexpected',
        ),
    ],
)
def test_refraction_static_request_rejects_top_level_contract_errors(
    mutator: Any,
    match: str,
) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


def test_refraction_pick_source_defaults_batch_artifact_name() -> None:
    payload = _payload()
    del payload['pick_source']['artifact_name']

    req = _validate(payload)

    assert req.pick_source.artifact_name == 'predicted_picks_time_s.npz'


def test_refraction_static_pick_source_uploaded_npz_valid_without_job_id() -> None:
    payload = _payload()
    payload['pick_source'] = {'kind': 'uploaded_npz'}

    req = _validate(payload)

    assert req.pick_source.kind == 'uploaded_npz'
    assert req.pick_source.job_id is None
    assert req.pick_source.artifact_name is None


def test_refraction_static_pick_source_uploaded_npz_rejects_job_id() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'uploaded_npz',
        'job_id': 'first-break-job-id',
    }

    with pytest.raises(ValidationError, match='uploaded_npz'):
        _validate(payload)


def test_refraction_static_pick_source_uploaded_npz_rejects_artifact_name() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'uploaded_npz',
        'artifact_name': 'predicted_picks_time_s.npz',
    }

    with pytest.raises(ValidationError, match='uploaded_npz'):
        _validate(payload)


def test_refraction_static_legacy_pick_source_still_valid() -> None:
    req = _validate(_payload())

    assert req.pick_source.kind == 'batch_predicted_npz'
    assert req.pick_source.job_id == 'first-break-job-id'
    assert req.pick_source.artifact_name == 'predicted_picks_time_s.npz'


def test_refraction_pick_source_rejects_artifact_source_without_job_id() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'batch_predicted_npz',
        'artifact_name': 'predicted_picks_time_s.npz',
    }

    with pytest.raises(ValidationError, match='pick_source.job_id'):
        _validate(payload)


def test_refraction_pick_source_rejects_manual_artifact_without_job_id() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'manual_npz_artifact',
        'artifact_name': 'manual_picks.npz',
    }

    with pytest.raises(ValidationError, match='pick_source.job_id'):
        _validate(payload)


def test_refraction_pick_source_rejects_manual_artifact_without_npz_suffix() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'manual_npz_artifact',
        'job_id': 'manual-job-id',
        'artifact_name': 'manual_picks.txt',
    }

    with pytest.raises(ValidationError, match='.npz'):
        _validate(payload)


def test_refraction_pick_source_rejects_manual_memmap_with_job_id() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'manual_memmap',
        'job_id': 'manual-job-id',
    }

    with pytest.raises(ValidationError, match='manual_memmap'):
        _validate(payload)


def test_refraction_pick_source_rejects_manual_memmap_with_artifact_name() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'manual_memmap',
        'artifact_name': 'manual_picks.npz',
    }

    with pytest.raises(ValidationError, match='manual_memmap'):
        _validate(payload)


@pytest.mark.parametrize(
    'artifact_name',
    ['', '.', '..', '/tmp/picks.npz', '../picks.npz', 'nested/picks.npz'],
)
def test_refraction_request_rejects_plain_artifact_name_violations(
    artifact_name: str,
) -> None:
    payload = _payload()
    payload['pick_source']['artifact_name'] = artifact_name

    with pytest.raises(ValidationError):
        _validate(payload)


def test_refraction_linkage_rejects_path_artifact_name() -> None:
    payload = _payload()
    payload['linkage']['artifact_name'] = 'nested/geometry_linkage.npz'

    with pytest.raises(ValidationError, match='linkage.artifact_name'):
        _validate(payload)


def test_refraction_geometry_rejects_same_source_receiver_id_byte() -> None:
    payload = _payload()
    payload['geometry']['receiver_id_byte'] = payload['geometry']['source_id_byte']

    with pytest.raises(ValidationError, match='source_id_byte'):
        _validate(payload)


def test_refraction_geometry_accepts_null_source_depth_byte() -> None:
    payload = _payload()
    payload['geometry']['source_depth_byte'] = None

    req = _validate(payload)

    assert req.geometry.source_depth_byte is None


@pytest.mark.parametrize('unit_field', ['coordinate_unit', 'elevation_unit'])
def test_refraction_geometry_rejects_invalid_units(unit_field: str) -> None:
    payload = _payload()
    payload['geometry'][unit_field] = 'km'

    with pytest.raises(ValidationError, match=unit_field):
        _validate(payload)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('source_id_byte', 0),
        ('receiver_id_byte', 241),
        ('source_x_byte', True),
        ('source_depth_byte', 0),
    ],
)
def test_refraction_geometry_rejects_invalid_trace_header_bytes(
    field: str,
    value: object,
) -> None:
    payload = _payload()
    payload['geometry'][field] = value

    with pytest.raises(ValidationError, match=field):
        _validate(payload)


def test_refraction_linkage_requires_job_id_for_required_mode() -> None:
    payload = _payload()
    del payload['linkage']['job_id']

    with pytest.raises(ValidationError, match='linkage.job_id'):
        _validate(payload)


def test_refraction_linkage_accepts_none_mode_without_job_id() -> None:
    payload = _payload()
    payload['linkage'] = {'mode': 'none'}

    req = _validate(payload)

    assert req.linkage.mode == 'none'
    assert req.linkage.job_id is None
    assert req.linkage.artifact_name == 'geometry_linkage.npz'


def test_refraction_linkage_rejects_job_id_for_none_mode() -> None:
    payload = _payload()
    payload['linkage']['mode'] = 'none'

    with pytest.raises(ValidationError, match='linkage.job_id'):
        _validate(payload)


def test_refraction_linkage_accepts_optional_mode_without_job_id() -> None:
    payload = _payload()
    payload['linkage'] = {'mode': 'optional'}

    req = _validate(payload)

    assert req.linkage.mode == 'optional'
    assert req.linkage.job_id is None


@pytest.mark.parametrize(
    ('mutator', 'match'),
    [
        (
            lambda payload: payload['model'].update({'method': 'plus_minus'}),
            'model.method',
        ),
        (
            lambda payload: payload['model'].update(
                {'weathering_velocity_m_s': 0.0}
            ),
            'weathering_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {
                    'min_bedrock_velocity_m_s': 6000.0,
                    'max_bedrock_velocity_m_s': 6000.0,
                }
            ),
            'min_bedrock_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {'min_bedrock_velocity_m_s': 700.0}
            ),
            'min_bedrock_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {'max_bedrock_velocity_m_s': 700.0}
            ),
            'max_bedrock_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {
                    'bedrock_velocity_mode': 'fixed_global',
                    'bedrock_velocity_m_s': None,
                }
            ),
            'bedrock_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {
                    'bedrock_velocity_mode': 'fixed_global',
                    'bedrock_velocity_m_s': 7000.0,
                }
            ),
            'bedrock_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {
                    'bedrock_velocity_mode': 'fixed_global',
                    'bedrock_velocity_m_s': 700.0,
                }
            ),
            'bedrock_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {'bedrock_velocity_m_s': 2500.0}
            ),
            'bedrock_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {'initial_bedrock_velocity_m_s': 700.0}
            ),
            'initial_bedrock_velocity_m_s',
        ),
        (
            lambda payload: payload['model'].update(
                {'max_weathering_thickness_m': 0.0}
            ),
            'max_weathering_thickness_m',
        ),
    ],
)
def test_refraction_model_rejects_invalid_values(
    mutator: Any,
    match: str,
) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


def test_refraction_model_accepts_fixed_global_velocity() -> None:
    payload = _payload()
    payload['model'].update(
        {
            'bedrock_velocity_mode': 'fixed_global',
            'bedrock_velocity_m_s': 2500.0,
            'initial_bedrock_velocity_m_s': None,
        }
    )

    req = _validate(payload)

    assert req.model.bedrock_velocity_mode == 'fixed_global'
    assert req.model.bedrock_velocity_m_s == 2500.0


def test_refraction_model_accepts_solve_global_initial_velocity() -> None:
    payload = _payload()
    payload['model']['initial_bedrock_velocity_m_s'] = 3000.0

    req = _validate(payload)

    assert req.model.initial_bedrock_velocity_m_s == 3000.0


def test_refraction_moveout_requires_offset_byte_for_offset_header() -> None:
    payload = _payload()
    payload['moveout'].update({'distance_source': 'offset_header', 'offset_byte': None})

    with pytest.raises(ValidationError, match='moveout.offset_byte'):
        _validate(payload)


@pytest.mark.parametrize('distance_source', ['geometry', 'auto'])
def test_refraction_moveout_accepts_null_offset_byte_without_required_header(
    distance_source: str,
) -> None:
    payload = _payload()
    payload['moveout'].update(
        {'distance_source': distance_source, 'offset_byte': None}
    )

    req = _validate(payload)

    assert req.moveout.distance_source == distance_source
    assert req.moveout.offset_byte is None


@pytest.mark.parametrize(
    ('mutator', 'match'),
    [
        (
            lambda payload: payload['moveout'].update({'min_offset_m': -1.0}),
            'min_offset_m',
        ),
        (
            lambda payload: payload['moveout'].update({'max_offset_m': -1.0}),
            'max_offset_m',
        ),
        (
            lambda payload: payload['moveout'].update(
                {'min_offset_m': 100.0, 'max_offset_m': 100.0}
            ),
            'min_offset_m',
        ),
        (
            lambda payload: payload['moveout'].update(
                {'max_geometry_offset_mismatch_m': -1.0}
            ),
            'max_geometry_offset_mismatch_m',
        ),
        (
            lambda payload: payload['moveout'].update({'allow_missing_offset': 1}),
            'allow_missing_offset',
        ),
    ],
)
def test_refraction_moveout_rejects_invalid_values(
    mutator: Any,
    match: str,
) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


@pytest.mark.parametrize('method', ['mad', 'sigma'])
def test_refraction_robust_accepts_supported_methods(method: str) -> None:
    payload = _payload()
    payload['solver']['robust']['method'] = method

    req = _validate(payload)

    assert req.solver.robust.method == method


@pytest.mark.parametrize(
    ('mutator', 'match'),
    [
        (
            lambda payload: payload['solver']['robust'].update(
                {'method': 'median'}
            ),
            'method',
        ),
        (
            lambda payload: payload['solver']['robust'].update({'enabled': 1}),
            'enabled',
        ),
        (
            lambda payload: payload['solver']['robust'].update({'threshold': 0.0}),
            'threshold',
        ),
        (
            lambda payload: payload['solver']['robust'].update(
                {'max_iterations': 0}
            ),
            'max_iterations',
        ),
        (
            lambda payload: payload['solver']['robust'].update(
                {'min_used_fraction': 0.0}
            ),
            'min_used_fraction',
        ),
        (
            lambda payload: payload['solver']['robust'].update(
                {'min_used_fraction': 1.1}
            ),
            'min_used_fraction',
        ),
        (
            lambda payload: payload['solver']['robust'].update(
                {'min_used_observations': 0}
            ),
            'min_used_observations',
        ),
    ],
)
def test_refraction_robust_rejects_invalid_values(
    mutator: Any,
    match: str,
) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


@pytest.mark.parametrize(
    ('mutator', 'match'),
    [
        (lambda payload: payload['solver'].update({'damping': -0.1}), 'damping'),
        (
            lambda payload: payload['solver'].update({'min_picks_per_node': 0}),
            'min_picks_per_node',
        ),
        (
            lambda payload: payload['solver'].update(
                {'max_abs_half_intercept_time_ms': 0.0}
            ),
            'max_abs_half_intercept_time_ms',
        ),
    ],
)
def test_refraction_solver_rejects_invalid_values(
    mutator: Any,
    match: str,
) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


def test_refraction_datum_accepts_floating_and_flat_request() -> None:
    payload = _payload()
    payload['datum'] = {
        'mode': 'floating_and_flat',
        'floating_datum_mode': 'constant',
        'flat_datum_elevation_m': 200.0,
        'floating_datum_elevation_m': 100.0,
        'smoothing_radius_m': None,
        'smoothing_window_nodes': 11,
        'smoothing_method': 'moving_average',
        'floating_datum_job_id': None,
        'floating_datum_artifact_name': None,
        'allow_flat_datum_above_topography': True,
        'allow_flat_datum_below_refractor': False,
    }

    req = _validate(payload)

    assert req.datum.mode == 'floating_and_flat'
    assert req.datum.floating_datum_mode == 'constant'
    assert req.datum.flat_datum_elevation_m == 200.0
    assert req.datum.floating_datum_elevation_m == 100.0


@pytest.mark.parametrize(
    ('datum_payload', 'match'),
    [
        (
            {
                'mode': 'floating_and_flat',
                'floating_datum_mode': 'surface',
            },
            'flat_datum_elevation_m',
        ),
        (
            {
                'mode': 'flat_only',
                'floating_datum_mode': 'surface',
            },
            'flat_datum_elevation_m',
        ),
        (
            {
                'mode': 'floating_only',
                'floating_datum_mode': 'constant',
            },
            'floating_datum_elevation_m',
        ),
        (
            {
                'mode': 'floating_only',
                'floating_datum_mode': 'smoothed_topography',
                'smoothing_radius_m': 0.0,
            },
            'smoothing_radius_m',
        ),
        (
            {
                'mode': 'floating_only',
                'floating_datum_mode': 'smoothed_topography',
                'smoothing_window_nodes': 4,
            },
            'smoothing_window_nodes',
        ),
        (
            {
                'mode': 'floating_only',
                'floating_datum_mode': 'from_artifact',
                'floating_datum_job_id': 'job-id',
                'floating_datum_artifact_name': 'nested/floating.csv',
            },
            'floating_datum_artifact_name',
        ),
        (
            {
                'mode': 'none',
                'allow_flat_datum_below_refractor': 0,
            },
            'allow_flat_datum_below_refractor',
        ),
    ],
)
def test_refraction_datum_rejects_invalid_values(
    datum_payload: dict[str, Any],
    match: str,
) -> None:
    payload = _payload()
    payload['datum'] = datum_payload

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


@pytest.mark.parametrize(
    ('mutator', 'match'),
    [
        (
            lambda payload: payload['apply'].update({'mode': 'weathering_only'}),
            'mode',
        ),
        (
            lambda payload: payload['apply'].update({'interpolation': 'nearest'}),
            'interpolation',
        ),
        (
            lambda payload: payload['apply'].update({'output_dtype': 'float64'}),
            'output_dtype',
        ),
        (
            lambda payload: payload['apply'].update({'fill_value': float('nan')}),
            'fill_value',
        ),
        (
            lambda payload: payload['apply'].update({'max_abs_shift_ms': 0.0}),
            'max_abs_shift_ms',
        ),
        (
            lambda payload: payload['apply'].update({'register_corrected_file': 0}),
            'register_corrected_file',
        ),
    ],
)
def test_refraction_apply_rejects_invalid_values(
    mutator: Any,
    match: str,
) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


def test_refraction_static_apply_response_validates() -> None:
    response = RefractionStaticApplyResponse.model_validate(
        {'job_id': 'x', 'state': 'queued'}
    )

    assert response.job_id == 'x'
    assert response.state == 'queued'
