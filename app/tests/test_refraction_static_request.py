from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from app.api.schemas import RefractionStaticApplyRequest


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
        },
        'offset': {
            'distance_source': 'geometry',
            'offset_byte': 37,
            'min_offset_m': None,
            'max_offset_m': None,
            'allow_missing_offset': False,
            'max_geometry_offset_mismatch_m': None,
        },
        'solver': {
            'damping': 0.01,
            'robust': {
                'enabled': True,
                'method': 'mad',
                'threshold': 3.5,
                'max_iterations': 5,
                'min_used_fraction': 0.5,
                'min_used_observations': 1,
            },
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


def _validate(payload: dict[str, Any]) -> RefractionStaticApplyRequest:
    return RefractionStaticApplyRequest.model_validate(deepcopy(payload))


def test_refraction_static_request_accepts_valid_payload() -> None:
    req = _validate(_payload())

    assert req.file_id == 'raw-file-id'
    assert req.pick_source.artifact_name == 'predicted_picks_time_s.npz'
    assert req.linkage.artifact_name == 'geometry_linkage.npz'
    assert req.model.bedrock_velocity_m_s is None
    assert req.apply.register_corrected_file is False


def test_refraction_static_request_accepts_register_corrected_file_true() -> None:
    payload = _payload()
    payload['apply']['register_corrected_file'] = True

    req = _validate(payload)

    assert req.apply.register_corrected_file is True


def test_refraction_pick_source_defaults_batch_artifact_name() -> None:
    payload = _payload()
    del payload['pick_source']['artifact_name']

    req = _validate(payload)

    assert req.pick_source.artifact_name == 'predicted_picks_time_s.npz'


@pytest.mark.parametrize(
    ('mutator', 'match'),
    [
        (lambda payload: payload.update({'file_id': ''}), 'file_id'),
        (lambda payload: payload.update({'key1_byte': 0}), 'key1_byte'),
        (
            lambda payload: payload['geometry'].update({'receiver_id_byte': 9}),
            'source_id_byte',
        ),
        (
            lambda payload: payload['geometry'].update({'coordinate_unit': 'km'}),
            'coordinate_unit',
        ),
        (
            lambda payload: payload['linkage'].pop('job_id'),
            'linkage.job_id',
        ),
        (
            lambda payload: payload['linkage'].update(
                {'mode': 'none', 'job_id': 'linkage-job-id'}
            ),
            'linkage.job_id',
        ),
        (
            lambda payload: payload['model'].update({'method': 'plus_minus'}),
            'method',
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
            lambda payload: payload['offset'].update(
                {'distance_source': 'offset_header', 'offset_byte': None}
            ),
            'offset.offset_byte',
        ),
        (
            lambda payload: payload['offset'].update({'min_offset_m': -1.0}),
            'min_offset_m',
        ),
        (
            lambda payload: payload['offset'].update(
                {'min_offset_m': 100.0, 'max_offset_m': 100.0}
            ),
            'min_offset_m',
        ),
        (
            lambda payload: payload['solver'].update({'damping': -0.1}),
            'solver.damping',
        ),
        (
            lambda payload: payload['solver']['robust'].update(
                {'max_iterations': 0}
            ),
            'max_iterations',
        ),
        (
            lambda payload: payload['solver']['robust'].update({'threshold': 0.0}),
            'threshold',
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
        (
            lambda payload: payload['apply'].update({'mode': 'time_term'}),
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
            lambda payload: payload['apply'].update({'max_abs_shift_ms': 0.0}),
            'max_abs_shift_ms',
        ),
    ],
)
def test_refraction_static_request_rejects_invalid_values(
    mutator: Any,
    match: str,
) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(ValidationError, match=match):
        _validate(payload)


def test_refraction_pick_source_rejects_manual_memmap_with_artifact_ref() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'manual_memmap',
        'job_id': 'manual-job-id',
        'artifact_name': 'manual_picks.npz',
    }

    with pytest.raises(ValidationError, match='manual_memmap'):
        _validate(payload)


def test_refraction_pick_source_rejects_artifact_source_without_job_id() -> None:
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
