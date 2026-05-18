from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from app.api.schemas import ResidualStaticApplyRequest, ResidualStaticApplyResponse


def _payload() -> dict[str, Any]:
    return {
        'file_id': 'datum-corrected-file-id',
        'key1_byte': 189,
        'key2_byte': 193,
        'datum_solution': {
            'job_id': 'datum-job',
            'name': 'datum_static_solution.npz',
        },
        'pick_source': {
            'kind': 'batch_job_artifact',
            'job_id': 'pick-job',
            'name': 'predicted_picks_time_s.npz',
        },
        'geometry': {
            'source_id_byte': 17,
            'receiver_id_byte': 13,
        },
        'offset': {'offset_byte': 37},
        'moveout': {'model': 'linear_abs_offset'},
        'solver': {
            'gauge': 'zero_mean_source_receiver',
            'damping_lambda': 0.0,
            'min_valid_picks': 10,
            'min_picks_per_source': 1,
            'min_picks_per_receiver': 1,
            'max_abs_estimated_delay_ms': 250.0,
        },
        'robust': {
            'enabled': True,
            'method': 'mad',
            'max_iterations': 3,
            'threshold': 4.0,
            'min_used_fraction': 0.5,
        },
        'apply': {
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
            'register_corrected_file': True,
        },
    }


def _validate(payload: dict[str, Any]) -> ResidualStaticApplyRequest:
    return ResidualStaticApplyRequest.model_validate(deepcopy(payload))


def test_residual_static_apply_request_accepts_nested_linear_abs_offset() -> None:
    req = _validate(_payload())

    assert req.file_id == 'datum-corrected-file-id'
    assert req.datum_solution.name == 'datum_static_solution.npz'
    assert req.pick_source.kind == 'batch_job_artifact'
    assert req.geometry.source_id_byte == 17
    assert req.geometry.receiver_id_byte == 13
    assert req.offset.offset_byte == 37
    assert req.source_id_byte == 17
    assert req.receiver_id_byte == 13
    assert req.offset_byte == 37
    assert req.moveout.model == 'linear_abs_offset'
    assert req.solver.min_valid_picks == 10
    assert req.robust.enabled is True
    assert req.apply.output_dtype == 'float32'


def test_residual_static_apply_request_accepts_moveout_none_with_null_offset() -> None:
    payload = _payload()
    payload['moveout'] = {'model': 'none'}
    payload['offset'] = {'offset_byte': None}
    payload['pick_source'] = {'kind': 'manual_memmap'}

    req = _validate(payload)

    assert req.moveout.model == 'none'
    assert req.offset_byte is None


def test_residual_static_apply_request_rejects_flattened_source_receiver_fields() -> None:
    payload = _payload()
    payload['source_id_byte'] = 17
    payload['receiver_id_byte'] = 13
    payload['offset_byte'] = 37

    with pytest.raises(ValidationError):
        _validate(payload)


def test_residual_static_apply_request_rejects_moveout_none_with_offset_byte() -> None:
    payload = _payload()
    payload['moveout'] = {'model': 'none'}

    with pytest.raises(ValidationError, match='offset.offset_byte must be null'):
        _validate(payload)


def test_residual_static_apply_request_rejects_linear_abs_offset_without_offset() -> None:
    payload = _payload()
    payload['offset'] = {'offset_byte': None}

    with pytest.raises(ValidationError, match='offset.offset_byte is required'):
        _validate(payload)


def test_residual_static_apply_request_rejects_header_byte_zero() -> None:
    payload = _payload()
    payload['key1_byte'] = 0

    with pytest.raises(ValidationError, match='range 1..240'):
        _validate(payload)


def test_residual_static_apply_request_rejects_header_byte_241() -> None:
    payload = _payload()
    payload['geometry']['source_id_byte'] = 241

    with pytest.raises(ValidationError, match='range 1..240'):
        _validate(payload)


def test_residual_static_apply_request_rejects_bool_header_byte() -> None:
    payload = _payload()
    payload['offset'] = {'offset_byte': True}

    with pytest.raises(ValidationError, match='integer SEG-Y trace header byte'):
        _validate(payload)


def test_residual_static_apply_request_rejects_same_source_receiver_id_byte() -> None:
    payload = _payload()
    payload['geometry']['receiver_id_byte'] = payload['geometry']['source_id_byte']

    with pytest.raises(ValidationError, match='source_id_byte and receiver_id_byte'):
        _validate(payload)


def test_residual_static_apply_request_rejects_negative_damping_lambda() -> None:
    payload = _payload()
    payload['solver']['damping_lambda'] = -1.0

    with pytest.raises(ValidationError, match='damping_lambda'):
        _validate(payload)


def test_residual_static_apply_request_rejects_non_finite_damping_lambda() -> None:
    payload = _payload()
    payload['solver']['damping_lambda'] = float('inf')

    with pytest.raises(ValidationError, match='damping_lambda'):
        _validate(payload)


@pytest.mark.parametrize(
    'field',
    ['min_valid_picks', 'min_picks_per_source', 'min_picks_per_receiver'],
)
def test_residual_static_apply_request_rejects_non_positive_solver_ints(
    field: str,
) -> None:
    payload = _payload()
    payload['solver'][field] = 0

    with pytest.raises(ValidationError, match=field):
        _validate(payload)


def test_residual_static_apply_request_rejects_bool_positive_int() -> None:
    payload = _payload()
    payload['solver']['min_valid_picks'] = True

    with pytest.raises(ValidationError, match='min_valid_picks'):
        _validate(payload)


def test_residual_static_apply_request_rejects_non_positive_max_abs_delay() -> None:
    payload = _payload()
    payload['solver']['max_abs_estimated_delay_ms'] = 0.0

    with pytest.raises(ValidationError, match='max_abs_estimated_delay_ms'):
        _validate(payload)


def test_residual_static_apply_request_rejects_invalid_robust_threshold() -> None:
    payload = _payload()
    payload['robust']['threshold'] = 0.0

    with pytest.raises(ValidationError, match='robust.threshold'):
        _validate(payload)


@pytest.mark.parametrize('min_used_fraction', [0.0, 1.1])
def test_residual_static_apply_request_rejects_invalid_min_used_fraction(
    min_used_fraction: float,
) -> None:
    payload = _payload()
    payload['robust']['min_used_fraction'] = min_used_fraction

    with pytest.raises(ValidationError, match='min_used_fraction'):
        _validate(payload)


def test_residual_static_apply_request_rejects_register_corrected_file_false() -> None:
    payload = _payload()
    payload['apply']['register_corrected_file'] = False

    with pytest.raises(ValidationError, match='register_corrected_file'):
        _validate(payload)


def test_residual_static_apply_request_rejects_non_finite_fill_value() -> None:
    payload = _payload()
    payload['apply']['fill_value'] = float('nan')

    with pytest.raises(ValidationError, match='apply.fill_value'):
        _validate(payload)


@pytest.mark.parametrize(
    'name',
    ['', '.', '..', '/tmp/datum_static_solution.npz', '../x.npz', 'nested/x.npz'],
)
def test_residual_static_apply_request_rejects_path_artifact_names(
    name: str,
) -> None:
    payload = _payload()
    payload['datum_solution']['name'] = name

    with pytest.raises(ValidationError):
        _validate(payload)


def test_residual_static_apply_request_rejects_manual_memmap_artifact_ref() -> None:
    payload = _payload()
    payload['pick_source'] = {
        'kind': 'manual_memmap',
        'job_id': 'pick-job',
        'name': 'manual_picks.npz',
    }

    with pytest.raises(ValidationError, match='manual_memmap'):
        _validate(payload)


def test_residual_static_apply_response_accepts_job_id_and_state() -> None:
    response = ResidualStaticApplyResponse.model_validate(
        {'job_id': 'residual-job', 'state': 'queued'}
    )

    assert response.job_id == 'residual-job'
    assert response.state == 'queued'
