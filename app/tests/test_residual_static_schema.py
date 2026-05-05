from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.api.schemas import ResidualStaticApplyRequest


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
        'source_id_byte': 17,
        'receiver_id_byte': 13,
        'offset_byte': 37,
        'moveout': {'model': 'linear_abs_offset'},
    }


def test_residual_static_apply_request_accepts_linear_abs_offset() -> None:
    req = ResidualStaticApplyRequest.model_validate(_payload())

    assert req.file_id == 'datum-corrected-file-id'
    assert req.datum_solution.name == 'datum_static_solution.npz'
    assert req.pick_source.kind == 'batch_job_artifact'
    assert req.source_id_byte == 17
    assert req.receiver_id_byte == 13
    assert req.offset_byte == 37
    assert req.moveout.model == 'linear_abs_offset'


def test_residual_static_apply_request_accepts_moveout_none_without_offset() -> None:
    payload = _payload()
    payload['moveout'] = {'model': 'none'}
    payload['offset_byte'] = None

    req = ResidualStaticApplyRequest.model_validate(payload)

    assert req.moveout.model == 'none'
    assert req.offset_byte is None


def test_residual_static_apply_request_rejects_same_source_receiver_id_byte() -> None:
    payload = _payload()
    payload['receiver_id_byte'] = payload['source_id_byte']

    with pytest.raises(ValidationError, match='source_id_byte and receiver_id_byte'):
        ResidualStaticApplyRequest.model_validate(payload)


def test_residual_static_apply_request_rejects_linear_abs_offset_without_offset() -> None:
    payload = _payload()
    payload['offset_byte'] = None

    with pytest.raises(ValidationError, match='offset_byte is required'):
        ResidualStaticApplyRequest.model_validate(payload)


def test_residual_static_apply_request_rejects_path_artifact_names() -> None:
    payload = _payload()
    payload['datum_solution']['name'] = '../datum_static_solution.npz'

    with pytest.raises(ValidationError, match='plain file name'):
        ResidualStaticApplyRequest.model_validate(payload)
