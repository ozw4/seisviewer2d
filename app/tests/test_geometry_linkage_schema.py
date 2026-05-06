from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from app.api.schemas import StaticLinkageBuildRequest


def _payload() -> dict[str, Any]:
    return {
        'file_id': 'file-1',
        'key1_byte': 189,
        'key2_byte': 193,
        'linkage': {
            'mode': 'auto_threshold',
            'threshold_m': 25.0,
        },
    }


def _validate(payload: dict[str, Any]) -> StaticLinkageBuildRequest:
    return StaticLinkageBuildRequest.model_validate(deepcopy(payload))


def test_static_linkage_request_accepts_auto_threshold_defaults() -> None:
    req = _validate(_payload())

    assert req.file_id == 'file-1'
    assert req.key1_byte == 189
    assert req.key2_byte == 193
    assert req.geometry.source_x_byte == 73
    assert req.geometry.source_y_byte == 77
    assert req.geometry.receiver_x_byte == 81
    assert req.geometry.receiver_y_byte == 85
    assert req.geometry.coordinate_scalar_byte == 71
    assert req.linkage.mode == 'auto_threshold'
    assert req.linkage.threshold_m == 25.0
    assert req.linkage.receiver_location_interval_m is None
    assert req.linkage.prefer_receiver_anchor is True


def test_static_linkage_request_accepts_none_mode() -> None:
    payload = _payload()
    payload['linkage'] = {
        'mode': 'none',
        'threshold_m': None,
        'prefer_receiver_anchor': False,
    }

    req = _validate(payload)

    assert req.linkage.mode == 'none'
    assert req.linkage.threshold_m is None
    assert req.linkage.prefer_receiver_anchor is False


def test_static_linkage_request_rejects_extra_fields() -> None:
    payload = _payload()
    payload['unknown'] = True

    with pytest.raises(ValidationError):
        _validate(payload)


def test_static_linkage_request_rejects_nested_extra_fields() -> None:
    payload = _payload()
    payload['geometry'] = {'source_x_byte': 73, 'unknown': True}

    with pytest.raises(ValidationError):
        _validate(payload)


def test_static_linkage_request_rejects_bool_header_byte() -> None:
    payload = _payload()
    payload['geometry'] = {'source_x_byte': True}

    with pytest.raises(ValidationError, match='integer SEG-Y trace header byte'):
        _validate(payload)


@pytest.mark.parametrize('bad_byte', [0, 241])
def test_static_linkage_request_rejects_out_of_range_header_byte(
    bad_byte: int,
) -> None:
    payload = _payload()
    payload['key1_byte'] = bad_byte

    with pytest.raises(ValidationError, match='range 1..240'):
        _validate(payload)


def test_static_linkage_request_rejects_duplicate_geometry_header_bytes() -> None:
    payload = _payload()
    payload['geometry'] = {
        'source_x_byte': 73,
        'source_y_byte': 73,
        'receiver_x_byte': 81,
        'receiver_y_byte': 85,
        'coordinate_scalar_byte': 71,
    }

    with pytest.raises(ValidationError, match='geometry header bytes must be unique'):
        _validate(payload)


def test_static_linkage_request_rejects_auto_threshold_without_threshold() -> None:
    payload = _payload()
    payload['linkage'] = {
        'mode': 'auto_threshold',
        'threshold_m': None,
    }

    with pytest.raises(ValidationError, match='threshold_m is required'):
        _validate(payload)


def test_static_linkage_request_rejects_none_mode_with_threshold() -> None:
    payload = _payload()
    payload['linkage'] = {
        'mode': 'none',
        'threshold_m': 10.0,
    }

    with pytest.raises(ValidationError, match='threshold_m must be null'):
        _validate(payload)


@pytest.mark.parametrize('threshold_m', [0.0, -1.0])
def test_static_linkage_request_rejects_non_positive_threshold(
    threshold_m: float,
) -> None:
    payload = _payload()
    payload['linkage']['threshold_m'] = threshold_m

    with pytest.raises(ValidationError, match='threshold_m'):
        _validate(payload)


@pytest.mark.parametrize('threshold_m', [float('nan'), float('inf'), float('-inf')])
def test_static_linkage_request_rejects_non_finite_threshold(
    threshold_m: float,
) -> None:
    payload = _payload()
    payload['linkage']['threshold_m'] = threshold_m

    with pytest.raises(ValidationError, match='threshold_m'):
        _validate(payload)


@pytest.mark.parametrize(
    'receiver_location_interval_m',
    [0.0, -1.0, float('nan'), float('inf'), float('-inf')],
)
def test_static_linkage_request_rejects_invalid_receiver_location_interval(
    receiver_location_interval_m: float,
) -> None:
    payload = _payload()
    payload['linkage']['receiver_location_interval_m'] = (
        receiver_location_interval_m
    )

    with pytest.raises(ValidationError, match='receiver_location_interval_m'):
        _validate(payload)


def test_static_linkage_request_rejects_none_mode_with_receiver_interval() -> None:
    payload = _payload()
    payload['linkage'] = {
        'mode': 'none',
        'threshold_m': None,
        'receiver_location_interval_m': 10.0,
    }

    with pytest.raises(ValidationError, match='receiver_location_interval_m'):
        _validate(payload)


def test_static_linkage_request_rejects_non_bool_prefer_receiver_anchor() -> None:
    payload = _payload()
    payload['linkage']['prefer_receiver_anchor'] = 1

    with pytest.raises(ValidationError, match='prefer_receiver_anchor'):
        _validate(payload)


def test_static_linkage_request_rejects_empty_file_id() -> None:
    payload = _payload()
    payload['file_id'] = ''

    with pytest.raises(ValidationError, match='file_id'):
        _validate(payload)
