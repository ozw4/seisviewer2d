from __future__ import annotations

import inspect

import pytest

from app.services.section_window_request import (
    DEFAULT_FIXED_OFFSET_BYTE,
    SectionWindowRequest,
)

def _request(**overrides: object) -> SectionWindowRequest:
    values: dict[str, object] = {
        'file_id': 'file-a',
        'key1': 10,
        'key1_byte': 189,
        'key2_byte': 193,
        'normalization_file_id': None,
        'offset_byte': None,
        'x0': 1,
        'x1': 20,
        'y0': 2,
        'y1': 80,
        'step_x': 2,
        'step_y': 3,
        'transpose': True,
        'pipeline_key': None,
        'tap_label': None,
        'reference_pipeline_key': None,
        'reference_tap_label': None,
        'scaling': None,
        'lmo_enabled': False,
        'lmo_velocity_mps': None,
        'lmo_offset_byte': 37,
        'lmo_offset_scale': 1.0,
        'lmo_offset_mode': 'absolute',
        'lmo_ref_mode': 'min',
        'lmo_ref_trace': None,
        'lmo_polarity': 1,
        'default_fbpick_model_id': 'fbpick_edgenext_small.pt',
    }
    values.update(overrides)
    return SectionWindowRequest(**values)  # type: ignore[arg-type]


def test_scaling_none_defaults_to_amax() -> None:
    assert _request(scaling=None).scaling_mode == 'amax'


def test_scaling_input_is_normalized_to_lowercase() -> None:
    assert _request(scaling='TRACEWISE').scaling_mode == 'tracewise'


def test_scaling_bad_value_raises_value_error() -> None:
    with pytest.raises(ValueError, match='Unsupported scaling mode'):
        _request(scaling='bad')


def test_normalization_file_defaults_to_file_id() -> None:
    req = _request(normalization_file_id=None)

    assert req.resolved_normalization_file_id == 'file-a'


def test_raw_source_applies_requested_normalization_file() -> None:
    req = _request(normalization_file_id='file-b')

    assert req.normalization_applies_to_raw is True
    assert req.raw_normalization_file_id == 'file-b'


def test_raw_source_baseline_request_uses_reference_file_id() -> None:
    baseline_request = _request(normalization_file_id='file-b').raw_baseline_request()

    assert baseline_request is not None
    assert baseline_request.file_id == 'file-b'
    assert baseline_request.key1_byte == 189
    assert baseline_request.key2_byte == 193


@pytest.mark.parametrize('field', ['pipeline_key', 'tap_label'])
def test_pipeline_source_disables_raw_baseline_normalization(field: str) -> None:
    req = _request(normalization_file_id='file-b', **{field: 'source'})

    assert req.uses_pipeline_source is True
    assert req.normalization_applies_to_raw is False
    assert req.raw_normalization_file_id == 'file-a'
    assert req.raw_baseline_request() is None


@pytest.mark.parametrize('field', ['reference_pipeline_key', 'reference_tap_label'])
def test_reference_source_disables_raw_baseline_normalization(field: str) -> None:
    req = _request(normalization_file_id='file-b', **{field: 'reference'})

    assert req.uses_reference_source is True
    assert req.normalization_applies_to_raw is False
    assert req.raw_normalization_file_id == 'file-a'
    assert req.raw_baseline_request() is None


def test_lmo_disabled_cache_key_ignores_lmo_parameter_differences() -> None:
    req_a = _request(lmo_enabled=False, lmo_velocity_mps=1500.0, lmo_offset_byte=37)
    req_b = _request(lmo_enabled=False, lmo_velocity_mps=2500.0, lmo_offset_byte=41)

    assert req_a.cache_key() == req_b.cache_key()


def test_lmo_enabled_cache_key_includes_lmo_parameter_differences() -> None:
    req_a = _request(lmo_enabled=True, lmo_velocity_mps=1500.0, lmo_offset_byte=37)
    req_b = _request(lmo_enabled=True, lmo_velocity_mps=2500.0, lmo_offset_byte=41)

    assert req_a.cache_key() != req_b.cache_key()


def test_offset_model_forces_fixed_offset_byte_for_payload() -> None:
    req = _request(
        offset_byte=41,
        default_fbpick_model_id='fbpick_offset_edgenext_small.pt',
    )

    assert req.offset_byte_for_payload == DEFAULT_FIXED_OFFSET_BYTE


def test_payload_kwargs_match_build_section_window_payload_request_args() -> None:
    from app.services.section_service import build_section_window_payload

    service_injected_args = {
        'trace_stats_cache',
        'reader_getter',
        'pipeline_section_getter',
        'store_dir_resolver',
        'trace_stats_lock',
        'dt_resolver',
        'perf_timings_ms',
    }
    expected = set(inspect.signature(build_section_window_payload).parameters)
    expected -= service_injected_args

    assert set(_request().payload_kwargs()) == expected


def test_payload_kwargs_offset_byte_reflects_forced_offset_byte() -> None:
    kwargs = _request(
        offset_byte=41,
        default_fbpick_model_id='fbpick_offset_edgenext_small.pt',
    ).payload_kwargs()

    assert kwargs['offset_byte'] == DEFAULT_FIXED_OFFSET_BYTE


def test_payload_kwargs_scaling_mode_is_normalized_to_lowercase() -> None:
    kwargs = _request(scaling='TRACEWISE').payload_kwargs()

    assert kwargs['scaling_mode'] == 'tracewise'


def test_cache_key_field_order_is_explicit_for_lmo_disabled() -> None:
    request = _request(
        file_id='B',
        normalization_file_id='A',
        key1=7,
        x0=0,
        x1=1,
        y0=0,
        y1=99,
        step_x=1,
        step_y=2,
        transpose=False,
        scaling='amax',
    )

    assert request.cache_key() == (
        'B',
        'A',
        7,
        189,
        193,
        None,
        0,
        1,
        0,
        99,
        1,
        2,
        False,
        None,
        None,
        None,
        None,
        'amax',
    )


def test_cache_key_field_order_is_explicit_for_lmo_enabled() -> None:
    request = _request(
        file_id='B',
        normalization_file_id='A',
        key1=7,
        x0=0,
        x1=1,
        y0=0,
        y1=99,
        step_x=1,
        step_y=2,
        transpose=False,
        scaling='amax',
        lmo_enabled=True,
        lmo_velocity_mps='1800',
        lmo_offset_byte='41',
        lmo_offset_scale='0.5',
        lmo_offset_mode='signed',
        lmo_ref_mode='trace',
        lmo_ref_trace='7',
        lmo_polarity='-1',
    )

    assert request.cache_key() == (
        'B',
        'A',
        7,
        189,
        193,
        None,
        0,
        1,
        0,
        99,
        1,
        2,
        False,
        None,
        None,
        None,
        None,
        'amax',
        'lmo',
        True,
        1800.0,
        41,
        0.5,
        'signed',
        'trace',
        7,
        -1,
    )


def test_cache_key_offset_model_reflects_forced_offset_byte() -> None:
    request = _request(
        offset_byte=41,
        default_fbpick_model_id='fbpick_offset_edgenext_small.pt',
    )

    assert request.cache_key()[5] == DEFAULT_FIXED_OFFSET_BYTE
