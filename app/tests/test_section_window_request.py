from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from app.services.section_window_request import (
    DEFAULT_FIXED_OFFSET_BYTE,
    SectionWindowRequest,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.parametrize('field', ['pipeline_key', 'tap_label'])
def test_pipeline_source_disables_raw_baseline_normalization(field: str) -> None:
    req = _request(normalization_file_id='file-b', **{field: 'source'})

    assert req.uses_pipeline_source is True
    assert req.normalization_applies_to_raw is False
    assert req.raw_normalization_file_id == 'file-a'


@pytest.mark.parametrize('field', ['reference_pipeline_key', 'reference_tap_label'])
def test_reference_source_disables_raw_baseline_normalization(field: str) -> None:
    req = _request(normalization_file_id='file-b', **{field: 'reference'})

    assert req.uses_reference_source is True
    assert req.normalization_applies_to_raw is False
    assert req.raw_normalization_file_id == 'file-a'


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


@pytest.mark.parametrize(
    'overrides',
    [
        {},
        {'normalization_file_id': 'file-b'},
        {'pipeline_key': 'pipe', 'tap_label': 'fbpick'},
        {'reference_pipeline_key': 'ref-pipe', 'reference_tap_label': 'raw'},
        {
            'lmo_enabled': True,
            'lmo_velocity_mps': '1800',
            'lmo_offset_byte': '41',
            'lmo_offset_scale': '0.5',
            'lmo_offset_mode': 'signed',
            'lmo_ref_mode': 'trace',
            'lmo_ref_trace': '7',
            'lmo_polarity': '-1',
        },
    ],
)
def test_cache_key_matches_existing_section_router_helper(
    overrides: dict[str, object],
) -> None:
    from app.api.routers import section as sec

    req = _request(**overrides)

    assert req.cache_key() == sec._build_window_section_cache_key(
        file_id=req.file_id,
        normalization_file_id=req.raw_normalization_file_id,
        key1=req.key1,
        key1_byte=req.key1_byte,
        key2_byte=req.key2_byte,
        offset_byte=req.offset_byte_for_payload,
        x0=req.x0,
        x1=req.x1,
        y0=req.y0,
        y1=req.y1,
        step_x=req.step_x,
        step_y=req.step_y,
        transpose=req.transpose,
        pipeline_key=req.pipeline_key,
        tap_label=req.tap_label,
        reference_pipeline_key=req.reference_pipeline_key,
        reference_tap_label=req.reference_tap_label,
        scaling_mode=req.scaling_mode,
        lmo_enabled=req.lmo_enabled,
        lmo_velocity_mps=req.lmo_velocity_mps,
        lmo_offset_byte=req.lmo_offset_byte,
        lmo_offset_scale=req.lmo_offset_scale,
        lmo_offset_mode=req.lmo_offset_mode,
        lmo_ref_mode=req.lmo_ref_mode,
        lmo_ref_trace=req.lmo_ref_trace,
        lmo_polarity=req.lmo_polarity,
    )


def test_section_window_request_import_does_not_load_runtime_modules() -> None:
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import sys

        forbidden = {
            'app.main',
            'app.api.routers.section',
            'segyio',
        }
        for name in list(sys.modules):
            if name in forbidden or name.startswith('app.api.routers.'):
                sys.modules.pop(name, None)

        module = importlib.import_module('app.services.section_window_request')
        assert module.SectionWindowRequest.__name__ == 'SectionWindowRequest'
        for name in forbidden:
            assert name not in sys.modules, name
        """
    )

    subprocess.run(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
