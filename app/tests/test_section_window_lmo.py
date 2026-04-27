from __future__ import annotations

import gzip
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.routers import section as sec
from app.main import app
from app.services import section_service as svc
from app.services.fbpick_support import OFFSET_BYTE_FIXED
from app.tests._stubs import make_stub_reader, write_baseline_raw


def _decode_window_payload(resp) -> tuple[np.ndarray, tuple[int, int], float]:
    assert resp.headers.get('Content-Encoding') == 'gzip'
    payload = msgpack.unpackb(gzip.decompress(resp.body))
    shape = tuple(int(x) for x in payload['shape'])
    scale = float(payload['scale'])
    data = np.frombuffer(payload['data'], dtype=np.int8).reshape(shape)
    return data, (shape[0], shape[1]), scale


@pytest.fixture(autouse=True)
def _clean_section_lmo_env(monkeypatch):
    monkeypatch.setenv('FIXED_INT8_SCALE', '1')

    app.state.sv.file_registry.clear()
    state = sec.get_state(app)
    state.window_section_cache.clear()
    state.pipeline_tap_cache.clear()
    state.trace_stats_cache.clear()
    state.cached_readers.clear()

    monkeypatch.setattr(
        app.state.sv.file_registry,
        'get_dt',
        lambda _fid: 1.0,
        raising=True,
    )
    yield
    app.state.sv.file_registry.clear()
    state.window_section_cache.clear()
    state.pipeline_tap_cache.clear()
    state.trace_stats_cache.clear()
    state.cached_readers.clear()


def _install_reader(
    monkeypatch,
    tmp_path,
    section: np.ndarray,
    *,
    offsets: np.ndarray | list[float] | None = None,
):
    reader = make_stub_reader(section, offsets=offsets)
    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda _fid, _kb1, _kb2, state=None: reader,
        raising=True,
    )
    app.state.sv.file_registry.set_record(
        'f',
        {'store_path': str(tmp_path), 'dt': 1.0},
    )
    write_baseline_raw(
        tmp_path,
        key1=7,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0] * int(section.shape[0]),
        trace_stds=[1.0] * int(section.shape[0]),
    )
    return reader


def _window_response(**overrides):
    params = dict(
        file_id='f',
        key1=7,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        x0=0,
        x1=1,
        y0=0,
        y1=3,
        step_x=1,
        step_y=1,
        transpose=False,
        pipeline_key=None,
        tap_label=None,
        scaling='amax',
        request=SimpleNamespace(app=app),
    )
    params.update(overrides)
    return sec.get_section_window_bin(**params)


def test_lmo_disabled_ignores_lmo_params_for_payload_and_cache(monkeypatch, tmp_path):
    section = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    _install_reader(monkeypatch, tmp_path, section)

    first = _window_response(
        lmo_enabled=False,
        lmo_velocity_mps=1000.0,
        lmo_offset_mode='invalid',
    )
    second = _window_response(
        lmo_enabled=False,
        lmo_velocity_mps=2500.0,
        lmo_offset_mode='signed',
        lmo_ref_mode='trace',
        lmo_ref_trace=None,
    )

    assert first.body == second.body
    assert first.headers['X-SV-Cache'] == 'miss'
    assert second.headers['X-SV-Cache'] == 'hit'


def test_lmo_zero_shift_matches_existing_window(monkeypatch, tmp_path):
    section = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    _install_reader(monkeypatch, tmp_path, section, offsets=[10.0, 10.0, 10.0])

    plain = _window_response(x0=0, x1=2, y0=1, y1=4)
    lmo = _window_response(
        x0=0,
        x1=2,
        y0=1,
        y1=4,
        lmo_enabled=True,
        lmo_velocity_mps=1000.0,
    )

    assert plain.body == lmo.body


def test_lmo_default_offset_byte_uses_fixed_offset_constant(monkeypatch, tmp_path):
    section = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    requested_offset_bytes: list[int] = []
    reader = _install_reader(monkeypatch, tmp_path, section, offsets=[0.0, 0.0])
    original_get_offsets = reader.get_offsets_for_section

    def _recording_get_offsets(key1_val: int, offset_byte: int):
        requested_offset_bytes.append(int(offset_byte))
        return original_get_offsets(key1_val, offset_byte)

    reader.get_offsets_for_section = _recording_get_offsets

    _window_response(lmo_enabled=True, lmo_velocity_mps=1000.0)

    assert requested_offset_bytes == [OFFSET_BYTE_FIXED]


def test_lmo_fractional_sample_shift_uses_linear_interpolation(monkeypatch, tmp_path):
    section = np.array(
        [
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        ],
        dtype=np.float32,
    )
    _install_reader(monkeypatch, tmp_path, section, offsets=[0.0, 2.0])

    resp = _window_response(
        x0=0,
        x1=1,
        y0=1,
        y1=3,
        lmo_enabled=True,
        lmo_velocity_mps=4.0,
        lmo_offset_mode='signed',
        lmo_ref_mode='zero',
    )

    q, shape, scale = _decode_window_payload(resp)
    assert scale == pytest.approx(1.0)
    assert shape == (2, 3)
    assert np.array_equal(
        q,
        np.array(
            [
                [10, 20, 30],
                [15, 25, 35],
            ],
            dtype=np.int8,
        ),
    )


def test_lmo_out_of_range_samples_are_filled_with_zero(monkeypatch, tmp_path):
    section = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    _install_reader(monkeypatch, tmp_path, section, offsets=[3.0])

    resp = _window_response(
        x0=0,
        x1=0,
        y0=3,
        y1=5,
        lmo_enabled=True,
        lmo_velocity_mps=1.0,
        lmo_offset_mode='signed',
        lmo_ref_mode='zero',
    )

    q, shape, _ = _decode_window_payload(resp)
    assert shape == (1, 3)
    assert np.array_equal(q, np.zeros((1, 3), dtype=np.int8))


def test_lmo_downsampled_transposed_window_uses_display_grid(monkeypatch, tmp_path):
    section = np.array(
        [
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        ],
        dtype=np.float32,
    )
    _install_reader(monkeypatch, tmp_path, section, offsets=[0.0, 2.0, 4.0])

    resp = _window_response(
        x0=0,
        x1=2,
        y0=0,
        y1=4,
        step_x=2,
        step_y=2,
        transpose=True,
        lmo_enabled=True,
        lmo_velocity_mps=2.0,
        lmo_offset_mode='signed',
        lmo_ref_mode='zero',
    )

    q, shape, _ = _decode_window_payload(resp)
    assert shape == (3, 2)
    assert np.array_equal(
        q,
        np.array(
            [
                [0, 20],
                [20, 40],
                [40, 0],
            ],
            dtype=np.int8,
        ),
    )


def test_lmo_pipeline_tap_uses_raw_reader_offsets(monkeypatch, tmp_path):
    state = sec.get_state(app)
    tap = np.array(
        [
            [0.0, 10.0, 20.0, 30.0],
            [0.0, 10.0, 20.0, 30.0],
        ],
        dtype=np.float32,
    )
    raw = tap + 100.0
    _install_reader(monkeypatch, tmp_path, raw, offsets=[0.0, 1.0])
    pipeline_key = 'pk-lmo'
    base_key = ('f', 7, 189, 193, pipeline_key, None, None)
    state.pipeline_tap_cache.set((*base_key, 'tapA'), {'data': tap})

    resp = _window_response(
        x0=0,
        x1=1,
        y0=0,
        y1=2,
        pipeline_key=pipeline_key,
        tap_label='tapA',
        lmo_enabled=True,
        lmo_velocity_mps=1.0,
        lmo_offset_mode='signed',
        lmo_ref_mode='zero',
    )

    q, shape, _ = _decode_window_payload(resp)
    assert shape == (2, 3)
    assert np.array_equal(
        q,
        np.array(
            [
                [0, 10, 20],
                [10, 20, 30],
            ],
            dtype=np.int8,
        ),
    )


def test_lmo_enabled_cache_key_includes_lmo_params(monkeypatch, tmp_path):
    section = np.array(
        [
            [0.0, 10.0, 20.0, 30.0],
            [0.0, 10.0, 20.0, 30.0],
        ],
        dtype=np.float32,
    )
    _install_reader(monkeypatch, tmp_path, section, offsets=[0.0, 2.0])

    slow = _window_response(
        y0=0,
        y1=2,
        lmo_enabled=True,
        lmo_velocity_mps=2.0,
        lmo_offset_mode='signed',
        lmo_ref_mode='zero',
    )
    fast = _window_response(
        y0=0,
        y1=2,
        lmo_enabled=True,
        lmo_velocity_mps=1.0,
        lmo_offset_mode='signed',
        lmo_ref_mode='zero',
    )

    assert slow.headers['X-SV-Cache'] == 'miss'
    assert fast.headers['X-SV-Cache'] == 'miss'
    assert slow.body != fast.body


@pytest.mark.parametrize(
    ('overrides', 'detail'),
    [
        ({'lmo_velocity_mps': None}, 'lmo_velocity_mps is required'),
        ({'lmo_velocity_mps': 0.0}, 'velocity_mps must be finite'),
        ({'lmo_offset_scale': 0.0}, 'offset_scale must be finite'),
        ({'lmo_offset_mode': 'bad'}, 'offset_mode must be'),
        ({'lmo_ref_mode': 'bad'}, 'ref_mode must be'),
        ({'lmo_ref_mode': 'trace'}, 'ref_trace is required'),
        ({'lmo_ref_mode': 'trace', 'lmo_ref_trace': 9}, 'ref_trace is out of range'),
        ({'lmo_polarity': 0}, 'polarity must be 1 or -1'),
    ],
)
def test_lmo_invalid_parameters_return_400(
    monkeypatch,
    tmp_path,
    overrides: dict[str, object],
    detail: str,
):
    section = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    _install_reader(monkeypatch, tmp_path, section, offsets=[0.0, 1.0])

    params = {'lmo_enabled': True, 'lmo_velocity_mps': 1.0}
    params.update(overrides)
    with pytest.raises(HTTPException) as exc_info:
        _window_response(**params)

    assert exc_info.value.status_code == 400
    assert detail in str(exc_info.value.detail)


@pytest.mark.parametrize('lmo_offset_byte', [0, 241, -1])
def test_lmo_invalid_offset_byte_query_is_422(lmo_offset_byte: int):
    with TestClient(app) as client:
        resp = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'f',
                'key1': 7,
                'x0': 0,
                'x1': 1,
                'y0': 0,
                'y1': 3,
                'lmo_enabled': True,
                'lmo_velocity_mps': 1.0,
                'lmo_offset_byte': lmo_offset_byte,
            },
        )

    assert resp.status_code == 422


@pytest.mark.parametrize('lmo_offset_byte', [0, 241, -1])
def test_lmo_invalid_offset_byte_service_raises_value_error(
    monkeypatch,
    tmp_path,
    lmo_offset_byte: int,
):
    section = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    reader = _install_reader(monkeypatch, tmp_path, section, offsets=[0.0, 1.0])

    with pytest.raises(
        ValueError,
        match='lmo_offset_byte must be between 1 and 240',
    ):
        svc.build_section_window_payload(
            file_id='f',
            key1=7,
            key1_byte=189,
            key2_byte=193,
            offset_byte=None,
            x0=0,
            x1=1,
            y0=0,
            y1=3,
            step_x=1,
            step_y=1,
            transpose=False,
            pipeline_key=None,
            tap_label=None,
            scaling_mode='amax',
            lmo_enabled=True,
            lmo_velocity_mps=1.0,
            lmo_offset_byte=lmo_offset_byte,
            trace_stats_cache={},
            reader_getter=lambda _fid, _kb1, _kb2: reader,
            pipeline_section_getter=lambda **_kwargs: None,
            store_dir_resolver=lambda _fid: str(tmp_path),
            dt_resolver=lambda _fid: 1.0,
        )


@pytest.mark.parametrize(
    ('offsets', 'detail'),
    [
        ([], 'offsets must not be empty'),
        ([0.0], 'Offsets length does not match section trace count'),
        ([0.0, np.nan], 'offsets contain NaN or Inf'),
    ],
)
def test_lmo_invalid_offsets_return_400(monkeypatch, tmp_path, offsets, detail: str):
    section = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    _install_reader(monkeypatch, tmp_path, section, offsets=offsets)

    with pytest.raises(HTTPException) as exc_info:
        _window_response(lmo_enabled=True, lmo_velocity_mps=1.0)

    assert exc_info.value.status_code == 400
    assert detail in str(exc_info.value.detail)
