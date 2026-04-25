# app/tests/test_section_window_bin_options.py

from __future__ import annotations

import gzip
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from app.api.routers import section as sec
from app.main import app
from app.tests._stubs import make_stub_reader, write_baseline_raw


def _decode_window_payload(resp) -> tuple[np.ndarray, tuple[int, int], float]:
    assert resp.headers.get('Content-Encoding') == 'gzip'
    payload = msgpack.unpackb(gzip.decompress(resp.body))
    shape = tuple(int(x) for x in payload['shape'])
    scale = float(payload['scale'])
    data = np.frombuffer(payload['data'], dtype=np.int8).reshape(shape)
    return data, (shape[0], shape[1]), scale


@pytest.fixture(autouse=True)
def _clean_section_env(monkeypatch):
    # Make quantization deterministic: q = round(x) with scale=1.
    monkeypatch.setenv('FIXED_INT8_SCALE', '1')

    # Ensure a clean registry and caches.
    app.state.sv.file_registry.clear()
    state = sec.get_state(app)
    state.window_section_cache.clear()
    state.pipeline_tap_cache.clear()
    state.trace_stats_cache.clear()
    state.cached_readers.clear()

    # Avoid relying on real dt resolution.
    monkeypatch.setattr(
        app.state.sv.file_registry,
        'get_dt',
        lambda _fid: 0.004,
        raising=True,
    )
    yield
    app.state.sv.file_registry.clear()


def test_get_section_window_bin_step_xy_downsample_shape_and_values(
    monkeypatch, tmp_path
):
    stub_reader = make_stub_reader(np.arange(5 * 6, dtype=np.float32).reshape(5, 6))

    monkeypatch.setattr(
        sec, 'get_reader', lambda fid, kb1, kb2, state=None: stub_reader, raising=True
    )

    app.state.sv.file_registry.set_record(
        'f', {'store_path': str(tmp_path), 'dt': 0.004}
    )
    write_baseline_raw(
        tmp_path,
        key1=7,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0] * 5,
        trace_stds=[1.0] * 5,
    )

    res = sec.get_section_window_bin(
        file_id='f',
        key1=7,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        x0=0,
        x1=4,
        y0=1,
        y1=5,
        step_x=2,
        step_y=2,
        transpose=False,
        pipeline_key=None,
        tap_label=None,
        scaling='amax',
        request=SimpleNamespace(app=app),
    )
    q, (h, w), scale = _decode_window_payload(res)
    assert scale == pytest.approx(1.0)
    assert (h, w) == (3, 3)

    base = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    expected = base[0 : 4 + 1 : 2, 1 : 5 + 1 : 2]
    expected_q = expected.astype(np.int8)
    assert np.array_equal(q, expected_q)


def test_get_section_window_bin_transpose_true_swaps_axes_and_transposes_values(
    monkeypatch, tmp_path
):
    stub_reader = make_stub_reader(np.arange(5 * 6, dtype=np.float32).reshape(5, 6))

    monkeypatch.setattr(
        sec, 'get_reader', lambda fid, kb1, kb2, state=None: stub_reader, raising=True
    )

    app.state.sv.file_registry.set_record(
        'f', {'store_path': str(tmp_path), 'dt': 0.004}
    )
    write_baseline_raw(
        tmp_path,
        key1=7,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0] * 5,
        trace_stds=[1.0] * 5,
    )

    # Pick a non-square window so axis swap is visible: (4 traces x 3 samples) -> (3 x 4)
    res = sec.get_section_window_bin(
        file_id='f',
        key1=7,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        x0=0,
        x1=3,
        y0=1,
        y1=3,
        step_x=1,
        step_y=1,
        transpose=True,
        pipeline_key=None,
        tap_label=None,
        scaling='amax',
        request=SimpleNamespace(app=app),
    )
    q, (h, w), scale = _decode_window_payload(res)
    assert scale == pytest.approx(1.0)
    assert (h, w) == (3, 4)

    base = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    expected = base[0 : 3 + 1 : 1, 1 : 3 + 1 : 1]
    expected_q = expected.T.astype(np.int8)
    assert np.array_equal(q, expected_q)


def test_get_section_window_bin_scaling_amax_vs_tracewise(monkeypatch, tmp_path):
    stub_reader = make_stub_reader(
        np.array(
            [
                [0, 1, 2, 3],
                [10, 11, 12, 13],
                [20, 21, 22, 23],
            ],
            dtype=np.float32,
        )
    )

    monkeypatch.setattr(
        sec, 'get_reader', lambda fid, kb1, kb2, state=None: stub_reader, raising=True
    )

    app.state.sv.file_registry.set_record(
        'f', {'store_path': str(tmp_path), 'dt': 0.004}
    )
    write_baseline_raw(
        tmp_path,
        key1=7,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0, 10.0, 20.0],
        trace_stds=[1.0, 1.0, 1.0],
    )

    params = dict(
        file_id='f',
        key1=7,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        x0=0,
        x1=2,
        y0=0,
        y1=3,
        step_x=1,
        step_y=1,
        transpose=False,
        pipeline_key=None,
        tap_label=None,
        request=SimpleNamespace(app=app),
    )

    res_amax = sec.get_section_window_bin(**params, scaling='amax')
    q_amax, (_, _), _ = _decode_window_payload(res_amax)

    res_tw = sec.get_section_window_bin(**params, scaling='tracewise')
    q_tw, (_, _), _ = _decode_window_payload(res_tw)

    base = np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]], dtype=np.int8)
    expected_tw = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int8)

    assert np.array_equal(q_amax, base)
    assert np.array_equal(q_tw, expected_tw)
    assert not np.array_equal(q_amax, q_tw)


def test_get_section_window_bin_tracewise_clamp_saturates_int8(monkeypatch, tmp_path):
    stub_reader = make_stub_reader(
        np.array(
            [
                [0, 1, 2, 3],
                [10, 11, 12, 13],
                [20, 21, 22, 23],
            ],
            dtype=np.float32,
        )
    )

    monkeypatch.setattr(
        sec, 'get_reader', lambda fid, kb1, kb2, state=None: stub_reader, raising=True
    )

    app.state.sv.file_registry.set_record(
        'f', {'store_path': str(tmp_path), 'dt': 0.004}
    )
    # trace_stds[1]=0 triggers clamp to eps -> very large inv_std -> int8 saturation
    write_baseline_raw(
        tmp_path,
        key1=7,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0, 10.0, 20.0],
        trace_stds=[1.0, 0.0, 1.0],
    )

    res = sec.get_section_window_bin(
        file_id='f',
        key1=7,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        x0=0,
        x1=2,
        y0=0,
        y1=3,
        step_x=1,
        step_y=1,
        transpose=False,
        pipeline_key=None,
        tap_label=None,
        scaling='tracewise',
        request=SimpleNamespace(app=app),
    )
    q, (h, w), scale = _decode_window_payload(res)
    assert scale == pytest.approx(1.0)
    assert (h, w) == (3, 4)

    expected = np.array(
        [
            [0, 1, 2, 3],
            [0, 127, 127, 127],
            [0, 1, 2, 3],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(q, expected)


def test_get_section_window_bin_pipeline_key_tap_label_window_uses_expected_tap(
    monkeypatch, tmp_path
):
    state = sec.get_state(app)
    state.window_section_cache.clear()
    state.pipeline_tap_cache.clear()
    state.trace_stats_cache.clear()

    app.state.sv.file_registry.set_record(
        'f', {'store_path': str(tmp_path), 'dt': 0.004}
    )
    write_baseline_raw(
        tmp_path,
        key1=7,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0] * 4,
        trace_stds=[1.0] * 4,
    )

    # Put two distinct tap outputs in the in-memory cache.
    arr_a = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    arr_b = np.arange(4 * 5, dtype=np.float32).reshape(4, 5) + 100.0

    pipeline_key = 'pk1'
    base_key = ('f', 7, 189, 193, pipeline_key, None, None)
    state.pipeline_tap_cache.set((*base_key, 'tapA'), {'data': arr_a})
    state.pipeline_tap_cache.set((*base_key, 'tapB'), {'data': arr_b})

    # Ensure raw reader path is not used when pipeline_key+tap_label are provided.
    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('get_reader called')),
        raising=True,
    )

    res = sec.get_section_window_bin(
        file_id='f',
        key1=7,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        x0=0,
        x1=3,
        y0=1,
        y1=4,
        step_x=2,
        step_y=2,
        transpose=False,
        pipeline_key=pipeline_key,
        tap_label='tapB',
        scaling='amax',
        request=SimpleNamespace(app=app),
    )
    q, (h, w), scale = _decode_window_payload(res)
    assert scale == pytest.approx(1.0)
    assert (h, w) == (2, 2)

    expected = arr_b[0 : 3 + 1 : 2, 1 : 4 + 1 : 2]
    assert np.array_equal(q, expected.astype(np.int8))


def test_get_section_window_bin_amax_can_use_pipeline_tap_reference(
    monkeypatch, tmp_path
):
    state = sec.get_state(app)

    app.state.sv.file_registry.set_record(
        'f', {'store_path': str(tmp_path), 'dt': 0.004}
    )
    write_baseline_raw(
        tmp_path,
        key1=7,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0] * 2,
        trace_stds=[1.0] * 2,
    )

    arr_a = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    arr_b = np.array([[3.0, 5.0], [7.0, 9.0]], dtype=np.float32)
    pipeline_key = 'pk1'
    base_key = ('f', 7, 189, 193, pipeline_key, None, None)
    state.pipeline_tap_cache.set((*base_key, 'tapA'), {'data': arr_a})
    state.pipeline_tap_cache.set((*base_key, 'tapB'), {'data': arr_b})

    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('get_reader called')),
        raising=True,
    )

    res = sec.get_section_window_bin(
        file_id='f',
        key1=7,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        step_x=1,
        step_y=1,
        transpose=False,
        pipeline_key=pipeline_key,
        tap_label='tapB',
        reference_pipeline_key=pipeline_key,
        reference_tap_label='tapA',
        scaling='amax',
        request=SimpleNamespace(app=app),
    )
    q, (h, w), scale = _decode_window_payload(res)
    assert scale == pytest.approx(1.0)
    assert (h, w) == (2, 2)
    assert np.array_equal(q, np.array([[2, 4], [6, 8]], dtype=np.int8))


def test_get_section_window_bin_tracewise_can_use_pipeline_tap_reference(
    monkeypatch, tmp_path
):
    state = sec.get_state(app)

    app.state.sv.file_registry.set_record(
        'f', {'store_path': str(tmp_path), 'dt': 0.004}
    )
    write_baseline_raw(
        tmp_path,
        key1=7,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0] * 2,
        trace_stds=[1.0] * 2,
    )

    arr_a = np.array([[0.0, 2.0], [10.0, 12.0]], dtype=np.float32)
    arr_b = np.array([[2.0, 4.0], [14.0, 16.0]], dtype=np.float32)
    pipeline_key = 'pk1'
    base_key = ('f', 7, 189, 193, pipeline_key, None, None)
    state.pipeline_tap_cache.set((*base_key, 'tapA'), {'data': arr_a})
    state.pipeline_tap_cache.set((*base_key, 'tapB'), {'data': arr_b})

    monkeypatch.setattr(
        sec,
        'get_reader',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('get_reader called')),
        raising=True,
    )

    res = sec.get_section_window_bin(
        file_id='f',
        key1=7,
        key1_byte=189,
        key2_byte=193,
        offset_byte=None,
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        step_x=1,
        step_y=1,
        transpose=False,
        pipeline_key=pipeline_key,
        tap_label='tapB',
        reference_pipeline_key=pipeline_key,
        reference_tap_label='tapA',
        scaling='tracewise',
        request=SimpleNamespace(app=app),
    )
    q, (h, w), scale = _decode_window_payload(res)
    assert scale == pytest.approx(1.0)
    assert (h, w) == (2, 2)
    assert np.array_equal(q, np.array([[1, 3], [3, 5]], dtype=np.int8))
