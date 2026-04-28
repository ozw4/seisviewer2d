from __future__ import annotations

import gzip
import statistics
import time
from pathlib import Path

import msgpack
import numpy as np
import pytest

from app.api._helpers import get_state
from app.api.binary_codec import pack_msgpack_gzip
from app.codec.quantize import quantize_float32
from app.main import app
from app.services import section_service as svc
from app.tests._stubs import make_stub_reader, write_baseline_raw
from app.trace_store.types import SectionView
from app.utils.segy_meta import _BASELINE_CACHE

DT = 0.004
KEY1 = 189
KEY2 = 193
KEY1_VALUE = 7
PERF_WARMUP_RUNS = 2
PERF_MEASURED_RUNS = 7


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setenv('FIXED_INT8_SCALE', '1')
    state = get_state(app)
    app.state.sv.file_registry.clear()
    state.cached_readers.clear()
    state.pipeline_tap_cache.clear()
    state.window_section_cache.clear()
    state.section_offsets_cache.clear()
    state.trace_stats_cache.clear()
    _BASELINE_CACHE.clear()
    yield
    app.state.sv.file_registry.clear()
    state.cached_readers.clear()
    state.pipeline_tap_cache.clear()
    state.window_section_cache.clear()
    state.section_offsets_cache.clear()
    state.trace_stats_cache.clear()
    _BASELINE_CACHE.clear()


def _decode_payload(payload_bytes: bytes) -> dict[str, object]:
    payload = msgpack.unpackb(gzip.decompress(payload_bytes), raw=False)
    shape = tuple(int(x) for x in payload['shape'])
    return {
        'shape': shape,
        'scale': float(payload['scale']),
        'dt': float(payload['dt']),
        'data': np.frombuffer(payload['data'], dtype=np.int8).reshape(shape),
    }


def _legacy_pack_quantized_array_gzip(
    arr_f32: np.ndarray,
    *,
    scale: float | None,
    dt: float | None,
    extra: dict[str, object] | None = None,
) -> bytes:
    scale_val, q = quantize_float32(arr_f32, fixed_scale=scale)
    obj: dict[str, object] = {
        'scale': scale_val,
        'shape': arr_f32.shape,
        'data': q.tobytes(),
    }
    if dt is not None:
        obj['dt'] = float(dt)
    if extra:
        obj.update(extra)
    return pack_msgpack_gzip(obj)


def _legacy_build_section_window_payload(
    *,
    file_id: str,
    key1: int,
    key1_byte: int,
    key2_byte: int,
    offset_byte: int | None,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    step_x: int,
    step_y: int,
    transpose: bool,
    pipeline_key: str | None,
    tap_label: str | None,
    scaling_mode: str,
    trace_stats_cache: object,
    reader_getter,
    pipeline_section_getter,
    store_dir_resolver,
    trace_stats_lock=None,
    dt_resolver=None,
    perf_timings_ms: dict[str, float] | None = None,
) -> bytes:
    build_started = time.perf_counter()
    mode = scaling_mode.lower()
    section_view, reader = svc._load_section_view(
        file_id=file_id,
        key1=key1,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        offset_byte=offset_byte,
        pipeline_key=pipeline_key,
        tap_label=tap_label,
        reader_getter=reader_getter,
        pipeline_section_getter=pipeline_section_getter,
    )
    base = section_view.arr
    if base.ndim != svc.EXPECTED_SECTION_NDIM:
        raise svc.SectionServiceInternalError('Section data must be 2D')

    sub = base[x0 : x1 + 1 : step_x, y0 : y1 + 1 : step_y]
    prepared = svc.coerce_section_f32(sub, section_view.scale)
    if dt_resolver is None:
        raise svc.SectionServiceInternalError('dt resolver is required')
    store_dir = svc._resolve_store_dir(
        file_id=file_id,
        reader=reader,
        store_dir_resolver=store_dir_resolver,
    )
    prepared = svc.apply_scaling_from_baseline(
        prepared,
        scaling=mode,
        file_id=file_id,
        key1=key1,
        store_dir=store_dir,
        key1_byte=key1_byte,
        key2_byte=key2_byte,
        trace_stats_cache=trace_stats_cache,
        trace_stats_lock=trace_stats_lock,
        x0=x0,
        x1=x1,
        step_x=step_x,
    )
    view = prepared.T if transpose else prepared
    window_view = np.ascontiguousarray(view, dtype=np.float32)
    dt_val = dt_resolver(file_id)
    build_ms = (time.perf_counter() - build_started) * 1000.0
    pack_started = time.perf_counter()
    payload = _legacy_pack_quantized_array_gzip(
        window_view,
        scale=None,
        dt=dt_val,
        extra=None,
    )
    if perf_timings_ms is not None:
        perf_timings_ms['build_ms'] = build_ms
        perf_timings_ms['pack_ms'] = (time.perf_counter() - pack_started) * 1000.0
    return payload


def _clear_runtime_caches() -> None:
    state = get_state(app)
    app.state.sv.file_registry.clear()
    state.cached_readers.clear()
    state.pipeline_tap_cache.clear()
    state.window_section_cache.clear()
    state.section_offsets_cache.clear()
    state.trace_stats_cache.clear()
    _BASELINE_CACHE.clear()


def test_build_section_window_payload_defers_transpose_to_packer(
    monkeypatch, tmp_path: Path
):
    section = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)
    stub_reader = make_stub_reader(section)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        svc,
        'apply_scaling_from_baseline',
        lambda arr, **_kwargs: arr,
        raising=True,
    )

    def _pack(arr_f32, *, scale, dt, extra=None, transpose=False):
        captured['shape'] = tuple(int(x) for x in arr_f32.shape)
        captured['transpose'] = bool(transpose)
        captured['c_contiguous'] = bool(arr_f32.flags.c_contiguous)
        return b'payload'

    monkeypatch.setattr(svc, 'pack_quantized_array_gzip', _pack, raising=True)

    payload = svc.build_section_window_payload(
        file_id='f',
        key1=KEY1_VALUE,
        key1_byte=KEY1,
        key2_byte=KEY2,
        offset_byte=None,
        x0=0,
        x1=6,
        y0=1,
        y1=6,
        step_x=1,
        step_y=1,
        transpose=True,
        pipeline_key=None,
        tap_label=None,
        scaling_mode='amax',
        trace_stats_cache={},
        reader_getter=lambda _fid, _kb1, _kb2: stub_reader,
        pipeline_section_getter=lambda **_kwargs: None,
        store_dir_resolver=lambda _fid: str(tmp_path),
        dt_resolver=lambda _fid: DT,
    )

    assert payload == b'payload'
    assert captured == {
        'shape': (7, 6),
        'transpose': True,
        'c_contiguous': True,
    }


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('step_x,step_y', [(1, 1), (2, 2)])
@pytest.mark.parametrize('scaling_mode', ['amax', 'tracewise'])
def test_build_section_window_payload_matches_legacy_reference(
    tmp_path: Path,
    transpose: bool,
    step_x: int,
    step_y: int,
    scaling_mode: str,
):
    section = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)
    stub_reader = make_stub_reader(section)
    write_baseline_raw(
        tmp_path,
        key1=KEY1_VALUE,
        key1_byte=KEY1,
        key2_byte=KEY2,
        section_mean=0.0,
        section_std=1.0,
        trace_means=[0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0],
        trace_stds=[1.0] * 8,
    )

    kwargs = dict(
        file_id='f',
        key1=KEY1_VALUE,
        key1_byte=KEY1,
        key2_byte=KEY2,
        offset_byte=None,
        x0=0,
        x1=6,
        y0=1,
        y1=6,
        step_x=step_x,
        step_y=step_y,
        transpose=transpose,
        pipeline_key=None,
        tap_label=None,
        scaling_mode=scaling_mode,
        reader_getter=lambda _fid, _kb1, _kb2: stub_reader,
        pipeline_section_getter=lambda **_kwargs: None,
        store_dir_resolver=lambda _fid: str(tmp_path),
        dt_resolver=lambda _fid: DT,
    )

    legacy = _decode_payload(
        _legacy_build_section_window_payload(
            **kwargs,
            trace_stats_cache={},
        )
    )
    actual = _decode_payload(
        svc.build_section_window_payload(
            **kwargs,
            trace_stats_cache={},
        )
    )

    assert actual['shape'] == legacy['shape']
    assert actual['scale'] == pytest.approx(legacy['scale'])
    assert actual['dt'] == pytest.approx(legacy['dt'])
    assert np.array_equal(actual['data'], legacy['data'])


class _StaticReader:
    def __init__(self, section: np.ndarray):
        self._section = np.asarray(section, dtype=np.float32, order='C')

    def get_section(self, _key1: int) -> SectionView:
        return SectionView(
            arr=self._section,
            dtype=self._section.dtype,
            scale=None,
        )


def _measure_total_ms(builder) -> dict[str, float]:
    perf_timings_ms: dict[str, float] = {}
    payload = builder(perf_timings_ms=perf_timings_ms)
    assert payload
    build_ms = perf_timings_ms['build_ms']
    pack_ms = perf_timings_ms['pack_ms']
    return {
        'build_ms': build_ms,
        'pack_ms': pack_ms,
        'total_ms': build_ms + pack_ms,
    }


def _format_samples(label: str, samples: list[dict[str, float]]) -> str:
    builds = [round(sample['build_ms'], 3) for sample in samples]
    packs = [round(sample['pack_ms'], 3) for sample in samples]
    totals = [round(sample['total_ms'], 3) for sample in samples]
    return (
        f'{label}: builds={builds} packs={packs} totals={totals} '
        f'p50={statistics.median([sample["total_ms"] for sample in samples]):.3f}ms'
    )


def test_build_section_window_payload_total_ms_improves_on_transpose_cold_path(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        svc,
        'apply_scaling_from_baseline',
        lambda arr, **_kwargs: arr,
        raising=True,
    )

    section = np.zeros((4096, 1536), dtype=np.float32)
    reader = _StaticReader(section)
    kwargs = dict(
        file_id='perf-file',
        key1=KEY1_VALUE,
        key1_byte=KEY1,
        key2_byte=KEY2,
        offset_byte=None,
        x0=0,
        x1=4095,
        y0=0,
        y1=1535,
        step_x=1,
        step_y=1,
        transpose=True,
        pipeline_key=None,
        tap_label=None,
        scaling_mode='amax',
        reader_getter=lambda _fid, _kb1, _kb2: reader,
        pipeline_section_getter=lambda **_kwargs: None,
        store_dir_resolver=lambda _fid: str(tmp_path),
        dt_resolver=lambda _fid: DT,
    )

    legacy_samples: list[dict[str, float]] = []
    actual_samples: list[dict[str, float]] = []
    total_rounds = PERF_WARMUP_RUNS + PERF_MEASURED_RUNS

    for round_idx in range(total_rounds):
        round_samples: dict[str, dict[str, float]] = {}
        order = ('legacy', 'actual') if round_idx % 2 == 0 else ('actual', 'legacy')
        for name in order:
            _clear_runtime_caches()
            if name == 'legacy':
                sample = _measure_total_ms(
                    lambda *, perf_timings_ms: _legacy_build_section_window_payload(
                        **kwargs,
                        trace_stats_cache={},
                        perf_timings_ms=perf_timings_ms,
                    )
                )
            else:
                sample = _measure_total_ms(
                    lambda *, perf_timings_ms: svc.build_section_window_payload(
                        **kwargs,
                        trace_stats_cache={},
                        perf_timings_ms=perf_timings_ms,
                    )
                )
            round_samples[name] = sample
        if round_idx >= PERF_WARMUP_RUNS:
            legacy_samples.append(round_samples['legacy'])
            actual_samples.append(round_samples['actual'])

    legacy_p50 = statistics.median(
        sample['total_ms'] for sample in legacy_samples
    )
    actual_p50 = statistics.median(
        sample['total_ms'] for sample in actual_samples
    )

    allowed_noise_ms = max(1.0, legacy_p50 * 0.05)
    assert actual_p50 <= legacy_p50 + allowed_noise_ms, (
        'Expected quantize-then-int8-transpose total_ms median to be at least as fast '
        'as the legacy float32-transpose cold path for transpose=True within timing '
        'noise. '
        f'{_format_samples("actual", actual_samples)}; '
        f'{_format_samples("legacy", legacy_samples)}; '
        f'allowed_noise_ms={allowed_noise_ms:.3f}'
    )
