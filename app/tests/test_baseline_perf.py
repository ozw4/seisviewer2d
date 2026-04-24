from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import baselines as baselines_mod
from app.api._helpers import get_state
from app.main import app
from app.services import section_service as svc
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.baseline_artifacts import (
    BASELINE_STAGE_RAW,
    build_legacy_baseline_path,
    build_baseline_manifest_path,
    build_baseline_npz_path,
    build_raw_baseline_payload,
    build_trace_spans_by_key1,
    write_raw_baseline_artifacts,
)
from app.utils.segy_meta import _BASELINE_CACHE

KEY1 = 189
KEY2 = 193
DT = 0.004
N_SECTIONS = 32
TRACES_PER_SECTION = 4096
N_SAMPLES = 16
ZERO_STD_EPS = 1e-12


@pytest.fixture(autouse=True)
def _clean_state():
    state = get_state(app)
    app.state.sv.file_registry.clear()
    state.cached_readers.clear()
    state.window_section_cache.clear()
    state.trace_stats_cache.clear()
    _BASELINE_CACHE.clear()
    yield
    app.state.sv.file_registry.clear()
    state.cached_readers.clear()
    state.window_section_cache.clear()
    state.trace_stats_cache.clear()
    _BASELINE_CACHE.clear()


def _clear_runtime_caches() -> None:
    state = get_state(app)
    state.cached_readers.clear()
    state.window_section_cache.clear()
    state.trace_stats_cache.clear()
    _BASELINE_CACHE.clear()


def _build_perf_payload(
    *,
    traces: np.ndarray,
    key1_values: np.ndarray,
    key1_offsets: np.ndarray,
    key1_counts: np.ndarray,
) -> dict[str, object]:
    trace_mean = traces.mean(axis=1, dtype=np.float64)
    trace_var = traces.var(axis=1, dtype=np.float64)
    trace_std = np.sqrt(np.maximum(trace_var, 0.0))
    zero_mask = trace_std <= ZERO_STD_EPS
    if zero_mask.any():
        trace_std = trace_std.copy()
        trace_std[zero_mask] = 1.0

    trace_sum = traces.sum(axis=1, dtype=np.float64)
    trace_sumsq = np.einsum('ij,ij->i', traces, traces, dtype=np.float64)
    total_samples = key1_counts.astype(np.float64) * float(traces.shape[1])
    section_sum = np.add.reduceat(trace_sum, key1_offsets.astype(np.int64, copy=False))
    section_sumsq = np.add.reduceat(
        trace_sumsq,
        key1_offsets.astype(np.int64, copy=False),
    )
    section_mean = section_sum / total_samples
    section_std = np.sqrt(
        np.maximum((section_sumsq / total_samples) - np.square(section_mean), 0.0)
    )
    return build_raw_baseline_payload(
        dtype_base='float32',
        dt=DT,
        key1_values=key1_values,
        mu_sections=section_mean,
        sigma_sections=section_std,
        mu_traces=trace_mean,
        sigma_traces=trace_std,
        zero_var_mask=zero_mask,
        trace_spans_by_key1=build_trace_spans_by_key1(
            key1_values,
            key1_offsets,
            key1_counts,
        ),
        source_sha256='perf-sha',
        key1_byte=KEY1,
        key2_byte=KEY2,
    )


def _write_perf_store(store_dir: Path, *, baseline_mode: str) -> int:
    store_dir.mkdir(parents=True, exist_ok=True)
    key1_values = np.arange(10_001, 10_001 + N_SECTIONS, dtype=np.int32)
    key1s = np.repeat(key1_values, TRACES_PER_SECTION)
    key2s = np.tile(np.arange(1, TRACES_PER_SECTION + 1, dtype=np.int32), N_SECTIONS)
    key1_offsets = np.arange(0, key1s.size, TRACES_PER_SECTION, dtype=np.int64)
    key1_counts = np.full(N_SECTIONS, TRACES_PER_SECTION, dtype=np.int64)
    section_idx = np.repeat(np.arange(N_SECTIONS, dtype=np.float32), TRACES_PER_SECTION)
    trace_idx = np.tile(np.arange(TRACES_PER_SECTION, dtype=np.float32), N_SECTIONS)
    sample_idx = np.arange(N_SAMPLES, dtype=np.float32)[None, :]
    traces = (
        np.sin(sample_idx * 0.13 + trace_idx[:, None] * 0.003)
        + section_idx[:, None] * 0.02
    ).astype(np.float32)

    np.save(store_dir / 'traces.npy', traces)
    np.save(store_dir / f'headers_byte_{KEY1}.npy', key1s)
    np.save(store_dir / f'headers_byte_{KEY2}.npy', key2s)
    np.savez(
        store_dir / 'index.npz',
        key1_values=key1_values,
        key1_offsets=key1_offsets,
        key1_counts=key1_counts,
        sorted_to_original=np.arange(key1s.size, dtype=np.int64),
    )
    meta = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': int(key1s.size),
        'n_samples': int(N_SAMPLES),
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'sorted_by': ['key1', 'key2'],
        'dt': DT,
        'original_segy_path': str(store_dir / 'perf.sgy'),
        'original_mtime': 0.0,
        'original_size': 0,
        'source_sha256': 'perf-sha',
    }
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')

    if baseline_mode != 'none':
        payload = _build_perf_payload(
            traces=traces,
            key1_values=key1_values,
            key1_offsets=key1_offsets,
            key1_counts=key1_counts,
        )
        if baseline_mode == 'split':
            write_raw_baseline_artifacts(
                store_dir,
                key1_byte=KEY1,
                key2_byte=KEY2,
                payload=payload,
            )
        elif baseline_mode == 'legacy':
            (store_dir / 'baseline_raw.json').write_text(
                json.dumps(payload),
                encoding='utf-8',
            )
        else:
            raise AssertionError(baseline_mode)
    return int(key1_values[N_SECTIONS // 2])


def _fetch_meta_baseline_ms(
    client: TestClient,
    *,
    file_id: str,
) -> tuple[float, str]:
    _clear_runtime_caches()
    resp = client.get(
        '/get_section_meta',
        params={'file_id': file_id, 'key1_byte': KEY1, 'key2_byte': KEY2},
    )
    assert resp.status_code == 200, resp.text
    assert 'sv_baseline;dur=' in resp.headers['server-timing']
    return (
        float(resp.headers['x-sv-baseline-ms']),
        resp.headers['x-sv-baseline-source'],
    )


def _build_payload(store_dir: Path, *, key1_value: int) -> tuple[bytes, float, str]:
    _clear_runtime_caches()
    reader = TraceStoreSectionReader(store_dir, KEY1, KEY2)
    perf_timings_ms: dict[str, float] = {}
    payload = svc.build_section_window_payload(
        file_id='perf-file',
        key1=key1_value,
        key1_byte=KEY1,
        key2_byte=KEY2,
        offset_byte=None,
        x0=0,
        x1=255,
        y0=0,
        y1=N_SAMPLES - 1,
        step_x=1,
        step_y=1,
        transpose=False,
        pipeline_key=None,
        tap_label=None,
        scaling_mode='tracewise',
        trace_stats_cache={},
        reader_getter=lambda _fid, _kb1, _kb2: reader,
        pipeline_section_getter=lambda **_kwargs: None,
        store_dir_resolver=lambda _fid: str(store_dir),
        dt_resolver=lambda _fid: DT,
        perf_timings_ms=perf_timings_ms,
    )
    cache_keys = list(_BASELINE_CACHE)
    assert len(cache_keys) == 1
    return payload, perf_timings_ms['build_ms'], cache_keys[0]


def _remove_all_baseline_artifacts(store_dir: Path) -> None:
    build_baseline_manifest_path(
        store_dir,
        stage=BASELINE_STAGE_RAW,
        key1_byte=KEY1,
        key2_byte=KEY2,
    ).unlink(missing_ok=True)
    build_baseline_npz_path(
        store_dir,
        stage=BASELINE_STAGE_RAW,
        key1_byte=KEY1,
        key2_byte=KEY2,
    ).unlink(missing_ok=True)
    build_legacy_baseline_path(store_dir).unlink(missing_ok=True)


def _measure_meta_baseline_runs(
    client: TestClient,
    *,
    file_id: str,
    runs: int,
    prepare: Callable[[], None] | None = None,
) -> tuple[list[float], list[str]]:
    timings_ms: list[float] = []
    sources: list[str] = []
    for _ in range(runs):
        if prepare is not None:
            prepare()
        baseline_ms, source = _fetch_meta_baseline_ms(client, file_id=file_id)
        timings_ms.append(baseline_ms)
        sources.append(source)
    return timings_ms, sources


def _measure_window_build_runs(
    store_dir: Path,
    *,
    key1_value: int,
    runs: int,
) -> tuple[bytes, list[float], list[str]]:
    payload: bytes | None = None
    timings_ms: list[float] = []
    cache_keys: list[str] = []
    for _ in range(runs):
        run_payload, build_ms, cache_key = _build_payload(
            store_dir,
            key1_value=key1_value,
        )
        if payload is None:
            payload = run_payload
        timings_ms.append(build_ms)
        cache_keys.append(cache_key)
    assert payload is not None
    return payload, timings_ms, cache_keys


def test_get_section_meta_cold_path_precomputed_is_faster_than_fallback(
    tmp_path,
    monkeypatch,
):
    state = get_state(app)
    compute_calls = {'count': 0}
    original_compute = baselines_mod._compute_baseline

    def _wrapped_compute(*args, **kwargs):
        compute_calls['count'] += 1
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(baselines_mod, '_compute_baseline', _wrapped_compute)

    with TestClient(app) as client:
        precomputed_store = tmp_path / 'precomputed'
        fallback_store = tmp_path / 'fallback'
        _write_perf_store(precomputed_store, baseline_mode='split')
        _write_perf_store(fallback_store, baseline_mode='none')

        precomputed_file_id = 'precomputed-file'
        fallback_file_id = 'fallback-file'
        state.file_registry.set_record(
            precomputed_file_id,
            {'store_path': str(precomputed_store), 'dt': DT},
        )
        state.file_registry.set_record(
            fallback_file_id,
            {'store_path': str(fallback_store), 'dt': DT},
        )
        runs = 5

        precomputed_ms, precomputed_sources = _measure_meta_baseline_runs(
            client,
            file_id=precomputed_file_id,
            runs=runs,
        )
        fallback_ms, fallback_sources = _measure_meta_baseline_runs(
            client,
            file_id=fallback_file_id,
            runs=runs,
            prepare=lambda: _remove_all_baseline_artifacts(fallback_store),
        )
        assert precomputed_sources == ['precomputed'] * runs
        assert fallback_sources == ['computed-fallback'] * runs
        assert compute_calls['count'] == runs
        assert build_baseline_manifest_path(
            fallback_store,
            stage=BASELINE_STAGE_RAW,
            key1_byte=KEY1,
            key2_byte=KEY2,
        ).is_file()
        assert build_baseline_npz_path(
            fallback_store,
            stage=BASELINE_STAGE_RAW,
            key1_byte=KEY1,
            key2_byte=KEY2,
        ).is_file()
        precomputed_p50 = statistics.median(precomputed_ms)
        fallback_p50 = statistics.median(fallback_ms)
        assert precomputed_p50 < fallback_p50, (
            'expected precomputed /get_section_meta baseline path to beat '
            f'fallback compute path, but medians were '
            f'precomputed={precomputed_p50:.3f}ms ({precomputed_ms}) and '
            f'fallback={fallback_p50:.3f}ms ({fallback_ms})'
        )


def test_build_section_window_payload_cold_path_split_is_faster_than_legacy_json(
    tmp_path,
):
    split_store = tmp_path / 'split'
    legacy_store = tmp_path / 'legacy'
    key1_value = _write_perf_store(split_store, baseline_mode='split')
    _write_perf_store(legacy_store, baseline_mode='legacy')
    runs = 5
    split_payload, split_build_ms, split_cache_keys = _measure_window_build_runs(
        split_store,
        key1_value=key1_value,
        runs=runs,
    )
    legacy_payload, legacy_build_ms, legacy_cache_keys = _measure_window_build_runs(
        legacy_store,
        key1_value=key1_value,
        runs=runs,
    )
    assert split_payload
    assert legacy_payload
    assert all(cache_key.startswith('split|') for cache_key in split_cache_keys)
    assert all(
        cache_key.startswith('legacy-json|') for cache_key in legacy_cache_keys
    )
    split_p50 = statistics.median(split_build_ms)
    legacy_p50 = statistics.median(legacy_build_ms)
    allowed_noise_ms = max(1.0, legacy_p50 * 0.05)
    assert split_p50 <= legacy_p50 + allowed_noise_ms, (
        'expected split baseline artifacts to be at least as fast as legacy JSON '
        'within measurement noise for cold window builds, but medians were '
        f'split={split_p50:.3f}ms ({split_build_ms}) and '
        f'legacy={legacy_p50:.3f}ms ({legacy_build_ms}); '
        f'allowed_noise_ms={allowed_noise_ms:.3f}'
    )
