from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import median

import numpy as np

from app.services import section_service as svc
from app.trace_store.reader import TraceStoreSectionReader

KEY1 = 189
KEY2 = 193
N_SECTIONS = 256
TRACES_PER_SECTION = 4096
N_SAMPLES = 4
WARMUP_RUNS = 2
MEASURED_RUNS = 7


def _write_perf_store(store_dir: Path) -> np.ndarray:
    store_dir.mkdir(parents=True, exist_ok=True)
    key1_values = np.arange(10_000, 10_000 + N_SECTIONS, dtype=np.int32)
    key1s = np.repeat(key1_values, TRACES_PER_SECTION)
    key2s = np.tile(np.arange(TRACES_PER_SECTION, dtype=np.int32), N_SECTIONS)
    n_traces = int(key1s.size)
    traces = np.zeros((n_traces, N_SAMPLES), dtype=np.float32)

    np.save(store_dir / 'traces.npy', traces)
    np.save(store_dir / f'headers_byte_{KEY1}.npy', key1s)
    np.save(store_dir / f'headers_byte_{KEY2}.npy', key2s)
    np.savez(
        store_dir / 'index.npz',
        key1_values=key1_values,
        key1_offsets=np.arange(0, n_traces, TRACES_PER_SECTION, dtype=np.int64),
        key1_counts=np.full(N_SECTIONS, TRACES_PER_SECTION, dtype=np.int64),
        sorted_to_original=np.arange(n_traces, dtype=np.int64),
    )
    meta = {
        'dt': 0.004,
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'original_segy_path': 'dummy.sgy',
        'original_mtime': 0.0,
        'original_size': 0,
    }
    (store_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return key1_values


def _measure_build_ms_samples(store_dir: Path, *, key1_val: int) -> list[float]:
    def _run_once() -> float:
        reader = TraceStoreSectionReader(store_dir, KEY1, KEY2)
        perf_timings_ms: dict[str, float] = {}
        payload = svc.build_section_window_payload(
            file_id='perf-file',
            key1=key1_val,
            key1_byte=KEY1,
            key2_byte=KEY2,
            offset_byte=None,
            x0=0,
            x1=TRACES_PER_SECTION - 1,
            y0=0,
            y1=N_SAMPLES - 1,
            step_x=1,
            step_y=1,
            transpose=False,
            pipeline_key=None,
            tap_label=None,
            scaling_mode='amax',
            trace_stats_cache={},
            reader_getter=lambda _fid, _kb1, _kb2: reader,
            pipeline_section_getter=lambda **_kwargs: None,
            store_dir_resolver=lambda _fid: str(store_dir),
            dt_resolver=lambda _fid: 0.004,
            perf_timings_ms=perf_timings_ms,
        )
        assert payload == b'payload'
        return perf_timings_ms['build_ms']

    for _ in range(WARMUP_RUNS):
        _run_once()
    return [_run_once() for _ in range(MEASURED_RUNS)]


def test_build_section_window_payload_prefers_index_lookup_in_build_ms(
    monkeypatch, tmp_path: Path
):
    index_store = tmp_path / 'with_index'
    fallback_store = tmp_path / 'without_index'
    key1_values = _write_perf_store(index_store)
    shutil.copytree(index_store, fallback_store)
    (fallback_store / 'index.npz').unlink()

    monkeypatch.setattr(
        svc,
        'apply_scaling_from_baseline',
        lambda arr, **_kwargs: arr,
        raising=True,
    )
    monkeypatch.setattr(
        svc,
        'pack_quantized_array_gzip',
        lambda *_args, **_kwargs: b'payload',
        raising=True,
    )

    key1_val = int(key1_values[N_SECTIONS // 2])
    index_samples = _measure_build_ms_samples(index_store, key1_val=key1_val)
    fallback_samples = _measure_build_ms_samples(fallback_store, key1_val=key1_val)
    index_median = float(median(index_samples))
    fallback_median = float(median(fallback_samples))

    assert index_median < fallback_median, (
        'Expected index-backed build_ms median to be lower than fallback build_ms '
        f'median for the same cold request. index={index_median:.3f}ms '
        f'samples={index_samples}; fallback={fallback_median:.3f}ms '
        f'samples={fallback_samples}'
    )
