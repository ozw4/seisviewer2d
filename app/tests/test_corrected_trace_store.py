from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import app.services.corrected_trace_store as svc
from app.services.corrected_trace_store import build_time_shifted_trace_store
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.baseline_artifacts import (
    BASELINE_STAGE_RAW,
    build_baseline_manifest_path,
    build_baseline_npz_path,
)
from app.utils.time_shift import shift_traces_linear

KEY1 = 189
KEY2 = 193
HEADER_BYTE = 37


def _write_source_store(
    store: Path,
    *,
    traces: np.ndarray | None = None,
    dt: float = 1.0,
    dtype: str = 'float32',
    scale: float | None = None,
    include_sorted_to_original: bool = True,
    key1_values: np.ndarray | None = None,
    key1_offsets: np.ndarray | None = None,
    key1_counts: np.ndarray | None = None,
    sorted_to_original: np.ndarray | None = None,
    headers: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    store.mkdir(parents=True, exist_ok=True)
    if traces is None:
        traces = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
            dtype=np.float32,
        )
    traces_arr = np.asarray(traces)
    n_traces, n_samples = traces_arr.shape
    np.save(store / 'traces.npy', traces_arr)

    if key1_values is None:
        key1_values = np.array([10], dtype=np.int64)
    if key1_offsets is None:
        key1_offsets = np.array([0], dtype=np.int64)
    if key1_counts is None:
        key1_counts = np.array([n_traces], dtype=np.int64)
    if sorted_to_original is None:
        sorted_to_original = np.arange(n_traces, dtype=np.int64)

    index_payload = {
        'key1_values': np.asarray(key1_values),
        'key1_offsets': np.asarray(key1_offsets),
        'key1_counts': np.asarray(key1_counts),
    }
    if include_sorted_to_original:
        index_payload['sorted_to_original'] = np.asarray(sorted_to_original)
    np.savez(store / 'index.npz', **index_payload)

    key1_header = np.empty(n_traces, dtype=np.int32)
    for key1, offset, count in zip(
        np.asarray(key1_values, dtype=np.int32),
        np.asarray(key1_offsets, dtype=np.int64),
        np.asarray(key1_counts, dtype=np.int64),
        strict=True,
    ):
        key1_header[int(offset) : int(offset + count)] = int(key1)
    header_payload = {
        KEY1: key1_header,
        KEY2: np.arange(1, n_traces + 1, dtype=np.int32),
    }
    header_payload.update(headers or {})
    for byte, values in header_payload.items():
        np.save(store / f'headers_byte_{int(byte)}.npy', np.asarray(values))

    meta = {
        'schema_version': 1,
        'dtype': dtype,
        'n_traces': int(n_traces),
        'n_samples': int(n_samples),
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'sorted_by': ['key1', 'key2'],
        'dt': dt,
        'original_segy_path': str(store / 'source.sgy'),
        'source_sha256': 'source-sha',
        'original_name': 'source.sgy',
    }
    if scale is not None:
        meta['scale'] = float(scale)
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return traces_arr


def _build(
    tmp_path: Path,
    *,
    traces: np.ndarray | None = None,
    shifts: np.ndarray | None = None,
    **kwargs,
) -> tuple[Path, Path, np.ndarray]:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces_arr = _write_source_store(source, traces=traces, **kwargs)
    if shifts is None:
        shifts = np.zeros(traces_arr.shape[0], dtype=np.float64)
    build_time_shifted_trace_store(
        source_store_path=source,
        output_store_path=output,
        trace_shift_s_sorted=shifts,
        chunk_size=2,
    )
    return source, output, traces_arr


def _tmp_stores(tmp_path: Path) -> list[Path]:
    return list(tmp_path.glob('corrected.tmp-*'))


def test_build_time_shifted_trace_store_writes_expected_shifted_traces(
    tmp_path: Path,
) -> None:
    traces = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    shifts = np.asarray([1.0, -1.0, 0.5], dtype=np.float64)
    _, output, _ = _build(tmp_path, traces=traces, shifts=shifts, dt=1.0)

    corrected = np.load(output / 'traces.npy', mmap_mode='r')
    np.testing.assert_allclose(corrected, shift_traces_linear(traces, shifts, dt=1.0))
    assert corrected.dtype == np.float32


def test_build_time_shifted_trace_store_positive_shift_delays_impulse(
    tmp_path: Path,
) -> None:
    traces = np.zeros((2, 8), dtype=np.float32)
    traces[:, 3] = 1.0
    _, output, _ = _build(
        tmp_path,
        traces=traces,
        shifts=np.asarray([0.008, 0.008], dtype=np.float64),
        dt=0.004,
    )

    corrected = np.load(output / 'traces.npy', mmap_mode='r')
    np.testing.assert_allclose(corrected[:, 5], np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(corrected[:, 3], np.zeros(2, dtype=np.float32))


def test_build_time_shifted_trace_store_negative_shift_advances_impulse(
    tmp_path: Path,
) -> None:
    traces = np.zeros((2, 8), dtype=np.float32)
    traces[:, 3] = 1.0
    _, output, _ = _build(
        tmp_path,
        traces=traces,
        shifts=np.asarray([-0.008, -0.008], dtype=np.float64),
        dt=0.004,
    )

    corrected = np.load(output / 'traces.npy', mmap_mode='r')
    np.testing.assert_allclose(corrected[:, 1], np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(corrected[:, 3], np.zeros(2, dtype=np.float32))


def test_build_time_shifted_trace_store_fractional_shift_uses_linear_interpolation(
    tmp_path: Path,
) -> None:
    traces = np.asarray([[0.0, 10.0, 20.0, 30.0]], dtype=np.float32)
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    _write_source_store(source, traces=traces, dt=1.0)

    build_time_shifted_trace_store(
        source_store_path=source,
        output_store_path=output,
        trace_shift_s_sorted=np.asarray([0.5], dtype=np.float64),
        fill_value=-1.0,
    )

    corrected = np.load(output / 'traces.npy', mmap_mode='r')
    np.testing.assert_allclose(
        corrected,
        np.asarray([[-1.0, 5.0, 15.0, 25.0]], dtype=np.float32),
    )


def test_build_time_shifted_trace_store_preserves_index_npz(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = np.zeros((3, 4), dtype=np.float32)
    _write_source_store(
        source,
        traces=traces,
        key1_values=np.asarray([10, 20], dtype=np.int64),
        key1_offsets=np.asarray([0, 2], dtype=np.int64),
        key1_counts=np.asarray([2, 1], dtype=np.int64),
        sorted_to_original=np.asarray([2, 0, 1], dtype=np.int64),
    )

    build_time_shifted_trace_store(
        source_store_path=source,
        output_store_path=output,
        trace_shift_s_sorted=np.zeros(3, dtype=np.float64),
    )

    with np.load(source / 'index.npz', allow_pickle=False) as src:
        with np.load(output / 'index.npz', allow_pickle=False) as dst:
            assert set(dst.files) == set(src.files)
            for key in src.files:
                np.testing.assert_array_equal(dst[key], src[key])


def test_build_time_shifted_trace_store_copies_existing_headers(tmp_path: Path) -> None:
    header = np.asarray([100, 200, 300], dtype=np.int64)
    _, output, _ = _build(tmp_path, headers={HEADER_BYTE: header})

    copied = np.load(output / f'headers_byte_{HEADER_BYTE}.npy', mmap_mode='r')
    assert copied.dtype == np.int32
    np.testing.assert_array_equal(copied, header.astype(np.int32))


def test_build_time_shifted_trace_store_materializes_requested_headers_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)
    materialized = np.asarray([70, 80, 90], dtype=np.int32)
    calls: list[int] = []

    def _ensure_header(self: TraceStoreSectionReader, byte: int) -> np.ndarray:
        calls.append(int(byte))
        path = self.store_dir / f'headers_byte_{int(byte)}.npy'
        np.save(path, materialized)
        return np.load(path, mmap_mode='r')

    monkeypatch.setattr(svc.TraceStoreSectionReader, 'ensure_header', _ensure_header)

    build_time_shifted_trace_store(
        source_store_path=source,
        output_store_path=output,
        trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
        header_bytes_to_materialize=[HEADER_BYTE],
    )

    assert calls == [HEADER_BYTE]
    np.testing.assert_array_equal(
        np.load(output / f'headers_byte_{HEADER_BYTE}.npy'),
        materialized,
    )


def test_build_time_shifted_trace_store_writes_derived_meta(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source, dt=0.004)

    result = build_time_shifted_trace_store(
        source_store_path=source,
        output_store_path=output,
        trace_shift_s_sorted=np.asarray([0.0, 0.004, -0.004], dtype=np.float64),
        fill_value=-5.0,
        derived_metadata={'method': 'unit-test'},
        from_file_id='file-1',
    )

    meta = json.loads((output / 'meta.json').read_text(encoding='utf-8'))
    assert meta['dtype'] == 'float32'
    assert meta['n_traces'] == traces.shape[0]
    assert meta['n_samples'] == traces.shape[1]
    assert meta['key_bytes'] == {'key1': KEY1, 'key2': KEY2}
    assert meta['dt'] == 0.004
    assert meta['original_segy_path'] is None
    assert meta['source_sha256'] is None
    assert 'scale' not in meta
    derived = meta['derived']
    assert derived['from_store_path'] == str(source.resolve())
    assert derived['header_source_store_path'] == str(source.resolve())
    assert derived['from_file_id'] == 'file-1'
    assert derived['fill_value'] == -5.0
    assert derived['output_dtype'] == 'float32'
    assert derived['method'] == 'unit-test'
    assert derived['applied_shift_summary'] == result.shift_summary


def test_build_time_shifted_trace_store_does_not_set_original_name(
    tmp_path: Path,
) -> None:
    _, output, _ = _build(tmp_path)

    meta = json.loads((output / 'meta.json').read_text(encoding='utf-8'))
    assert 'original_name' not in meta


def test_build_time_shifted_trace_store_writes_baseline_artifacts_from_corrected_traces(
    tmp_path: Path,
) -> None:
    traces = np.asarray([[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]], dtype=np.float32)
    _, output, _ = _build(
        tmp_path,
        traces=traces,
        shifts=np.asarray([1.0, -1.0], dtype=np.float64),
        dt=1.0,
    )
    corrected = np.load(output / 'traces.npy')
    manifest_path = build_baseline_manifest_path(
        output,
        stage=BASELINE_STAGE_RAW,
        key1_byte=KEY1,
        key2_byte=KEY2,
    )
    npz_path = build_baseline_npz_path(
        output,
        stage=BASELINE_STAGE_RAW,
        key1_byte=KEY1,
        key2_byte=KEY2,
    )

    assert manifest_path.is_file()
    assert npz_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['dtype_base'] == 'float32'
    assert manifest['source_sha256'] is None
    with np.load(npz_path, allow_pickle=False) as arrays:
        np.testing.assert_allclose(
            arrays['mu_traces'],
            corrected.mean(axis=1, dtype=np.float64).astype(np.float32),
        )


def test_build_time_shifted_trace_store_reader_can_open_generated_store(
    tmp_path: Path,
) -> None:
    _, output, traces = _build(tmp_path)

    reader = TraceStoreSectionReader(output, key1_byte=KEY1, key2_byte=KEY2)
    view = reader.get_section(10)

    np.testing.assert_allclose(view.arr, traces)
    assert view.scale is None


def test_build_time_shifted_trace_store_float32_output_for_int8_source_with_existing_scale_semantics(
    tmp_path: Path,
) -> None:
    traces = np.asarray([[2, 4, 6], [8, 10, 12]], dtype=np.int8)
    _, output, _ = _build(tmp_path, traces=traces, dtype='int8', scale=0.5)

    corrected = np.load(output / 'traces.npy')
    assert corrected.dtype == np.float32
    np.testing.assert_allclose(corrected, traces.astype(np.float32) * 0.5)


def test_build_time_shifted_trace_store_rejects_existing_output_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)
    output.mkdir()

    with pytest.raises(ValueError, match='already exists'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
        )


def test_build_time_shifted_trace_store_rejects_source_and_output_same_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    traces = _write_source_store(source)

    with pytest.raises(ValueError, match='different'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=source,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
        )


def test_build_time_shifted_trace_store_rejects_non_float32_output_dtype(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)

    with pytest.raises(ValueError, match='output_dtype'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
            output_dtype='int8',
        )


def test_build_time_shifted_trace_store_rejects_shift_shape_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    _write_source_store(source)

    with pytest.raises(ValueError, match='shape mismatch'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(2, dtype=np.float64),
        )


def test_build_time_shifted_trace_store_rejects_non_finite_shifts(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)
    shifts = np.zeros(traces.shape[0], dtype=np.float64)
    shifts[1] = np.nan

    with pytest.raises(ValueError, match='finite'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=shifts,
        )


def test_build_time_shifted_trace_store_rejects_invalid_dt(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source, dt=0.0)

    with pytest.raises(ValueError, match='dt'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
        )


def test_build_time_shifted_trace_store_rejects_invalid_chunk_size(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)

    with pytest.raises(ValueError, match='chunk_size'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
            chunk_size=0,
        )


@pytest.mark.parametrize('header_byte', [0, 241, 'bad'])
def test_build_time_shifted_trace_store_rejects_invalid_header_byte(
    tmp_path: Path,
    header_byte: object,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)

    with pytest.raises(ValueError, match='header byte'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
            header_bytes_to_materialize=[header_byte],
        )


def test_build_time_shifted_trace_store_rejects_index_without_sorted_to_original(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source, include_sorted_to_original=False)

    with pytest.raises(ValueError, match='sorted_to_original'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
        )


def test_build_time_shifted_trace_store_cleans_tmp_store_on_shift_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)

    def _boom(*_args, **_kwargs) -> np.ndarray:
        raise ValueError('shift failed')

    monkeypatch.setattr(svc, 'shift_traces_linear', _boom)

    with pytest.raises(ValueError, match='shift failed'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
        )

    assert not output.exists()
    assert not _tmp_stores(tmp_path)


def test_build_time_shifted_trace_store_cleans_tmp_store_on_baseline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)

    def _boom(**_kwargs) -> dict:
        raise RuntimeError('baseline failed')

    monkeypatch.setattr(svc, 'write_trace_store_raw_baseline_artifacts', _boom)

    with pytest.raises(RuntimeError, match='baseline failed'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
        )

    assert not output.exists()
    assert not _tmp_stores(tmp_path)


def test_build_time_shifted_trace_store_cleans_tmp_store_on_cancel(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)
    calls = {'count': 0}

    def _cancel() -> bool:
        calls['count'] += 1
        return calls['count'] >= 4

    with pytest.raises(RuntimeError, match='cancelled'):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=np.zeros(traces.shape[0], dtype=np.float64),
            cancel_check=_cancel,
        )

    assert not output.exists()
    assert not _tmp_stores(tmp_path)


def test_build_time_shifted_trace_store_does_not_create_output_store_on_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'source'
    output = tmp_path / 'corrected'
    traces = _write_source_store(source)
    shifts = np.zeros(traces.shape[0] + 1, dtype=np.float64)

    with pytest.raises(ValueError):
        build_time_shifted_trace_store(
            source_store_path=source,
            output_store_path=output,
            trace_shift_s_sorted=shifts,
        )

    assert not output.exists()
    assert not _tmp_stores(tmp_path)
