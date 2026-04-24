import json
from pathlib import Path

import numpy as np
import pytest

from app.trace_store.reader import TraceStoreSectionReader

KEY1 = 189
KEY2 = 193


def _write_store(
    tmp_path: Path,
    *,
    key1s: np.ndarray,
    key2s: np.ndarray,
    n_samples: int = 4,
    traces: np.ndarray | None = None,
    include_index: bool = True,
    index_values: np.ndarray | None = None,
    index_offsets: np.ndarray | None = None,
    index_counts: np.ndarray | None = None,
    meta_key_bytes: tuple[int, int] | None = (KEY1, KEY2),
    store_name: str = 'store',
) -> tuple[Path, np.ndarray]:
    store = tmp_path / store_name
    store.mkdir(parents=True, exist_ok=True)

    n_traces = int(key1s.size)
    if traces is None:
        traces = np.arange(n_traces * n_samples, dtype=np.float32).reshape(
            n_traces, n_samples
        )

    np.save(store / 'traces.npy', np.asarray(traces, dtype=np.float32))
    np.save(store / f'headers_byte_{KEY1}.npy', key1s.astype(np.int32, copy=False))
    np.save(store / f'headers_byte_{KEY2}.npy', key2s.astype(np.int32, copy=False))

    if include_index:
        if index_values is None or index_offsets is None or index_counts is None:
            index_values, index_offsets, index_counts = np.unique(
                key1s,
                return_index=True,
                return_counts=True,
            )
        np.savez(
            store / 'index.npz',
            key1_values=np.asarray(index_values),
            key1_offsets=np.asarray(index_offsets),
            key1_counts=np.asarray(index_counts),
            sorted_to_original=np.arange(n_traces, dtype=np.int64),
        )

    meta = {
        'dt': 0.004,
        'original_segy_path': 'dummy.sgy',
        'original_mtime': 0.0,
        'original_size': 0,
    }
    if meta_key_bytes is not None:
        meta['key_bytes'] = {
            'key1': int(meta_key_bytes[0]),
            'key2': int(meta_key_bytes[1]),
        }
    (store / 'meta.json').write_text(json.dumps(meta))
    return store, np.asarray(traces, dtype=np.float32)


def _legacy_display_indices(
    key1s: np.ndarray,
    key2s: np.ndarray,
    key1_val: int,
) -> np.ndarray:
    indices = np.flatnonzero(key1s == key1_val).astype(np.int64)
    return indices[np.argsort(key2s[indices], kind='stable')]


def test_get_key1_values_uses_index_without_loading_headers(tmp_path: Path):
    key1s = np.array([10, 10, 10, 20, 20, 30], dtype=np.int32)
    key2s = np.array([1, 2, 3, 1, 4, 2], dtype=np.int32)
    store, _ = _write_store(tmp_path, key1s=key1s, key2s=key2s)
    reader = TraceStoreSectionReader(store, KEY1, KEY2)

    def _raise(byte: int) -> np.ndarray:
        raise AssertionError(f'get_header({byte}) should not be called')

    reader.get_header = _raise  # type: ignore[method-assign]

    np.testing.assert_array_equal(reader.get_key1_values(), np.array([10, 20, 30]))


def test_get_section_uses_index_slice_without_loading_headers(tmp_path: Path):
    key1s = np.array([10, 10, 10, 20, 20, 30], dtype=np.int32)
    key2s = np.array([1, 2, 3, 1, 4, 2], dtype=np.int32)
    traces = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    store, traces = _write_store(tmp_path, key1s=key1s, key2s=key2s, traces=traces)
    reader = TraceStoreSectionReader(store, KEY1, KEY2)

    def _raise(byte: int) -> np.ndarray:
        raise AssertionError(f'get_header({byte}) should not be called')

    reader.get_header = _raise  # type: ignore[method-assign]

    view = reader.get_section(20)
    np.testing.assert_array_equal(view.arr, traces[3:5])
    assert np.shares_memory(view.arr, reader.traces)
    np.testing.assert_array_equal(
        reader.get_trace_seq_for_value(20, align_to='display'),
        np.array([3, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        reader.get_trace_seq_for_value(20, align_to='original'),
        np.array([3, 4], dtype=np.int64),
    )


def test_index_path_matches_fallback_results_for_sorted_store(tmp_path: Path):
    key1s = np.array([10, 10, 10, 20, 20, 30], dtype=np.int32)
    key2s = np.array([1, 2, 3, 1, 4, 2], dtype=np.int32)
    traces = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    index_store, _ = _write_store(
        tmp_path,
        key1s=key1s,
        key2s=key2s,
        traces=traces,
        store_name='with_index',
    )
    fallback_store, _ = _write_store(
        tmp_path,
        key1s=key1s,
        key2s=key2s,
        traces=traces,
        include_index=False,
        store_name='without_index',
    )
    reader = TraceStoreSectionReader(index_store, KEY1, KEY2)
    fallback_reader = TraceStoreSectionReader(fallback_store, KEY1, KEY2)

    np.testing.assert_array_equal(
        reader.get_key1_values(),
        fallback_reader.get_key1_values(),
    )
    for key1_val in (10, 20, 30):
        np.testing.assert_array_equal(
            reader.get_trace_seq_for_value(key1_val, align_to='display'),
            fallback_reader.get_trace_seq_for_value(key1_val, align_to='display'),
        )
        np.testing.assert_array_equal(
            reader.get_section(key1_val).arr,
            fallback_reader.get_section(key1_val).arr,
        )


def test_missing_index_falls_back_to_header_scan(tmp_path: Path):
    key1s = np.array([10, 20, 10, 30, 20, 10], dtype=np.int32)
    key2s = np.array([3, 1, 1, 4, 2, 2], dtype=np.int32)
    store, traces = _write_store(
        tmp_path,
        key1s=key1s,
        key2s=key2s,
        include_index=False,
    )
    reader = TraceStoreSectionReader(store, KEY1, KEY2)
    expected = _legacy_display_indices(key1s, key2s, 10)

    np.testing.assert_array_equal(reader.get_key1_values(), np.unique(key1s))
    np.testing.assert_array_equal(
        reader.get_trace_seq_for_value(10, align_to='display'),
        expected,
    )
    np.testing.assert_array_equal(reader.get_section(10).arr, traces[expected])


def test_invalid_index_arrays_fall_back_to_header_scan(tmp_path: Path):
    key1s = np.array([10, 20, 10, 30, 20, 10], dtype=np.int32)
    key2s = np.array([3, 1, 1, 4, 2, 2], dtype=np.int32)
    store, traces = _write_store(
        tmp_path,
        key1s=key1s,
        key2s=key2s,
        index_values=np.array([10, 20], dtype=np.int32),
        index_offsets=np.array([0], dtype=np.int64),
        index_counts=np.array([3, 2], dtype=np.int64),
    )
    reader = TraceStoreSectionReader(store, KEY1, KEY2)
    expected = _legacy_display_indices(key1s, key2s, 10)

    np.testing.assert_array_equal(reader.get_key1_values(), np.unique(key1s))
    np.testing.assert_array_equal(
        reader.get_trace_seq_for_value(10, align_to='display'),
        expected,
    )
    np.testing.assert_array_equal(reader.get_section(10).arr, traces[expected])


@pytest.mark.parametrize(
    ('index_offsets', 'index_counts'),
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([2, 2], dtype=np.int64),
        ),
        (
            np.array([0, 3], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
        ),
    ],
)
def test_malformed_index_ranges_fall_back_to_header_scan(
    tmp_path: Path,
    index_offsets: np.ndarray,
    index_counts: np.ndarray,
):
    key1s = np.array([10, 20, 10, 20], dtype=np.int32)
    key2s = np.array([2, 1, 1, 2], dtype=np.int32)
    store, traces = _write_store(
        tmp_path,
        key1s=key1s,
        key2s=key2s,
        index_values=np.array([10, 20], dtype=np.int32),
        index_offsets=index_offsets,
        index_counts=index_counts,
    )
    reader = TraceStoreSectionReader(store, KEY1, KEY2)
    expected = _legacy_display_indices(key1s, key2s, 20)

    np.testing.assert_array_equal(reader.get_key1_values(), np.unique(key1s))
    np.testing.assert_array_equal(
        reader.get_trace_seq_for_value(20, align_to='display'),
        expected,
    )
    np.testing.assert_array_equal(reader.get_section(20).arr, traces[expected])


def test_mismatched_ingest_key_bytes_fall_back_to_header_scan(tmp_path: Path):
    key1s = np.array([10, 20, 10, 30, 20, 10], dtype=np.int32)
    key2s = np.array([3, 1, 1, 4, 2, 2], dtype=np.int32)
    store, traces = _write_store(
        tmp_path,
        key1s=key1s,
        key2s=key2s,
        index_values=np.array([999], dtype=np.int32),
        index_offsets=np.array([0], dtype=np.int64),
        index_counts=np.array([key1s.size], dtype=np.int64),
        meta_key_bytes=(17, KEY2),
    )
    reader = TraceStoreSectionReader(store, KEY1, KEY2)
    expected = _legacy_display_indices(key1s, key2s, 10)

    np.testing.assert_array_equal(reader.get_key1_values(), np.unique(key1s))
    np.testing.assert_array_equal(
        reader.get_trace_seq_for_value(10, align_to='display'),
        expected,
    )
    np.testing.assert_array_equal(reader.get_section(10).arr, traces[expected])


def test_trace_seq_raises_for_missing_key1(tmp_path: Path):
    key1s = np.array([1, 1, 2, 3], dtype=np.int32)
    key2s = np.array([0, 1, 2, 3], dtype=np.int32)
    store, _ = _write_store(tmp_path, key1s=key1s, key2s=key2s, include_index=False)
    reader = TraceStoreSectionReader(store, KEY1, KEY2)

    with pytest.raises(ValueError):
        reader.get_trace_seq_for_value(999999, align_to='display')


def test_get_n_samples(tmp_path: Path):
    key1s = np.array([1, 1, 2], dtype=np.int32)
    key2s = np.array([0, 1, 2], dtype=np.int32)
    store, _ = _write_store(
        tmp_path,
        key1s=key1s,
        key2s=key2s,
        n_samples=7,
        include_index=False,
    )
    reader = TraceStoreSectionReader(store, KEY1, KEY2)
    assert reader.get_n_samples() == 7
