from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.trace_store.reader import TraceStoreSectionReader

KEY1 = 189
KEY2 = 193
HEADER_BYTE = 37


class _Attr:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values)

    def __getitem__(self, _key):
        return self._values


class _FakeSegy:
    def __init__(self, headers: dict[int, np.ndarray]) -> None:
        self._headers = {int(key): np.asarray(value) for key, value in headers.items()}

    def mmap(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def attributes(self, byte: int):
        return _Attr(self._headers[int(byte)])


def _patch_segyio_header(monkeypatch, headers: dict[int, np.ndarray]) -> None:
    import segyio

    def _open_stub(_path, _mode='r', ignore_geometry=True):
        return _FakeSegy(headers)

    monkeypatch.setattr(segyio, 'open', _open_stub, raising=True)


def _write_store(
    store: Path,
    *,
    sorted_to_original: np.ndarray | None = None,
    include_sorted_to_original: bool = True,
    original_segy_path: str | None = None,
    derived: dict | None = None,
    header_values: np.ndarray | None = None,
    n_traces: int | None = None,
) -> None:
    store.mkdir(parents=True, exist_ok=True)
    if sorted_to_original is None:
        if n_traces is None:
            n_traces = 3
        sorted_to_original = np.arange(n_traces, dtype=np.int64)
    else:
        sorted_to_original = np.asarray(sorted_to_original)
        if n_traces is None:
            n_traces = int(sorted_to_original.shape[0])

    np.save(store / 'traces.npy', np.zeros((n_traces, 4), dtype=np.float32))
    np.save(store / f'headers_byte_{KEY1}.npy', np.full(n_traces, 10, dtype=np.int32))
    np.save(store / f'headers_byte_{KEY2}.npy', np.arange(n_traces, dtype=np.int32))
    if header_values is not None:
        np.save(store / f'headers_byte_{HEADER_BYTE}.npy', np.asarray(header_values))

    index_values = {
        'key1_values': np.array([10], dtype=np.int32),
        'key1_offsets': np.array([0], dtype=np.int64),
        'key1_counts': np.array([n_traces], dtype=np.int64),
    }
    if include_sorted_to_original:
        index_values['sorted_to_original'] = sorted_to_original
    np.savez(store / 'index.npz', **index_values)

    meta: dict = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': n_traces,
        'n_samples': 4,
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'sorted_by': ['key1', 'key2'],
        'dt': 0.004,
        'original_segy_path': original_segy_path,
        'source_sha256': None,
    }
    if derived is not None:
        meta['derived'] = derived
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')


def _reader(store: Path) -> TraceStoreSectionReader:
    return TraceStoreSectionReader(store, key1_byte=KEY1, key2_byte=KEY2)


def test_derived_reader_materializes_missing_header_from_header_source_store_path(
    tmp_path: Path,
) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    sorted_to_original = np.array([2, 0, 1], dtype=np.int64)
    parent_header_sorted = np.array([900, 100, 500], dtype=np.int32)
    _write_store(
        parent,
        sorted_to_original=sorted_to_original,
        header_values=parent_header_sorted,
    )
    _write_store(
        child,
        sorted_to_original=sorted_to_original,
        derived={
            'kind': 'time_shifted_trace_store',
            'from_store_path': str(parent),
            'header_source_store_path': str(parent),
        },
    )

    got = _reader(child).ensure_header(HEADER_BYTE)

    np.testing.assert_array_equal(got, parent_header_sorted)
    assert (child / f'headers_byte_{HEADER_BYTE}.npy').exists()


def test_derived_reader_materializes_missing_header_from_from_store_path_when_header_source_missing(
    tmp_path: Path,
) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    parent_header = np.array([11, 22, 33], dtype=np.int32)
    _write_store(parent, header_values=parent_header)
    _write_store(
        child,
        derived={
            'kind': 'time_shifted_trace_store',
            'from_store_path': str(parent),
        },
    )

    got = _reader(child).ensure_header(HEADER_BYTE)

    np.testing.assert_array_equal(got, parent_header)


def test_derived_reader_prefers_local_header_without_touching_parent(
    tmp_path: Path,
) -> None:
    child = tmp_path / 'child'
    local_header = np.array([7, 8, 9], dtype=np.int32)
    _write_store(
        child,
        derived={'header_source_store_path': str(tmp_path / 'missing-parent')},
        header_values=local_header,
    )

    got = _reader(child).ensure_header(HEADER_BYTE)

    np.testing.assert_array_equal(got, local_header)


def test_derived_reader_does_not_apply_sorted_to_original_to_parent_header(
    tmp_path: Path,
) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    sorted_to_original = np.array([2, 0, 1], dtype=np.int64)
    parent_header_sorted = np.array([20, 30, 10], dtype=np.int32)
    _write_store(
        parent,
        sorted_to_original=sorted_to_original,
        header_values=parent_header_sorted,
    )
    _write_store(
        child,
        sorted_to_original=sorted_to_original,
        derived={'header_source_store_path': str(parent)},
    )

    got = _reader(child).ensure_header(HEADER_BYTE)

    np.testing.assert_array_equal(got, parent_header_sorted)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        got,
        parent_header_sorted[sorted_to_original],
    )


def test_derived_reader_writes_materialized_header_as_int32_mmap(
    tmp_path: Path,
) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    _write_store(parent, header_values=np.array([1, 2, 3], dtype=np.int64))
    _write_store(child, derived={'header_source_store_path': str(parent)})

    got = _reader(child).ensure_header(HEADER_BYTE)

    assert isinstance(got, np.memmap)
    assert got.dtype == np.int32
    assert got.shape == (3,)
    assert got.flags.c_contiguous
    assert not list(child.glob(f'headers_byte_{HEADER_BYTE}.npy.*.tmp'))


def test_derived_reader_rejects_relative_header_source_path(tmp_path: Path) -> None:
    child = tmp_path / 'child'
    _write_store(child, derived={'header_source_store_path': 'relative/store'})

    with pytest.raises(ValueError, match='absolute'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_derived_reader_rejects_missing_header_source_store(tmp_path: Path) -> None:
    child = tmp_path / 'child'
    _write_store(
        child,
        derived={'header_source_store_path': str(tmp_path / 'missing-parent')},
    )

    with pytest.raises(ValueError, match='does not exist'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_derived_reader_rejects_self_header_source_store(tmp_path: Path) -> None:
    child = tmp_path / 'child'
    _write_store(child, derived={'header_source_store_path': str(child)})

    with pytest.raises(ValueError, match='target store'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_derived_reader_rejects_missing_sorted_to_original_in_child_index(
    tmp_path: Path,
) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    _write_store(parent, header_values=np.array([1, 2, 3], dtype=np.int32))
    _write_store(
        child,
        include_sorted_to_original=False,
        derived={'header_source_store_path': str(parent)},
    )

    with pytest.raises(ValueError, match='target.*sorted_to_original'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_derived_reader_rejects_missing_sorted_to_original_in_parent_index(
    tmp_path: Path,
) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    _write_store(
        parent,
        include_sorted_to_original=False,
        header_values=np.array([1, 2, 3], dtype=np.int32),
    )
    _write_store(child, derived={'header_source_store_path': str(parent)})

    with pytest.raises(ValueError, match='source.*sorted_to_original'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_derived_reader_rejects_sorted_to_original_mismatch(tmp_path: Path) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    _write_store(
        parent,
        sorted_to_original=np.array([0, 2, 1], dtype=np.int64),
        header_values=np.array([1, 2, 3], dtype=np.int32),
    )
    _write_store(
        child,
        sorted_to_original=np.array([0, 1, 2], dtype=np.int64),
        derived={'header_source_store_path': str(parent)},
    )

    with pytest.raises(ValueError, match='do not match'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_derived_reader_rejects_parent_header_shape_mismatch(tmp_path: Path) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    _write_store(parent, header_values=np.array([1, 2], dtype=np.int32))
    _write_store(child, derived={'header_source_store_path': str(parent)})

    with pytest.raises(ValueError, match='shape mismatch'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_derived_reader_rejects_parent_header_non_integer_dtype(
    tmp_path: Path,
) -> None:
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    _write_store(parent, header_values=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    _write_store(child, derived={'header_source_store_path': str(parent)})

    with pytest.raises(ValueError, match='integer dtype'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_derived_reader_rejects_when_no_parent_and_no_original_segy(
    tmp_path: Path,
) -> None:
    child = tmp_path / 'child'
    _write_store(child, original_segy_path=None)

    with pytest.raises(ValueError, match='No usable header source'):
        _reader(child).ensure_header(HEADER_BYTE)


def test_raw_reader_keeps_existing_segy_header_materialization_behavior(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = tmp_path / 'raw'
    segy_path = tmp_path / 'raw.sgy'
    segy_path.write_bytes(b'stub')
    sorted_to_original = np.array([2, 0, 1], dtype=np.int64)
    raw_header_original_order = np.array([100, 500, 900], dtype=np.int64)
    _write_store(
        store,
        sorted_to_original=sorted_to_original,
        original_segy_path=str(segy_path),
    )
    _patch_segyio_header(monkeypatch, {HEADER_BYTE: raw_header_original_order})

    got = _reader(store).ensure_header(HEADER_BYTE)

    np.testing.assert_array_equal(got, raw_header_original_order[sorted_to_original])
    assert got.dtype == np.int32
    assert (store / f'headers_byte_{HEADER_BYTE}.npy').exists()
