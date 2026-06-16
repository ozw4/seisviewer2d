from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _import_reader_without_segyio(monkeypatch: pytest.MonkeyPatch):
    sys.modules.pop('app.trace_store.reader', None)
    monkeypatch.setitem(sys.modules, 'segyio', None)
    return importlib.import_module('app.trace_store.reader')


def _write_minimal_store(
    store: Path,
    *,
    original_segy_path: str = 'dummy.sgy',
    materialized_header_byte: int | None = 189,
) -> None:
    store.mkdir(parents=True)
    np.save(store / 'traces.npy', np.zeros((3, 4), dtype=np.float32))
    if materialized_header_byte is not None:
        np.save(
            store / f'headers_byte_{materialized_header_byte}.npy',
            np.array([10, 20, 30], dtype=np.int32),
        )
    meta = {
        'dt': 0.004,
        'original_segy_path': original_segy_path,
        'original_mtime': 0.0,
        'original_size': 0,
        'key_bytes': {'key1': 189, 'key2': 193},
    }
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')


def test_trace_store_reader_imports_without_segyio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_reader_without_segyio(monkeypatch)

    assert hasattr(module, 'TraceStoreSectionReader')


def test_reader_gets_existing_header_without_segyio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_reader_without_segyio(monkeypatch)
    store = tmp_path / 'store'
    _write_minimal_store(store)

    reader = module.TraceStoreSectionReader(store)
    got = reader.get_header(189)

    np.testing.assert_array_equal(got, np.array([10, 20, 30], dtype=np.int32))


def test_reader_requires_segyio_only_when_materializing_original_segy_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_reader_without_segyio(monkeypatch)
    segy_path = tmp_path / 'original.sgy'
    segy_path.write_bytes(b'stub')
    store = tmp_path / 'store'
    _write_minimal_store(
        store,
        original_segy_path=str(segy_path),
        materialized_header_byte=None,
    )

    reader = module.TraceStoreSectionReader(store)
    with pytest.raises(
        ModuleNotFoundError,
        match='segyio.*materialize.*original SEG-Y',
    ):
        reader.get_header(37)
