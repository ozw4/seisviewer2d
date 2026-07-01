from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import raw_compare_validation


class _ValidationReader:
    def __init__(
        self,
        key2_by_key1: Mapping[int, Sequence[int]],
        *,
        n_samples: int = 1500,
    ) -> None:
        self._key1_values = np.asarray(list(key2_by_key1), dtype=np.int64)
        self._n_samples = int(n_samples)
        self._seq_by_key1: dict[int, np.ndarray] = {}
        chunks: list[np.ndarray] = []
        offset = 0
        for key1, key2_values in key2_by_key1.items():
            key2_arr = np.asarray(key2_values, dtype=np.int64)
            chunks.append(key2_arr)
            self._seq_by_key1[int(key1)] = np.arange(
                offset,
                offset + key2_arr.size,
                dtype=np.int64,
            )
            offset += int(key2_arr.size)
        self._key2_header = (
            np.concatenate(chunks)
            if chunks
            else np.asarray([], dtype=np.int64)
        )

    def get_key1_values(self) -> np.ndarray:
        return np.array(self._key1_values, copy=True)

    def get_n_samples(self) -> int:
        return self._n_samples

    def get_header(self, _byte: int) -> np.ndarray:
        return np.array(self._key2_header, copy=True)

    def get_trace_seq_for_value(
        self,
        key1_val: int,
        align_to: str = 'display',
    ) -> np.ndarray:
        if align_to != 'display':
            raise ValueError("align_to must be 'display'")
        return np.array(self._seq_by_key1[int(key1_val)], copy=True)


@pytest.fixture()
def client() -> TestClient:
    state = app.state.sv
    state.file_registry.clear()
    state.cached_readers.clear()
    with TestClient(app) as test_client:
        yield test_client
    state.file_registry.clear()
    state.cached_readers.clear()


def _install_readers(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    readers: Mapping[str, _ValidationReader],
    *,
    dts: Mapping[str, float] | None = None,
    tmp_path: Path,
) -> None:
    dt_by_file = dict(dts or {})
    state = client.app.state.sv
    for file_id in readers:
        state.file_registry.set_record(
            file_id,
            {
                'path': str(tmp_path / f'{file_id}.sgy'),
                'store_path': str(tmp_path / f'{file_id}.store'),
                'dt': dt_by_file.get(file_id, 0.002),
            },
        )

    def _get_reader(
        file_id: str,
        _key1_byte: int,
        _key2_byte: int,
        *,
        state: Any,
    ) -> _ValidationReader:
        del state
        return readers[file_id]

    monkeypatch.setattr(
        raw_compare_validation,
        'get_reader',
        _get_reader,
        raising=True,
    )


def _validate(client: TestClient) -> dict[str, Any]:
    response = client.get(
        '/compare/raw/validate',
        params={
            'file_id_a': 'line_a',
            'file_id_b': 'line_b',
            'key1_byte': 189,
            'key2_byte': 193,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_compare_raw_validate_route_registered_once() -> None:
    def _collect_paths(routes: Sequence[Any]) -> list[str]:
        paths: list[str] = []
        for route in routes:
            path = getattr(route, 'path', None)
            if isinstance(path, str):
                paths.append(path)
            original_router = getattr(route, 'original_router', None)
            if original_router is not None:
                paths.extend(_collect_paths(original_router.routes))
        return paths

    paths = _collect_paths(app.routes)

    assert paths.count('/compare/raw/validate') == 1


def test_compare_raw_validation_matching_readers_ok(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': _ValidationReader({10: [100, 101], 20: [200, 201, 202]}),
        'line_b': _ValidationReader({10: [100, 101], 20: [200, 201, 202]}),
    }
    _install_readers(monkeypatch, client, readers, tmp_path=tmp_path)

    body = _validate(client)

    assert body['ok'] is True
    assert body['reason'] == ''
    assert body['checked_key1_count'] == 2
    assert body['files'] == [
        {
            'role': 'A',
            'file_id': 'line_a',
            'file_name': 'line_a.sgy',
            'key1_count': 2,
            'n_samples': 1500,
            'dt': 0.002,
        },
        {
            'role': 'B',
            'file_id': 'line_b',
            'file_name': 'line_b.sgy',
            'key1_count': 2,
            'n_samples': 1500,
            'dt': 0.002,
        },
    ]
    assert 'key1_values' not in body
    assert 'key2_values' not in body


def test_compare_raw_validation_key1_values_mismatch(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': _ValidationReader({10: [100, 101], 20: [200, 201]}),
        'line_b': _ValidationReader({10: [100, 101], 30: [200, 201]}),
    }
    _install_readers(monkeypatch, client, readers, tmp_path=tmp_path)

    body = _validate(client)

    assert body['ok'] is False
    assert body['reason'] == 'key1_values'
    assert body['checked_key1_count'] == 0
    assert body['mismatch']['type'] == 'key1_values'
    assert body['mismatch']['index'] == 1
    assert body['files'] == []


def test_compare_raw_validation_n_samples_mismatch(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': _ValidationReader({10: [100, 101]}, n_samples=1500),
        'line_b': _ValidationReader({10: [100, 101]}, n_samples=1501),
    }
    _install_readers(monkeypatch, client, readers, tmp_path=tmp_path)

    body = _validate(client)

    assert body['ok'] is False
    assert body['reason'] == 'n_samples'
    assert body['mismatch'] == {
        'type': 'n_samples',
        'a_n_samples': 1500,
        'b_n_samples': 1501,
    }


def test_compare_raw_validation_dt_mismatch(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': _ValidationReader({10: [100, 101]}),
        'line_b': _ValidationReader({10: [100, 101]}),
    }
    _install_readers(
        monkeypatch,
        client,
        readers,
        dts={'line_a': 0.002, 'line_b': 0.002000002},
        tmp_path=tmp_path,
    )

    body = _validate(client)

    assert body['ok'] is False
    assert body['reason'] == 'dt'
    assert body['mismatch']['type'] == 'dt'
    assert body['mismatch']['a_dt'] == 0.002
    assert body['mismatch']['b_dt'] == 0.002000002


def test_compare_raw_validation_key2_order_mismatch(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': _ValidationReader({10: [100, 101], 20: [200, 201]}),
        'line_b': _ValidationReader({10: [101, 100], 20: [200, 201]}),
    }
    _install_readers(monkeypatch, client, readers, tmp_path=tmp_path)

    body = _validate(client)

    assert body['ok'] is False
    assert body['reason'] == 'key2_sequence'
    assert body['checked_key1_count'] == 1
    assert body['mismatch'] == {
        'type': 'key2_sequence',
        'key1': 10,
        'a_count': 2,
        'b_count': 2,
    }
    assert 'key2_values' not in body['mismatch']
