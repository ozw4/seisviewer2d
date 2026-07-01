from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.services import raw_compare_validation
from app.tests._raw_compare_validation_stubs import DummyState, ValidationReader


def _install_readers(
    monkeypatch: pytest.MonkeyPatch,
    readers: Mapping[str, ValidationReader],
) -> None:
    def _get_reader(
        file_id: str,
        _key1_byte: int,
        _key2_byte: int,
        *,
        state: Any,
    ) -> ValidationReader:
        del state
        return readers[file_id]

    monkeypatch.setattr(
        raw_compare_validation,
        'get_reader',
        _get_reader,
        raising=True,
    )


def _validate(
    readers: Mapping[str, ValidationReader],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    dts: Mapping[str, float] | None = None,
) -> dict[str, object]:
    _install_readers(monkeypatch, readers)
    state = DummyState(tuple(readers), dts=dts, base_path=tmp_path)

    return raw_compare_validation.validate_raw_compare_grid(
        file_id_a='line_a',
        file_id_b='line_b',
        key1_byte=189,
        key2_byte=193,
        state=state,  # type: ignore[arg-type]
    )


def test_raw_compare_validation_matching_readers_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': ValidationReader({10: [100, 101], 20: [200, 201, 202]}),
        'line_b': ValidationReader({10: [100, 101], 20: [200, 201, 202]}),
    }

    body = _validate(readers, monkeypatch, tmp_path)

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


def test_raw_compare_validation_key1_values_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': ValidationReader({10: [100, 101], 20: [200, 201]}),
        'line_b': ValidationReader({10: [100, 101], 30: [200, 201]}),
    }

    body = _validate(readers, monkeypatch, tmp_path)

    assert body['ok'] is False
    assert body['reason'] == 'key1_values'
    assert body['checked_key1_count'] == 0
    assert body['mismatch']['type'] == 'key1_values'  # type: ignore[index]
    assert body['mismatch']['index'] == 1  # type: ignore[index]
    assert body['files'] == []


def test_raw_compare_validation_n_samples_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': ValidationReader({10: [100, 101]}, n_samples=1500),
        'line_b': ValidationReader({10: [100, 101]}, n_samples=1501),
    }

    body = _validate(readers, monkeypatch, tmp_path)

    assert body['ok'] is False
    assert body['reason'] == 'n_samples'
    assert body['mismatch'] == {
        'type': 'n_samples',
        'a_n_samples': 1500,
        'b_n_samples': 1501,
    }


def test_raw_compare_validation_dt_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': ValidationReader({10: [100, 101]}),
        'line_b': ValidationReader({10: [100, 101]}),
    }

    body = _validate(
        readers,
        monkeypatch,
        tmp_path,
        dts={'line_a': 0.002, 'line_b': 0.002000002},
    )

    assert body['ok'] is False
    assert body['reason'] == 'dt'
    assert body['mismatch']['type'] == 'dt'  # type: ignore[index]
    assert body['mismatch']['a_dt'] == 0.002  # type: ignore[index]
    assert body['mismatch']['b_dt'] == 0.002000002  # type: ignore[index]


def test_raw_compare_validation_key2_order_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readers = {
        'line_a': ValidationReader({10: [100, 101], 20: [200, 201]}),
        'line_b': ValidationReader({10: [101, 100], 20: [200, 201]}),
    }

    body = _validate(readers, monkeypatch, tmp_path)

    assert body['ok'] is False
    assert body['reason'] == 'key2_sequence'
    assert body['checked_key1_count'] == 1
    assert body['mismatch'] == {
        'type': 'key2_sequence',
        'key1': 10,
        'a_count': 2,
        'b_count': 2,
    }
    assert 'key2_values' not in body['mismatch']  # type: ignore[operator]
