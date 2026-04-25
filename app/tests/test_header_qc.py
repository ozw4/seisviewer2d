from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.utils.header_qc import inspect_segy_header_qc


class _Attr:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values)

    def __getitem__(self, _key):
        return self._values


class _FakeBin:
    def __init__(self, interval_us: int | None) -> None:
        self._interval_us = interval_us

    def __getitem__(self, _key):
        if self._interval_us is None:
            raise KeyError(_key)
        return self._interval_us


class _FakeSegy:
    def __init__(
        self,
        *,
        headers: dict[int, np.ndarray],
        n_samples: int,
        interval_us: int | None,
    ) -> None:
        self._headers = {int(k): np.asarray(v) for k, v in headers.items()}
        first = next(iter(self._headers.values()))
        self.tracecount = int(first.size)
        self.samples = np.arange(n_samples, dtype=np.int32)
        self.bin = _FakeBin(interval_us)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def attributes(self, byte: int):
        values = self._headers.get(int(byte))
        if values is None:
            values = np.zeros(self.tracecount, dtype=np.int32)
        return _Attr(values)


def _patch_segyio(
    monkeypatch,
    *,
    headers: dict[int, np.ndarray],
    n_samples: int = 1500,
    interval_us: int | None = 2000,
) -> None:
    import app.utils.header_qc as header_qc

    def _open_stub(_path, _mode='r', ignore_geometry=True):
        assert ignore_geometry is True
        return _FakeSegy(
            headers=headers,
            n_samples=n_samples,
            interval_us=interval_us,
        )

    monkeypatch.setattr(header_qc.segyio, 'open', _open_stub, raising=True)

    class _BinField:
        Interval = 'Interval'

    monkeypatch.setattr(header_qc.segyio, 'BinField', _BinField, raising=True)


def _find_header(result: dict, byte: int) -> dict:
    return next(item for item in result['headers'] if item['byte'] == byte)


def _find_pair(result: dict, key1_byte: int, key2_byte: int) -> dict:
    return next(
        item
        for item in result['recommended_pairs']
        if item['key1_byte'] == key1_byte and item['key2_byte'] == key2_byte
    )


def test_inline_crossline_headers_rank_highly(monkeypatch, tmp_path: Path):
    inline = np.repeat(np.arange(10), 20)
    crossline = np.tile(np.arange(20), 10)
    dt = np.full(200, 2000)
    _patch_segyio(
        monkeypatch,
        headers={189: inline, 193: crossline, 117: dt},
    )

    result = inspect_segy_header_qc(tmp_path / 'line.sgy')

    assert result['segy'] == {
        'n_traces': 200,
        'n_samples': 1500,
        'dt': pytest.approx(0.002),
    }
    best = result['recommended_pairs'][0]
    assert best['key1_byte'] == 189
    assert best['key2_byte'] == 193
    assert best['confidence'] == 'high'
    assert best['score'] > 0.8


def test_constant_header_gets_low_score_and_warning(monkeypatch, tmp_path: Path):
    inline = np.repeat(np.arange(10), 20)
    crossline = np.tile(np.arange(20), 10)
    constant = np.full(200, 7)
    _patch_segyio(
        monkeypatch,
        headers={189: inline, 193: crossline, 21: constant},
    )

    result = inspect_segy_header_qc(tmp_path / 'line.sgy')
    header = _find_header(result, 21)

    assert header['key1_score'] < 0.2
    assert any('constant' in warning for warning in header['warnings'])


def test_all_unique_header_gets_low_key1_score(monkeypatch, tmp_path: Path):
    inline = np.repeat(np.arange(10), 20)
    crossline = np.tile(np.arange(20), 10)
    all_unique = np.arange(200)
    _patch_segyio(
        monkeypatch,
        headers={189: inline, 193: crossline, 1: all_unique},
    )

    result = inspect_segy_header_qc(tmp_path / 'line.sgy')
    header = _find_header(result, 1)

    assert header['key1_score'] < 0.2
    assert any('unique for every trace' in warning for warning in header['warnings'])


def test_bad_key2_duplicate_behavior_warns(monkeypatch, tmp_path: Path):
    inline = np.repeat(np.arange(10), 20)
    duplicated_key2 = np.tile(np.repeat(np.arange(5), 4), 10)
    _patch_segyio(
        monkeypatch,
        headers={189: inline, 193: duplicated_key2},
    )

    result = inspect_segy_header_qc(tmp_path / 'line.sgy')
    pair = _find_pair(result, 189, 193)

    assert pair['confidence'] in {'low', 'medium'}
    assert any('duplicates' in warning for warning in pair['warnings'])


def test_recommended_pairs_are_sorted_by_descending_score(monkeypatch, tmp_path: Path):
    inline = np.repeat(np.arange(10), 20)
    crossline = np.tile(np.arange(20), 10)
    _patch_segyio(
        monkeypatch,
        headers={189: inline, 193: crossline},
    )

    result = inspect_segy_header_qc(tmp_path / 'line.sgy')
    scores = [item['score'] for item in result['recommended_pairs']]

    assert scores == sorted(scores, reverse=True)
