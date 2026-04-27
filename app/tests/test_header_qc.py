from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.utils import header_qc
from app.utils.header_qc import HEADER_CANDIDATES, inspect_segy_header_qc


class _Attr:
    def __init__(
        self,
        values: np.ndarray,
        *,
        byte: int,
        tracecount: int,
        access_log: list[dict] | None = None,
        reject_full_slice: bool = False,
    ) -> None:
        self._values = np.asarray(values)
        self._byte = int(byte)
        self._tracecount = int(tracecount)
        self._access_log = access_log
        self._reject_full_slice = reject_full_slice

    def __getitem__(self, key):
        is_full_slice = (
            isinstance(key, slice)
            and key.start is None
            and key.stop is None
            and key.step is None
        )
        if self._reject_full_slice and is_full_slice:
            raise AssertionError('full header slice was requested')
        if isinstance(key, slice):
            length = len(range(*key.indices(self._tracecount)))
        else:
            length = int(np.asarray(key).size)
        if self._access_log is not None:
            self._access_log.append(
                {
                    'byte': self._byte,
                    'length': length,
                    'full_slice': is_full_slice,
                }
            )
        return self._values[key]


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
        tracecount: int | None = None,
        access_log: list[dict] | None = None,
        reject_full_slice: bool = False,
    ) -> None:
        self._headers = {int(k): np.asarray(v) for k, v in headers.items()}
        if tracecount is None:
            first = next(iter(self._headers.values()))
            tracecount = int(first.size)
        self.tracecount = int(tracecount)
        self.samples = np.arange(n_samples, dtype=np.int32)
        self.bin = _FakeBin(interval_us)
        self.access_log = access_log if access_log is not None else []
        self.reject_full_slice = reject_full_slice

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def attributes(self, byte: int):
        values = self._headers.get(int(byte))
        if values is None:
            values = np.zeros(self.tracecount, dtype=np.int32)
        return _Attr(
            values,
            byte=int(byte),
            tracecount=self.tracecount,
            access_log=self.access_log,
            reject_full_slice=self.reject_full_slice,
        )


def _patch_segyio(
    monkeypatch,
    *,
    headers: dict[int, np.ndarray],
    n_samples: int = 1500,
    interval_us: int | None = 2000,
    tracecount: int | None = None,
    reject_full_slice: bool = False,
) -> dict:
    import app.utils.header_qc as header_qc

    opened: dict[str, _FakeSegy] = {}

    def _open_stub(_path, _mode='r', ignore_geometry=True):
        assert ignore_geometry is True
        fake = _FakeSegy(
            headers=headers,
            n_samples=n_samples,
            interval_us=interval_us,
            tracecount=tracecount,
            reject_full_slice=reject_full_slice,
        )
        opened['fake'] = fake
        return fake

    monkeypatch.setattr(header_qc.segyio, 'open', _open_stub, raising=True)

    class _BinField:
        Interval = 'Interval'

    monkeypatch.setattr(header_qc.segyio, 'BinField', _BinField, raising=True)
    return opened


def _find_header(result: dict, byte: int) -> dict:
    return next(item for item in result['headers'] if item['byte'] == byte)


def _find_pair(result: dict, key1_byte: int, key2_byte: int) -> dict:
    return next(
        item
        for item in result['recommended_pairs']
        if item['key1_byte'] == key1_byte and item['key2_byte'] == key2_byte
    )


def _all_candidate_headers(n_sections: int = 10, section_size: int = 20) -> dict[int, np.ndarray]:
    n_traces = n_sections * section_size
    headers: dict[int, np.ndarray] = {}
    for idx, (byte, _name) in enumerate(HEADER_CANDIDATES):
        offset = idx * 1_000
        if idx % 2 == 0:
            values = np.repeat(np.arange(n_sections), section_size)
        else:
            values = np.tile(np.arange(section_size), n_sections)
        headers[byte] = (values + offset).astype(np.int32)
    assert all(values.size == n_traces for values in headers.values())
    return headers


def test_header_candidates_are_reduced_to_nine_key_bytes():
    assert HEADER_CANDIDATES == [
        (1, "TRACE_SEQUENCE_LINE"),
        (5, "TRACE_SEQUENCE_FILE"),
        (9, "FIELD_RECORD"),
        (13, "TRACE_NUMBER"),
        (21, "CDP"),
        (25, "CDP_TRACE"),
        (37, "OFFSET"),
        (189, "INLINE_3D"),
        (193, "CROSSLINE_3D"),
    ]


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


def test_excluded_metadata_headers_are_not_scored(monkeypatch, tmp_path: Path):
    headers = _all_candidate_headers()
    _patch_segyio(monkeypatch, headers=headers)

    result = inspect_segy_header_qc(tmp_path / 'line.sgy')

    excluded = {115, 117, 197}
    assert {item['byte'] for item in result['headers']} == {
        byte for byte, _name in HEADER_CANDIDATES
    }
    assert not excluded & {item['byte'] for item in result['headers']}
    assert len(result['recommended_pairs']) == 72
    for pair in result['recommended_pairs']:
        assert pair['key1_byte'] not in excluded
        assert pair['key2_byte'] not in excluded


def test_cdp_headers_rank_as_useful_pair(monkeypatch, tmp_path: Path):
    cdp = np.repeat(np.arange(12), 16)
    cdp_trace = np.tile(np.arange(16), 12)
    _patch_segyio(
        monkeypatch,
        headers={21: cdp, 25: cdp_trace},
    )

    result = inspect_segy_header_qc(tmp_path / 'cdp.sgy')
    best = result['recommended_pairs'][0]

    assert best['key1_byte'] == 21
    assert best['key2_byte'] == 25
    assert best['score'] >= 0.75


def test_field_record_trace_number_headers_rank_as_useful_pair(
    monkeypatch,
    tmp_path: Path,
):
    field_record = np.repeat(np.arange(8), 24)
    trace_number = np.tile(np.arange(24), 8)
    _patch_segyio(
        monkeypatch,
        headers={9: field_record, 13: trace_number},
    )

    result = inspect_segy_header_qc(tmp_path / 'records.sgy')
    best = result['recommended_pairs'][0]

    assert best['key1_byte'] == 9
    assert best['key2_byte'] == 13
    assert best['score'] >= 0.75


def test_trace_sequence_and_offset_can_participate_in_scoring(
    monkeypatch,
    tmp_path: Path,
):
    sequence_line = np.repeat(np.arange(10), 20)
    offset = np.tile(np.arange(20) * 25, 10)
    _patch_segyio(
        monkeypatch,
        headers={1: sequence_line, 37: offset},
    )

    result = inspect_segy_header_qc(tmp_path / 'offsets.sgy')
    pair = _find_pair(result, 1, 37)

    assert pair['score'] >= 0.55


def test_dt_fallback_reads_metadata_header_without_scoring_it(
    monkeypatch,
    tmp_path: Path,
):
    inline = np.repeat(np.arange(10), 20)
    crossline = np.tile(np.arange(20), 10)
    dt = np.full(200, 4000)
    _patch_segyio(
        monkeypatch,
        headers={189: inline, 193: crossline, 117: dt},
        interval_us=None,
    )

    result = inspect_segy_header_qc(tmp_path / 'line.sgy')

    assert result['segy']['dt'] == pytest.approx(0.004)
    assert 117 not in {item['byte'] for item in result['headers']}
    for pair in result['recommended_pairs']:
        assert pair['key1_byte'] != 117
        assert pair['key2_byte'] != 117


def test_large_file_header_reads_are_bounded_to_sample(
    monkeypatch,
    tmp_path: Path,
):
    n_traces = header_qc._MAX_QC_TRACES + 123
    headers = {
        byte: np.arange(n_traces, dtype=np.int32)
        for byte, _name in HEADER_CANDIDATES
    }
    opened = _patch_segyio(
        monkeypatch,
        headers=headers,
        tracecount=n_traces,
        reject_full_slice=True,
    )

    inspect_segy_header_qc(tmp_path / 'large.sgy')

    fake = opened['fake']
    assert len(fake.access_log) == len(HEADER_CANDIDATES)
    assert {entry['byte'] for entry in fake.access_log} == {
        byte for byte, _name in HEADER_CANDIDATES
    }
    assert all(entry['length'] <= header_qc._MAX_QC_TRACES for entry in fake.access_log)
    assert not any(entry['full_slice'] for entry in fake.access_log)


def test_pair_ranking_reuses_key1_grouping_per_candidate(
    monkeypatch,
    tmp_path: Path,
):
    _patch_segyio(monkeypatch, headers=_all_candidate_headers())
    calls = 0
    original = header_qc._build_key1_grouping

    def _spy_build_key1_grouping(values: np.ndarray):
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(
        header_qc,
        '_build_key1_grouping',
        _spy_build_key1_grouping,
        raising=True,
    )

    result = inspect_segy_header_qc(tmp_path / 'line.sgy')

    assert len(result['recommended_pairs']) == 72
    assert calls == len(HEADER_CANDIDATES)


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
