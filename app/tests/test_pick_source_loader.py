from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.state import AppState
from app.services.pick_source_loader import (
    load_manual_memmap_pick_source,
    load_npz_pick_source,
)
from app.trace_store.reader import TraceStoreSectionReader
from app.utils.pick_cache_file1d_mem import open_for_write

KEY1 = 189
KEY2 = 193
DT = 0.002
N_SAMPLES = 5
SORTED_TO_ORIGINAL = np.asarray([2, 0, 3, 1], dtype=np.int64)
PICKS_ORIGINAL = np.asarray([0.002, np.nan, 0.006, 0.008], dtype=np.float32)
PICKS_SORTED = np.asarray([0.006, 0.002, 0.008, np.nan], dtype=np.float64)
_ABSENT = object()


def _write_store(
    tmp_path: Path,
    *,
    sorted_to_original: np.ndarray = SORTED_TO_ORIGINAL,
    n_samples: int = N_SAMPLES,
) -> Path:
    store = tmp_path / 'store'
    store.mkdir(parents=True, exist_ok=True)
    n_traces = int(np.asarray(sorted_to_original).shape[0])
    np.save(store / 'traces.npy', np.zeros((n_traces, n_samples), dtype=np.float32))
    np.savez(
        store / 'index.npz',
        sorted_to_original=np.asarray(sorted_to_original),
    )
    meta = {
        'schema_version': 1,
        'dtype': 'float32',
        'n_traces': n_traces,
        'n_samples': int(n_samples),
        'key_bytes': {'key1': KEY1, 'key2': KEY2},
        'dt': DT,
        'original_segy_path': str(tmp_path / 'line.sgy'),
    }
    (store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return store


def _reader(tmp_path: Path) -> TraceStoreSectionReader:
    return TraceStoreSectionReader(_write_store(tmp_path), KEY1, KEY2)


def _write_npz(
    path: Path,
    *,
    picks_time_s: np.ndarray | object = PICKS_ORIGINAL,
    n_traces: int | object = 4,
    n_samples: int | object = N_SAMPLES,
    dt: float | object = DT,
    sorted_to_original: np.ndarray | object = _ABSENT,
) -> Path:
    payload: dict[str, object] = {}
    if picks_time_s is not _ABSENT:
        payload['picks_time_s'] = picks_time_s
    if n_traces is not _ABSENT:
        payload['n_traces'] = np.int64(n_traces)
    if n_samples is not _ABSENT:
        payload['n_samples'] = np.int64(n_samples)
    if dt is not _ABSENT:
        payload['dt'] = np.float64(dt)
    if sorted_to_original is not _ABSENT:
        payload['sorted_to_original'] = sorted_to_original
    np.savez(path, **payload)
    return path


def _load_npz(
    path: Path,
    reader: TraceStoreSectionReader,
    *,
    source_kind='batch_npz',
):
    return load_npz_pick_source(
        path,
        reader=reader,
        expected_dt=DT,
        expected_n_samples=N_SAMPLES,
        source_kind=source_kind,
    )


def test_pick_source_loader_batch_npz_original_to_sorted(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / 'batch.npz')

    loaded = _load_npz(path, _reader(tmp_path), source_kind='batch_npz')

    assert loaded.source_kind == 'batch_npz'
    np.testing.assert_allclose(
        loaded.picks_time_s_sorted,
        PICKS_SORTED,
        equal_nan=True,
    )


def test_pick_source_loader_manual_npz_original_to_sorted(tmp_path: Path) -> None:
    path = _write_npz(
        tmp_path / 'manual.npz',
        sorted_to_original=SORTED_TO_ORIGINAL,
    )

    loaded = _load_npz(path, _reader(tmp_path), source_kind='manual_npz')

    assert loaded.source_kind == 'manual_npz'
    np.testing.assert_allclose(
        loaded.picks_time_s_sorted,
        PICKS_SORTED,
        equal_nan=True,
    )


def test_pick_source_loader_manual_npz_validates_sorted_to_original(
    tmp_path: Path,
) -> None:
    reader = _reader(tmp_path)
    path = _write_npz(
        tmp_path / 'manual.npz',
        sorted_to_original=SORTED_TO_ORIGINAL,
    )

    loaded = _load_npz(path, reader, source_kind='manual_npz')

    assert loaded.metadata['has_sorted_to_original'] is True
    bad_path = _write_npz(
        tmp_path / 'manual_bad_order.npz',
        sorted_to_original=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )
    with pytest.raises(ValueError, match='sorted_to_original mismatch'):
        _load_npz(bad_path, reader, source_kind='manual_npz')


def test_pick_source_loader_manual_memmap_already_sorted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('PICKS_NPY_DIR', str(tmp_path / 'picks_npy'))
    store = _write_store(tmp_path)
    file_id = 'manual-file'
    file_name = 'LineA.sgy'
    values_sorted = np.asarray([0.008, 0.002, np.nan, 0.006], dtype=np.float32)
    mm = open_for_write(file_name, int(values_sorted.shape[0]))
    mm[:] = values_sorted
    mm.flush()
    del mm

    state = AppState()
    state.file_registry.set_record(
        file_id,
        {
            'path': str(tmp_path / file_name),
            'store_path': str(store),
            'dt': DT,
        },
    )

    loaded = load_manual_memmap_pick_source(
        file_id=file_id,
        key1_byte=KEY1,
        key2_byte=KEY2,
        state=state,
    )

    assert loaded.source_kind == 'manual_memmap'
    np.testing.assert_allclose(
        loaded.picks_time_s_sorted,
        values_sorted.astype(np.float64),
        equal_nan=True,
    )


def test_pick_source_loader_allows_nan_and_reports_counts(tmp_path: Path) -> None:
    picks = np.asarray([np.nan, 0.002, np.nan, 0.006], dtype=np.float32)
    path = _write_npz(tmp_path / 'picks.npz', picks_time_s=picks)

    loaded = _load_npz(path, _reader(tmp_path))

    assert loaded.n_valid == 2
    assert loaded.n_nan == 2
    np.testing.assert_array_equal(
        loaded.valid_mask_sorted,
        np.asarray([False, False, True, True]),
    )


def test_pick_source_loader_rejects_missing_picks_time_s(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / 'picks.npz', picks_time_s=_ABSENT)

    with pytest.raises(ValueError, match='Missing key: picks_time_s'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_non_1d_picks(tmp_path: Path) -> None:
    path = _write_npz(
        tmp_path / 'picks.npz',
        picks_time_s=np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match='picks_time_s must be 1D'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_non_float_picks(tmp_path: Path) -> None:
    path = _write_npz(
        tmp_path / 'picks.npz',
        picks_time_s=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )

    with pytest.raises(ValueError, match='floating dtype'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_inf(tmp_path: Path) -> None:
    picks = PICKS_ORIGINAL.copy()
    picks[1] = np.inf
    path = _write_npz(tmp_path / 'picks.npz', picks_time_s=picks)

    with pytest.raises(ValueError, match='contains inf'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_negative_pick_time(tmp_path: Path) -> None:
    picks = PICKS_ORIGINAL.copy()
    picks[0] = -0.002
    path = _write_npz(tmp_path / 'picks.npz', picks_time_s=picks)

    with pytest.raises(ValueError, match='negative pick times'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_pick_time_beyond_n_samples(
    tmp_path: Path,
) -> None:
    picks = PICKS_ORIGINAL.copy()
    picks[0] = 0.010
    path = _write_npz(tmp_path / 'picks.npz', picks_time_s=picks)

    with pytest.raises(ValueError, match='beyond n_samples'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_n_traces_mismatch(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / 'picks.npz', n_traces=5)

    with pytest.raises(ValueError, match='n_traces mismatch'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_n_samples_mismatch(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / 'picks.npz', n_samples=6)

    with pytest.raises(ValueError, match='n_samples mismatch'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_dt_mismatch(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / 'picks.npz', dt=0.003)

    with pytest.raises(ValueError, match='dt mismatch'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_sorted_to_original_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_npz(
        tmp_path / 'picks.npz',
        sorted_to_original=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )

    with pytest.raises(ValueError, match='sorted_to_original mismatch'):
        _load_npz(path, _reader(tmp_path))


def test_pick_source_loader_rejects_non_permutation_sorted_to_original(
    tmp_path: Path,
) -> None:
    path = _write_npz(
        tmp_path / 'picks.npz',
        sorted_to_original=np.asarray([0, 0, 2, 3], dtype=np.int64),
    )

    with pytest.raises(ValueError, match='must be a permutation'):
        _load_npz(path, _reader(tmp_path))
