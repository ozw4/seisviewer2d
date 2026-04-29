from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.state import create_app_state


class _StubReader:
    created: list["_StubReader"] = []

    def __init__(self, store_dir: Path, key1_byte: int, key2_byte: int):
        self.store_dir = Path(store_dir)
        self.key1_byte = key1_byte
        self.key2_byte = key2_byte
        _StubReader.created.append(self)

    def preload_all_sections(self) -> None:
        pass

    def ensure_header(self, _header_byte: int) -> None:
        pass


class _CapturedThread:
    created: list["_CapturedThread"] = []

    def __init__(self, *, target, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        _CapturedThread.created.append(self)

    def start(self) -> None:
        self.started = True


@pytest.fixture()
def registration_env(monkeypatch):
    from app.services import trace_store_registration as registration

    _StubReader.created.clear()
    _CapturedThread.created.clear()
    monkeypatch.setattr(
        registration,
        'TraceStoreSectionReader',
        _StubReader,
        raising=True,
    )
    monkeypatch.setattr(
        registration,
        'threading',
        SimpleNamespace(Thread=_CapturedThread),
        raising=True,
    )
    return registration, create_app_state()


def _target_name(thread: _CapturedThread) -> str:
    return getattr(thread.target, '__name__', '')


def test_trace_store_cache_key_keeps_existing_format(registration_env):
    registration, _state = registration_env

    assert registration.trace_store_cache_key('file-a', 189, 193) == 'file-a_189_193'


def test_register_trace_store_adds_reader_to_state_cache(
    registration_env,
    tmp_path: Path,
):
    registration, state = registration_env

    reader = registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        update_registry=False,
        touch_meta=False,
    )

    assert reader is _StubReader.created[0]
    assert reader.store_dir == tmp_path
    with state.lock:
        assert state.cached_readers['file-a_189_193'] is reader


def test_register_trace_store_starts_preload_all_sections_thread(
    registration_env,
    tmp_path: Path,
):
    registration, state = registration_env

    reader = registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        update_registry=False,
        touch_meta=False,
    )

    preload_threads = [
        thread
        for thread in _CapturedThread.created
        if _target_name(thread) == 'preload_all_sections'
    ]
    assert len(preload_threads) == 1
    thread = preload_threads[0]
    assert thread.target.__self__ is reader
    assert thread.daemon is True
    assert thread.started is True


def test_register_trace_store_starts_unique_header_preload_threads(
    registration_env,
    tmp_path: Path,
):
    registration, state = registration_env

    reader = registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        update_registry=False,
        touch_meta=False,
        preload_header_bytes=[193, 201, 189, 201],
    )

    header_threads = [
        thread
        for thread in _CapturedThread.created
        if _target_name(thread) == 'ensure_header'
    ]
    assert sorted(thread.args[0] for thread in header_threads) == [189, 193, 201]
    assert all(thread.target.__self__ is reader for thread in header_threads)
    assert all(thread.daemon is True for thread in header_threads)
    assert all(thread.started is True for thread in header_threads)


def test_register_trace_store_updates_file_registry_with_store_path_and_dt(
    registration_env,
    tmp_path: Path,
):
    registration, state = registration_env

    registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        dt=0.004,
        touch_meta=False,
    )

    record = state.file_registry.get_record('file-a')
    assert isinstance(record, dict)
    assert record['store_path'] == str(tmp_path)
    assert record['dt'] == pytest.approx(0.004)


def test_register_trace_store_updates_file_registry_with_store_path_when_dt_is_none(
    registration_env,
    tmp_path: Path,
):
    registration, state = registration_env

    registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        dt=None,
        touch_meta=False,
    )

    record = state.file_registry.get_record('file-a')
    assert isinstance(record, dict)
    assert record['store_path'] == str(tmp_path)
    assert 'dt' not in record


def test_register_trace_store_can_skip_file_registry_update(
    registration_env,
    tmp_path: Path,
):
    registration, state = registration_env

    registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        dt=0.004,
        update_registry=False,
        touch_meta=False,
    )

    assert state.file_registry.get_record('file-a') is None


def test_register_trace_store_touches_existing_meta(registration_env, tmp_path: Path):
    registration, state = registration_env
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text('{}', encoding='utf-8')
    old_mtime = 1000.0
    os.utime(meta_path, (old_mtime, old_mtime))

    registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        update_registry=False,
    )

    assert meta_path.stat().st_mtime > old_mtime


def test_register_trace_store_missing_meta_is_noop(registration_env, tmp_path: Path):
    registration, state = registration_env

    registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        update_registry=False,
    )

    assert not (tmp_path / 'meta.json').exists()


def test_register_trace_store_can_skip_meta_touch(registration_env, tmp_path: Path):
    registration, state = registration_env
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text('{}', encoding='utf-8')
    old_mtime = 1000.0
    os.utime(meta_path, (old_mtime, old_mtime))

    registration.register_trace_store(
        state=state,
        file_id='file-a',
        store_dir=tmp_path,
        key1_byte=189,
        key2_byte=193,
        update_registry=False,
        touch_meta=False,
    )

    assert meta_path.stat().st_mtime == pytest.approx(old_mtime)
