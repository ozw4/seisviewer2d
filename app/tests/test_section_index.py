import numpy as np
import pytest

from app.core.state import create_app_state
from app.services import section_index as secidx
from app.services.errors import ConflictError


def test_get_ntraces_for_returns_reader_ntraces(monkeypatch):
    state = create_app_state()
    reader = type('_Reader', (), {'ntraces': 6})()
    monkeypatch.setattr(
        secidx, 'get_reader', lambda file_id, key1_byte, key2_byte, state=None: reader
    )

    got = secidx.get_ntraces_for('f', 189, 193, state=state)
    assert got == 6


def test_get_ntraces_for_raises_when_reader_has_no_trace_count(monkeypatch):
    state = create_app_state()
    reader = type('_Reader', (), {})()
    monkeypatch.setattr(
        secidx, 'get_reader', lambda file_id, key1_byte, key2_byte, state=None: reader
    )

    with pytest.raises(ConflictError, match='number of traces'):
        secidx.get_ntraces_for('f', 189, 193, state=state)


def test_get_trace_seq_for_value_uses_reader_public_method(monkeypatch):
    state = create_app_state()
    calls: list[tuple[int, str]] = []

    class _Reader:
        def get_trace_seq_for_value(self, key1: int, align_to: str = 'display'):
            calls.append((int(key1), align_to))
            return np.array([3, 1], dtype=np.int32)

    monkeypatch.setattr(
        secidx,
        'get_reader',
        lambda file_id, key1_byte, key2_byte, state=None: _Reader(),
    )

    got = secidx.get_trace_seq_for_value('f', 42, 189, 193, state=state)
    assert got.dtype == np.int64
    assert got.tolist() == [3, 1]
    assert calls == [(42, 'display')]


def test_get_trace_seq_for_value_requires_reader_method(monkeypatch):
    state = create_app_state()
    reader = type('_Reader', (), {})()
    monkeypatch.setattr(
        secidx, 'get_reader', lambda file_id, key1_byte, key2_byte, state=None: reader
    )

    with pytest.raises(ConflictError, match='trace sequence'):
        secidx.get_trace_seq_for_value('f', 42, 189, 193, state=state)
