"""In-memory job table cleanup helpers."""

from __future__ import annotations

import time

from app.core.state import AppState


def cleanup_in_memory_state(state: AppState) -> None:
    """Evict terminal jobs by TTL and cap the total in-memory jobs count."""
    with state.lock:
        state.jobs.cleanup_in_memory(now_ts=time.time(), settings=state.settings)


__all__ = ['cleanup_in_memory_state']
