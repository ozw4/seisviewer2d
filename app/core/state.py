"""Application state container for shared in-memory objects."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.utils.utils import TraceStoreSectionReader


class LRUCache(OrderedDict):
    """A tiny ordered cache used for in-memory tap storage."""

    def __init__(self, capacity: int = 16):
        super().__init__()
        self.capacity = capacity

    def get(self, key):
        if key in self:
            self.move_to_end(key)
            return super().__getitem__(key)
        return None

    def set(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)


@dataclass
class AppState:
    """Shared mutable state used by API helpers and routers."""

    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cached_readers: dict[str, TraceStoreSectionReader] = field(default_factory=dict)
    fbpick_cache: dict[tuple[Any, ...], bytes] = field(default_factory=dict)
    jobs: dict[str, dict[str, object]] = field(default_factory=dict)
    pipeline_tap_cache: LRUCache = field(default_factory=lambda: LRUCache(16))
    window_section_cache: LRUCache = field(default_factory=lambda: LRUCache(32))
    trace_stats_cache: dict[
        tuple[str, int, int], tuple[np.ndarray, np.ndarray, int]
    ] = field(default_factory=dict)


def create_app_state() -> AppState:
    """Create a fresh application state object."""
    return AppState()


DEFAULT_STATE = create_app_state()


__all__ = ['AppState', 'DEFAULT_STATE', 'LRUCache', 'create_app_state']
