"""Application state container for shared in-memory objects."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f'{name} must be > 0')
    return value


class LRUCache(OrderedDict):
    """A tiny ordered cache used for in-memory tap storage."""

    def __init__(self, capacity: int = 16):
        super().__init__()
        if capacity <= 0:
            raise ValueError('capacity must be > 0')
        self.capacity = capacity

    def __setitem__(self, key, value):
        if super().__contains__(key):
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)

    def get(self, key, default=None):
        if super().__contains__(key):
            self.move_to_end(key)
            return super().__getitem__(key)
        return default

    def set(self, key, value):
        self[key] = value


class ExpiringLRUCache(OrderedDict):
    """A small LRU cache with TTL semantics for each entry."""

    def __init__(
        self,
        capacity: int = 128,
        ttl_sec: int = 1800,
        *,
        time_fn: Any = time.time,
    ):
        super().__init__()
        if capacity <= 0:
            raise ValueError('capacity must be > 0')
        if ttl_sec <= 0:
            raise ValueError('ttl_sec must be > 0')
        self.capacity = capacity
        self.ttl_sec = ttl_sec
        self._time_fn = time_fn

    def __contains__(self, key) -> bool:
        if not super().__contains__(key):
            return False
        expires_at, _ = super().__getitem__(key)
        if expires_at <= self._time_fn():
            super().pop(key, None)
            return False
        return True

    def __setitem__(self, key, value):
        if super().__contains__(key):
            self.move_to_end(key)
        expires_at = float(self._time_fn()) + float(self.ttl_sec)
        super().__setitem__(key, (expires_at, value))
        if len(self) > self.capacity:
            self.popitem(last=False)

    def get(self, key, default=None):
        item = super().get(key)
        if item is None:
            return default
        expires_at, value = item
        if expires_at <= self._time_fn():
            super().pop(key, None)
            return default
        self.move_to_end(key)
        return value

    def pop(self, key, default=None):
        item = super().pop(key, None)
        if item is None:
            return default
        expires_at, value = item
        if expires_at <= self._time_fn():
            return default
        return value

    def set(self, key, value):
        self[key] = value


class TraceStatsCache:
    """Grouped LRU cache for per-section trace statistics."""

    def __init__(self, *, max_sections: int = 128, max_windows: int = 16):
        if max_sections <= 0:
            raise ValueError('max_sections must be > 0')
        if max_windows <= 0:
            raise ValueError('max_windows must be > 0')
        self.max_sections = max_sections
        self.max_windows = max_windows
        self._groups: OrderedDict[
            tuple[Any, int], dict[str, tuple[Any, ...] | LRUCache | None]
        ] = OrderedDict()

    def _parse_key(self, key: tuple[Any, ...]) -> tuple[tuple[Any, int], str, Any]:
        if len(key) == 3 and key[2] == 'idx_full':
            return (key[0], int(key[1])), 'idx_full', None
        if len(key) == 5:
            return (
                (key[0], int(key[1])),
                'window',
                (
                    int(key[2]),
                    int(key[3]),
                    int(key[4]),
                ),
            )
        raise KeyError('Unsupported trace stats cache key')

    def _evict_groups(self) -> None:
        while len(self._groups) > self.max_sections:
            self._groups.popitem(last=False)

    def _get_group(
        self, group_key: tuple[Any, int]
    ) -> dict[str, tuple[Any, ...] | LRUCache | None] | None:
        group = self._groups.get(group_key)
        if group is not None:
            self._groups.move_to_end(group_key)
        return group

    def _ensure_group(
        self, group_key: tuple[Any, int]
    ) -> dict[str, tuple[Any, ...] | LRUCache | None]:
        group = self._groups.get(group_key)
        if group is None:
            group = {'idx_full': None, 'windows': LRUCache(self.max_windows)}
            self._groups[group_key] = group
            self._evict_groups()
        else:
            self._groups.move_to_end(group_key)
        return group

    def get(self, key: tuple[Any, ...], default=None):
        try:
            group_key, entry_kind, entry_key = self._parse_key(key)
        except KeyError:
            return default
        group = self._get_group(group_key)
        if group is None:
            return default
        if entry_kind == 'idx_full':
            payload = group.get('idx_full')
            return default if payload is None else payload
        windows = group.get('windows')
        if not isinstance(windows, LRUCache):
            return default
        return windows.get(entry_key, default)

    def setdefault(self, key: tuple[Any, ...], default):
        group_key, entry_kind, entry_key = self._parse_key(key)
        group = self._ensure_group(group_key)
        if entry_kind == 'idx_full':
            payload = group.get('idx_full')
            if payload is None:
                group['idx_full'] = default
                return default
            return payload
        windows = group.get('windows')
        if not isinstance(windows, LRUCache):
            windows = LRUCache(self.max_windows)
            group['windows'] = windows
        cached = windows.get(entry_key)
        if cached is not None:
            return cached
        windows[entry_key] = default
        return default

    def clear(self) -> None:
        self._groups.clear()


@dataclass
class AppState:
    """Shared mutable state used by API helpers and routers."""

    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cached_readers: LRUCache = field(
        default_factory=lambda: LRUCache(
            _env_positive_int('SV_CACHED_READERS_CAPACITY', 8)
        )
    )
    fbpick_cache: ExpiringLRUCache = field(
        default_factory=lambda: ExpiringLRUCache(
            capacity=_env_positive_int('SV_FBPICK_CACHE_CAPACITY', 128),
            ttl_sec=_env_positive_int('SV_FBPICK_CACHE_TTL_SEC', 1800),
        )
    )
    jobs: dict[str, dict[str, object]] = field(default_factory=dict)
    pipeline_tap_cache: LRUCache = field(default_factory=lambda: LRUCache(16))
    window_section_cache: LRUCache = field(default_factory=lambda: LRUCache(32))
    trace_stats_cache: TraceStatsCache = field(
        default_factory=lambda: TraceStatsCache(
            max_sections=_env_positive_int('SV_TRACE_STATS_MAX_SECTIONS', 128),
            max_windows=_env_positive_int('SV_TRACE_STATS_MAX_WINDOWS', 16),
        )
    )


def create_app_state() -> AppState:
    """Create a fresh application state object."""
    return AppState()


DEFAULT_STATE = create_app_state()


__all__ = [
    'AppState',
    'DEFAULT_STATE',
    'ExpiringLRUCache',
    'LRUCache',
    'TraceStatsCache',
    'create_app_state',
]
