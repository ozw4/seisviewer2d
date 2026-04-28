"""Application state container for shared in-memory objects."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.settings import Settings
from app.services.file_registry import FileRegistry
from app.services.job_manager import JobManager


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
        on_discard: Callable[[Any, Any, str], None] | None = None,
    ):
        super().__init__()
        if capacity <= 0:
            raise ValueError('capacity must be > 0')
        if ttl_sec <= 0:
            raise ValueError('ttl_sec must be > 0')
        self.capacity = capacity
        self.ttl_sec = ttl_sec
        self._time_fn = time_fn
        self._on_discard = on_discard

    def set_on_discard(
        self, on_discard: Callable[[Any, Any, str], None] | None
    ) -> None:
        """Set a callback invoked when an entry is expired or evicted."""
        self._on_discard = on_discard

    def _notify_discard(self, key, item, reason: str) -> None:
        if self._on_discard is None:
            return
        _, value = item
        self._on_discard(key, value, reason)

    def _pop_stored_item(self, key):
        if not super().__contains__(key):
            return None
        item = super().__getitem__(key)
        super().__delitem__(key)
        return item

    def __contains__(self, key) -> bool:
        if not super().__contains__(key):
            return False
        expires_at, _ = super().__getitem__(key)
        if expires_at <= self._time_fn():
            item = self._pop_stored_item(key)
            if item is not None:
                self._notify_discard(key, item, 'expired')
            return False
        return True

    def __setitem__(self, key, value):
        if super().__contains__(key):
            self.move_to_end(key)
        expires_at = float(self._time_fn()) + float(self.ttl_sec)
        super().__setitem__(key, (expires_at, value))
        if len(self) > self.capacity:
            evicted_key, evicted_item = self.popitem(last=False)
            self._notify_discard(evicted_key, evicted_item, 'capacity')

    def get(self, key, default=None):
        item = super().get(key)
        if item is None:
            return default
        expires_at, value = item
        if expires_at <= self._time_fn():
            item = self._pop_stored_item(key)
            if item is not None:
                self._notify_discard(key, item, 'expired')
            return default
        self.move_to_end(key)
        return value

    def pop(self, key, default=None):
        item = self._pop_stored_item(key)
        if item is None:
            return default
        expires_at, value = item
        if expires_at <= self._time_fn():
            self._notify_discard(key, item, 'expired')
            return default
        return value

    def purge_expired(self) -> int:
        """Remove expired entries and return the number removed."""
        now = self._time_fn()
        expired_keys = [
            key for key, (expires_at, _) in super().items() if expires_at <= now
        ]
        for key in expired_keys:
            item = self._pop_stored_item(key)
            if item is not None:
                self._notify_discard(key, item, 'expired')
        return len(expired_keys)

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
    settings: Settings = field(default_factory=Settings.from_env)
    file_registry: FileRegistry = field(default_factory=FileRegistry)
    cached_readers: LRUCache = field(init=False)
    fbpick_cache: ExpiringLRUCache = field(init=False)
    staged_uploads: ExpiringLRUCache = field(init=False)
    jobs: JobManager = field(default_factory=JobManager)
    pipeline_tap_cache: LRUCache = field(default_factory=lambda: LRUCache(16))
    window_section_cache: LRUCache = field(default_factory=lambda: LRUCache(32))
    section_offsets_cache: LRUCache = field(default_factory=lambda: LRUCache(64))
    trace_stats_cache: TraceStatsCache = field(init=False)

    def __post_init__(self) -> None:
        self.cached_readers = LRUCache(self.settings.sv_cached_readers_capacity)
        self.fbpick_cache = ExpiringLRUCache(
            capacity=self.settings.sv_fbpick_cache_capacity,
            ttl_sec=self.settings.sv_fbpick_cache_ttl_sec,
        )
        self.staged_uploads = ExpiringLRUCache(capacity=16, ttl_sec=3600)
        self.trace_stats_cache = TraceStatsCache(
            max_sections=self.settings.sv_trace_stats_max_sections,
            max_windows=self.settings.sv_trace_stats_max_windows,
        )


def create_app_state(settings: Settings | None = None) -> AppState:
    """Create a fresh application state object."""
    resolved = settings if settings is not None else Settings.from_env()
    return AppState(settings=resolved)


DEFAULT_STATE = create_app_state()


__all__ = [
    'AppState',
    'DEFAULT_STATE',
    'ExpiringLRUCache',
    'LRUCache',
    'TraceStatsCache',
    'create_app_state',
]
