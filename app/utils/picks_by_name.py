"""Filename-scoped manual pick persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STORE_PATH = Path('/workspace/app_data/picks_by_name.json')

_store: dict[str, dict[str, Any]] = {}


def _normalize_name(file_name: str | None) -> str | None:
        if not file_name:
                return None
        return Path(file_name).name.casefold()


def _section_key(key1_idx: int, key1_byte: int) -> str:
        return f'{key1_idx}:{key1_byte}'


def load() -> None:
        """Load stored picks from disk if available."""
        global _store
        if STORE_PATH.exists():
                try:
                        _store = json.loads(STORE_PATH.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                        _store = {}
        else:
                _store = {}



def save() -> None:
        """Persist picks to disk atomically."""
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE_PATH.with_suffix(STORE_PATH.suffix  ".tmp")  # e.g. picks_by_name.json.tmp
        data = json.dumps(_store, ensure_ascii=False, indent=2)
        with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                import os
                os.fsync(f.fileno())  # ensure bytes hit disk before swap
        tmp.replace(STORE_PATH)  # atomic on same filesystem


def _ensure_entry(file_name: str | None) -> tuple[str, dict[str, Any]] | None:
        norm = _normalize_name(file_name)
        if norm is None:
                return None
        entry = _store.setdefault(
                norm,
                {
                        'sections': {},
                        'meta': {'filename': Path(file_name).name if file_name else ''},
                },
        )
        if 'meta' not in entry:
                entry['meta'] = {'filename': Path(file_name).name if file_name else ''}
        elif 'filename' not in entry['meta'] and file_name:
                entry['meta']['filename'] = Path(file_name).name
        return norm, entry


def list_picks(file_name: str, key1_idx: int, key1_byte: int) -> list[dict[str, Any]]:
        """Return picks stored for the provided filename and section."""
        norm = _normalize_name(file_name)
        if norm is None:
                return []
        entry = _store.get(norm)
        if not entry:
                return []
        picks = entry.get('sections', {}).get(_section_key(key1_idx, key1_byte), [])
        return [
                {'trace': int(p['trace']), 'time': float(p['time'])}
                for p in picks
                if 'trace' in p and 'time' in p
        ]


def set_picks(
        file_name: str,
        key1_idx: int,
        key1_byte: int,
        manual: list[dict[str, Any]],
) -> None:
        """Replace picks for the given filename and section."""
        ensured = _ensure_entry(file_name)
        if ensured is None:
                return
        norm, entry = ensured
        if not manual:
                entry.get('sections', {}).pop(_section_key(key1_idx, key1_byte), None)
        else:
                section = [
                        {'trace': int(p['trace']), 'time': float(p['time'])}
                        for p in manual
                        if 'trace' in p and 'time' in p
                ]
                entry.setdefault('sections', {})[_section_key(key1_idx, key1_byte)] = section
        if not entry.get('sections'):
                _store.pop(norm, None)


def add_pick(
        file_name: str,
        trace: int,
        time: float,
        key1_idx: int,
        key1_byte: int,
) -> None:
        """Add or update a manual pick entry for the specified trace."""
        ensured = _ensure_entry(file_name)
        if ensured is None:
                return
        _, entry = ensured
        section_key = _section_key(key1_idx, key1_byte)
        section = entry.setdefault('sections', {}).setdefault(section_key, [])
        for pick in section:
                if pick.get('trace') == trace:
                        pick['time'] = float(time)
                        break
        else:
                        section.append({'trace': int(trace), 'time': float(time)})


def delete_pick(
        file_name: str,
        trace: int | None,
        key1_idx: int,
        key1_byte: int,
) -> None:
        """Delete picks for a filename, optionally filtered by trace."""
        norm = _normalize_name(file_name)
        if norm is None:
                return
        entry = _store.get(norm)
        if not entry:
                return
        sections = entry.setdefault('sections', {})
        section_key = _section_key(key1_idx, key1_byte)
        picks = sections.get(section_key)
        if not picks:
                return
        if trace is None:
                sections.pop(section_key, None)
        else:
                sections[section_key] = [p for p in picks if p.get('trace') != trace]
                if not sections[section_key]:
                        sections.pop(section_key)
        if not sections:
                _store.pop(norm, None)
