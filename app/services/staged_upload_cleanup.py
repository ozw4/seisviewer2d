"""Disk cleanup helpers for staged SEG-Y uploads."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _safe_direct_staged_dir(path: Path, staged_root: Path) -> Path | None:
    root = _resolved(staged_root)
    candidate = path.expanduser()
    if candidate.parent.resolve(strict=False) != root:
        logger.warning(
            'Refusing to delete staged upload path outside staged root: %s',
            path,
        )
        return None
    safe_dir = root / candidate.name
    if safe_dir.is_symlink():
        logger.warning('Refusing to delete staged upload symlink: %s', safe_dir)
        return None
    return safe_dir


def _safe_staged_file(raw_path: Path, staged_root: Path) -> Path | None:
    candidate = raw_path.expanduser()
    staged_dir = _safe_direct_staged_dir(candidate.parent, staged_root)
    if staged_dir is None:
        return None
    return staged_dir / candidate.name


def remove_staged_upload_dir(staged_dir: Path, *, staged_root: Path) -> bool:
    """Remove one direct child directory below ``staged_root``."""
    safe_dir = _safe_direct_staged_dir(staged_dir, staged_root)
    if safe_dir is None:
        return False
    if not safe_dir.exists():
        return True
    try:
        shutil.rmtree(safe_dir)
    except OSError as exc:
        logger.warning('Unable to delete staged SEG-Y directory: %s: %s', safe_dir, exc)
        return False
    return True


def cleanup_staged_upload(raw_path: Path, *, staged_root: Path) -> bool:
    """Remove a staged SEG-Y file and its direct staged directory."""
    safe_raw_path = _safe_staged_file(raw_path, staged_root)
    if safe_raw_path is None:
        return False

    try:
        safe_raw_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning('Unable to delete staged SEG-Y file: %s: %s', safe_raw_path, exc)
        return False

    return remove_staged_upload_dir(safe_raw_path.parent, staged_root=staged_root)


def cleanup_stale_staged_upload_dirs(
    *,
    staged_root: Path,
    ttl_sec: int | float,
    active_ids: Iterable[str] = (),
    now_ts: float | None = None,
) -> int:
    """Remove stale staged-upload directories with no active in-memory record."""
    if not staged_root.is_dir():
        return 0

    now = time.time() if now_ts is None else float(now_ts)
    active = {str(item) for item in active_ids}
    removed = 0

    for child in staged_root.iterdir():
        if child.name in active:
            continue
        safe_dir = _safe_direct_staged_dir(child, staged_root)
        if safe_dir is None or not safe_dir.is_dir():
            continue
        try:
            age = now - safe_dir.stat().st_mtime
        except OSError as exc:
            logger.warning(
                'Unable to inspect staged SEG-Y directory: %s: %s',
                safe_dir,
                exc,
            )
            continue
        if age <= float(ttl_sec):
            continue
        if remove_staged_upload_dir(safe_dir, staged_root=staged_root):
            removed += 1

    return removed


__all__ = [
    'cleanup_staged_upload',
    'cleanup_stale_staged_upload_dirs',
    'remove_staged_upload_dir',
]
