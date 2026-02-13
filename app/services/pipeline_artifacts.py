"""Disk-backed persistence for ``/pipeline/all`` artifacts."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import msgpack

from app.core.paths import get_pipeline_jobs_dir

_SAFE_LABEL_RE = re.compile(r'[^A-Za-z0-9._-]+')
logger = logging.getLogger(__name__)

_DEFAULT_TTL_HOURS = 48
_DEFAULT_CLEANUP_INTERVAL_SEC = 600
_last_cleanup_ts = 0.0


def get_pipeline_jobs_base_dir() -> Path:
    """Compatibility wrapper for the pipeline artifacts base directory."""
    return get_pipeline_jobs_dir()


def get_job_dir(job_id: str) -> Path:
    """Return the artifact directory path for ``job_id``."""
    return get_pipeline_jobs_base_dir() / str(job_id)


def safe_filename(label: str) -> str:
    """Return a deterministic filesystem-safe name for a tap label."""
    if not isinstance(label, str):
        raise TypeError('tap label must be a string')
    clean = _SAFE_LABEL_RE.sub('_', label).strip('._')
    if not clean:
        clean = 'tap'
    digest = hashlib.sha256(label.encode('utf-8')).hexdigest()[:12]
    return f'{clean}__{digest}'


def _artifact_path(job_id: str, key1_val: int, tap_label: str) -> Path:
    return get_job_dir(job_id) / str(int(key1_val)) / f'{safe_filename(tap_label)}.bin'


def _ttl_seconds() -> int:
    raw = os.getenv('PIPELINE_JOBS_TTL_HOURS')
    if raw is None:
        return _DEFAULT_TTL_HOURS * 3600
    ttl_hours = int(raw)
    if ttl_hours <= 0:
        raise ValueError('PIPELINE_JOBS_TTL_HOURS must be > 0')
    return ttl_hours * 3600


def cleanup_expired_jobs(*, now_ts: float | None = None) -> int:
    """Remove job directories older than TTL and return removed count."""
    base = get_pipeline_jobs_base_dir()
    if not base.is_dir():
        return 0
    now = time.time() if now_ts is None else float(now_ts)
    ttl_sec = _ttl_seconds()
    removed = 0
    for job_dir in base.iterdir():
        if not job_dir.is_dir():
            continue
        age = now - job_dir.stat().st_mtime
        if age <= ttl_sec:
            continue
        try:
            shutil.rmtree(job_dir)
        except OSError as exc:
            logger.warning('Failed to remove expired job dir %s: %s', job_dir, exc)
            continue
        removed += 1
    return removed


def maybe_cleanup_expired_jobs() -> int:
    """Run TTL cleanup at most once per interval and return removed count."""
    global _last_cleanup_ts
    now = time.time()
    if now - _last_cleanup_ts < _DEFAULT_CLEANUP_INTERVAL_SEC:
        return 0
    _last_cleanup_ts = now
    return cleanup_expired_jobs(now_ts=now)


def write_artifact(*, job_id: str, key1_val: int, tap_label: str, payload: Any) -> Path:
    """Persist one artifact payload atomically and return its output path."""
    out_path = _artifact_path(job_id, key1_val, tap_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packed = msgpack.packb(payload, use_bin_type=True)
    tmp_path = out_path.with_name(f'{out_path.name}.tmp')
    tmp_path.write_bytes(packed)
    tmp_path.replace(out_path)
    job_dir = get_job_dir(job_id)
    os.utime(job_dir, None)
    return out_path


def read_artifact(*, job_id: str, key1_val: int, tap_label: str) -> Any | None:
    """Load a persisted artifact payload, returning ``None`` when absent."""
    in_path = _artifact_path(job_id, key1_val, tap_label)
    if not in_path.is_file():
        return None
    packed = in_path.read_bytes()
    return msgpack.unpackb(packed, raw=False)


__all__ = [
    'cleanup_expired_jobs',
    'get_job_dir',
    'get_pipeline_jobs_base_dir',
    'maybe_cleanup_expired_jobs',
    'read_artifact',
    'safe_filename',
    'write_artifact',
]
