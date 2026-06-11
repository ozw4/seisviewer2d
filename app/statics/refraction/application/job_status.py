"""Pure job-status helpers for refraction application code."""

from __future__ import annotations

from typing import Literal

RefractionJobStatus = Literal[
    'queued',
    'running',
    'cancel_requested',
    'done',
    'error',
    'cancelled',
    'expired',
    'unknown',
]


def normalize_status_value(status: object) -> RefractionJobStatus:
    """Normalize job status values without importing the app job manager."""
    if not isinstance(status, str):
        return 'unknown'
    normalized = status.strip().lower()
    if normalized == 'completed':
        return 'done'
    if normalized == 'failed':
        return 'error'
    if normalized in {
        'queued',
        'running',
        'cancel_requested',
        'done',
        'error',
        'cancelled',
        'expired',
        'unknown',
    }:
        return normalized
    return 'unknown'


def is_ready_status_value(status: object) -> bool:
    """Return true when a job status represents completed usable output."""
    return normalize_status_value(status) == 'done'


__all__ = ['RefractionJobStatus', 'is_ready_status_value', 'normalize_status_value']
