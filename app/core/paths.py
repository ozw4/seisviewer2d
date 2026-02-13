"""Path resolution for writable application data directories."""

from __future__ import annotations

import os
from pathlib import Path

APP_CACHE_DIRNAME = 'seisviewer2d'
APP_DATA_DIRNAME = 'seisviewer2d_app_data'


def get_app_data_dir() -> Path:
    """Return the base writable directory for app-owned data."""
    override = os.getenv('SV_APP_DATA_DIR')
    if override:
        return Path(override).expanduser()

    runner_temp = os.getenv('RUNNER_TEMP')
    if runner_temp:
        return Path(runner_temp).expanduser() / APP_DATA_DIRNAME

    xdg_cache_home = os.getenv('XDG_CACHE_HOME')
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / APP_CACHE_DIRNAME

    return Path.home() / '.cache' / APP_CACHE_DIRNAME


def get_picks_npy_dir() -> Path:
    """Return the manual-picks memmap directory path (without creating it)."""
    override = os.getenv('PICKS_NPY_DIR')
    if override:
        return Path(override).expanduser()
    return get_app_data_dir() / 'picks_npy'


def get_pipeline_jobs_dir() -> Path:
    """Return the pipeline artifact directory path (without creating it)."""
    override = os.getenv('PIPELINE_JOBS_DIR')
    if override:
        return Path(override).expanduser()
    return get_app_data_dir() / 'pipeline_jobs'


def get_upload_dir() -> Path:
    """Return the upload root directory path (without creating it)."""
    override = os.getenv('SV_UPLOAD_DIR')
    if override:
        return Path(override).expanduser()
    return get_app_data_dir() / 'uploads'


def get_processed_upload_dir() -> Path:
    """Return the processed-upload directory path (without creating it)."""
    override = os.getenv('SV_PROCESSED_DIR')
    if override:
        return Path(override).expanduser()
    return get_upload_dir() / 'processed'


def get_trace_store_dir() -> Path:
    """Return the trace-store directory path (without creating it)."""
    override = os.getenv('SV_TRACE_DIR')
    if override:
        return Path(override).expanduser()
    return get_processed_upload_dir() / 'traces'
