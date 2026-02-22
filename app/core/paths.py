"""Path resolution for writable application data directories."""

from __future__ import annotations

from pathlib import Path

from app.core.settings import (
    APP_CACHE_DIRNAME as _APP_CACHE_DIRNAME,
    APP_DATA_DIRNAME as _APP_DATA_DIRNAME,
    resolve_app_data_dir,
    resolve_picks_npy_dir,
    resolve_pipeline_jobs_dir,
    resolve_processed_upload_dir,
    resolve_trace_store_dir,
    resolve_upload_dir,
)

APP_CACHE_DIRNAME = _APP_CACHE_DIRNAME
APP_DATA_DIRNAME = _APP_DATA_DIRNAME


def get_app_data_dir() -> Path:
    """Return the base writable directory for app-owned data."""
    return resolve_app_data_dir()


def get_picks_npy_dir() -> Path:
    """Return the manual-picks memmap directory path (without creating it)."""
    return resolve_picks_npy_dir()


def get_pipeline_jobs_dir() -> Path:
    """Return the pipeline artifact directory path (without creating it)."""
    return resolve_pipeline_jobs_dir()


def get_upload_dir() -> Path:
    """Return the upload root directory path (without creating it)."""
    return resolve_upload_dir()


def get_processed_upload_dir() -> Path:
    """Return the processed-upload directory path (without creating it)."""
    return resolve_processed_upload_dir()


def get_trace_store_dir() -> Path:
    """Return the trace-store directory path (without creating it)."""
    return resolve_trace_store_dir()
