"""Environment-backed settings and path resolution helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

APP_CACHE_DIRNAME = 'seisviewer2d'
APP_DATA_DIRNAME = 'seisviewer2d_app_data'
_PIPELINE_JOBS_CLEANUP_INTERVAL_SEC = 600


def _resolve_environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is not None:
        return environ
    return os.environ


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    return int(raw)


def env_positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = _env_int(environ, name, default)
    if value <= 0:
        raise ValueError(f'{name} must be > 0')
    return value


def resolve_pipeline_jobs_ttl_hours(
    environ: Mapping[str, str] | None = None,
) -> int:
    env = _resolve_environ(environ)
    return env_positive_int(env, 'PIPELINE_JOBS_TTL_HOURS', 48)


def resolve_pipeline_jobs_ttl_seconds(
    environ: Mapping[str, str] | None = None,
) -> int:
    return resolve_pipeline_jobs_ttl_hours(environ) * 3600


def resolve_pipeline_jobs_cleanup_interval_sec() -> int:
    return _PIPELINE_JOBS_CLEANUP_INTERVAL_SEC


@dataclass(frozen=True, slots=True)
class Settings:
    sv_cached_readers_capacity: int
    sv_fbpick_cache_capacity: int
    sv_fbpick_cache_ttl_sec: int
    sv_trace_stats_max_sections: int
    sv_trace_stats_max_windows: int
    sv_jobs_max: int
    sv_fbpick_job_ttl_sec: int
    pipeline_jobs_ttl_hours: int
    pipeline_jobs_cleanup_interval_sec: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = _resolve_environ(environ)
        return cls(
            sv_cached_readers_capacity=env_positive_int(
                env,
                'SV_CACHED_READERS_CAPACITY',
                8,
            ),
            sv_fbpick_cache_capacity=env_positive_int(
                env,
                'SV_FBPICK_CACHE_CAPACITY',
                128,
            ),
            sv_fbpick_cache_ttl_sec=env_positive_int(
                env,
                'SV_FBPICK_CACHE_TTL_SEC',
                1800,
            ),
            sv_trace_stats_max_sections=env_positive_int(
                env,
                'SV_TRACE_STATS_MAX_SECTIONS',
                128,
            ),
            sv_trace_stats_max_windows=env_positive_int(
                env,
                'SV_TRACE_STATS_MAX_WINDOWS',
                16,
            ),
            sv_jobs_max=env_positive_int(env, 'SV_JOBS_MAX', 2000),
            sv_fbpick_job_ttl_sec=env_positive_int(
                env,
                'SV_FBPICK_JOB_TTL_SEC',
                1800,
            ),
            pipeline_jobs_ttl_hours=resolve_pipeline_jobs_ttl_hours(env),
            pipeline_jobs_cleanup_interval_sec=(
                resolve_pipeline_jobs_cleanup_interval_sec()
            ),
        )


def resolve_app_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = _resolve_environ(environ)
    override = env.get('SV_APP_DATA_DIR')
    if override:
        return Path(override).expanduser()

    runner_temp = env.get('RUNNER_TEMP')
    if runner_temp:
        return Path(runner_temp).expanduser() / APP_DATA_DIRNAME

    xdg_cache_home = env.get('XDG_CACHE_HOME')
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / APP_CACHE_DIRNAME

    return Path.home() / '.cache' / APP_CACHE_DIRNAME


def resolve_picks_npy_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = _resolve_environ(environ)
    override = env.get('PICKS_NPY_DIR')
    if override:
        return Path(override).expanduser()
    return resolve_app_data_dir(env) / 'picks_npy'


def resolve_pipeline_jobs_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = _resolve_environ(environ)
    override = env.get('PIPELINE_JOBS_DIR')
    if override:
        return Path(override).expanduser()
    return resolve_app_data_dir(env) / 'pipeline_jobs'


def resolve_upload_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = _resolve_environ(environ)
    override = env.get('SV_UPLOAD_DIR')
    if override:
        return Path(override).expanduser()
    return resolve_app_data_dir(env) / 'uploads'


def resolve_processed_upload_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = _resolve_environ(environ)
    override = env.get('SV_PROCESSED_DIR')
    if override:
        return Path(override).expanduser()
    return resolve_upload_dir(env) / 'processed'


def resolve_trace_store_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = _resolve_environ(environ)
    override = env.get('SV_TRACE_DIR')
    if override:
        return Path(override).expanduser()
    return resolve_processed_upload_dir(env) / 'traces'


__all__ = [
    'APP_CACHE_DIRNAME',
    'APP_DATA_DIRNAME',
    'Settings',
    'env_positive_int',
    'resolve_app_data_dir',
    'resolve_pipeline_jobs_cleanup_interval_sec',
    'resolve_pipeline_jobs_ttl_hours',
    'resolve_pipeline_jobs_ttl_seconds',
    'resolve_picks_npy_dir',
    'resolve_pipeline_jobs_dir',
    'resolve_processed_upload_dir',
    'resolve_trace_store_dir',
    'resolve_upload_dir',
]
