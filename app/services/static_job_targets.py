"""Lazy registry for static correction background job targets."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.services.job_runner import start_job_thread as _start_job_thread


@dataclass(frozen=True)
class StaticJobTargetSpec:
    module: str
    attribute: str


STATIC_JOB_TARGETS: dict[str, StaticJobTargetSpec] = {
    'datum': StaticJobTargetSpec(
        module='app.services.datum_static_service',
        attribute='run_datum_static_apply_job',
    ),
    'first_break_qc': StaticJobTargetSpec(
        module='app.services.first_break_qc_service',
        attribute='run_first_break_qc_job',
    ),
    'geometry_linkage': StaticJobTargetSpec(
        module='app.services.geometry_linkage_service',
        attribute='run_geometry_linkage_build_job',
    ),
    'residual': StaticJobTargetSpec(
        module='app.services.residual_static_service',
        attribute='run_residual_static_apply_job',
    ),
    'time_term': StaticJobTargetSpec(
        module='app.services.time_term_static_service',
        attribute='run_time_term_static_apply_job',
    ),
    'refraction': StaticJobTargetSpec(
        module='app.statics.refraction.adapters.seisviewer2d.workflow_runner',
        attribute='run_refraction_static_apply_job',
    ),
    'refraction_export': StaticJobTargetSpec(
        module='app.statics.refraction.adapters.seisviewer2d.export_runner',
        attribute='run_refraction_static_export_job',
    ),
    'refraction_static_table_apply': StaticJobTargetSpec(
        module='app.statics.refraction.adapters.seisviewer2d.table_apply_runner',
        attribute='run_refraction_static_table_apply_job',
    ),
}


def get_static_job_target(key: str) -> Callable[..., Any]:
    spec = STATIC_JOB_TARGETS[key]
    module = importlib.import_module(spec.module)
    target = getattr(module, spec.attribute)
    if not callable(target):
        raise TypeError(f'static job target {key!r} is not callable')
    return target


def start_static_job_thread(**kwargs: Any) -> object:
    return _start_job_thread(**kwargs)


__all__ = [
    'STATIC_JOB_TARGETS',
    'StaticJobTargetSpec',
    'get_static_job_target',
    'start_static_job_thread',
]
