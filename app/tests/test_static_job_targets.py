from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


EXPECTED_TARGETS = {
    'datum': (
        'app.services.datum_static_service',
        'run_datum_static_apply_job',
    ),
    'first_break_qc': (
        'app.services.first_break_qc_service',
        'run_first_break_qc_job',
    ),
    'geometry_linkage': (
        'app.services.geometry_linkage_service',
        'run_geometry_linkage_build_job',
    ),
    'residual': (
        'app.services.residual_static_service',
        'run_residual_static_apply_job',
    ),
    'time_term': (
        'app.services.time_term_static_service',
        'run_time_term_static_apply_job',
    ),
    'refraction': (
        'app.statics.refraction.adapters.seisviewer2d.workflow_runner',
        'run_refraction_static_apply_job',
    ),
    'refraction_export': (
        'app.statics.refraction.adapters.seisviewer2d.export_runner',
        'run_refraction_static_export_job',
    ),
    'refraction_static_table_apply': (
        'app.statics.refraction.adapters.seisviewer2d.table_apply_runner',
        'run_refraction_static_table_apply_job',
    ),
}


def test_get_static_job_target_returns_registered_callables() -> None:
    registry = importlib.import_module('app.services.static_job_targets')

    assert set(registry.STATIC_JOB_TARGETS) == set(EXPECTED_TARGETS)
    for key, (module_name, attribute) in EXPECTED_TARGETS.items():
        module = importlib.import_module(module_name)
        assert registry.get_static_job_target(key) is getattr(module, attribute)


def test_get_static_job_target_rejects_unknown_key() -> None:
    registry = importlib.import_module('app.services.static_job_targets')

    with pytest.raises(KeyError):
        registry.get_static_job_target('__missing__')


def test_get_static_job_target_rejects_non_callable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = importlib.import_module('app.services.static_job_targets')
    monkeypatch.setitem(
        registry.STATIC_JOB_TARGETS,
        'not_callable',
        registry.StaticJobTargetSpec(
            module='app.services.static_job_targets',
            attribute='STATIC_JOB_TARGETS',
        ),
    )

    with pytest.raises(TypeError):
        registry.get_static_job_target('not_callable')


def test_static_job_targets_import_does_not_import_worker_modules() -> None:
    worker_modules = [module_name for module_name, _attribute in EXPECTED_TARGETS.values()]
    code = f"""
import importlib
import sys

worker_modules = {worker_modules!r}
registry = importlib.import_module('app.services.static_job_targets')
assert set(registry.STATIC_JOB_TARGETS) == {set(EXPECTED_TARGETS)!r}
imported = [module_name for module_name in worker_modules if module_name in sys.modules]
assert imported == [], imported
"""

    subprocess.run(
        [sys.executable, '-c', code],
        check=True,
        capture_output=True,
        text=True,
    )
