from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
from fastapi import APIRouter


_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATICS_PACKAGE_OLD_REEXPORTS = {
    'run_datum_static_apply_job',
    'run_first_break_qc_job',
    'run_geometry_linkage_build_job',
    'run_refraction_static_apply_job',
    'run_refraction_static_export_job',
    'run_residual_static_apply_job',
    'run_time_term_static_apply_job',
    'start_job_thread',
}


def _launch_module_ast() -> ast.Module:
    path = _REPO_ROOT / 'app/api/routers/statics/launch.py'
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return ()


def test_statics_router_package_public_surface_is_router_only() -> None:
    statics = importlib.import_module('app.api.routers.statics')

    assert statics.__all__ == ['router']
    assert isinstance(statics.router, APIRouter)
    for name in _STATICS_PACKAGE_OLD_REEXPORTS:
        assert not hasattr(statics, name)


def test_launch_module_does_not_reverse_import_statics_router_package() -> None:
    tree = _launch_module_ast()

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert 'statics_router_module' not in names

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names = {alias.name for alias in node.names}
            assert not (
                node.module == 'app.api.routers' and 'statics' in imported_names
            )
        elif isinstance(node, ast.Import):
            imported_modules = {alias.name for alias in node.names}
            assert 'app.api.routers.statics' not in imported_modules


def test_launch_static_job_uses_static_job_target_thread_wrapper() -> None:
    tree = _launch_module_ast()
    start_thread_values: list[tuple[str, ...]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name) and node.func.id == 'launch_managed_job'
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'start_thread':
                start_thread_values.append(_attribute_path(keyword.value))

    assert start_thread_values == [
        ('static_job_targets', 'start_static_job_thread')
    ]


def test_static_job_target_registry_lazy_imports_worker_modules() -> None:
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import sys

        sys.modules.pop('app.services.static_job_targets', None)
        registry = importlib.import_module('app.services.static_job_targets')
        worker_modules = {
            spec.module for spec in registry.STATIC_JOB_TARGETS.values()
        }
        for module_name in worker_modules:
            sys.modules.pop(module_name, None)

        sys.modules.pop('app.services.static_job_targets', None)
        registry = importlib.import_module('app.services.static_job_targets')
        imported = sorted(
            module_name for module_name in worker_modules if module_name in sys.modules
        )
        assert imported == [], imported

        with_key_error = False
        try:
            registry.get_static_job_target('__missing__')
        except KeyError:
            with_key_error = True
        assert with_key_error
        imported = sorted(
            module_name for module_name in worker_modules if module_name in sys.modules
        )
        assert imported == [], imported

        target = registry.get_static_job_target('datum')
        datum_module = registry.STATIC_JOB_TARGETS['datum'].module
        assert target.__module__ == datum_module
        assert datum_module in sys.modules
        imported = sorted(
            module_name for module_name in worker_modules if module_name in sys.modules
        )
        assert imported == [datum_module], imported
        """
    )

    subprocess.run(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_static_job_target_registry_rejects_invalid_key() -> None:
    registry = importlib.import_module('app.services.static_job_targets')

    with pytest.raises(KeyError):
        registry.get_static_job_target('__missing__')
