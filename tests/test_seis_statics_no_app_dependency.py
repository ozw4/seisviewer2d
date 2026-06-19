"""Guardrail tests for the standalone seis_statics package boundary."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / 'src' / 'seis_statics'

FORBIDDEN_IMPORT_PREFIXES = (
    'app',
    'fastapi',
    'pydantic',
    'segyio',
)


def _is_forbidden_module(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f'{prefix}.')
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def _prohibited_imports(path: Path, *, root: Path) -> list[str]:
    violations: list[str] = []
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        module_names: list[str] = []
        if isinstance(node, ast.Import):
            module_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module_names.append(node.module)

        for module_name in module_names:
            if _is_forbidden_module(module_name):
                relative_path = path.relative_to(root)
                violations.append(f'{relative_path}:{node.lineno}:{module_name}')
    return violations


def _scan_for_prohibited_imports(root: Path) -> tuple[list[Path], list[str]]:
    assert root.exists(), f'package root does not exist: {root}'
    paths = sorted(root.rglob('*.py'))
    assert len(paths) >= 1, f'package root contains no Python files: {root}'

    violations: list[str] = []
    for path in paths:
        violations.extend(_prohibited_imports(path, root=root))

    return paths, violations


def test_seis_statics_source_does_not_import_app_runtime_dependencies() -> None:
    paths, violations = _scan_for_prohibited_imports(PACKAGE_ROOT)

    assert len(paths) >= 1
    assert violations == []


def test_prohibited_import_helper_rejects_runtime_dependencies(tmp_path: Path) -> None:
    source = tmp_path / 'bad_imports.py'
    source.write_text(
        '\n'.join(
            [
                'import app.services.job_manager',
                'from fastapi import APIRouter',
                'from pydantic import BaseModel',
                'import segyio',
            ]
        ),
        encoding='utf-8',
    )

    violations = _prohibited_imports(source, root=tmp_path)

    assert violations == [
        'bad_imports.py:1:app.services.job_manager',
        'bad_imports.py:2:fastapi',
        'bad_imports.py:3:pydantic',
        'bad_imports.py:4:segyio',
    ]
