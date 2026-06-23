from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / 'app'
SOURCE_ROOTS = tuple(
    path
    for path in (
        APP_ROOT,
        REPO_ROOT / 'tests',
        REPO_ROOT / 'scripts',
    )
    if path.exists()
)
DOMAIN_ROOT = APP_ROOT / 'statics' / 'refraction' / 'domain'
LEGACY_DOMAIN_PREFIX = '.'.join(('app', 'statics', 'refraction', 'domain'))


def test_refraction_domain_package_is_removed() -> None:
    assert not DOMAIN_ROOT.exists()


def test_refraction_domain_package_cannot_be_imported() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(LEGACY_DOMAIN_PREFIX)


def test_refraction_sources_do_not_import_legacy_domain_package() -> None:
    offenders: list[str] = []
    for source_path in _source_paths():
        if source_path == Path(__file__).resolve():
            continue
        tree = ast.parse(source_path.read_text(encoding='utf-8'))
        for module in _imported_modules(tree):
            if module == LEGACY_DOMAIN_PREFIX or module.startswith(
                f'{LEGACY_DOMAIN_PREFIX}.'
            ):
                offenders.append(
                    f'{source_path.relative_to(REPO_ROOT).as_posix()}: {module}'
                )

    assert offenders == []


def _source_paths() -> list[Path]:
    return sorted(
        source_path
        for root in SOURCE_ROOTS
        for source_path in root.rglob('*.py')
    )


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules
