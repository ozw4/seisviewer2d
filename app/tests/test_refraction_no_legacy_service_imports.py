from __future__ import annotations

import importlib
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
LEGACY_IMPORT_PREFIX = 'app.services.refraction_static'


def _production_python_sources() -> list[Path]:
    return sorted(
        path
        for path in APP_ROOT.rglob('*.py')
        if 'tests' not in path.relative_to(APP_ROOT).parts
    )


def test_production_code_has_no_legacy_refraction_service_imports() -> None:
    offenders = [
        path.relative_to(APP_ROOT).as_posix()
        for path in _production_python_sources()
        if LEGACY_IMPORT_PREFIX in path.read_text(encoding='utf-8')
    ]

    assert offenders == []


def test_legacy_refraction_service_files_are_removed() -> None:
    service_root = APP_ROOT / 'services'

    assert sorted(service_root.glob('refraction_static*.py')) == []
    assert not (service_root / 'refraction_static_artifacts').exists()


def test_refraction_package_imports_without_legacy_services() -> None:
    for module_name in (
        'app.statics.refraction.contracts',
        'app.statics.refraction.artifacts',
        'app.statics.refraction.application',
        'app.statics.refraction.api',
        'app.statics.common',
    ):
        importlib.import_module(module_name)
