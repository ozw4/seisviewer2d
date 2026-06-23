from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEIS_STATICS_VERSION = '0.4.0'


def test_local_seis_statics_shadow_package_is_absent() -> None:
    assert not (REPO_ROOT / 'seis_statics').exists()


def test_seis_statics_imports_from_external_distribution() -> None:
    seis_statics = importlib.import_module('seis_statics')

    origin = Path(seis_statics.__file__).resolve()
    assert not origin.is_relative_to(REPO_ROOT)
    assert metadata.version('seis-statics') == EXPECTED_SEIS_STATICS_VERSION


def test_required_external_seis_statics_surface_imports() -> None:
    module_names = [
        'seis_statics.validation',
        'seis_statics.datum',
        'seis_statics.residual',
        'seis_statics.trace_shift',
        'seis_statics.time_term',
        'seis_statics.time_term.apply_shift',
        'seis_statics.time_term.design_matrix',
        'seis_statics.time_term.moveout',
        'seis_statics.time_term.robust_solver',
        'seis_statics.time_term.sparse_solver',
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        origin = Path(module.__file__).resolve()
        assert not origin.is_relative_to(REPO_ROOT), module_name


def test_app_code_does_not_import_private_seis_statics_validation() -> None:
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / 'app').rglob('*.py')):
        source = path.read_text(encoding='utf-8')
        if 'seis_statics._validation' in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
