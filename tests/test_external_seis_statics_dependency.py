from __future__ import annotations

import ast
import importlib
from importlib import metadata
from pathlib import Path
import pkgutil


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEIS_STATICS_VERSION = '0.4.1'
EXPECTED_SEIS_STATICS_REQUIREMENT = (
    'seis-statics @ git+https://github.com/ozw4/seis-statics.git@v0.4.1'
)
REQUIREMENTS_PATH = REPO_ROOT / '.devcontainer' / 'requirements-dev.txt'

PRIVATE_SEIS_STATICS_IMPORT_PREFIXES = (
    'seis_statics._',
    'seis_statics.refraction._',
    'seis_statics.time_term._',
    'seis_statics.residual._',
    'seis_statics.datum._',
)
FORBIDDEN_REVERSE_IMPORT_PREFIXES = (
    'app',
    'fastapi',
    'pydantic',
    'segyio',
)


def test_local_seis_statics_shadow_package_is_absent() -> None:
    assert not (REPO_ROOT / 'seis_statics').exists()


def test_seis_statics_imports_from_external_distribution() -> None:
    seis_statics = importlib.import_module('seis_statics')

    origin = Path(seis_statics.__file__).resolve()
    assert not origin.is_relative_to(REPO_ROOT)
    assert metadata.version('seis-statics') == EXPECTED_SEIS_STATICS_VERSION


def test_seis_statics_dependency_pin_is_immutable_release_tag() -> None:
    lines = [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]
    pins = [line for line in lines if line.startswith('seis-statics')]

    assert pins == [EXPECTED_SEIS_STATICS_REQUIREMENT]
    assert '@main' not in pins[0]
    assert '-e ' not in pins[0]
    assert '../' not in pins[0]
    assert './' not in pins[0]


def test_required_external_seis_statics_surface_imports() -> None:
    from seis_statics.refraction import (
        resolve_smoothed_refraction_floating_datum,
        solve_refraction_multilayer_time_terms,
    )

    assert callable(resolve_smoothed_refraction_floating_datum)
    assert callable(solve_refraction_multilayer_time_terms)

    module_names = [
        'seis_statics.validation',
        'seis_statics.datum',
        'seis_statics.residual',
        'seis_statics.refraction',
        'seis_statics.refraction.bedrock',
        'seis_statics.refraction.cell_coordinates',
        'seis_statics.refraction.cell_grid',
        'seis_statics.refraction.cell_regularization',
        'seis_statics.refraction.datum',
        'seis_statics.refraction.design_matrix',
        'seis_statics.refraction.field_composition',
        'seis_statics.refraction.first_layer',
        'seis_statics.refraction.half_intercept',
        'seis_statics.refraction.layer_config',
        'seis_statics.refraction.layer_observations',
        'seis_statics.refraction.manual_static',
        'seis_statics.refraction.multilayer_conversion',
        'seis_statics.refraction.multilayer_solver',
        'seis_statics.refraction.source_depth',
        'seis_statics.refraction.solver',
        'seis_statics.refraction.status',
        'seis_statics.refraction.t1lsst',
        'seis_statics.refraction.uphole',
        'seis_statics.refraction.v1',
        'seis_statics.refraction.weathering',
        'seis_statics.refraction.weathering_replacement',
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


def test_app_code_does_not_import_private_seis_statics_modules() -> None:
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / 'app').rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for module in _imported_modules(tree):
            if module.startswith(PRIVATE_SEIS_STATICS_IMPORT_PREFIXES):
                offenders.append(f'{path.relative_to(REPO_ROOT)}: {module}')

    assert offenders == []


def test_external_seis_statics_package_has_no_reverse_app_dependency() -> None:
    package = importlib.import_module('seis_statics')
    offenders: list[str] = []

    for module_info in pkgutil.walk_packages(package.__path__, 'seis_statics.'):
        module = importlib.import_module(module_info.name)
        module_path = getattr(module, '__file__', None)
        if not module_path or not module_path.endswith('.py'):
            continue
        path = Path(module_path)
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for imported_module in _imported_modules(tree):
            if imported_module == 'app' or imported_module.startswith(
                tuple(f'{prefix}.' for prefix in FORBIDDEN_REVERSE_IMPORT_PREFIXES)
            ):
                offenders.append(f'{module_info.name}: {imported_module}')
            elif imported_module in FORBIDDEN_REVERSE_IMPORT_PREFIXES:
                offenders.append(f'{module_info.name}: {imported_module}')

    assert offenders == []


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules
