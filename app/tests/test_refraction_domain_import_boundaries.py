from __future__ import annotations

import ast
import fnmatch
import importlib
from pathlib import Path


DOMAIN_ROOT = Path(__file__).resolve().parents[1] / 'statics' / 'refraction' / 'domain'

DOMAIN_MODULES = {
    'types': 'RefractionStaticInputModel',
    'qc_types': 'RefractionFirstBreakQcSeries',
    'export_types': 'RefractionStaticEndpointExportRow',
    'export_units': 'normalize_export_units',
    'field_composition': 'compose_refraction_final_trace_shift',
    'manual_static': 'resolve_refraction_manual_static',
    'solver': 'solve_refraction_static_bounded_ls',
    'source_depth': 'resolve_refraction_source_depth',
    'table_import': 'import_refraction_static_tables',
    'table_validator': 'validate_canonical_static_table_rows',
    't1lsst': 'compute_t1lsst_1layer_thickness',
    'uphole': 'resolve_refraction_uphole',
    'v1': 'estimate_global_v1_from_direct_arrivals',
}

FORBIDDEN_IMPORTS = (
    'fastapi',
    'app.core.state',
    'app.services.reader',
    'app.services.job_manager',
    'app.services.job_runner',
    'app.services.job_artifact_refs',
)

FIELD_ARTIFACT_DOMAIN_MODULES = (
    'source_depth',
    't1lsst',
    'uphole',
    'v1',
)
FORBIDDEN_FIELD_ARTIFACT_IMPORT_MODULES = (
    'pathlib',
    'app.services.common.artifact_io',
)
FORBIDDEN_FIELD_ARTIFACT_SYMBOLS = {
    'Path',
    'write_csv_atomic',
    'write_json_atomic',
}
FORBIDDEN_FIELD_ARTIFACT_WRITER_PATTERNS = (
    'write_refraction_*_artifacts',
    'write_refraction_*_csv',
)


def test_refraction_domain_modules_import() -> None:
    importlib.import_module('app.statics.refraction.domain')

    for module_name, representative_name in DOMAIN_MODULES.items():
        new_module = importlib.import_module(
            f'app.statics.refraction.domain.{module_name}'
        )

        assert hasattr(new_module, representative_name)


def test_refraction_domain_sources_do_not_import_application_boundaries() -> None:
    for source_path in sorted(DOMAIN_ROOT.glob('*.py')):
        source = source_path.read_text()

        for forbidden_import in FORBIDDEN_IMPORTS:
            assert forbidden_import not in source, (
                f'{source_path.relative_to(DOMAIN_ROOT)} imports '
                f'application boundary {forbidden_import}'
            )


def test_refraction_field_domain_modules_do_not_implement_artifact_io() -> None:
    for module_name in FIELD_ARTIFACT_DOMAIN_MODULES:
        source_path = DOMAIN_ROOT / f'{module_name}.py'
        tree = ast.parse(source_path.read_text(encoding='utf-8'))

        imported_modules: list[str] = []
        imported_symbols: list[str] = []
        imported_writer_symbols: list[str] = []
        writer_functions: list[str] = []
        writer_calls: list[str] = []
        forbidden_calls: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
                imported_symbols.extend(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    imported_modules.append(node.module)
                imported_symbols.extend(alias.asname or alias.name for alias in node.names)
                imported_writer_symbols.extend(
                    alias.name
                    for alias in node.names
                    if _is_forbidden_field_artifact_writer_name(alias.name)
                )
            elif isinstance(node, ast.FunctionDef):
                if _is_forbidden_field_artifact_writer_name(node.name):
                    writer_functions.append(node.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    forbidden_calls.append('open')
                elif isinstance(
                    node.func, ast.Name
                ) and _is_forbidden_field_artifact_writer_name(node.func.id):
                    writer_calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute) and node.func.attr == 'open':
                    forbidden_calls.append('.open')
                elif isinstance(
                    node.func, ast.Attribute
                ) and _is_forbidden_field_artifact_writer_name(node.func.attr):
                    writer_calls.append(node.func.attr)

        assert not [
            imported_module
            for imported_module in imported_modules
            if imported_module in FORBIDDEN_FIELD_ARTIFACT_IMPORT_MODULES
            or imported_module.startswith(
                tuple(f'{name}.' for name in FORBIDDEN_FIELD_ARTIFACT_IMPORT_MODULES)
            )
        ], f'{source_path.name} imports artifact I/O modules'
        assert not (
            set(imported_symbols) & FORBIDDEN_FIELD_ARTIFACT_SYMBOLS
        ), f'{source_path.name} imports forbidden artifact I/O symbols'
        assert not imported_writer_symbols, (
            f'{source_path.name} imports artifact writer symbols'
        )
        assert not writer_functions, f'{source_path.name} defines artifact writers'
        assert not writer_calls, f'{source_path.name} calls artifact writers'
        assert not forbidden_calls, f'{source_path.name} uses file-open writer calls'


def _is_forbidden_field_artifact_writer_name(name: str) -> bool:
    return any(
        fnmatch.fnmatchcase(name, pattern)
        for pattern in FORBIDDEN_FIELD_ARTIFACT_WRITER_PATTERNS
    )
