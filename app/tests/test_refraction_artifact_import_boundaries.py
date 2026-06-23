from __future__ import annotations

import ast
from pathlib import Path

ARTIFACTS_ROOT = (
    Path(__file__).resolve().parents[1] / 'statics' / 'refraction' / 'artifacts'
)
ARTIFACT_NAME_IMPORT_SOURCES = {
    'REFRACTION_SOURCE_DEPTH_QC_JSON_NAME': (
        'app.statics.refraction.artifacts.source_depth'
    ),
    'REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME': (
        'app.statics.refraction.artifacts.source_depth'
    ),
    'REFRACTION_UPHOLE_QC_JSON_NAME': 'app.statics.refraction.artifacts.uphole',
    'REFRACTION_UPHOLE_SOURCES_CSV_NAME': 'app.statics.refraction.artifacts.uphole',
    'REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME': (
        'app.statics.refraction.artifacts.t1lsst'
    ),
    'REFRACTION_V1_ESTIMATES_CSV_NAME': 'app.statics.refraction.artifacts.v1',
    'REFRACTION_V1_QC_JSON_NAME': 'app.statics.refraction.artifacts.v1',
}
ARTIFACT_NAME_CONSUMERS = (
    ARTIFACTS_ROOT / 'registry.py',
    ARTIFACTS_ROOT / 'writer.py',
    ARTIFACTS_ROOT / 'qc.py',
)
FIELD_DOMAIN_MODULES: tuple[str, ...] = ()


def test_refraction_artifacts_do_not_import_application_layer() -> None:
    for source_path in sorted(ARTIFACTS_ROOT.glob('*.py')):
        tree = ast.parse(source_path.read_text(encoding='utf-8'))
        imported_modules = _imported_modules(tree)
        assert not [
            module
            for module in imported_modules
            if module == 'app.statics.refraction.application'
            or module.startswith('app.statics.refraction.application.')
        ], f'{source_path.relative_to(ARTIFACTS_ROOT)} imports application layer'


def test_refraction_field_artifact_names_are_imported_from_artifact_modules() -> None:
    for source_path in ARTIFACT_NAME_CONSUMERS:
        tree = ast.parse(source_path.read_text(encoding='utf-8'))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            assert not (
                node.module in FIELD_DOMAIN_MODULES
                or node.module.startswith(
                    tuple(f'{module}.' for module in FIELD_DOMAIN_MODULES)
                )
            ), f'{source_path.name} imports field artifact names from domain'

            for alias in node.names:
                expected_module = ARTIFACT_NAME_IMPORT_SOURCES.get(alias.name)
                if expected_module is not None:
                    assert node.module == expected_module, (
                        f'{source_path.name} imports {alias.name} from {node.module}; '
                        f'expected {expected_module}'
                    )


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
            modules.extend(
                f'{node.module}.{alias.name}'
                for alias in node.names
                if alias.name != '*'
            )
    return modules
