from __future__ import annotations

import ast
from pathlib import Path

APPLICATION_ROOT = (
    Path(__file__).resolve().parents[1] / 'statics' / 'refraction' / 'application'
)
REFRACTION_ROOT = Path(__file__).resolve().parents[1] / 'statics' / 'refraction'

FORBIDDEN_MODULES = (
    'app.core.state',
    'app.services.reader',
    'app.services.job_runner',
    'app.services.job_manager',
    'app.services.job_artifact_refs',
    'app.services.corrected_trace_store',
    'app.services.trace_store_registration',
    'app.trace_store.reader',
    'app.statics.refraction.adapters.seisviewer2d',
)

MIGRATED_LEGACY_HELPER_MODULES = (
    'app.statics.refraction.domain.first_layer',
    'app.statics.refraction.domain.status',
    'app.statics.refraction.domain.cell_coordinates',
    'app.statics.refraction.domain.cell_grid',
    'app.statics.refraction.domain.cell_regularization',
    'app.statics.refraction.domain.cell_velocity_status',
    'app.statics.refraction.domain.layer_config',
    'app.statics.refraction.domain.layer_observations',
)


def test_refraction_application_does_not_import_seisviewer2d_runtime() -> None:
    for source_path in sorted(APPLICATION_ROOT.glob('*.py')):
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or '',)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                modules = (node.value,)
            else:
                continue
            for module in modules:
                for forbidden in FORBIDDEN_MODULES:
                    assert module != forbidden and not module.startswith(
                        f'{forbidden}.'
                    ), (
                        f'{source_path.relative_to(APPLICATION_ROOT)} references '
                        f'{forbidden}'
                    )


def test_refraction_production_does_not_import_migrated_legacy_helpers() -> None:
    for source_path in sorted(REFRACTION_ROOT.rglob('*.py')):
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or '',)
            else:
                continue
            for module in modules:
                for legacy_module in MIGRATED_LEGACY_HELPER_MODULES:
                    assert module != legacy_module and not module.startswith(
                        f'{legacy_module}.'
                    ), (
                        f'{source_path.relative_to(REFRACTION_ROOT)} imports '
                        f'{legacy_module}; use seis_statics.refraction instead'
                    )
