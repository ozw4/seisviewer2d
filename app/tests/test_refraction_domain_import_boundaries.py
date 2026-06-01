from __future__ import annotations

import importlib
from pathlib import Path


DOMAIN_ROOT = Path(__file__).resolve().parents[1] / 'statics' / 'refraction' / 'domain'

DOMAIN_MODULES = {
    'types': 'RefractionStaticInputModel',
    'status': 'REFRACTION_STATIC_STATUSES',
    'qc_types': 'RefractionFirstBreakQcSeries',
    'export_types': 'RefractionStaticEndpointExportRow',
    'export_units': 'normalize_export_units',
    'first_layer': 'resolve_weathering_velocity_m_s',
    'layer_config': 'normalize_refraction_static_layers',
    'layer_observations': 'build_refraction_layer_observation_masks',
    'cell_coordinates': 'project_refraction_cell_coordinates',
    'cell_grid': 'build_refraction_cell_grid',
    'cell_regularization': 'build_cell_slowness_smoothing_rows',
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
