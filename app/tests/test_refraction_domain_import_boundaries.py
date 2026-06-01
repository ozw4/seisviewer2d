from __future__ import annotations

import importlib
from pathlib import Path


DOMAIN_ROOT = Path(__file__).resolve().parents[1] / 'statics' / 'refraction' / 'domain'

MOVED_MODULES = {
    'refraction_static_types': ('types', 'RefractionStaticInputModel'),
    'refraction_static_status': ('status', 'REFRACTION_STATIC_STATUSES'),
    'refraction_static_qc_types': ('qc_types', 'RefractionFirstBreakQcSeries'),
    'refraction_static_export_types': (
        'export_types',
        'RefractionStaticEndpointExportRow',
    ),
    'refraction_static_export_units': ('export_units', 'normalize_export_units'),
    'refraction_static_first_layer': (
        'first_layer',
        'resolve_weathering_velocity_m_s',
    ),
    'refraction_static_layer_config': (
        'layer_config',
        'normalize_refraction_static_layers',
    ),
    'refraction_static_layer_observations': (
        'layer_observations',
        'build_refraction_layer_observation_masks',
    ),
    'refraction_static_cell_coordinates': (
        'cell_coordinates',
        'project_refraction_cell_coordinates',
    ),
    'refraction_static_cell_grid': ('cell_grid', 'build_refraction_cell_grid'),
    'refraction_static_cell_regularization': (
        'cell_regularization',
        'build_cell_slowness_smoothing_rows',
    ),
    'refraction_static_field_composition': (
        'field_composition',
        'compose_refraction_final_trace_shift',
    ),
    'refraction_static_manual_static': (
        'manual_static',
        'resolve_refraction_manual_static',
    ),
    'refraction_static_solver': ('solver', 'solve_refraction_static_bounded_ls'),
    'refraction_static_source_depth': (
        'source_depth',
        'resolve_refraction_source_depth',
    ),
    'refraction_static_table_import': (
        'table_import',
        'import_refraction_static_tables',
    ),
    'refraction_static_table_validator': (
        'table_validator',
        'validate_canonical_static_table_rows',
    ),
    'refraction_static_t1lsst': ('t1lsst', 'compute_t1lsst_1layer_thickness'),
    'refraction_static_uphole': ('uphole', 'resolve_refraction_uphole'),
    'refraction_static_v1': ('v1', 'estimate_global_v1_from_direct_arrivals'),
}

FORBIDDEN_IMPORTS = (
    'fastapi',
    'app.core.state',
    'app.services.reader',
    'app.services.job_manager',
    'app.services.job_runner',
    'app.services.job_artifact_refs',
)


def test_refraction_domain_modules_import_and_shims_share_objects() -> None:
    importlib.import_module('app.statics.refraction.domain')

    for old_name, (new_name, representative_name) in MOVED_MODULES.items():
        old_module = importlib.import_module(f'app.services.{old_name}')
        new_module = importlib.import_module(
            f'app.statics.refraction.domain.{new_name}'
        )

        assert getattr(old_module, representative_name) is getattr(
            new_module,
            representative_name,
        )


def test_refraction_domain_sources_do_not_import_application_boundaries() -> None:
    for source_path in sorted(DOMAIN_ROOT.glob('*.py')):
        source = source_path.read_text()

        for forbidden_import in FORBIDDEN_IMPORTS:
            assert forbidden_import not in source, (
                f'{source_path.relative_to(DOMAIN_ROOT)} imports '
                f'application boundary {forbidden_import}'
            )
