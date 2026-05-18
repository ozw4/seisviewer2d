from __future__ import annotations

import importlib
import pkgutil


def _refraction_static_artifact_submodule_names() -> set[str]:
    import app.services.refraction_static_artifacts as pkg

    return {module.name for module in pkgutil.iter_modules(pkg.__path__)}


def test_refraction_static_artifact_package_submodules_import() -> None:
    import app.services.refraction_static_artifacts as pkg

    for module_name in _refraction_static_artifact_submodule_names():
        importlib.import_module(f'{pkg.__name__}.{module_name}')


def test_refraction_static_artifact_facade_exports_core_public_api() -> None:
    import app.services.refraction_static_artifacts as artifacts

    required_names = {
        'write_refraction_static_artifacts',
        'write_refraction_static_solution_npz',
        'write_refraction_statics_csv',
        'write_refraction_static_qc_json',
        'write_refraction_static_history_json',
        'write_refraction_first_break_fit_qc_csv',
        'write_refraction_reduced_time_qc_csv',
        'write_refraction_line_profile_qc_artifacts',
        'write_refraction_grid_map_qc_csv',
        'write_refraction_static_component_qc_artifacts',
        'REFRACTION_STATIC_ARTIFACTS_JSON_NAME',
        'REFRACTION_STATIC_QC_JSON_NAME',
        'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
    }

    missing = sorted(name for name in required_names if not hasattr(artifacts, name))

    assert missing == []


def test_refraction_static_artifact_deleted_modules_are_absent() -> None:
    deleted_module_names = {
        '_legacy',
        'first_break_qc',
        'history',
        'reduced_time_qc',
        'row_context',
    }

    remaining_deleted_modules = sorted(
        deleted_module_names & _refraction_static_artifact_submodule_names()
    )

    assert remaining_deleted_modules == []
