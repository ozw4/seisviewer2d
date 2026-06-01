from __future__ import annotations

import importlib


LEGACY_CONTRACT_IMPORTS = {
    'app.contracts.statics.refraction': (
        'RefractionStaticApplyRequest',
        'RefractionStaticQcBundleRequest',
        'RefractionStaticTableApplyRequest',
    ),
    'app.contracts.statics.refraction.apply': (
        'RefractionStaticApplyRequest',
        'RefractionStaticApplyResponse',
    ),
    'app.contracts.statics.refraction.export': (
        'RefractionStaticExportRequest',
        'RefractionStaticExportJobResponse',
    ),
    'app.contracts.statics.refraction.field_corrections': (
        'RefractionStaticFieldCorrectionsRequest',
    ),
    'app.contracts.statics.refraction.gather_preview': (
        'RefractionStaticGatherPreviewRequest',
        'RefractionStaticGatherPreviewResponse',
    ),
    'app.contracts.statics.refraction.inputs': (
        'RefractionStaticGeometryRequest',
        'RefractionStaticPickSourceRequest',
    ),
    'app.contracts.statics.refraction.model': (
        'RefractionStaticFirstLayerRequest',
        'RefractionStaticModelRequest',
    ),
    'app.contracts.statics.refraction.options': (
        'RefractionStaticMoveoutRequest',
        'RefractionStaticSolverRequest',
    ),
    'app.contracts.statics.refraction.qc': (
        'RefractionStaticPickMapRequest',
        'RefractionStaticQcBundleRequest',
    ),
    'app.contracts.statics.refraction.table_apply': (
        'RefractionStaticTableApplyRequest',
    ),
}

LEGACY_SERVICE_IMPORTS = {
    'app.services.refraction_static_export_service': (
        'run_refraction_static_export_job',
    ),
    'app.services.refraction_static_gather_preview': (
        'build_refraction_static_gather_preview',
    ),
    'app.services.refraction_static_inputs': (
        'build_refraction_static_input_model',
    ),
    'app.services.refraction_static_qc_bundle': (
        'build_refraction_static_qc_bundle',
    ),
    'app.services.refraction_static_qc_drilldown': (
        'build_refraction_static_qc_drilldown',
    ),
    'app.services.refraction_static_qc_endpoint_search': (
        'build_refraction_static_qc_endpoint_search',
    ),
    'app.services.refraction_static_service': (
        'run_refraction_static_apply_job',
    ),
    'app.services.refraction_static_station_structure': (
        'build_refraction_static_station_structure',
    ),
    'app.services.refraction_static_table_apply_service': (
        'run_refraction_static_table_apply_job',
    ),
}

LEGACY_ARTIFACT_IMPORTS = {
    'app.services.refraction_static_artifacts': (
        'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
        'write_refraction_static_artifacts',
        'write_refraction_static_solution_npz',
    ),
    'app.services.refraction_static_artifacts.contract': (
        'REFRACTION_STATIC_SOLUTION_NPZ_NAME',
    ),
    'app.services.refraction_static_artifacts.writer': (
        'write_refraction_static_artifacts',
    ),
}


def _assert_imports_resolve(imports: dict[str, tuple[str, ...]]) -> None:
    for module_name, names in imports.items():
        module = importlib.import_module(module_name)
        missing = sorted(name for name in names if not hasattr(module, name))

        assert missing == [], module_name


def test_legacy_refraction_contract_imports_resolve() -> None:
    _assert_imports_resolve(LEGACY_CONTRACT_IMPORTS)


def test_legacy_refraction_service_imports_resolve() -> None:
    _assert_imports_resolve(LEGACY_SERVICE_IMPORTS)


def test_legacy_refraction_artifact_imports_resolve() -> None:
    _assert_imports_resolve(LEGACY_ARTIFACT_IMPORTS)
