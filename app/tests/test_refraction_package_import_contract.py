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

APPLICATION_IMPORTS = {
    'app.statics.refraction.application.workflow': (
        'run_refraction_static_apply_job',
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


def test_refraction_contract_shims_return_same_class_objects() -> None:
    from app.contracts.statics.refraction.apply import (
        RefractionStaticApplyRequest as OldApply,
    )
    from app.contracts.statics.refraction.export import (
        RefractionStaticExportJobRequest as OldExportJob,
    )
    from app.contracts.statics.refraction.gather_preview import (
        RefractionStaticGatherPreviewRequest as OldGatherPreview,
    )
    from app.contracts.statics.refraction.qc import (
        RefractionStaticQcBundleRequest as OldQcBundle,
    )
    from app.contracts.statics.refraction.table_apply import (
        RefractionStaticTableApplyRequest as OldTableApply,
    )
    from app.statics.refraction.contracts.apply import (
        RefractionStaticApplyRequest as NewApply,
    )
    from app.statics.refraction.contracts.export import (
        RefractionStaticExportJobRequest as NewExportJob,
    )
    from app.statics.refraction.contracts.gather_preview import (
        RefractionStaticGatherPreviewRequest as NewGatherPreview,
    )
    from app.statics.refraction.contracts.qc import (
        RefractionStaticQcBundleRequest as NewQcBundle,
    )
    from app.statics.refraction.contracts.table_apply import (
        RefractionStaticTableApplyRequest as NewTableApply,
    )

    assert OldApply is NewApply
    assert OldQcBundle is NewQcBundle
    assert OldGatherPreview is NewGatherPreview
    assert OldExportJob is NewExportJob
    assert OldTableApply is NewTableApply


def test_legacy_refraction_service_imports_resolve() -> None:
    _assert_imports_resolve(LEGACY_SERVICE_IMPORTS)


def test_refraction_application_imports_resolve() -> None:
    _assert_imports_resolve(APPLICATION_IMPORTS)


def test_legacy_refraction_artifact_imports_resolve() -> None:
    _assert_imports_resolve(LEGACY_ARTIFACT_IMPORTS)
