from __future__ import annotations

import importlib
from typing import Any

import app.api.schemas as legacy


EXPECTED_SCHEMA_EXPORTS = [
    'TransformName',
    'AnalyzerName',
    'RefractionStaticExportFormat',
    'RefractionStaticQcBundleInclude',
    'RefractionStaticQcBundleCoordinateMode',
    'RefractionStaticGatherPreviewAxis',
    'RefractionStaticGatherPreviewOverlayLayer',
    'RefractionStaticGatherPreviewScaling',
    'RefractionStaticGatherPreviewSampleSource',
    'REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS',
    '_REFRACTION_STATIC_LAYER_ORDER',
    'BandpassParams',
    'DenoiseParams',
    'FbpickParams',
    'PipelineOp',
    'PipelineSpec',
    'PipelineSectionResponse',
    'PipelineAllResponse',
    'PipelineJobStatusResponse',
    'SnapOptions',
    'PickOptions',
    'BatchApplyRequest',
    'BatchApplyResponse',
    'BatchJobStatusResponse',
    'BatchJobFile',
    'BatchJobFilesResponse',
    'StaticJobStatusResponse',
    'StaticJobFile',
    'StaticJobFilesResponse',
    'DatumStaticGeometryRequest',
    'DatumStaticDatumRequest',
    'DatumStaticExistingStaticsRequest',
    'DatumStaticApplyOptions',
    'DatumStaticApplyRequest',
    'DatumStaticApplyResponse',
    'require_trace_header_byte',
    'StaticLinkageGeometryRequest',
    'StaticLinkageOptionsRequest',
    'StaticLinkageBuildRequest',
    'StaticLinkageBuildResponse',
    'FirstBreakQcDatumSolutionRequest',
    'FirstBreakQcPickSourceRequest',
    'FirstBreakQcOffsetRequest',
    'FirstBreakQcOptionsRequest',
    'FirstBreakQcRequest',
    'FirstBreakQcJobResponse',
    'ResidualStaticDatumSolutionRequest',
    'ResidualStaticPickSourceRequest',
    'ResidualStaticGeometryRequest',
    'ResidualStaticOffsetRequest',
    'ResidualStaticMoveoutRequest',
    'ResidualStaticSolverRequest',
    'ResidualStaticRobustRequest',
    'ResidualStaticApplyOptions',
    'ResidualStaticApplyRequest',
    'ResidualStaticApplyResponse',
    'TimeTermStaticPickSourceRequest',
    'TimeTermStaticGeometryRequest',
    'TimeTermStaticLinkageRequest',
    'TimeTermStaticVelocityRequest',
    'TimeTermStaticMoveoutRequest',
    'TimeTermStaticRobustRequest',
    'TimeTermStaticSolverRequest',
    'TimeTermStaticApplyOptions',
    'TimeTermStaticApplyRequest',
    'TimeTermStaticApplyResponse',
    'RefractionStaticPickSourceRequest',
    'RefractionStaticGeometryRequest',
    'RefractionStaticLinkageRequest',
    'RefractionStaticFirstLayerRequest',
    'RefractionStaticRefractorCellRequest',
    'RefractionStaticLayerKind',
    'RefractionStaticLayerVelocityMode',
    'RefractionStaticLayerRequest',
    'RefractionStaticModelRequest',
    'RefractionStaticMoveoutRequest',
    'RefractionStaticRobustRequest',
    'RefractionStaticSolverRequest',
    'RefractionStaticDatumRequest',
    'RefractionStaticApplyOptions',
    'RefractionStaticConversionRequest',
    'RefractionStaticReducedTimeQcRequest',
    'RefractionStaticFieldCorrectionArtifactRequest',
    'RefractionStaticSourceDepthCorrectionRequest',
    'RefractionStaticUpholeCorrectionRequest',
    'RefractionStaticManualStaticInlineEntry',
    'RefractionStaticManualStaticRequest',
    'RefractionStaticFieldCorrectionCompositionRequest',
    'RefractionStaticFieldCorrectionsRequest',
    'RefractionStaticExportRequest',
    'RefractionStaticApplyRequest',
    'RefractionStaticApplyResponse',
    'RefractionStaticQcBundleRequest',
    'RefractionStaticQcDownsamplingEntry',
    'RefractionStaticQcTabularView',
    'RefractionStaticQcBundleResponse',
    'RefractionStaticQcDrilldownEndpointTarget',
    'RefractionStaticQcDrilldownCellTarget',
    'RefractionStaticQcDrilldownTarget',
    'RefractionStaticQcDrilldownRequest',
    'RefractionStaticQcDrilldownObservations',
    'RefractionStaticQcDrilldownResponse',
    'RefractionStaticQcEndpointKind',
    'RefractionStaticQcEndpointRecordKind',
    'RefractionStaticQcEndpointSearchRecord',
    'RefractionStaticQcEndpointSearchRequest',
    'RefractionStaticQcEndpointSearchResponse',
    'RefractionStaticQcEndpointSort',
    'RefractionStaticQcEndpointStatusFilter',
    'RefractionStaticPickMapData',
    'RefractionStaticPickMapGeometryRequest',
    'RefractionStaticPickMapRequest',
    'RefractionStaticPickMapResponse',
    'RefractionStaticStationStructureDepthField',
    'RefractionStaticStationStructurePanel',
    'RefractionStaticStationStructureRequest',
    'RefractionStaticStationStructureResponse',
    'RefractionStaticStationStructureSeries',
    'RefractionStaticStationStructureVelocityField',
    'RefractionStaticStationStructureXAxis',
    'RefractionStaticGatherPreviewRequest',
    'RefractionStaticGatherPreviewResponse',
    'RefractionStaticTableApplyRequest',
    'RefractionStaticTableApplyResponse',
    'RefractionStaticExportJobRequest',
    'RefractionStaticExportJobResponse',
]


LEGACY_EXPORT_MODULES = {
    'PipelineOp': 'app.contracts.pipeline',
    'PipelineSpec': 'app.contracts.pipeline',
    'SnapOptions': 'app.contracts.batch',
    'BatchApplyRequest': 'app.contracts.batch',
    'require_trace_header_byte': 'app.contracts._validation',
    'StaticJobStatusResponse': 'app.contracts.statics.common',
    'DatumStaticApplyRequest': 'app.contracts.statics.datum',
    'FirstBreakQcRequest': 'app.contracts.statics.first_break_qc',
    'StaticLinkageBuildRequest': 'app.contracts.statics.geometry_linkage',
    'ResidualStaticOffsetRequest': 'app.contracts.statics.residual',
    'ResidualStaticApplyRequest': 'app.contracts.statics.residual',
    'TimeTermStaticMoveoutRequest': 'app.contracts.statics.time_term',
    'TimeTermStaticApplyRequest': 'app.contracts.statics.time_term',
    'REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS': (
        'app.contracts.statics.refraction.common'
    ),
    'RefractionStaticGeometryRequest': 'app.contracts.statics.refraction.inputs',
    'RefractionStaticModelRequest': 'app.contracts.statics.refraction.model',
    'RefractionStaticSolverRequest': 'app.contracts.statics.refraction.options',
    'RefractionStaticApplyOptions': 'app.contracts.statics.refraction.options',
    'RefractionStaticFieldCorrectionsRequest': (
        'app.contracts.statics.refraction.field_corrections'
    ),
    'RefractionStaticExportRequest': 'app.contracts.statics.refraction.export',
    'RefractionStaticApplyRequest': 'app.contracts.statics.refraction.apply',
    'RefractionStaticQcBundleRequest': 'app.contracts.statics.refraction.qc',
    'RefractionStaticGatherPreviewRequest': (
        'app.contracts.statics.refraction.gather_preview'
    ),
    'RefractionStaticTableApplyRequest': (
        'app.contracts.statics.refraction.table_apply'
    ),
}


def _legacy_schema_import(name: str) -> Any:
    module = __import__('app.api.schemas', fromlist=[name])
    return getattr(module, name)


def test_schema_public_exports_remain_available() -> None:
    missing = [
        name
        for name in EXPECTED_SCHEMA_EXPORTS
        if not hasattr(__import__('app.api.schemas', fromlist=[name]), name)
    ]

    assert missing == []


def test_schema_all_matches_public_exports() -> None:
    schemas = __import__('app.api.schemas', fromlist=['__all__'])

    assert list(schemas.__all__) == EXPECTED_SCHEMA_EXPORTS


def test_schema_all_has_no_duplicates() -> None:
    schemas = __import__('app.api.schemas', fromlist=['__all__'])

    assert len(schemas.__all__) == len(set(schemas.__all__))


def test_legacy_schema_exports_are_direct_reexports() -> None:
    for name, module_name in LEGACY_EXPORT_MODULES.items():
        module = importlib.import_module(module_name)

        assert getattr(legacy, name) is getattr(module, name)


def test_statics_package_exports_are_direct_reexports() -> None:
    statics = importlib.import_module('app.contracts.statics')
    datum = importlib.import_module('app.contracts.statics.datum')
    residual = importlib.import_module('app.contracts.statics.residual')
    time_term = importlib.import_module('app.contracts.statics.time_term')
    refraction = importlib.import_module('app.contracts.statics.refraction')

    assert statics.DatumStaticApplyRequest is datum.DatumStaticApplyRequest
    assert statics.ResidualStaticApplyRequest is residual.ResidualStaticApplyRequest
    assert statics.TimeTermStaticApplyRequest is time_term.TimeTermStaticApplyRequest
    assert statics.RefractionStaticApplyRequest is (
        refraction.RefractionStaticApplyRequest
    )


def test_refraction_package_exports_are_direct_reexports() -> None:
    refraction = importlib.import_module('app.contracts.statics.refraction')
    apply = importlib.import_module('app.contracts.statics.refraction.apply')
    model = importlib.import_module('app.contracts.statics.refraction.model')
    options = importlib.import_module('app.contracts.statics.refraction.options')
    field_corrections = importlib.import_module(
        'app.contracts.statics.refraction.field_corrections'
    )
    qc = importlib.import_module('app.contracts.statics.refraction.qc')

    assert refraction.RefractionStaticApplyRequest is (
        apply.RefractionStaticApplyRequest
    )
    assert refraction.RefractionStaticModelRequest is (
        model.RefractionStaticModelRequest
    )
    assert refraction.RefractionStaticSolverRequest is (
        options.RefractionStaticSolverRequest
    )
    assert refraction.RefractionStaticFieldCorrectionsRequest is (
        field_corrections.RefractionStaticFieldCorrectionsRequest
    )
    assert refraction.RefractionStaticQcBundleRequest is (
        qc.RefractionStaticQcBundleRequest
    )


def test_schema_public_models_instantiate_from_legacy_import_path() -> None:
    BandpassParams = _legacy_schema_import('BandpassParams')
    PipelineOp = _legacy_schema_import('PipelineOp')
    PipelineSpec = _legacy_schema_import('PipelineSpec')
    RefractionStaticQcBundleRequest = _legacy_schema_import(
        'RefractionStaticQcBundleRequest'
    )
    RefractionStaticGeometryRequest = _legacy_schema_import(
        'RefractionStaticGeometryRequest'
    )
    RefractionStaticMoveoutRequest = _legacy_schema_import(
        'RefractionStaticMoveoutRequest'
    )
    RefractionStaticApplyOptions = _legacy_schema_import(
        'RefractionStaticApplyOptions'
    )
    ResidualStaticOffsetRequest = _legacy_schema_import(
        'ResidualStaticOffsetRequest'
    )
    TimeTermStaticMoveoutRequest = _legacy_schema_import(
        'TimeTermStaticMoveoutRequest'
    )
    SnapOptions = _legacy_schema_import('SnapOptions')

    bandpass = BandpassParams(low_hz=5.0, high_hz=60.0)
    op = PipelineOp(
        kind='transform',
        name='bandpass',
        params={'low_hz': 5.0, 'high_hz': 60.0},
    )
    spec = PipelineSpec(steps=[op])
    qc_bundle = RefractionStaticQcBundleRequest(job_id='job-1')
    refraction_geometry = RefractionStaticGeometryRequest()
    refraction_moveout = RefractionStaticMoveoutRequest()
    refraction_apply = RefractionStaticApplyOptions()
    residual_offset = ResidualStaticOffsetRequest()
    time_term_moveout = TimeTermStaticMoveoutRequest()
    snap = SnapOptions()

    assert bandpass.low_hz == 5.0
    assert spec.steps[0].name == 'bandpass'
    assert qc_bundle.include == [
        'summary',
        'first_break',
        'profiles',
        'cells',
        'static_components',
    ]
    assert refraction_geometry.source_id_byte == 9
    assert refraction_moveout.offset_byte == 37
    assert refraction_apply.output_dtype == 'float32'
    assert residual_offset.offset_byte == 37
    assert time_term_moveout.distance_source == 'geometry'
    assert snap.enabled is False


def test_schema_public_constants_remain_available_from_legacy_import_path() -> None:
    default_formats = _legacy_schema_import(
        'REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS'
    )
    layer_order = _legacy_schema_import('_REFRACTION_STATIC_LAYER_ORDER')

    assert default_formats == ('canonical_static_table', 'time_term_spreadsheet')
    assert layer_order == {'v2_t1': 0, 'v3_t2': 1, 'vsub_t3': 2}
