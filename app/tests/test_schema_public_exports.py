from __future__ import annotations

from typing import Any


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
    'RefractionStaticGatherPreviewRequest',
    'RefractionStaticGatherPreviewResponse',
    'RefractionStaticTableApplyRequest',
    'RefractionStaticTableApplyResponse',
    'RefractionStaticExportJobRequest',
    'RefractionStaticExportJobResponse',
]


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


def test_schema_public_models_instantiate_from_legacy_import_path() -> None:
    BandpassParams = _legacy_schema_import('BandpassParams')
    PipelineOp = _legacy_schema_import('PipelineOp')
    PipelineSpec = _legacy_schema_import('PipelineSpec')
    RefractionStaticQcBundleRequest = _legacy_schema_import(
        'RefractionStaticQcBundleRequest'
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
    assert snap.enabled is False


def test_schema_public_constants_remain_available_from_legacy_import_path() -> None:
    default_formats = _legacy_schema_import(
        'REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS'
    )

    assert default_formats == ('canonical_static_table', 'time_term_spreadsheet')
