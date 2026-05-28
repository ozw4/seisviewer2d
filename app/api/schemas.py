"""Pydantic models for describing pipeline operations."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_finite_float,
    require_trace_header_byte,
)
from app.contracts.batch import (
    BatchApplyRequest as BatchApplyRequest,
    BatchApplyResponse as BatchApplyResponse,
    BatchJobFile as BatchJobFile,
    BatchJobFilesResponse as BatchJobFilesResponse,
    BatchJobStatusResponse as BatchJobStatusResponse,
    PickOptions as PickOptions,
    SnapOptions as SnapOptions,
)
from app.contracts.pipeline import (
    AnalyzerName as AnalyzerName,
    BandpassParams as BandpassParams,
    DenoiseParams as DenoiseParams,
    FbpickParams as FbpickParams,
    PipelineAllResponse as PipelineAllResponse,
    PipelineJobStatusResponse as PipelineJobStatusResponse,
    PipelineOp as PipelineOp,
    PipelineSectionResponse as PipelineSectionResponse,
    PipelineSpec as PipelineSpec,
    TransformName as TransformName,
)
from app.contracts.statics.common import (
    StaticJobFile as StaticJobFile,
    StaticJobFilesResponse as StaticJobFilesResponse,
    StaticJobStatusResponse as StaticJobStatusResponse,
)
from app.contracts.statics.datum import (
    DatumStaticApplyOptions as DatumStaticApplyOptions,
    DatumStaticApplyRequest as DatumStaticApplyRequest,
    DatumStaticApplyResponse as DatumStaticApplyResponse,
    DatumStaticDatumRequest as DatumStaticDatumRequest,
    DatumStaticExistingStaticsRequest as DatumStaticExistingStaticsRequest,
    DatumStaticGeometryRequest as DatumStaticGeometryRequest,
)
from app.contracts.statics.first_break_qc import (
    FirstBreakQcDatumSolutionRequest as FirstBreakQcDatumSolutionRequest,
    FirstBreakQcJobResponse as FirstBreakQcJobResponse,
    FirstBreakQcOffsetRequest as FirstBreakQcOffsetRequest,
    FirstBreakQcOptionsRequest as FirstBreakQcOptionsRequest,
    FirstBreakQcPickSourceRequest as FirstBreakQcPickSourceRequest,
    FirstBreakQcRequest as FirstBreakQcRequest,
)
from app.contracts.statics.geometry_linkage import (
    StaticLinkageBuildRequest as StaticLinkageBuildRequest,
    StaticLinkageBuildResponse as StaticLinkageBuildResponse,
    StaticLinkageGeometryRequest as StaticLinkageGeometryRequest,
    StaticLinkageOptionsRequest as StaticLinkageOptionsRequest,
)
from app.contracts.statics.residual import (
    ResidualStaticApplyOptions as ResidualStaticApplyOptions,
    ResidualStaticApplyRequest as ResidualStaticApplyRequest,
    ResidualStaticApplyResponse as ResidualStaticApplyResponse,
    ResidualStaticDatumSolutionRequest as ResidualStaticDatumSolutionRequest,
    ResidualStaticGeometryRequest as ResidualStaticGeometryRequest,
    ResidualStaticMoveoutRequest as ResidualStaticMoveoutRequest,
    ResidualStaticOffsetRequest as ResidualStaticOffsetRequest,
    ResidualStaticPickSourceRequest as ResidualStaticPickSourceRequest,
    ResidualStaticRobustRequest as ResidualStaticRobustRequest,
    ResidualStaticSolverRequest as ResidualStaticSolverRequest,
)
from app.contracts.statics.time_term import (
    TimeTermStaticApplyOptions as TimeTermStaticApplyOptions,
    TimeTermStaticApplyRequest as TimeTermStaticApplyRequest,
    TimeTermStaticApplyResponse as TimeTermStaticApplyResponse,
    TimeTermStaticGeometryRequest as TimeTermStaticGeometryRequest,
    TimeTermStaticLinkageRequest as TimeTermStaticLinkageRequest,
    TimeTermStaticMoveoutRequest as TimeTermStaticMoveoutRequest,
    TimeTermStaticPickSourceRequest as TimeTermStaticPickSourceRequest,
    TimeTermStaticRobustRequest as TimeTermStaticRobustRequest,
    TimeTermStaticSolverRequest as TimeTermStaticSolverRequest,
    TimeTermStaticVelocityRequest as TimeTermStaticVelocityRequest,
)
from app.contracts.statics.refraction.common import (
    _REFRACTION_STATIC_LAYER_ORDER as _REFRACTION_STATIC_LAYER_ORDER,
    REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS as REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS,
    RefractionStaticExportFormat as RefractionStaticExportFormat,
    RefractionStaticGatherPreviewAxis as RefractionStaticGatherPreviewAxis,
    RefractionStaticGatherPreviewOverlayLayer as RefractionStaticGatherPreviewOverlayLayer,
    RefractionStaticGatherPreviewSampleSource as RefractionStaticGatherPreviewSampleSource,
    RefractionStaticGatherPreviewScaling as RefractionStaticGatherPreviewScaling,
    RefractionStaticLayerKind as RefractionStaticLayerKind,
    RefractionStaticLayerVelocityMode as RefractionStaticLayerVelocityMode,
    RefractionStaticQcBundleCoordinateMode as RefractionStaticQcBundleCoordinateMode,
    RefractionStaticQcBundleInclude as RefractionStaticQcBundleInclude,
)
from app.contracts.statics.refraction.apply import (
    RefractionStaticApplyRequest as RefractionStaticApplyRequest,
    RefractionStaticApplyResponse as RefractionStaticApplyResponse,
)
from app.contracts.statics.refraction.export import (
    RefractionStaticExportJobRequest as RefractionStaticExportJobRequest,
    RefractionStaticExportJobResponse as RefractionStaticExportJobResponse,
    RefractionStaticExportRequest as RefractionStaticExportRequest,
)
from app.contracts.statics.refraction.gather_preview import (
    RefractionStaticGatherPreviewRequest as RefractionStaticGatherPreviewRequest,
    RefractionStaticGatherPreviewResponse as RefractionStaticGatherPreviewResponse,
)
from app.contracts.statics.refraction.field_corrections import (
    RefractionStaticFieldCorrectionArtifactRequest as RefractionStaticFieldCorrectionArtifactRequest,
    RefractionStaticFieldCorrectionCompositionRequest as RefractionStaticFieldCorrectionCompositionRequest,
    RefractionStaticFieldCorrectionsRequest as RefractionStaticFieldCorrectionsRequest,
    RefractionStaticManualStaticInlineEntry as RefractionStaticManualStaticInlineEntry,
    RefractionStaticManualStaticRequest as RefractionStaticManualStaticRequest,
    RefractionStaticSourceDepthCorrectionRequest as RefractionStaticSourceDepthCorrectionRequest,
    RefractionStaticUpholeCorrectionRequest as RefractionStaticUpholeCorrectionRequest,
)
from app.contracts.statics.refraction.inputs import (
    RefractionStaticGeometryRequest as RefractionStaticGeometryRequest,
    RefractionStaticLinkageRequest as RefractionStaticLinkageRequest,
    RefractionStaticPickSourceRequest as RefractionStaticPickSourceRequest,
)
from app.contracts.statics.refraction.model import (
    RefractionStaticFirstLayerRequest as RefractionStaticFirstLayerRequest,
    RefractionStaticLayerRequest as RefractionStaticLayerRequest,
    RefractionStaticModelRequest as RefractionStaticModelRequest,
    RefractionStaticRefractorCellRequest as RefractionStaticRefractorCellRequest,
)
from app.contracts.statics.refraction.options import (
    RefractionStaticApplyOptions as RefractionStaticApplyOptions,
    RefractionStaticConversionRequest as RefractionStaticConversionRequest,
    RefractionStaticDatumRequest as RefractionStaticDatumRequest,
    RefractionStaticMoveoutRequest as RefractionStaticMoveoutRequest,
    RefractionStaticReducedTimeQcRequest as RefractionStaticReducedTimeQcRequest,
    RefractionStaticRobustRequest as RefractionStaticRobustRequest,
    RefractionStaticSolverRequest as RefractionStaticSolverRequest,
)
from app.contracts.statics.refraction.qc import (
    RefractionStaticQcBundleRequest as RefractionStaticQcBundleRequest,
    RefractionStaticQcBundleResponse as RefractionStaticQcBundleResponse,
    RefractionStaticQcDownsamplingEntry as RefractionStaticQcDownsamplingEntry,
    RefractionStaticQcDrilldownCellTarget as RefractionStaticQcDrilldownCellTarget,
    RefractionStaticQcDrilldownEndpointTarget as RefractionStaticQcDrilldownEndpointTarget,
    RefractionStaticQcDrilldownObservations as RefractionStaticQcDrilldownObservations,
    RefractionStaticQcDrilldownRequest as RefractionStaticQcDrilldownRequest,
    RefractionStaticQcDrilldownResponse as RefractionStaticQcDrilldownResponse,
    RefractionStaticQcDrilldownTarget as RefractionStaticQcDrilldownTarget,
    RefractionStaticQcTabularView as RefractionStaticQcTabularView,
)
from app.contracts.statics.refraction.table_apply import (
    RefractionStaticTableApplyRequest as RefractionStaticTableApplyRequest,
    RefractionStaticTableApplyResponse as RefractionStaticTableApplyResponse,
)
from app.utils.validation import require_non_negative_int, require_positive_int

RefractionStaticStationStructureXAxis = Literal[
    'auto',
    'global_receiver_number',
    'station_number',
    'inline_m',
]
RefractionStaticStationStructureVelocityField = Literal[
    'auto',
    'v1',
    'v2',
    'v3',
    'vsub',
]
RefractionStaticStationStructureDepthField = Literal[
    'auto',
    'sh1',
    'sh2',
    'sh3',
    'refractor_depth',
    'refractor_elevation',
    'layer1_base_elevation',
    'layer2_base_elevation',
]
RefractionStaticQcEndpointKind = Literal['source', 'receiver', 'both']
RefractionStaticQcEndpointRecordKind = Literal['source', 'receiver']
RefractionStaticQcEndpointSort = Literal[
    'station_id_asc',
    'station_id_desc',
    'residual_rms_desc',
    'residual_rms_asc',
    'pick_count_desc',
    'pick_count_asc',
    'endpoint_key_asc',
]
RefractionStaticQcEndpointStatusFilter = Literal['all', 'ok', 'problem']


class RefractionStaticQcEndpointSearchRequest(BaseModel):
    """Request model for server-side refraction endpoint selector search."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    endpoint_kind: RefractionStaticQcEndpointKind = 'both'
    query: str = ''
    status_filter: RefractionStaticQcEndpointStatusFilter = 'all'
    sort: RefractionStaticQcEndpointSort = 'endpoint_key_asc'
    limit: int = 50
    offset: int = 0

    @field_validator('job_id')
    @classmethod
    def _check_job_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('job_id must be a non-empty string')
        return value

    @field_validator('query')
    @classmethod
    def _check_query(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError('query must be a string')
        return value.strip()

    @field_validator('limit', mode='before')
    @classmethod
    def _check_limit(cls, value: object) -> int:
        limit = require_positive_int(value, 'limit')
        if limit > 200:
            raise ValueError('limit must be <= 200')
        return limit

    @field_validator('offset', mode='before')
    @classmethod
    def _check_offset(cls, value: object) -> int:
        return require_non_negative_int(value, 'offset')


class RefractionStaticQcEndpointSearchRecord(BaseModel):
    """One source or receiver endpoint selector record."""

    model_config = ConfigDict(extra='forbid')

    endpoint_kind: RefractionStaticQcEndpointRecordKind
    endpoint_key: str
    label: str
    station_id: int | None = None
    node_id: int | None = None
    x_m: float | None = None
    y_m: float | None = None
    surface_elevation_m: float | None = None
    pick_count: int | None = None
    residual_rms_ms: float | None = None
    datum_status: str | None = None
    static_status: str | None = None


class RefractionStaticQcEndpointSearchResponse(BaseModel):
    """Server-side endpoint selector search response."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    statics_kind: Literal['refraction']
    endpoint_kind: RefractionStaticQcEndpointKind
    query: str
    total: int
    limit: int
    offset: int
    records: list[RefractionStaticQcEndpointSearchRecord]


class RefractionStaticPickMapGeometryRequest(RefractionStaticGeometryRequest):
    """Geometry configuration for all-gather refraction pick-map QC."""

    receiver_number_mode: Literal['global_sequential'] = 'global_sequential'


class RefractionStaticPickMapRequest(BaseModel):
    """Request model for all-gather refraction pick-map QC."""

    model_config = ConfigDict(extra='forbid')

    job_id: str | None = None
    file_id: str | None = None
    key1_byte: int = 189
    key2_byte: int = 193
    gather_start: float | None = None
    gather_end: float | None = None
    pick_source: RefractionStaticPickSourceRequest | None = None
    geometry: RefractionStaticPickMapGeometryRequest = Field(
        default_factory=RefractionStaticPickMapGeometryRequest,
    )

    @field_validator('job_id', 'file_id')
    @classmethod
    def _check_optional_text(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f'{info.field_name} must be a non-empty string')
        return value

    @field_validator('key1_byte', 'key2_byte', mode='before')
    @classmethod
    def _check_key_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @field_validator('gather_start', 'gather_end', mode='before')
    @classmethod
    def _check_optional_gather_bound(cls, value: object, info: Any) -> float | None:
        if value is None or value == '':
            return None
        return _require_finite_float(value, info.field_name)

    @model_validator(mode='after')
    def _check_source(self) -> 'RefractionStaticPickMapRequest':
        if self.job_id is not None and self.pick_source is not None:
            raise ValueError('pick_source must be omitted when job_id is provided')
        if self.job_id is None:
            if self.file_id is None:
                raise ValueError('file_id is required when job_id is omitted')
            if self.pick_source is None:
                raise ValueError('pick_source is required when job_id is omitted')
        return self


class RefractionStaticPickMapData(BaseModel):
    """Columnar arrays for all-gather pick-map plotting."""

    model_config = ConfigDict(extra='forbid')

    gather_id: list[int | str | None]
    receiver_number: list[int | float | None]
    pick_before_ms: list[float]
    trace_index: list[int | None] = Field(default_factory=list)
    shot_id: list[int | str | None] = Field(default_factory=list)
    source_id: list[int | str | None] = Field(default_factory=list)
    receiver_id: list[int | str | None] = Field(default_factory=list)
    offset_m: list[float | None] = Field(default_factory=list)
    used_in_statics: list[bool | None] = Field(default_factory=list)
    pick_after_ms: list[float | None] = Field(default_factory=list)
    applied_shift_ms: list[float | None] = Field(default_factory=list)
    offset_used: list[float | None] = Field(default_factory=list)


class RefractionStaticPickMapResponse(BaseModel):
    """All-gather refraction pick-map QC response."""

    model_config = ConfigDict(extra='forbid')

    job_id: str | None = None
    statics_kind: Literal['refraction']
    mode: Literal['pre_statics', 'completed_job']
    status_message: str
    has_after_statics: bool
    receiver_number_mode: Literal['global_sequential']
    gather_range: dict[str, int | float | str | None]
    pick_map: RefractionStaticPickMapData


class RefractionStaticStationStructureRequest(BaseModel):
    """Request model for station-structure refraction QC."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    gather_start: float | None = None
    gather_end: float | None = None
    x_axis: RefractionStaticStationStructureXAxis = 'auto'
    velocity_field: RefractionStaticStationStructureVelocityField = 'auto'
    depth_field: RefractionStaticStationStructureDepthField = 'auto'

    @field_validator('job_id')
    @classmethod
    def _check_job_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('job_id must be a non-empty string')
        return value

    @field_validator('gather_start', 'gather_end', mode='before')
    @classmethod
    def _check_optional_gather_bound(cls, value: object, info: Any) -> float | None:
        if value is None or value == '':
            return None
        return _require_finite_float(value, info.field_name)


class RefractionStaticStationStructureSeries(BaseModel):
    """Columnar series for one endpoint side in station-structure QC."""

    model_config = ConfigDict(extra='forbid')

    x: list[float | int]
    y: list[float]
    endpoint_key: list[str]
    status: list[str]


class RefractionStaticStationStructurePanel(BaseModel):
    """One station-structure QC plot payload."""

    model_config = ConfigDict(extra='forbid')

    field: str
    label: str
    unit: str
    source: RefractionStaticStationStructureSeries
    receiver: RefractionStaticStationStructureSeries


class RefractionStaticStationStructureResponse(BaseModel):
    """Station-structure QC payload for completed refraction static jobs."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    statics_kind: Literal['refraction']
    view_kind: Literal['station_structure']
    x_axis: str
    x_axis_label: str
    x_axis_status: str = 'ok'
    station_mapping: dict[str, Any] = Field(default_factory=dict)
    filter_status: str
    gather_range: dict[str, int | float | str | None]
    colors: dict[Literal['source', 'receiver'], str]
    time_term: RefractionStaticStationStructurePanel
    velocity: RefractionStaticStationStructurePanel
    depth: RefractionStaticStationStructurePanel
    warnings: list[str] = Field(default_factory=list)
