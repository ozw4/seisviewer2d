"""Pydantic models for describing pipeline operations."""

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_bool,
    _require_finite_float,
    _require_positive_finite_float,
    _require_positive_int,
    _validate_artifact_basename,
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

class RefractionStaticFieldCorrectionArtifactRequest(BaseModel):
    """Artifact-table reference used by M4 field-correction request blocks."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    artifact_name: str

    @field_validator('job_id', mode='before')
    @classmethod
    def _check_job_id(cls, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(
                'field_corrections artifact job_id must be a non-empty string'
            )
        return value

    @field_validator('artifact_name', mode='before')
    @classmethod
    def _check_artifact_name(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(
                'field_corrections artifact_name must be a plain file name'
            )
        return _validate_artifact_basename(
            value,
            'field_corrections artifact_name',
        )


class RefractionStaticSourceDepthCorrectionRequest(BaseModel):
    """M4 source-depth correction request contract."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['none', 'weathering_velocity_time'] = 'none'
    source_depth_byte: int | None = None
    source_depth_unit: Literal['m'] = 'm'
    positive_down: bool = True
    max_abs_source_depth_m: float = 100.0

    @field_validator('source_depth_byte', mode='before')
    @classmethod
    def _check_source_depth_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(
            value,
            'field_corrections.source_depth.source_depth_byte',
        )

    @field_validator('positive_down', mode='before')
    @classmethod
    def _check_positive_down(cls, value: object) -> bool:
        return _require_bool(value, 'field_corrections.source_depth.positive_down')

    @field_validator('max_abs_source_depth_m', mode='before')
    @classmethod
    def _check_max_abs_source_depth_m(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'field_corrections.source_depth.max_abs_source_depth_m',
        )


class RefractionStaticUpholeCorrectionRequest(BaseModel):
    """M4 uphole correction request contract."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['none', 'header_time', 'manual_table'] = 'none'
    uphole_time_byte: int | None = None
    uphole_time_unit: Literal['s', 'ms'] = 's'
    positive_time_means_delay: bool = True
    manual_table: RefractionStaticFieldCorrectionArtifactRequest | None = None
    max_abs_uphole_time_s: float = 1.0

    @field_validator('uphole_time_byte', mode='before')
    @classmethod
    def _check_uphole_time_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(
            value,
            'field_corrections.uphole.uphole_time_byte',
        )

    @field_validator('positive_time_means_delay', mode='before')
    @classmethod
    def _check_positive_time_means_delay(cls, value: object) -> bool:
        return _require_bool(
            value,
            'field_corrections.uphole.positive_time_means_delay',
        )

    @field_validator('max_abs_uphole_time_s', mode='before')
    @classmethod
    def _check_max_abs_uphole_time_s(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'field_corrections.uphole.max_abs_uphole_time_s',
        )

    @model_validator(mode='after')
    def _check_uphole_config(self) -> 'RefractionStaticUpholeCorrectionRequest':
        if self.mode == 'header_time' and self.uphole_time_byte is None:
            raise ValueError(
                'field_corrections.uphole.uphole_time_byte is required when '
                'field_corrections.uphole.mode is header_time'
            )
        if self.mode == 'manual_table' and self.manual_table is None:
            raise ValueError(
                'field_corrections.uphole.manual_table is required when '
                'field_corrections.uphole.mode is manual_table'
            )
        return self


class RefractionStaticManualStaticInlineEntry(BaseModel):
    """Inline endpoint manual-static value for the M4 request contract."""

    model_config = ConfigDict(extra='forbid')

    endpoint_id: int
    value: float

    @field_validator('endpoint_id', mode='before')
    @classmethod
    def _check_endpoint_id(cls, value: object) -> int:
        return require_non_negative_int(
            value,
            'field_corrections.manual_static.inline_table.endpoint_id',
        )

    @field_validator('value', mode='before')
    @classmethod
    def _check_value(cls, value: object) -> float:
        return _require_finite_float(
            value,
            'field_corrections.manual_static.inline_table.value',
        )


class RefractionStaticManualStaticRequest(BaseModel):
    """M4 manual source/receiver static request contract."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['none', 'artifact_table', 'inline_table'] = 'none'
    sign_convention: Literal['applied_shift_s', 'delay_positive_ms'] | None = None
    source_table_artifact: RefractionStaticFieldCorrectionArtifactRequest | None = None
    receiver_table_artifact: RefractionStaticFieldCorrectionArtifactRequest | None = None
    source_inline_table: list[RefractionStaticManualStaticInlineEntry] | None = None
    receiver_inline_table: list[RefractionStaticManualStaticInlineEntry] | None = None
    allow_missing_endpoints: bool = True

    @field_validator('allow_missing_endpoints', mode='before')
    @classmethod
    def _check_allow_missing_endpoints(cls, value: object) -> bool:
        return _require_bool(
            value,
            'field_corrections.manual_static.allow_missing_endpoints',
        )

    @model_validator(mode='after')
    def _check_manual_static_config(self) -> 'RefractionStaticManualStaticRequest':
        has_artifact = (
            self.source_table_artifact is not None
            or self.receiver_table_artifact is not None
        )
        has_inline = bool(self.source_inline_table) or bool(self.receiver_inline_table)
        has_manual_values = has_artifact or has_inline
        if self.mode == 'artifact_table' and not has_artifact:
            raise ValueError(
                'field_corrections.manual_static.source_table_artifact or '
                'receiver_table_artifact is required when '
                'field_corrections.manual_static.mode is artifact_table'
            )
        if self.mode == 'inline_table' and not has_inline:
            raise ValueError(
                'field_corrections.manual_static.source_inline_table or '
                'receiver_inline_table is required when '
                'field_corrections.manual_static.mode is inline_table'
            )
        if self.mode == 'none' and has_manual_values:
            raise ValueError(
                'field_corrections.manual_static.mode must be artifact_table or '
                'inline_table when manual static values are supplied'
            )
        if has_manual_values and self.sign_convention is None:
            raise ValueError(
                'field_corrections.manual_static.sign_convention is required '
                'when manual static values are supplied'
            )
        return self


class RefractionStaticFieldCorrectionCompositionRequest(BaseModel):
    """M4 field-correction component composition request contract."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    apply_to_trace_shift: bool = True
    invalid_component_policy: Literal['fail', 'skip_invalid_traces'] = 'fail'
    double_application_policy: Literal['warn', 'fail', 'allow'] = 'warn'

    @field_validator('enabled', 'apply_to_trace_shift', mode='before')
    @classmethod
    def _check_bool(cls, value: object, info: Any) -> bool:
        return _require_bool(
            value,
            f'field_corrections.composition.{info.field_name}',
        )


class RefractionStaticFieldCorrectionsRequest(BaseModel):
    """M4 source-depth, uphole, manual static, and composition contract."""

    model_config = ConfigDict(extra='forbid')

    source_depth: RefractionStaticSourceDepthCorrectionRequest = Field(
        default_factory=RefractionStaticSourceDepthCorrectionRequest,
    )
    uphole: RefractionStaticUpholeCorrectionRequest = Field(
        default_factory=RefractionStaticUpholeCorrectionRequest,
    )
    manual_static: RefractionStaticManualStaticRequest = Field(
        default_factory=RefractionStaticManualStaticRequest,
    )
    composition: RefractionStaticFieldCorrectionCompositionRequest = Field(
        default_factory=RefractionStaticFieldCorrectionCompositionRequest,
    )


class RefractionStaticExportRequest(BaseModel):
    """M5 export options accepted by public refraction statics endpoints."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    formats: list[RefractionStaticExportFormat] = Field(default_factory=list)
    units: Literal['seconds', 'milliseconds'] = 'milliseconds'
    rounding_ms: float | None = 0.001
    include_inactive_endpoints: bool = True
    include_legacy_alias_columns: bool = True
    fail_on_invalid_static_status: bool = True

    @field_validator('rounding_ms')
    @classmethod
    def _check_rounding_ms(cls, value: float | None) -> float | None:
        if value is None:
            return None
        rounded = float(value)
        if not math.isfinite(rounded) or rounded < 0.0:
            raise ValueError('export.rounding_ms must be finite and >= 0')
        return rounded

    @model_validator(mode='after')
    def _check_supported_export_options(self) -> 'RefractionStaticExportRequest':
        if self.units != 'milliseconds':
            raise ValueError('export.units must be "milliseconds"')
        if self.rounding_ms not in (None, 0.001):
            raise ValueError(
                'export.rounding_ms is reserved and must be null or 0.001'
            )
        if not self.include_legacy_alias_columns:
            raise ValueError('export.include_legacy_alias_columns must be true')
        return self


class RefractionStaticApplyRequest(BaseModel):
    """Request model for ``/statics/refraction/apply`` jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    pick_source: RefractionStaticPickSourceRequest
    geometry: RefractionStaticGeometryRequest = Field(
        default_factory=RefractionStaticGeometryRequest,
    )
    linkage: RefractionStaticLinkageRequest = Field(
        default_factory=RefractionStaticLinkageRequest,
    )
    model: RefractionStaticModelRequest
    moveout: RefractionStaticMoveoutRequest = Field(
        default_factory=RefractionStaticMoveoutRequest,
    )
    solver: RefractionStaticSolverRequest = Field(
        default_factory=RefractionStaticSolverRequest,
    )
    datum: RefractionStaticDatumRequest = Field(
        default_factory=RefractionStaticDatumRequest,
    )
    conversion: RefractionStaticConversionRequest = Field(
        default_factory=RefractionStaticConversionRequest,
    )
    reduced_time_qc: RefractionStaticReducedTimeQcRequest = Field(
        default_factory=RefractionStaticReducedTimeQcRequest,
    )
    field_corrections: RefractionStaticFieldCorrectionsRequest = Field(
        default_factory=RefractionStaticFieldCorrectionsRequest,
    )
    export: RefractionStaticExportRequest = Field(
        default_factory=RefractionStaticExportRequest,
    )
    apply: RefractionStaticApplyOptions = Field(
        default_factory=RefractionStaticApplyOptions,
    )

    @field_validator('file_id', mode='before')
    @classmethod
    def _check_file_id(cls, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('file_id must be a non-empty string')
        return value

    @field_validator('key1_byte', 'key2_byte', mode='before')
    @classmethod
    def _check_key_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @model_validator(mode='after')
    def _check_supported_refraction_apply_model(
        self,
    ) -> 'RefractionStaticApplyRequest':
        if self.conversion.mode == 't1lsst_multilayer':
            enabled_count = self.model.enabled_refraction_layer_count
            if self.conversion.layer_count != enabled_count:
                enabled_kinds = [
                    layer.kind
                    for layer in self.model.layers or []
                    if layer.enabled
                ]
                enabled_text = ', '.join(enabled_kinds) if enabled_kinds else 'none'
                raise ValueError(
                    'conversion.layer_count must match enabled refraction layers; '
                    f'conversion.layer_count={self.conversion.layer_count!r}, '
                    f'enabled layer kinds={enabled_text}'
                )
        if (
            self.field_corrections.source_depth.mode == 'weathering_velocity_time'
            and self.field_corrections.source_depth.source_depth_byte is None
            and self.geometry.source_depth_byte is None
        ):
            raise ValueError(
                'field_corrections.source_depth.source_depth_byte or '
                'geometry.source_depth_byte is required when '
                'field_corrections.source_depth.mode is weathering_velocity_time'
            )
        return self


class RefractionStaticApplyResponse(BaseModel):
    """Response model for creating a refraction static apply job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str
    requested_formats: list[RefractionStaticExportFormat] | None = None


class RefractionStaticQcBundleRequest(BaseModel):
    """Request model for compact refraction static QC bundles.

    ``max_points`` is applied independently to each tabular view with
    deterministic even-index sampling that keeps the first and last row when
    downsampling is needed.
    """

    model_config = ConfigDict(extra='forbid')

    job_id: str
    include: list[RefractionStaticQcBundleInclude] = Field(
        default_factory=lambda: [
            'summary',
            'first_break',
            'profiles',
            'cells',
            'static_components',
        ],
    )
    max_points: int = 20000
    coordinate_mode: RefractionStaticQcBundleCoordinateMode = 'auto'

    @field_validator('job_id')
    @classmethod
    def _check_job_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('job_id must be a non-empty string')
        return value

    @field_validator('include')
    @classmethod
    def _check_include(
        cls,
        value: list[RefractionStaticQcBundleInclude],
    ) -> list[RefractionStaticQcBundleInclude]:
        if not value:
            raise ValueError('include must contain at least one view')
        seen: set[str] = set()
        unique: list[RefractionStaticQcBundleInclude] = []
        for item in value:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    @field_validator('max_points', mode='before')
    @classmethod
    def _check_max_points(cls, value: object) -> int:
        return require_positive_int(value, 'max_points')


class RefractionStaticQcDownsamplingEntry(BaseModel):
    """Downsampling metadata for one QC bundle tabular view."""

    model_config = ConfigDict(extra='forbid')

    total_points: int
    returned_points: int
    downsampled: bool
    method: str


class RefractionStaticQcTabularView(BaseModel):
    """One sampled tabular artifact exposed as JSON records."""

    model_config = ConfigDict(extra='forbid')

    artifact: str
    columns: list[str]
    total_points: int
    returned_points: int
    downsampled: bool
    downsampling_method: str
    records: list[dict[str, Any]]


class RefractionStaticQcBundleResponse(BaseModel):
    """Compact QC bundle assembled from completed refraction static artifacts."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    statics_kind: Literal['refraction']
    sign_convention: str
    coordinate_mode: RefractionStaticQcBundleCoordinateMode
    summary: dict[str, Any]
    artifacts: dict[str, str]
    available_views: list[str]
    unavailable_views: list[str] = Field(default_factory=list)
    unavailable_view_reasons: dict[str, str] = Field(default_factory=dict)
    views: dict[str, RefractionStaticQcTabularView] = Field(default_factory=dict)
    downsampling: dict[str, RefractionStaticQcDownsamplingEntry] = Field(
        default_factory=dict,
    )


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


class RefractionStaticQcDrilldownEndpointTarget(BaseModel):
    """Endpoint target for detailed refraction QC drilldown."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['endpoint']
    endpoint_kind: Literal['source', 'receiver']
    endpoint_key: str

    @field_validator('endpoint_key')
    @classmethod
    def _check_endpoint_key(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('target.endpoint_key must be a non-empty string')
        return value


class RefractionStaticQcDrilldownCellTarget(BaseModel):
    """Cell target for detailed refraction QC drilldown."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['cell']
    layer_kind: RefractionStaticLayerKind = 'v2_t1'
    cell_ix: int
    cell_iy: int

    @field_validator('cell_ix', 'cell_iy', mode='before')
    @classmethod
    def _check_cell_index(cls, value: object, info: Any) -> int:
        return require_non_negative_int(value, f'target.{info.field_name}')


RefractionStaticQcDrilldownTarget = Annotated[
    RefractionStaticQcDrilldownEndpointTarget | RefractionStaticQcDrilldownCellTarget,
    Field(discriminator='kind'),
]


class RefractionStaticQcDrilldownRequest(BaseModel):
    """Request model for detailed refraction QC drilldown."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    target: RefractionStaticQcDrilldownTarget
    max_observations: int = 200

    @field_validator('job_id')
    @classmethod
    def _check_job_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('job_id must be a non-empty string')
        return value

    @field_validator('max_observations', mode='before')
    @classmethod
    def _check_max_observations(cls, value: object) -> int:
        return require_positive_int(value, 'max_observations')


class RefractionStaticQcDrilldownObservations(BaseModel):
    """Capped contributing-observation records for a drilldown target."""

    model_config = ConfigDict(extra='forbid')

    total_count: int
    returned_count: int
    capped: bool
    cap_method: str
    records: list[dict[str, Any]]


class RefractionStaticQcDrilldownResponse(BaseModel):
    """Detailed refraction QC drilldown assembled from completed artifacts."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    statics_kind: Literal['refraction']
    sign_convention: str
    drilldown_kind: Literal['endpoint', 'cell']
    target: dict[str, Any]
    max_observations: int
    artifacts: dict[str, str]
    observations: RefractionStaticQcDrilldownObservations
    residual_summary: dict[str, Any]
    endpoint: dict[str, Any] | None = None
    cell: dict[str, Any] | None = None
    static_components: dict[str, Any] | None = None
    time_terms: dict[str, Any] | None = None
    thicknesses: dict[str, Any] | None = None
    velocities: dict[str, Any] | None = None
    pick_counts: dict[str, Any] | None = None
    statuses: dict[str, Any] | None = None
    velocity: dict[str, Any] | None = None
    fold: dict[str, Any] | None = None
    endpoint_counts: dict[str, Any] | None = None
    neighbor_velocity_summary: dict[str, Any] | None = None


class RefractionStaticGatherPreviewRequest(BaseModel):
    """Request model for refraction gather preview data."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    file_id: str
    key1: int | None = None
    key1_byte: int = 189
    key2_byte: int = 193
    gather_axis: RefractionStaticGatherPreviewAxis = 'section'
    endpoint_key: str | None = None
    x0: int | None = None
    x1: int | None = None
    y0: int | None = None
    y1: int | None = None
    time_start_s: float | None = None
    time_end_s: float | None = None
    step_x: int = 1
    step_y: int = 1
    scaling: RefractionStaticGatherPreviewScaling = 'amax'
    reduction_velocity_m_s: float | None = None
    overlay_layers: list[RefractionStaticGatherPreviewOverlayLayer] = Field(
        default_factory=lambda: [
            'observed_first_break',
            'modeled_first_break',
            'reduced_time',
            'static_shift_trace_curve',
        ],
    )
    max_traces: int = 500
    max_samples: int = 4000

    @field_validator('job_id', 'file_id')
    @classmethod
    def _check_non_empty_text(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f'{info.field_name} must be a non-empty string')
        return value

    @field_validator('endpoint_key')
    @classmethod
    def _check_endpoint_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError('endpoint_key must be a non-empty string')
        return value

    @field_validator('key1', mode='before')
    @classmethod
    def _check_key1_int(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('key1 must be an integer')
        return value

    @field_validator('x0', 'x1', 'y0', 'y1', mode='before')
    @classmethod
    def _check_nonnegative_int(cls, value: object, info: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{info.field_name} must be a non-negative integer')
        if value < 0:
            raise ValueError(f'{info.field_name} must be a non-negative integer')
        return value

    @field_validator('key1_byte', 'key2_byte', mode='before')
    @classmethod
    def _check_key_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @field_validator('step_x', 'step_y', 'max_traces', 'max_samples', mode='before')
    @classmethod
    def _check_positive_count(cls, value: object, info: Any) -> int:
        return _require_positive_int(value, info.field_name)

    @field_validator('reduction_velocity_m_s', mode='before')
    @classmethod
    def _check_reduction_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        velocity = _require_finite_float(value, 'reduction_velocity_m_s')
        if velocity <= 0.0:
            raise ValueError('reduction_velocity_m_s must be finite and > 0')
        return velocity

    @field_validator('time_start_s', 'time_end_s', mode='before')
    @classmethod
    def _check_time_range_value(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        out = _require_finite_float(value, info.field_name)
        if out < 0.0:
            raise ValueError(f'{info.field_name} must be finite and >= 0')
        return out

    @field_validator('overlay_layers')
    @classmethod
    def _check_overlay_layers(
        cls,
        value: list[RefractionStaticGatherPreviewOverlayLayer],
    ) -> list[RefractionStaticGatherPreviewOverlayLayer]:
        if not value:
            raise ValueError('overlay_layers must not be empty')
        if len(set(value)) != len(value):
            raise ValueError('overlay_layers must not contain duplicates')
        return value

    @model_validator(mode='after')
    def _check_window_and_target(self) -> 'RefractionStaticGatherPreviewRequest':
        sample_fields = (self.y0, self.y1)
        time_fields = (self.time_start_s, self.time_end_s)
        has_sample_range = any(value is not None for value in sample_fields)
        has_time_range = any(value is not None for value in time_fields)
        if has_sample_range and has_time_range:
            raise ValueError('provide either y0/y1 or time_start_s/time_end_s')
        if not has_sample_range and not has_time_range:
            raise ValueError('y0/y1 or time_start_s/time_end_s is required')
        if has_sample_range and (self.y0 is None or self.y1 is None):
            raise ValueError('y0 and y1 must be provided together')
        if has_time_range and (
            self.time_start_s is None or self.time_end_s is None
        ):
            raise ValueError('time_start_s and time_end_s must be provided together')
        if self.y0 is not None and self.y1 is not None and self.y1 < self.y0:
            raise ValueError('y1 must be greater than or equal to y0')
        if (
            self.time_start_s is not None
            and self.time_end_s is not None
            and self.time_end_s <= self.time_start_s
        ):
            raise ValueError('time_end_s must be greater than time_start_s')

        has_trace_range = self.x0 is not None or self.x1 is not None
        if has_trace_range and (self.x0 is None or self.x1 is None):
            raise ValueError('x0 and x1 must be provided together')
        if self.x0 is not None and self.x1 is not None and self.x1 < self.x0:
            raise ValueError('x1 must be greater than or equal to x0')
        if self.gather_axis == 'section' and self.key1 is None:
            raise ValueError('key1 is required for section gather axis')
        if self.gather_axis == 'section' and not has_trace_range:
            raise ValueError('x0 and x1 are required for section gather axis')
        if self.key1 is None and has_trace_range:
            raise ValueError('key1 is required when x0/x1 are provided')
        if self.gather_axis in {'source', 'receiver'} and self.endpoint_key is None:
            raise ValueError('endpoint_key is required for source/receiver gather axes')
        if self.gather_axis == 'section' and self.endpoint_key is not None:
            raise ValueError('endpoint_key is only valid for source/receiver gathers')
        return self


class RefractionStaticGatherPreviewResponse(BaseModel):
    """Refraction before/after gather preview manifest and overlays."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    statics_kind: Literal['refraction']
    sign_convention: str
    raw_window_ref: dict[str, Any]
    corrected_window_ref: dict[str, Any]
    raw_samples: list[list[float]]
    corrected_samples: list[list[float]]
    corrected_samples_source: RefractionStaticGatherPreviewSampleSource
    dt_s: float
    shape: list[int]
    window: dict[str, Any]
    gather: dict[str, Any]
    x_indices: list[int]
    trace_indices: list[int]
    offset_m: list[float | None]
    source_endpoint_key: list[str | None]
    receiver_endpoint_key: list[str | None]
    observed_pick_time_s: list[float | None]
    modeled_pick_time_s: list[float | None]
    residual_s: list[float | None]
    final_trace_shift_s: list[float | None]
    corrected_observed_pick_time_s: list[float | None]
    corrected_modeled_pick_time_s: list[float | None]
    reduced_observed_time_s: list[float | None] | None = None
    reduced_modeled_time_s: list[float | None] | None = None
    overlay_status: dict[str, Any]
    artifacts: dict[str, str]


class RefractionStaticTableApplyRequest(BaseModel):
    """Request model for standalone M5 static-table TraceStore apply jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    geometry: RefractionStaticGeometryRequest = Field(
        default_factory=RefractionStaticGeometryRequest,
    )
    source_table_artifact_id: str | None = None
    receiver_table_artifact_id: str | None = None
    combined_table_artifact_id: str | None = None
    source_key_header: Literal['endpoint_key', 'endpoint_id'] | None = None
    receiver_key_header: Literal['endpoint_key', 'endpoint_id'] | None = None
    register_corrected_file: bool = True
    output_name: str | None = None
    allow_missing_source_static: bool = False
    allow_missing_receiver_static: bool = False
    missing_static_policy: Literal['fail', 'zero'] = 'fail'
    double_application_policy: Literal['warn', 'fail', 'allow'] = 'fail'
    allow_reapply_same_static_table: bool = False
    fill_value: float = 0.0
    max_abs_shift_ms: float = 250.0

    @field_validator('file_id', mode='before')
    @classmethod
    def _check_file_id(cls, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('file_id must be a non-empty string')
        return value

    @field_validator('key1_byte', 'key2_byte', mode='before')
    @classmethod
    def _check_key_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @field_validator(
        'source_table_artifact_id',
        'receiver_table_artifact_id',
        'combined_table_artifact_id',
        mode='before',
    )
    @classmethod
    def _check_artifact_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError('static table artifact id must be a non-empty string')
        return value

    @field_validator('register_corrected_file', mode='before')
    @classmethod
    def _check_register_corrected_file_bool(cls, value: object) -> bool:
        return _require_bool(value, 'register_corrected_file')

    @field_validator(
        'allow_missing_source_static',
        'allow_missing_receiver_static',
        'allow_reapply_same_static_table',
        mode='before',
    )
    @classmethod
    def _check_allow_missing_bool(cls, value: object, info: Any) -> bool:
        return _require_bool(value, info.field_name)

    @field_validator('output_name', mode='before')
    @classmethod
    def _check_output_name(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError('output_name must be a plain file name')
        return _validate_artifact_basename(value, 'output_name')

    @field_validator('fill_value', mode='before')
    @classmethod
    def _check_fill_value(cls, value: object) -> float:
        return _require_finite_float(value, 'fill_value')

    @field_validator('max_abs_shift_ms', mode='before')
    @classmethod
    def _check_max_abs_shift_ms(cls, value: object) -> float:
        return _require_positive_finite_float(value, 'max_abs_shift_ms')

    @model_validator(mode='after')
    def _check_table_artifacts(self) -> 'RefractionStaticTableApplyRequest':
        has_combined = self.combined_table_artifact_id is not None
        has_separate = (
            self.source_table_artifact_id is not None
            or self.receiver_table_artifact_id is not None
        )
        if has_combined and has_separate:
            raise ValueError(
                'provide either combined_table_artifact_id or separate '
                'source/receiver table artifact ids'
            )
        if has_combined:
            return self
        if self.source_table_artifact_id is None or self.receiver_table_artifact_id is None:
            raise ValueError(
                'source_table_artifact_id and receiver_table_artifact_id are '
                'required when combined_table_artifact_id is omitted'
            )
        return self


class RefractionStaticTableApplyResponse(BaseModel):
    """Response model for creating a standalone static-table apply job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str


class RefractionStaticExportJobRequest(BaseModel):
    """Request model for ``/statics/refraction/export`` jobs."""

    model_config = ConfigDict(extra='forbid')

    source_job_id: str
    export: RefractionStaticExportRequest

    @field_validator('source_job_id')
    @classmethod
    def _check_source_job_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError('source_job_id must be a non-empty string')
        return value


class RefractionStaticExportJobResponse(BaseModel):
    """Response model for creating a standalone refraction export job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str
    source_job_id: str
    requested_formats: list[RefractionStaticExportFormat]
