"""Pydantic models for describing pipeline operations."""

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_bool,
    _require_finite_float,
    _require_nonnegative_finite_float,
    _require_positive_finite_float,
    _require_positive_int,
    _validate_artifact_basename,
    _velocity_values_match,
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
from app.utils.validation import require_non_negative_int, require_positive_int

RefractionStaticExportFormat = Literal[
    'canonical_static_table',
    'lsst',
    'lsst_plus',
    'time_term_spreadsheet',
    'first_break_time',
]
RefractionStaticQcBundleInclude = Literal[
    'summary',
    'first_break',
    'reduced_time',
    'profiles',
    'cells',
    'static_components',
    'gather_preview',
]
RefractionStaticQcBundleCoordinateMode = Literal[
    'auto',
    'line_2d_projected',
    'grid_3d',
]
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
RefractionStaticGatherPreviewAxis = Literal['section', 'source', 'receiver']
RefractionStaticGatherPreviewOverlayLayer = Literal[
    'observed_first_break',
    'modeled_first_break',
    'reduced_time',
    'static_shift_trace_curve',
]
RefractionStaticGatherPreviewScaling = Literal['amax', 'tracewise']
RefractionStaticGatherPreviewSampleSource = Literal[
    'corrected_tracestore',
    'raw_tracestore_shifted_on_the_fly',
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

REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS: tuple[
    RefractionStaticExportFormat, ...
] = (
    'canonical_static_table',
    'time_term_spreadsheet',
)


class RefractionStaticPickSourceRequest(BaseModel):
    """First-break pick source reference for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal[
        'uploaded_npz',
        'batch_predicted_npz',
        'manual_npz_artifact',
        'manual_memmap',
    ] = Field(
        description=(
            'Use uploaded_npz for Static Correction UI direct .npz uploads; '
            'artifact-backed kinds are retained for legacy job-artifact flows.'
        )
    )
    job_id: str | None = None
    artifact_name: str | None = None

    @model_validator(mode='after')
    def _check_ref(self) -> 'RefractionStaticPickSourceRequest':
        if self.kind == 'uploaded_npz':
            if self.job_id is not None:
                raise ValueError(
                    'pick_source.job_id must be omitted for uploaded_npz'
                )
            if self.artifact_name is not None:
                raise ValueError(
                    'pick_source.artifact_name must be omitted for uploaded_npz'
                )
            return self

        if self.kind == 'manual_memmap':
            if self.job_id is not None or self.artifact_name is not None:
                raise ValueError(
                    'pick_source.job_id/artifact_name must be omitted for manual_memmap'
                )
            return self

        if not self.job_id:
            raise ValueError('pick_source.job_id is required for artifact sources')
        if self.kind == 'batch_predicted_npz' and self.artifact_name is None:
            self.artifact_name = 'predicted_picks_time_s.npz'
        if not self.artifact_name:
            raise ValueError(
                'pick_source.artifact_name is required for artifact sources'
            )
        _validate_artifact_basename(
            self.artifact_name,
            'pick_source.artifact_name',
        )
        if self.kind == 'manual_npz_artifact' and not self.artifact_name.endswith(
            '.npz'
        ):
            raise ValueError('pick_source.artifact_name must end with .npz')
        return self


class RefractionStaticGeometryRequest(BaseModel):
    """Trace header configuration for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    source_id_byte: int = 9
    receiver_id_byte: int = 13
    source_x_byte: int = 73
    source_y_byte: int = 77
    receiver_x_byte: int = 81
    receiver_y_byte: int = 85
    source_elevation_byte: int = 45
    receiver_elevation_byte: int = 41
    source_depth_byte: int | None = None
    coordinate_scalar_byte: int = 71
    elevation_scalar_byte: int = 69
    coordinate_unit: Literal['m', 'ft'] = 'm'
    elevation_unit: Literal['m', 'ft'] = 'm'

    @field_validator(
        'source_id_byte',
        'receiver_id_byte',
        'source_x_byte',
        'source_y_byte',
        'receiver_x_byte',
        'receiver_y_byte',
        'source_elevation_byte',
        'receiver_elevation_byte',
        'coordinate_scalar_byte',
        'elevation_scalar_byte',
        mode='before',
    )
    @classmethod
    def _check_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, f'geometry.{info.field_name}')

    @field_validator('source_depth_byte', mode='before')
    @classmethod
    def _check_optional_header_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(value, 'geometry.source_depth_byte')

    @model_validator(mode='after')
    def _check_distinct_endpoint_ids(self) -> 'RefractionStaticGeometryRequest':
        if self.source_id_byte == self.receiver_id_byte:
            raise ValueError('geometry.source_id_byte and receiver_id_byte must differ')
        return self


class RefractionStaticLinkageRequest(BaseModel):
    """Source/receiver endpoint linkage artifact reference."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['required', 'optional', 'none'] = 'none'
    job_id: str | None = None
    artifact_name: str = 'geometry_linkage.npz'

    @field_validator('artifact_name', mode='before')
    @classmethod
    def _check_artifact_name(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError('linkage.artifact_name must be a plain file name')
        return _validate_artifact_basename(value, 'linkage.artifact_name')

    @model_validator(mode='after')
    def _check_ref(self) -> 'RefractionStaticLinkageRequest':
        if self.mode == 'required' and not self.job_id:
            raise ValueError('linkage.job_id is required when linkage.mode is required')
        if self.mode == 'none' and self.job_id is not None:
            raise ValueError('linkage.job_id must be omitted when linkage.mode is none')
        if self.job_id is not None and not self.job_id:
            raise ValueError('linkage.job_id must be a non-empty string')
        return self


class RefractionStaticFirstLayerRequest(BaseModel):
    """First-layer / V1 configuration for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['constant', 'estimate_direct_arrival'] = 'constant'
    weathering_velocity_m_s: float | None = None

    min_weathering_velocity_m_s: float = 250.0
    max_weathering_velocity_m_s: float = 1800.0

    min_direct_offset_m: float | None = None
    max_direct_offset_m: float | None = None

    min_picks_per_fit: int = 5
    min_groups: int = 3

    robust_enabled: bool = True
    robust_threshold: float = 3.5

    @field_validator('weathering_velocity_m_s', mode='before')
    @classmethod
    def _check_optional_weathering_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'model.first_layer.weathering_velocity_m_s',
        )

    @field_validator(
        'min_weathering_velocity_m_s',
        'max_weathering_velocity_m_s',
        mode='before',
    )
    @classmethod
    def _check_weathering_velocity_bound(cls, value: object, info: Any) -> float:
        return _require_positive_finite_float(
            value,
            f'model.first_layer.{info.field_name}',
        )

    @field_validator('min_direct_offset_m', 'max_direct_offset_m', mode='before')
    @classmethod
    def _check_direct_offset(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            f'model.first_layer.{info.field_name}',
        )

    @field_validator('min_picks_per_fit', 'min_groups', mode='before')
    @classmethod
    def _check_positive_count(cls, value: object, info: Any) -> int:
        return _require_positive_int(value, f'model.first_layer.{info.field_name}')

    @field_validator('robust_enabled', mode='before')
    @classmethod
    def _check_robust_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'model.first_layer.robust_enabled')

    @field_validator('robust_threshold', mode='before')
    @classmethod
    def _check_robust_threshold(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'model.first_layer.robust_threshold',
        )

    @model_validator(mode='after')
    def _check_first_layer_values(self) -> 'RefractionStaticFirstLayerRequest':
        if self.min_weathering_velocity_m_s >= self.max_weathering_velocity_m_s:
            raise ValueError(
                'model.first_layer.min_weathering_velocity_m_s must be less than '
                'model.first_layer.max_weathering_velocity_m_s'
            )
        if (
            self.mode == 'estimate_direct_arrival'
            and self.weathering_velocity_m_s is not None
        ):
            raise ValueError(
                'model.first_layer.weathering_velocity_m_s must be omitted when '
                'model.first_layer.mode is estimate_direct_arrival'
            )
        if self.mode == 'estimate_direct_arrival' and (
            self.min_direct_offset_m is None or self.max_direct_offset_m is None
        ):
            raise ValueError(
                'model.first_layer.min_direct_offset_m and '
                'model.first_layer.max_direct_offset_m are required when '
                'model.first_layer.mode is estimate_direct_arrival'
            )
        if (
            self.min_direct_offset_m is not None
            and self.max_direct_offset_m is not None
            and self.min_direct_offset_m >= self.max_direct_offset_m
        ):
            raise ValueError(
                'model.first_layer.min_direct_offset_m must be less than '
                'model.first_layer.max_direct_offset_m'
            )
        return self


class RefractionStaticRefractorCellRequest(BaseModel):
    """Spatial refractor V2 cell configuration for Phase 2 request contracts."""

    model_config = ConfigDict(extra='forbid')

    number_of_cell_x: int
    size_of_cell_x_m: float
    x_coordinate_origin_m: float

    number_of_cell_y: int = 1
    size_of_cell_y_m: float | None = None
    y_coordinate_origin_m: float = 0.0

    assignment_mode: Literal['midpoint'] = 'midpoint'
    outside_grid_policy: Literal['reject'] = 'reject'
    coordinate_mode: Literal['grid_3d', 'line_2d_projected'] = 'grid_3d'
    line_origin_x_m: float | None = None
    line_origin_y_m: float | None = None
    line_azimuth_deg: float | None = None

    min_observations_per_cell: int = 5
    velocity_smoothing_weight: float = 0.0
    smoothing_reference_distance_m: float | None = None

    @field_validator(
        'number_of_cell_x',
        'number_of_cell_y',
        'min_observations_per_cell',
        mode='before',
    )
    @classmethod
    def _check_positive_count(cls, value: object, info: Any) -> int:
        return _require_positive_int(
            value,
            f'model.refractor_cell.{info.field_name}',
        )

    @field_validator('size_of_cell_x_m', mode='before')
    @classmethod
    def _check_size_of_cell_x(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'model.refractor_cell.size_of_cell_x_m',
        )

    @field_validator('size_of_cell_y_m', mode='before')
    @classmethod
    def _check_size_of_cell_y(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'model.refractor_cell.size_of_cell_y_m',
        )

    @field_validator(
        'x_coordinate_origin_m',
        'y_coordinate_origin_m',
        mode='before',
    )
    @classmethod
    def _check_origin(cls, value: object, info: Any) -> float:
        return _require_finite_float(
            value,
            f'model.refractor_cell.{info.field_name}',
        )

    @field_validator('assignment_mode', mode='before')
    @classmethod
    def _check_assignment_mode(cls, value: object) -> Literal['midpoint']:
        if value != 'midpoint':
            raise ValueError('model.refractor_cell.assignment_mode must be midpoint')
        return 'midpoint'

    @field_validator('outside_grid_policy', mode='before')
    @classmethod
    def _check_outside_grid_policy(cls, value: object) -> Literal['reject']:
        if value != 'reject':
            raise ValueError(
                'model.refractor_cell.outside_grid_policy must be reject'
            )
        return 'reject'

    @field_validator(
        'line_origin_x_m',
        'line_origin_y_m',
        'line_azimuth_deg',
        mode='before',
    )
    @classmethod
    def _check_optional_line_coordinate(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_finite_float(
            value,
            f'model.refractor_cell.{info.field_name}',
        )

    @field_validator('velocity_smoothing_weight', mode='before')
    @classmethod
    def _check_velocity_smoothing_weight(cls, value: object) -> float:
        return _require_nonnegative_finite_float(
            value,
            'model.refractor_cell.velocity_smoothing_weight',
        )

    @field_validator('smoothing_reference_distance_m', mode='before')
    @classmethod
    def _check_smoothing_reference_distance(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'model.refractor_cell.smoothing_reference_distance_m',
        )

    @model_validator(mode='after')
    def _check_cell_values(self) -> 'RefractionStaticRefractorCellRequest':
        if self.number_of_cell_y > 1 and self.size_of_cell_y_m is None:
            raise ValueError(
                'model.refractor_cell.size_of_cell_y_m is required when '
                'model.refractor_cell.number_of_cell_y > 1'
            )
        if self.coordinate_mode == 'line_2d_projected':
            if (
                self.line_origin_x_m is None
                or self.line_origin_y_m is None
                or self.line_azimuth_deg is None
            ):
                raise ValueError(
                    'model.refractor_cell.line_origin_x_m, '
                    'model.refractor_cell.line_origin_y_m, and '
                    'model.refractor_cell.line_azimuth_deg are required when '
                    'model.refractor_cell.coordinate_mode is line_2d_projected'
                )
            if self.number_of_cell_y != 1:
                raise ValueError(
                    'model.refractor_cell.number_of_cell_y must be 1 when '
                    'model.refractor_cell.coordinate_mode is line_2d_projected'
                )
        return self


RefractionStaticLayerKind = Literal['v2_t1', 'v3_t2', 'vsub_t3']
RefractionStaticLayerVelocityMode = Literal[
    'fixed_global',
    'solve_global',
    'solve_cell',
]
_REFRACTION_STATIC_LAYER_ORDER: dict[RefractionStaticLayerKind, int] = {
    'v2_t1': 0,
    'v3_t2': 1,
    'vsub_t3': 2,
}


class RefractionStaticLayerRequest(BaseModel):
    """Layer-specific time-term configuration for multi-layer refraction statics."""

    model_config = ConfigDict(extra='forbid')

    kind: RefractionStaticLayerKind
    enabled: bool = True
    min_offset_m: float | None = None
    max_offset_m: float | None = None
    velocity_mode: RefractionStaticLayerVelocityMode = 'solve_global'
    initial_velocity_m_s: float | None = None
    fixed_velocity_m_s: float | None = None
    min_velocity_m_s: float | None = None
    max_velocity_m_s: float | None = None
    min_observations_per_cell: int | None = None
    smoothing_weight: float | None = None

    @field_validator('enabled', mode='before')
    @classmethod
    def _check_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'model.layers.enabled')

    @field_validator('min_offset_m', 'max_offset_m', mode='before')
    @classmethod
    def _check_offset_gate(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            f'model.layers.{info.field_name}',
        )

    @field_validator(
        'initial_velocity_m_s',
        'fixed_velocity_m_s',
        'min_velocity_m_s',
        'max_velocity_m_s',
        mode='before',
    )
    @classmethod
    def _check_optional_velocity(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            f'model.layers.{info.field_name}',
        )

    @field_validator('min_observations_per_cell', mode='before')
    @classmethod
    def _check_min_observations_per_cell(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_positive_int(value, 'model.layers.min_observations_per_cell')

    @field_validator('smoothing_weight', mode='before')
    @classmethod
    def _check_smoothing_weight(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            'model.layers.smoothing_weight',
        )

    @model_validator(mode='after')
    def _check_layer_values(self) -> 'RefractionStaticLayerRequest':
        if (
            self.min_offset_m is not None
            and self.max_offset_m is not None
            and self.min_offset_m >= self.max_offset_m
        ):
            raise ValueError(
                'model.layers.min_offset_m must be less than '
                'model.layers.max_offset_m'
            )
        if (
            self.min_velocity_m_s is not None
            and self.max_velocity_m_s is not None
            and self.min_velocity_m_s >= self.max_velocity_m_s
        ):
            raise ValueError(
                'model.layers.min_velocity_m_s must be less than '
                'model.layers.max_velocity_m_s'
            )
        return self


class RefractionStaticModelRequest(BaseModel):
    """Near-surface model options for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    method: Literal['gli_variable_thickness', 'multilayer_time_term'] = (
        'gli_variable_thickness'
    )
    weathering_velocity_m_s: float | None = None
    first_layer: RefractionStaticFirstLayerRequest | None = None
    bedrock_velocity_mode: Literal[
        'solve_global',
        'fixed_global',
        'solve_cell',
    ] = 'solve_global'
    bedrock_velocity_m_s: float | None = None
    initial_bedrock_velocity_m_s: float | None = None
    min_bedrock_velocity_m_s: float = 1200.0
    max_bedrock_velocity_m_s: float = 6000.0
    max_weathering_thickness_m: float | None = None
    refractor_cell: RefractionStaticRefractorCellRequest | None = None
    layers: list[RefractionStaticLayerRequest] | None = None
    allow_overlapping_layer_gates: bool = False

    @field_validator('weathering_velocity_m_s', mode='before')
    @classmethod
    def _check_optional_weathering_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(value, 'model.weathering_velocity_m_s')

    @field_validator('min_bedrock_velocity_m_s', 'max_bedrock_velocity_m_s', mode='before')
    @classmethod
    def _check_positive_velocity(cls, value: object, info: Any) -> float:
        return _require_positive_finite_float(value, f'model.{info.field_name}')

    @field_validator(
        'bedrock_velocity_m_s',
        'initial_bedrock_velocity_m_s',
        'max_weathering_thickness_m',
        mode='before',
    )
    @classmethod
    def _check_optional_positive_velocity(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(value, f'model.{info.field_name}')

    @field_validator('allow_overlapping_layer_gates', mode='before')
    @classmethod
    def _check_allow_overlapping_layer_gates(cls, value: object) -> bool:
        return _require_bool(value, 'model.allow_overlapping_layer_gates')

    @model_validator(mode='after')
    def _check_velocity_values(self) -> 'RefractionStaticModelRequest':
        resolved_weathering_velocity = self._constant_weathering_velocity_or_none()
        if self.method == 'multilayer_time_term':
            self._check_multilayer_values(resolved_weathering_velocity)
            return self

        if self.layers is not None:
            raise ValueError(
                'model.layers is only allowed when '
                'model.method is multilayer_time_term'
            )
        if self.min_bedrock_velocity_m_s >= self.max_bedrock_velocity_m_s:
            raise ValueError(
                'model.min_bedrock_velocity_m_s must be less than '
                'model.max_bedrock_velocity_m_s'
            )
        if resolved_weathering_velocity is not None:
            if self.min_bedrock_velocity_m_s <= resolved_weathering_velocity:
                raise ValueError(
                    'model.min_bedrock_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
            if self.max_bedrock_velocity_m_s <= resolved_weathering_velocity:
                raise ValueError(
                    'model.max_bedrock_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
        if (
            self.initial_bedrock_velocity_m_s is not None
            and resolved_weathering_velocity is not None
            and self.initial_bedrock_velocity_m_s <= resolved_weathering_velocity
        ):
            raise ValueError(
                'model.initial_bedrock_velocity_m_s must be greater than '
                'model.resolved_weathering_velocity_m_s'
            )
        if self.initial_bedrock_velocity_m_s is not None and not (
            self.min_bedrock_velocity_m_s
            <= self.initial_bedrock_velocity_m_s
            <= self.max_bedrock_velocity_m_s
        ):
            raise ValueError(
                'model.initial_bedrock_velocity_m_s must be within '
                'bedrock velocity bounds'
            )
        if self.bedrock_velocity_mode == 'fixed_global':
            if self.bedrock_velocity_m_s is None:
                raise ValueError(
                    'model.bedrock_velocity_m_s is required when '
                    'model.bedrock_velocity_mode is fixed_global'
                )
            if (
                resolved_weathering_velocity is not None
                and self.bedrock_velocity_m_s <= resolved_weathering_velocity
            ):
                raise ValueError(
                    'model.bedrock_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
            if not (
                self.min_bedrock_velocity_m_s
                <= self.bedrock_velocity_m_s
                <= self.max_bedrock_velocity_m_s
            ):
                raise ValueError(
                    'model.bedrock_velocity_m_s must be within bedrock velocity bounds'
                )
        elif self.bedrock_velocity_m_s is not None:
            raise ValueError(
                'model.bedrock_velocity_m_s is only allowed when '
                'model.bedrock_velocity_mode is fixed_global'
            )
        if self.bedrock_velocity_mode == 'solve_cell':
            if self.refractor_cell is None:
                raise ValueError(
                    'model.refractor_cell is required when '
                    'model.bedrock_velocity_mode is solve_cell'
                )
        elif self.refractor_cell is not None:
            raise ValueError(
                'model.refractor_cell is only allowed when '
                'model.bedrock_velocity_mode is solve_cell'
            )
        return self

    def _check_multilayer_values(
        self,
        resolved_weathering_velocity: float | None,
    ) -> None:
        layers = self.layers
        if not layers:
            raise ValueError(
                'model.layers must include enabled v2_t1 when '
                'model.method is multilayer_time_term'
            )

        seen_kinds: set[RefractionStaticLayerKind] = set()
        previous_order = -1
        for layer in layers:
            if layer.kind in seen_kinds:
                raise ValueError('model.layers must not contain duplicate layer kinds')
            order = _REFRACTION_STATIC_LAYER_ORDER[layer.kind]
            if order < previous_order:
                raise ValueError(
                    'model.layers must be ordered v2_t1, v3_t2, vsub_t3'
                )
            seen_kinds.add(layer.kind)
            previous_order = order

        enabled_layers = [layer for layer in layers if layer.enabled]
        enabled_kinds = {layer.kind for layer in enabled_layers}
        if 'v2_t1' not in enabled_kinds:
            raise ValueError(
                'model.layers must include an enabled v2_t1 layer when '
                'model.method is multilayer_time_term'
            )
        if 'vsub_t3' in enabled_kinds and 'v3_t2' not in enabled_kinds:
            raise ValueError(
                'model.layers cannot enable vsub_t3 unless v3_t2 is enabled'
            )

        deepest_enabled_order = max(
            _REFRACTION_STATIC_LAYER_ORDER[layer.kind] for layer in enabled_layers
        )
        for layer in enabled_layers:
            if layer.min_offset_m is None and layer.max_offset_m is None:
                raise ValueError(
                    'model.layers.min_offset_m or model.layers.max_offset_m is '
                    'required for each enabled layer'
                )
            if (
                layer.max_offset_m is None
                and _REFRACTION_STATIC_LAYER_ORDER[layer.kind] != deepest_enabled_order
            ):
                raise ValueError(
                    'model.layers.max_offset_m may be null only for the deepest '
                    'enabled layer'
                )
            self._check_multilayer_velocity_layer(
                layer,
                resolved_weathering_velocity=resolved_weathering_velocity,
            )

        if not self.allow_overlapping_layer_gates:
            self._check_multilayer_layer_gate_overlap(enabled_layers)
        self._check_multilayer_legacy_aliases()
        self._check_multilayer_velocity_sequence(enabled_layers)

    def _check_multilayer_legacy_aliases(self) -> None:
        v2_layer = self._layer_by_kind('v2_t1')
        if (
            self.bedrock_velocity_m_s is not None
            and v2_layer is not None
            and v2_layer.velocity_mode != 'fixed_global'
        ):
            raise ValueError(
                'model.bedrock_velocity_m_s is only allowed as a v2_t1 fixed '
                'velocity when model.method is multilayer_time_term'
            )
        if (
            self.bedrock_velocity_m_s is not None
            and v2_layer is not None
            and v2_layer.fixed_velocity_m_s is not None
            and not _velocity_values_match(
                self.bedrock_velocity_m_s,
                v2_layer.fixed_velocity_m_s,
            )
        ):
            raise ValueError(
                'model.bedrock_velocity_m_s and '
                'model.layers.fixed_velocity_m_s must match for v2_t1'
            )
        if (
            self.initial_bedrock_velocity_m_s is not None
            and v2_layer is not None
            and v2_layer.initial_velocity_m_s is not None
            and not _velocity_values_match(
                self.initial_bedrock_velocity_m_s,
                v2_layer.initial_velocity_m_s,
            )
        ):
            raise ValueError(
                'model.initial_bedrock_velocity_m_s and '
                'model.layers.initial_velocity_m_s must match for v2_t1'
            )
        has_enabled_solve_cell_layer = any(
            layer.enabled and layer.velocity_mode == 'solve_cell'
            for layer in self.layers or []
        )
        if has_enabled_solve_cell_layer and self.refractor_cell is None:
            raise ValueError(
                'model.refractor_cell is required when an enabled '
                'multi-layer refraction layer uses solve_cell'
            )
        if self.refractor_cell is not None and not has_enabled_solve_cell_layer:
            raise ValueError(
                'model.refractor_cell is only allowed when an enabled '
                'multi-layer refraction layer uses solve_cell'
            )

    def _check_multilayer_velocity_layer(
        self,
        layer: RefractionStaticLayerRequest,
        *,
        resolved_weathering_velocity: float | None,
    ) -> None:
        min_velocity = self._layer_min_velocity_m_s(layer)
        max_velocity = self._layer_max_velocity_m_s(layer)
        if (
            min_velocity is not None
            and max_velocity is not None
            and min_velocity >= max_velocity
        ):
            raise ValueError(
                'model.layers.min_velocity_m_s must be less than '
                'model.layers.max_velocity_m_s'
            )
        if resolved_weathering_velocity is not None:
            if min_velocity is not None and min_velocity <= resolved_weathering_velocity:
                raise ValueError(
                    'model.layers.min_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
            if max_velocity is not None and max_velocity <= resolved_weathering_velocity:
                raise ValueError(
                    'model.layers.max_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )

        if layer.velocity_mode == 'fixed_global':
            fixed_velocity = self._layer_fixed_velocity_m_s(layer)
            if fixed_velocity is None:
                raise ValueError(
                    'model.layers.fixed_velocity_m_s is required when '
                    'model.layers.velocity_mode is fixed_global'
                )
            self._check_layer_velocity_in_bounds(
                fixed_velocity,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                field_name='fixed_velocity_m_s',
            )
            if (
                resolved_weathering_velocity is not None
                and fixed_velocity <= resolved_weathering_velocity
            ):
                raise ValueError(
                    'model.layers.fixed_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
            return

        initial_velocity = self._layer_initial_velocity_m_s(layer)
        if initial_velocity is None:
            raise ValueError(
                'model.layers.initial_velocity_m_s or '
                'model.initial_bedrock_velocity_m_s is required when '
                'model.layers.velocity_mode is solve_global or solve_cell'
            )
        self._check_layer_velocity_in_bounds(
            initial_velocity,
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            field_name='initial_velocity_m_s',
        )
        if (
            resolved_weathering_velocity is not None
            and initial_velocity <= resolved_weathering_velocity
        ):
            raise ValueError(
                'model.layers.initial_velocity_m_s must be greater than '
                'model.resolved_weathering_velocity_m_s'
            )

    def _check_layer_velocity_in_bounds(
        self,
        velocity: float,
        *,
        min_velocity: float | None,
        max_velocity: float | None,
        field_name: str,
    ) -> None:
        if min_velocity is not None and velocity < min_velocity:
            raise ValueError(f'model.layers.{field_name} must be within velocity bounds')
        if max_velocity is not None and velocity > max_velocity:
            raise ValueError(f'model.layers.{field_name} must be within velocity bounds')

    def _check_multilayer_velocity_sequence(
        self,
        enabled_layers: list[RefractionStaticLayerRequest],
    ) -> None:
        enabled_by_kind = {layer.kind: layer for layer in enabled_layers}
        for shallow_kind, deep_kind in (
            ('v2_t1', 'v3_t2'),
            ('v3_t2', 'vsub_t3'),
        ):
            shallow = enabled_by_kind.get(shallow_kind)
            deep = enabled_by_kind.get(deep_kind)
            if shallow is None or deep is None:
                continue
            shallow_min = self._layer_min_velocity_m_s(shallow)
            if shallow_min is None:
                continue
            deep_min = self._layer_min_velocity_m_s(deep)
            if deep_min is not None and deep_min <= shallow_min:
                raise ValueError(
                    f'model.layers {deep_kind} velocity bounds must be greater '
                    f'than {shallow_kind} minimum velocity'
                )
            deep_max = self._layer_max_velocity_m_s(deep)
            if deep_max is not None and deep_max <= shallow_min:
                raise ValueError(
                    f'model.layers {deep_kind} velocity bounds must allow '
                    f'velocities greater than {shallow_kind} minimum velocity'
                )
            configured_velocity = (
                self._layer_fixed_velocity_m_s(deep)
                if deep.velocity_mode == 'fixed_global'
                else self._layer_initial_velocity_m_s(deep)
            )
            if configured_velocity is not None and configured_velocity <= shallow_min:
                raise ValueError(
                    f'model.layers {deep_kind} configured velocity must be '
                    f'greater than {shallow_kind} minimum velocity'
                )

    def _check_multilayer_layer_gate_overlap(
        self,
        enabled_layers: list[RefractionStaticLayerRequest],
    ) -> None:
        for index, layer in enumerate(enabled_layers):
            layer_min = self._layer_gate_min_offset(layer)
            layer_max = self._layer_gate_max_offset(layer)
            for other in enabled_layers[index + 1 :]:
                other_min = self._layer_gate_min_offset(other)
                other_max = self._layer_gate_max_offset(other)
                if max(layer_min, other_min) < min(layer_max, other_max):
                    raise ValueError(
                        'model.layers offset gates must not overlap unless '
                        'model.allow_overlapping_layer_gates is true'
                    )

    def _layer_gate_min_offset(self, layer: RefractionStaticLayerRequest) -> float:
        if layer.min_offset_m is None:
            return float('-inf')
        return float(layer.min_offset_m)

    def _layer_gate_max_offset(self, layer: RefractionStaticLayerRequest) -> float:
        if layer.max_offset_m is None:
            return float('inf')
        return float(layer.max_offset_m)

    def _layer_by_kind(
        self,
        kind: RefractionStaticLayerKind,
    ) -> RefractionStaticLayerRequest | None:
        for layer in self.layers or []:
            if layer.kind == kind:
                return layer
        return None

    def _layer_initial_velocity_m_s(
        self,
        layer: RefractionStaticLayerRequest,
    ) -> float | None:
        if layer.initial_velocity_m_s is not None:
            return layer.initial_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.initial_bedrock_velocity_m_s
        return None

    def _layer_fixed_velocity_m_s(
        self,
        layer: RefractionStaticLayerRequest,
    ) -> float | None:
        if layer.fixed_velocity_m_s is not None:
            return layer.fixed_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.bedrock_velocity_m_s
        return None

    def _layer_min_velocity_m_s(
        self,
        layer: RefractionStaticLayerRequest,
    ) -> float | None:
        if layer.min_velocity_m_s is not None:
            return layer.min_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.min_bedrock_velocity_m_s
        return None

    def _layer_max_velocity_m_s(
        self,
        layer: RefractionStaticLayerRequest,
    ) -> float | None:
        if layer.max_velocity_m_s is not None:
            return layer.max_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.max_bedrock_velocity_m_s
        return None

    @property
    def enabled_refraction_layer_count(self) -> int:
        if self.layers is None:
            return 1
        return sum(1 for layer in self.layers if layer.enabled)

    @property
    def first_layer_mode(self) -> Literal['constant', 'estimate_direct_arrival']:
        first_layer = self.first_layer
        if first_layer is None:
            return 'constant'
        return first_layer.mode

    @property
    def resolved_weathering_velocity_m_s(self) -> float:
        value = self._constant_weathering_velocity_or_none()
        if value is None:
            raise ValueError(
                'model.first_layer.mode="estimate_direct_arrival" requires a '
                'resolved weathering velocity before downstream processing'
            )
        return value

    def _constant_weathering_velocity_or_none(self) -> float | None:
        legacy_velocity = self.weathering_velocity_m_s
        first_layer = self.first_layer
        if first_layer is None:
            if legacy_velocity is None:
                raise ValueError(
                    'model.weathering_velocity_m_s is required when '
                    'model.first_layer is omitted'
                )
            return legacy_velocity

        first_layer_velocity = first_layer.weathering_velocity_m_s
        if first_layer.mode == 'estimate_direct_arrival':
            if legacy_velocity is not None:
                raise ValueError(
                    'model.weathering_velocity_m_s must be omitted when '
                    'model.first_layer.mode is estimate_direct_arrival'
                )
            if first_layer_velocity is not None:
                raise ValueError(
                    'model.first_layer.weathering_velocity_m_s must be omitted when '
                    'model.first_layer.mode is estimate_direct_arrival'
                )
            return None
        if (
            legacy_velocity is not None
            and first_layer_velocity is not None
            and not _velocity_values_match(legacy_velocity, first_layer_velocity)
        ):
            raise ValueError(
                'model.weathering_velocity_m_s and '
                'model.first_layer.weathering_velocity_m_s must match when both '
                'are specified'
            )
        if first_layer_velocity is None:
            raise ValueError(
                'model.first_layer.weathering_velocity_m_s is required when '
                'model.first_layer.mode is constant'
            )
        return first_layer_velocity


class RefractionStaticMoveoutRequest(BaseModel):
    """Moveout distance source and filtering options for refraction statics."""

    model_config = ConfigDict(extra='forbid')

    model: Literal['head_wave_linear_offset'] = 'head_wave_linear_offset'
    distance_source: Literal['geometry', 'offset_header', 'auto'] = 'geometry'
    offset_byte: int | None = 37
    min_offset_m: float | None = None
    max_offset_m: float | None = None
    allow_missing_offset: bool = False
    max_geometry_offset_mismatch_m: float | None = None

    @field_validator('offset_byte', mode='before')
    @classmethod
    def _check_offset_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(value, 'moveout.offset_byte')

    @field_validator('min_offset_m', 'max_offset_m', mode='before')
    @classmethod
    def _check_offset_gate(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(value, f'moveout.{info.field_name}')

    @field_validator('allow_missing_offset', mode='before')
    @classmethod
    def _check_allow_missing_offset(cls, value: object) -> bool:
        return _require_bool(value, 'moveout.allow_missing_offset')

    @field_validator('max_geometry_offset_mismatch_m', mode='before')
    @classmethod
    def _check_max_geometry_offset_mismatch_m(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            'moveout.max_geometry_offset_mismatch_m',
        )

    @model_validator(mode='after')
    def _check_offset_values(self) -> 'RefractionStaticMoveoutRequest':
        if self.distance_source == 'offset_header' and self.offset_byte is None:
            raise ValueError(
                'moveout.offset_byte is required when '
                'moveout.distance_source is offset_header'
            )
        if (
            self.min_offset_m is not None
            and self.max_offset_m is not None
            and self.min_offset_m >= self.max_offset_m
        ):
            raise ValueError(
                'moveout.min_offset_m must be less than moveout.max_offset_m'
            )
        return self


class RefractionStaticRobustRequest(BaseModel):
    """Robust outlier-rejection options for refraction inversion."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    method: Literal['mad', 'sigma'] = 'mad'
    threshold: float = 3.5
    scale_floor_ms: float = 0.05
    max_iterations: int = 5
    min_used_fraction: float = 0.5
    min_used_observations: int = 1

    @field_validator('enabled', mode='before')
    @classmethod
    def _check_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'solver.robust.enabled')

    @field_validator('threshold', mode='before')
    @classmethod
    def _check_threshold(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'solver.robust.threshold',
        )

    @field_validator('scale_floor_ms', mode='before')
    @classmethod
    def _check_scale_floor_ms(cls, value: object) -> float:
        return _require_nonnegative_finite_float(
            value,
            'solver.robust.scale_floor_ms',
        )

    @field_validator('max_iterations', mode='before')
    @classmethod
    def _check_max_iterations(cls, value: object) -> int:
        return _require_positive_int(value, 'solver.robust.max_iterations')

    @field_validator('min_used_fraction', mode='before')
    @classmethod
    def _check_min_used_fraction(cls, value: object) -> float:
        fraction = _require_positive_finite_float(
            value,
            'solver.robust.min_used_fraction',
        )
        if fraction > 1.0:
            raise ValueError('solver.robust.min_used_fraction must be <= 1')
        return fraction

    @field_validator('min_used_observations', mode='before')
    @classmethod
    def _check_min_used_observations(cls, value: object) -> int:
        return _require_positive_int(value, 'solver.robust.min_used_observations')


class RefractionStaticSolverRequest(BaseModel):
    """Solver options for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    damping: float = 0.01
    min_picks_per_node: int = 1
    max_abs_half_intercept_time_ms: float = 500.0
    robust: RefractionStaticRobustRequest = Field(
        default_factory=RefractionStaticRobustRequest,
    )

    @field_validator('damping', mode='before')
    @classmethod
    def _check_damping(cls, value: object) -> float:
        return _require_nonnegative_finite_float(value, 'solver.damping')

    @field_validator('min_picks_per_node', mode='before')
    @classmethod
    def _check_min_picks_per_node(cls, value: object) -> int:
        return _require_positive_int(value, 'solver.min_picks_per_node')

    @field_validator('max_abs_half_intercept_time_ms', mode='before')
    @classmethod
    def _check_max_abs_half_intercept_time_ms(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'solver.max_abs_half_intercept_time_ms',
        )


class RefractionStaticDatumRequest(BaseModel):
    """Datum options for GLI refraction static composition."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal[
        'floating_and_flat',
        'floating_only',
        'flat_only',
        'none',
    ] = 'none'
    floating_datum_mode: Literal[
        'smoothed_topography',
        'constant',
        'surface',
        'from_artifact',
    ] = 'smoothed_topography'
    flat_datum_elevation_m: float | None = None
    floating_datum_elevation_m: float | None = None
    smoothing_radius_m: float | None = None
    smoothing_window_nodes: int | None = 11
    smoothing_method: Literal['moving_average', 'median'] = 'moving_average'
    floating_datum_job_id: str | None = None
    floating_datum_artifact_name: str | None = None
    allow_flat_datum_above_topography: bool = True
    allow_flat_datum_below_refractor: bool = False

    @field_validator(
        'flat_datum_elevation_m',
        'floating_datum_elevation_m',
        mode='before',
    )
    @classmethod
    def _check_optional_elevation(
        cls,
        value: object,
        info: Any,
    ) -> float | None:
        if value is None:
            return None
        return _require_finite_float(value, f'datum.{info.field_name}')

    @field_validator('smoothing_radius_m', mode='before')
    @classmethod
    def _check_smoothing_radius(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(value, 'datum.smoothing_radius_m')

    @field_validator('smoothing_window_nodes', mode='before')
    @classmethod
    def _check_smoothing_window_nodes(cls, value: object) -> int | None:
        if value is None:
            return None
        window = _require_positive_int(value, 'datum.smoothing_window_nodes')
        if window % 2 == 0:
            raise ValueError('datum.smoothing_window_nodes must be odd')
        return window

    @field_validator('floating_datum_artifact_name', mode='before')
    @classmethod
    def _check_floating_datum_artifact_name(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                'datum.floating_datum_artifact_name must be a plain file name'
            )
        return _validate_artifact_basename(
            value,
            'datum.floating_datum_artifact_name',
        )

    @field_validator(
        'allow_flat_datum_above_topography',
        'allow_flat_datum_below_refractor',
        mode='before',
    )
    @classmethod
    def _check_bool(cls, value: object, info: Any) -> bool:
        return _require_bool(value, f'datum.{info.field_name}')

    @model_validator(mode='after')
    def _check_datum_config(self) -> 'RefractionStaticDatumRequest':
        if self.mode in {'flat_only', 'floating_and_flat'}:
            if self.flat_datum_elevation_m is None:
                raise ValueError(
                    'datum.flat_datum_elevation_m is required for flat datum modes'
                )
        if self.floating_datum_mode == 'constant':
            if self.floating_datum_elevation_m is None:
                raise ValueError(
                    'datum.floating_datum_elevation_m is required when '
                    'floating_datum_mode is constant'
                )
        if self.floating_datum_mode == 'from_artifact':
            if not self.floating_datum_job_id:
                raise ValueError(
                    'datum.floating_datum_job_id is required when '
                    'floating_datum_mode is from_artifact'
                )
            if not self.floating_datum_artifact_name:
                raise ValueError(
                    'datum.floating_datum_artifact_name is required when '
                    'floating_datum_mode is from_artifact'
                )
        elif self.floating_datum_job_id is not None:
            raise ValueError(
                'datum.floating_datum_job_id is only allowed when '
                'floating_datum_mode is from_artifact'
            )
        elif self.floating_datum_artifact_name is not None:
            raise ValueError(
                'datum.floating_datum_artifact_name is only allowed when '
                'floating_datum_mode is from_artifact'
            )
        return self


class RefractionStaticApplyOptions(BaseModel):
    """Options for eventual refraction static TraceStore application."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['refraction_from_raw'] = 'refraction_from_raw'
    interpolation: Literal['linear'] = 'linear'
    fill_value: float = 0.0
    max_abs_shift_ms: float = 250.0
    output_dtype: Literal['float32'] = 'float32'
    register_corrected_file: bool = False

    @field_validator('fill_value', mode='before')
    @classmethod
    def _check_fill_value(cls, value: object) -> float:
        return _require_finite_float(value, 'apply.fill_value')

    @field_validator('max_abs_shift_ms', mode='before')
    @classmethod
    def _check_max_abs_shift_ms(cls, value: object) -> float:
        return _require_positive_finite_float(value, 'apply.max_abs_shift_ms')

    @field_validator('register_corrected_file', mode='before')
    @classmethod
    def _check_register_corrected_file_bool(cls, value: object) -> bool:
        return _require_bool(value, 'apply.register_corrected_file')


class RefractionStaticConversionRequest(BaseModel):
    """Conversion/output mode for refraction static component artifacts."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['existing', 't1lsst_1layer', 't1lsst_multilayer'] = 'existing'
    layer_count: int | None = None

    @field_validator('layer_count', mode='before')
    @classmethod
    def _check_layer_count(cls, value: object) -> int | None:
        if value is None:
            return None
        count = _require_positive_int(value, 'conversion.layer_count')
        if count > 3:
            raise ValueError('conversion.layer_count must be 1, 2, or 3')
        return count

    @model_validator(mode='after')
    def _check_multilayer_count(self) -> 'RefractionStaticConversionRequest':
        if self.mode == 't1lsst_multilayer':
            if self.layer_count is None:
                raise ValueError(
                    'conversion.layer_count is required when '
                    'conversion.mode is t1lsst_multilayer'
                )
            return self
        if self.layer_count is not None:
            raise ValueError(
                'conversion.layer_count is only allowed when '
                'conversion.mode is t1lsst_multilayer'
            )
        return self


class RefractionStaticReducedTimeQcRequest(BaseModel):
    """Reduced-time QC velocity selection for refraction first-break artifacts."""

    model_config = ConfigDict(extra='forbid')

    reduction_velocity_mode: Literal[
        'layer_velocity',
        'fixed',
        'initial_velocity',
    ] = 'layer_velocity'
    fixed_velocity_m_s: float | None = None

    @field_validator('fixed_velocity_m_s', mode='before')
    @classmethod
    def _check_fixed_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'reduced_time_qc.fixed_velocity_m_s',
        )

    @model_validator(mode='after')
    def _check_reduction_velocity_values(
        self,
    ) -> 'RefractionStaticReducedTimeQcRequest':
        if self.reduction_velocity_mode == 'fixed':
            if self.fixed_velocity_m_s is None:
                raise ValueError(
                    'reduced_time_qc.fixed_velocity_m_s is required when '
                    'reduced_time_qc.reduction_velocity_mode is fixed'
                )
            return self
        if self.fixed_velocity_m_s is not None:
            raise ValueError(
                'reduced_time_qc.fixed_velocity_m_s is only allowed when '
                'reduced_time_qc.reduction_velocity_mode is fixed'
            )
        return self


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
