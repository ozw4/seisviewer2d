"""QC bundle and drilldown contracts for refraction static workflows."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_finite_float,
    _require_positive_int,
    require_trace_header_byte,
)
from app.statics.refraction.contracts.common import (
    RefractionStaticLayerKind,
    RefractionStaticQcBundleCoordinateMode,
    RefractionStaticQcBundleInclude,
)
from app.statics.refraction.contracts.inputs import (
    RefractionStaticGeometryRequest,
    RefractionStaticPickSourceRequest,
)
from app.utils.validation import require_non_negative_int, require_positive_int


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
        return _require_positive_int(value, 'max_observations')


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
