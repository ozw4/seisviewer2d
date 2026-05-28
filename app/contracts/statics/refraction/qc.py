"""QC bundle and drilldown contracts for refraction static workflows."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.contracts._validation import _require_positive_int
from app.contracts.statics.refraction.common import (
    RefractionStaticLayerKind,
    RefractionStaticQcBundleCoordinateMode,
    RefractionStaticQcBundleInclude,
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
