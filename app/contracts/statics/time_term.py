"""Contracts for time-term static correction requests and responses."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_bool,
    _require_finite_float,
    _require_nonnegative_finite_float,
    _require_positive_finite_float,
    _require_positive_int,
    _validate_artifact_basename,
    require_trace_header_byte,
)


class TimeTermStaticPickSourceRequest(BaseModel):
    """First-break pick source reference for time-term static inversion."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['batch_predicted_npz', 'manual_npz_artifact', 'manual_memmap']
    job_id: str | None = None
    artifact_name: str | None = None

    @model_validator(mode='after')
    def _check_ref(self) -> 'TimeTermStaticPickSourceRequest':
        if self.kind == 'manual_memmap':
            if self.job_id is not None or self.artifact_name is not None:
                raise ValueError(
                    'pick_source.job_id/artifact_name must be omitted for manual_memmap'
                )
            return self

        if not self.job_id:
            raise ValueError('pick_source.job_id is required for artifact sources')
        if self.kind == 'batch_predicted_npz':
            self.artifact_name = self.artifact_name or 'predicted_picks_time_s.npz'
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


class TimeTermStaticGeometryRequest(BaseModel):
    """Trace header configuration for time-term static inversion."""

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
    def _check_distinct_endpoint_ids(self) -> 'TimeTermStaticGeometryRequest':
        if self.source_id_byte == self.receiver_id_byte:
            raise ValueError('geometry.source_id_byte and receiver_id_byte must differ')
        return self


class TimeTermStaticLinkageRequest(BaseModel):
    """Source/receiver endpoint linkage artifact reference."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['required', 'optional', 'none'] = 'required'
    job_id: str | None = None
    artifact_name: str = 'geometry_linkage.npz'

    @field_validator('artifact_name', mode='before')
    @classmethod
    def _check_artifact_name(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError('linkage.artifact_name must be a plain file name')
        return _validate_artifact_basename(value, 'linkage.artifact_name')

    @model_validator(mode='after')
    def _check_ref(self) -> 'TimeTermStaticLinkageRequest':
        if self.mode == 'required' and not self.job_id:
            raise ValueError('linkage.job_id is required when linkage.mode is required')
        if self.mode == 'none' and self.job_id is not None:
            raise ValueError('linkage.job_id must be omitted when linkage.mode is none')
        if self.job_id is not None and not self.job_id:
            raise ValueError('linkage.job_id must be a non-empty string')
        return self


class TimeTermStaticVelocityRequest(BaseModel):
    """Velocity parameters for time-term static inversion."""

    model_config = ConfigDict(extra='forbid')

    replacement_velocity_m_s: float
    refractor_velocity_m_s: float
    weathering_velocity_m_s: float | None = None

    @field_validator(
        'replacement_velocity_m_s',
        'refractor_velocity_m_s',
        mode='before',
    )
    @classmethod
    def _check_positive_velocity(cls, value: object, info: Any) -> float:
        return _require_positive_finite_float(value, f'velocity.{info.field_name}')

    @field_validator('weathering_velocity_m_s', mode='before')
    @classmethod
    def _check_optional_positive_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'velocity.weathering_velocity_m_s',
        )

    @model_validator(mode='after')
    def _check_head_wave_velocity_order(self) -> 'TimeTermStaticVelocityRequest':
        if self.refractor_velocity_m_s <= self.replacement_velocity_m_s:
            raise ValueError(
                'velocity.refractor_velocity_m_s must be greater than '
                'velocity.replacement_velocity_m_s'
            )
        return self


class TimeTermStaticMoveoutRequest(BaseModel):
    """Moveout model configuration for time-term static inversion."""

    model_config = ConfigDict(extra='forbid')

    model: Literal[
        'head_wave_linear_offset',
        'reciprocal_head_wave',
        'linear_offset',
        'none',
    ] = 'head_wave_linear_offset'
    distance_source: Literal['geometry', 'offset_header', 'auto'] = 'geometry'
    offset_byte: int | None = 37
    allow_missing_offset: bool = False
    max_geometry_offset_mismatch_m: float | None = None

    @field_validator('offset_byte', mode='before')
    @classmethod
    def _check_offset_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(value, 'moveout.offset_byte')

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
    def _check_offset_requirement(self) -> 'TimeTermStaticMoveoutRequest':
        if self.distance_source == 'offset_header' and self.offset_byte is None:
            raise ValueError(
                'moveout.offset_byte is required when '
                'moveout.distance_source is offset_header'
            )
        return self


class TimeTermStaticRobustRequest(BaseModel):
    """Robust outlier-rejection options for time-term inversion."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    method: Literal['mad', 'sigma'] = 'mad'
    threshold: float = 3.5
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


class TimeTermStaticSolverRequest(BaseModel):
    """Solver options for time-term static inversion."""

    model_config = ConfigDict(extra='forbid')

    damping: float = 0.01
    gauge: Literal['mean_zero', 'reference_node'] = 'mean_zero'
    reference_node_id: int | None = None
    robust: TimeTermStaticRobustRequest = Field(
        default_factory=TimeTermStaticRobustRequest,
    )

    @field_validator('damping', mode='before')
    @classmethod
    def _check_damping(cls, value: object) -> float:
        return _require_nonnegative_finite_float(value, 'solver.damping')

    @field_validator('reference_node_id', mode='before')
    @classmethod
    def _check_reference_node_id(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('solver.reference_node_id must be a non-negative integer')
        if value < 0:
            raise ValueError('solver.reference_node_id must be a non-negative integer')
        return value

    @model_validator(mode='after')
    def _check_reference_node_gauge(self) -> 'TimeTermStaticSolverRequest':
        if self.gauge == 'reference_node' and self.reference_node_id is None:
            raise ValueError(
                'solver.reference_node_id is required when '
                "solver.gauge is 'reference_node'"
            )
        if self.gauge != 'reference_node' and self.reference_node_id is not None:
            raise ValueError(
                'solver.reference_node_id is only allowed when '
                "solver.gauge is 'reference_node'"
            )
        return self


class TimeTermStaticApplyOptions(BaseModel):
    """Options for time-term artifact output and corrected TraceStore registration."""

    model_config = ConfigDict(extra='forbid')

    interpolation: Literal['linear'] = 'linear'
    fill_value: float = 0.0
    mode: Literal['weathering_only'] = 'weathering_only'
    register_corrected_file: bool = False
    max_abs_shift_ms: float = 250.0
    output_dtype: Literal['float32'] = 'float32'

    @field_validator('fill_value', mode='before')
    @classmethod
    def _check_fill_value(cls, value: object) -> float:
        return _require_finite_float(value, 'apply.fill_value')

    @field_validator('register_corrected_file', mode='before')
    @classmethod
    def _check_register_corrected_file_bool(cls, value: object) -> bool:
        return _require_bool(value, 'apply.register_corrected_file')

    @field_validator('max_abs_shift_ms', mode='before')
    @classmethod
    def _check_max_abs_shift_ms(cls, value: object) -> float:
        return _require_positive_finite_float(value, 'apply.max_abs_shift_ms')


class TimeTermStaticApplyRequest(BaseModel):
    """Request model for future ``/statics/time-term/apply`` jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    pick_source: TimeTermStaticPickSourceRequest
    geometry: TimeTermStaticGeometryRequest = Field(
        default_factory=TimeTermStaticGeometryRequest,
    )
    linkage: TimeTermStaticLinkageRequest = Field(
        default_factory=TimeTermStaticLinkageRequest,
    )
    velocity: TimeTermStaticVelocityRequest
    moveout: TimeTermStaticMoveoutRequest = Field(
        default_factory=TimeTermStaticMoveoutRequest,
    )
    solver: TimeTermStaticSolverRequest = Field(
        default_factory=TimeTermStaticSolverRequest,
    )
    apply: TimeTermStaticApplyOptions = Field(
        default_factory=TimeTermStaticApplyOptions,
    )

    @field_validator('key1_byte', 'key2_byte', mode='before')
    @classmethod
    def _check_key_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @model_validator(mode='after')
    def _check_values(self) -> 'TimeTermStaticApplyRequest':
        if not self.file_id:
            raise ValueError('file_id must be a non-empty string')
        return self


class TimeTermStaticApplyResponse(BaseModel):
    """Response model for creating a time-term static apply job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str
