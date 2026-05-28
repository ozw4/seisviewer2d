"""Contracts for residual static correction requests and responses."""

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


class ResidualStaticDatumSolutionRequest(BaseModel):
    """Datum static solution artifact reference for residual statics."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    name: str = 'datum_static_solution.npz'

    @model_validator(mode='after')
    def _check_values(self) -> 'ResidualStaticDatumSolutionRequest':
        if not self.job_id:
            raise ValueError('datum_solution.job_id must be a non-empty string')
        _validate_artifact_basename(self.name, 'datum_solution.name')
        return self


class ResidualStaticPickSourceRequest(BaseModel):
    """First-break pick source reference for residual static estimation."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['batch_job_artifact', 'manual_npz_artifact', 'manual_memmap']
    job_id: str | None = None
    name: str | None = None

    @model_validator(mode='after')
    def _check_ref(self) -> 'ResidualStaticPickSourceRequest':
        if self.kind in {'batch_job_artifact', 'manual_npz_artifact'}:
            if not self.job_id:
                raise ValueError('pick_source.job_id is required for artifact sources')
            if not self.name:
                raise ValueError('pick_source.name is required for artifact sources')
            _validate_artifact_basename(self.name, 'pick_source.name')
            return self

        if self.job_id is not None or self.name is not None:
            raise ValueError('pick_source.job_id/name must be omitted for manual_memmap')
        return self


class ResidualStaticGeometryRequest(BaseModel):
    """Source and receiver header configuration for residual statics."""

    model_config = ConfigDict(extra='forbid')

    source_id_byte: int
    receiver_id_byte: int

    @field_validator('source_id_byte', 'receiver_id_byte', mode='before')
    @classmethod
    def _check_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @model_validator(mode='after')
    def _check_distinct_headers(self) -> 'ResidualStaticGeometryRequest':
        if self.source_id_byte == self.receiver_id_byte:
            raise ValueError('source_id_byte and receiver_id_byte must differ')
        return self


class ResidualStaticOffsetRequest(BaseModel):
    """Offset header configuration for residual statics."""

    model_config = ConfigDict(extra='forbid')

    offset_byte: int | None = 37

    @field_validator('offset_byte', mode='before')
    @classmethod
    def _check_offset_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(value, 'offset.offset_byte')


class ResidualStaticMoveoutRequest(BaseModel):
    """Moveout model configuration for residual static estimation."""

    model_config = ConfigDict(extra='forbid')

    model: Literal['linear_abs_offset', 'none'] = 'linear_abs_offset'


class ResidualStaticSolverRequest(BaseModel):
    """Sparse solver stabilization options for residual statics."""

    model_config = ConfigDict(extra='forbid')

    gauge: Literal['zero_mean_source_receiver'] = 'zero_mean_source_receiver'
    damping_lambda: float = 0.0
    min_valid_picks: int = 10
    min_picks_per_source: int = 1
    min_picks_per_receiver: int = 1
    max_abs_estimated_delay_ms: float = 250.0

    @field_validator('damping_lambda', mode='before')
    @classmethod
    def _check_damping_lambda(cls, value: object) -> float:
        return _require_nonnegative_finite_float(value, 'solver.damping_lambda')

    @field_validator(
        'min_valid_picks',
        'min_picks_per_source',
        'min_picks_per_receiver',
        mode='before',
    )
    @classmethod
    def _check_positive_int(cls, value: object, info: Any) -> int:
        return _require_positive_int(value, f'solver.{info.field_name}')

    @field_validator('max_abs_estimated_delay_ms', mode='before')
    @classmethod
    def _check_max_abs_estimated_delay_ms(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'solver.max_abs_estimated_delay_ms',
        )


class ResidualStaticRobustRequest(BaseModel):
    """Robust outlier-rejection options for residual statics."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    method: Literal['mad', 'sigma'] = 'mad'
    max_iterations: int = 3
    threshold: float = 4.0
    min_used_fraction: float = 0.5

    @field_validator('enabled', mode='before')
    @classmethod
    def _check_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'robust.enabled')

    @field_validator('max_iterations', mode='before')
    @classmethod
    def _check_max_iterations(cls, value: object) -> int:
        return _require_positive_int(value, 'robust.max_iterations')

    @field_validator('threshold', mode='before')
    @classmethod
    def _check_threshold(cls, value: object) -> float:
        return _require_positive_finite_float(value, 'robust.threshold')

    @field_validator('min_used_fraction', mode='before')
    @classmethod
    def _check_min_used_fraction(cls, value: object) -> float:
        fraction = _require_positive_finite_float(value, 'robust.min_used_fraction')
        if fraction > 1.0:
            raise ValueError('robust.min_used_fraction must be <= 1')
        return fraction


class ResidualStaticApplyOptions(BaseModel):
    """Options for applying residual statics to a corrected TraceStore."""

    model_config = ConfigDict(extra='forbid')

    interpolation: Literal['linear'] = 'linear'
    fill_value: float = 0.0
    max_abs_shift_ms: float = 250.0
    output_dtype: Literal['float32'] = 'float32'
    register_corrected_file: bool = True

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

    @model_validator(mode='after')
    def _check_values(self) -> 'ResidualStaticApplyOptions':
        if self.register_corrected_file is not True:
            raise ValueError('apply.register_corrected_file must be true')
        return self


class ResidualStaticApplyRequest(BaseModel):
    """Request model for future ``/statics/residual/apply`` jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    datum_solution: ResidualStaticDatumSolutionRequest
    pick_source: ResidualStaticPickSourceRequest
    geometry: ResidualStaticGeometryRequest
    offset: ResidualStaticOffsetRequest = Field(
        default_factory=ResidualStaticOffsetRequest,
    )
    moveout: ResidualStaticMoveoutRequest = Field(
        default_factory=ResidualStaticMoveoutRequest,
    )
    solver: ResidualStaticSolverRequest = Field(
        default_factory=ResidualStaticSolverRequest,
    )
    robust: ResidualStaticRobustRequest = Field(
        default_factory=ResidualStaticRobustRequest,
    )
    apply: ResidualStaticApplyOptions = Field(
        default_factory=ResidualStaticApplyOptions,
    )

    @field_validator('key1_byte', 'key2_byte', mode='before')
    @classmethod
    def _check_key_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @model_validator(mode='after')
    def _check_values(self) -> 'ResidualStaticApplyRequest':
        if not self.file_id:
            raise ValueError('file_id must be a non-empty string')
        if self.moveout.model == 'linear_abs_offset' and self.offset.offset_byte is None:
            raise ValueError(
                'offset.offset_byte is required for linear_abs_offset moveout'
            )
        if self.moveout.model == 'none' and self.offset.offset_byte is not None:
            raise ValueError('offset.offset_byte must be null for none moveout')
        return self

    @property
    def source_id_byte(self) -> int:
        return self.geometry.source_id_byte

    @property
    def receiver_id_byte(self) -> int:
        return self.geometry.receiver_id_byte

    @property
    def offset_byte(self) -> int | None:
        return self.offset.offset_byte


class ResidualStaticApplyResponse(BaseModel):
    """Response model for creating a residual static apply job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str
