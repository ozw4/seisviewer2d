"""Contracts for static linkage geometry building."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_bool,
    _require_positive_finite_float,
    require_trace_header_byte,
)


class StaticLinkageGeometryRequest(BaseModel):
    """Geometry header configuration for static linkage."""

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
        return require_trace_header_byte(value, info.field_name)

    @field_validator('source_depth_byte', mode='before')
    @classmethod
    def _check_optional_header_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(value, 'source_depth_byte')

    @model_validator(mode='after')
    def _check_unique_headers(self) -> 'StaticLinkageGeometryRequest':
        header_bytes = (
            self.source_x_byte,
            self.source_y_byte,
            self.receiver_x_byte,
            self.receiver_y_byte,
            self.coordinate_scalar_byte,
        )
        if len(set(header_bytes)) != len(header_bytes):
            raise ValueError('geometry header bytes must be unique')
        if self.source_id_byte == self.receiver_id_byte:
            raise ValueError('source_id_byte and receiver_id_byte must differ')
        return self


class StaticLinkageOptionsRequest(BaseModel):
    """Linkage options for static linkage geometry building."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['none', 'auto_threshold']
    threshold_m: float | None = None
    receiver_location_interval_m: float | None = None
    prefer_receiver_anchor: bool = True

    @field_validator('threshold_m', mode='before')
    @classmethod
    def _check_threshold_m(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(value, 'linkage.threshold_m')

    @field_validator('receiver_location_interval_m', mode='before')
    @classmethod
    def _check_receiver_location_interval_m(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'linkage.receiver_location_interval_m',
        )

    @field_validator('prefer_receiver_anchor', mode='before')
    @classmethod
    def _check_prefer_receiver_anchor(cls, value: object) -> bool:
        return _require_bool(value, 'linkage.prefer_receiver_anchor')

    @model_validator(mode='after')
    def _check_mode_options(self) -> 'StaticLinkageOptionsRequest':
        if self.mode == 'auto_threshold' and self.threshold_m is None:
            raise ValueError('linkage.threshold_m is required for auto_threshold')
        if self.mode == 'none':
            if self.threshold_m is not None:
                raise ValueError('linkage.threshold_m must be null for none mode')
            if self.receiver_location_interval_m is not None:
                raise ValueError(
                    'linkage.receiver_location_interval_m must be null for none mode'
                )
        return self


class StaticLinkageBuildRequest(BaseModel):
    """Request model for future ``/statics/linkage/build`` jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    geometry: StaticLinkageGeometryRequest = Field(
        default_factory=StaticLinkageGeometryRequest,
    )
    linkage: StaticLinkageOptionsRequest

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


class StaticLinkageBuildResponse(BaseModel):
    """Response model for creating a static linkage build job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str
