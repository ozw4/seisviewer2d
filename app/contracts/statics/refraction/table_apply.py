"""Static-table apply contracts for refraction static workflows."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_bool,
    _require_finite_float,
    _require_positive_finite_float,
    _validate_artifact_basename,
    require_trace_header_byte,
)
from app.contracts.statics.refraction.inputs import RefractionStaticGeometryRequest


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
