"""Field-correction request contracts for refraction static workflows."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_bool,
    _require_finite_float,
    _require_positive_finite_float,
    _validate_artifact_basename,
    require_trace_header_byte,
)
from app.utils.validation import require_non_negative_int


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
