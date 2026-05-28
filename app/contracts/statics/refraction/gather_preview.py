"""Gather preview contracts for refraction static QC workflows."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_finite_float,
    _require_positive_int,
    require_trace_header_byte,
)
from app.contracts.statics.refraction.common import (
    RefractionStaticGatherPreviewAxis,
    RefractionStaticGatherPreviewOverlayLayer,
    RefractionStaticGatherPreviewSampleSource,
    RefractionStaticGatherPreviewScaling,
)


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
