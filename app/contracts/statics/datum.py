"""Contracts for datum static correction requests and responses."""

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatumStaticGeometryRequest(BaseModel):
    """Geometry header configuration for datum static correction."""

    source_elevation_byte: int = 45
    receiver_elevation_byte: int = 41
    elevation_scalar_byte: int = 69
    source_depth_byte: int | None = None
    elevation_unit: Literal['m', 'ft'] = 'm'


class DatumStaticDatumRequest(BaseModel):
    """Datum plane and replacement velocity parameters."""

    mode: Literal['constant'] = 'constant'
    elevation_m: float
    replacement_velocity_m_s: float

    @model_validator(mode='after')
    def _check_values(self) -> 'DatumStaticDatumRequest':
        if not math.isfinite(float(self.elevation_m)):
            raise ValueError('datum.elevation_m must be finite')
        velocity = float(self.replacement_velocity_m_s)
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError('datum.replacement_velocity_m_s must be finite and > 0')
        return self


class DatumStaticExistingStaticsRequest(BaseModel):
    """Existing static-header validation options."""

    policy: Literal['fail_if_nonzero'] = 'fail_if_nonzero'
    source_static_byte: int | None = 99
    receiver_static_byte: int | None = 101
    total_static_byte: int | None = 103


class DatumStaticApplyOptions(BaseModel):
    """Options for building the corrected TraceStore."""

    interpolation: Literal['linear'] = 'linear'
    fill_value: float = 0.0
    max_abs_shift_ms: float = 250.0
    output_dtype: Literal['float32'] = 'float32'
    register_corrected_file: bool = True

    @model_validator(mode='after')
    def _check_values(self) -> 'DatumStaticApplyOptions':
        if not math.isfinite(float(self.fill_value)):
            raise ValueError('apply.fill_value must be finite')
        max_abs_shift_ms = float(self.max_abs_shift_ms)
        if not math.isfinite(max_abs_shift_ms) or max_abs_shift_ms <= 0.0:
            raise ValueError('apply.max_abs_shift_ms must be finite and > 0')
        if self.register_corrected_file is not True:
            raise ValueError('apply.register_corrected_file must be true')
        return self


class DatumStaticApplyRequest(BaseModel):
    """Request model for ``/statics/datum/apply``."""

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    geometry: DatumStaticGeometryRequest = Field(
        default_factory=DatumStaticGeometryRequest
    )
    datum: DatumStaticDatumRequest
    existing_statics: DatumStaticExistingStaticsRequest = Field(
        default_factory=DatumStaticExistingStaticsRequest,
    )
    apply: DatumStaticApplyOptions = Field(default_factory=DatumStaticApplyOptions)


class DatumStaticApplyResponse(BaseModel):
    """Response model for creating a datum static apply job."""

    job_id: str
    state: str

