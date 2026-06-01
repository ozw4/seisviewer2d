"""Export request contracts for refraction static workflows."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.statics.refraction.contracts.common import RefractionStaticExportFormat


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
