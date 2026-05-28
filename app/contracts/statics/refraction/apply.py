"""Top-level apply contracts for refraction static workflows."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import require_trace_header_byte
from app.contracts.statics.refraction.common import RefractionStaticExportFormat
from app.contracts.statics.refraction.export import RefractionStaticExportRequest
from app.contracts.statics.refraction.field_corrections import (
    RefractionStaticFieldCorrectionsRequest,
)
from app.contracts.statics.refraction.inputs import (
    RefractionStaticGeometryRequest,
    RefractionStaticLinkageRequest,
    RefractionStaticPickSourceRequest,
)
from app.contracts.statics.refraction.model import RefractionStaticModelRequest
from app.contracts.statics.refraction.options import (
    RefractionStaticApplyOptions,
    RefractionStaticConversionRequest,
    RefractionStaticDatumRequest,
    RefractionStaticMoveoutRequest,
    RefractionStaticReducedTimeQcRequest,
    RefractionStaticSolverRequest,
)


class RefractionStaticApplyRequest(BaseModel):
    """Request model for ``/statics/refraction/apply`` jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    pick_source: RefractionStaticPickSourceRequest
    geometry: RefractionStaticGeometryRequest = Field(
        default_factory=RefractionStaticGeometryRequest,
    )
    linkage: RefractionStaticLinkageRequest = Field(
        default_factory=RefractionStaticLinkageRequest,
    )
    model: RefractionStaticModelRequest
    moveout: RefractionStaticMoveoutRequest = Field(
        default_factory=RefractionStaticMoveoutRequest,
    )
    solver: RefractionStaticSolverRequest = Field(
        default_factory=RefractionStaticSolverRequest,
    )
    datum: RefractionStaticDatumRequest = Field(
        default_factory=RefractionStaticDatumRequest,
    )
    conversion: RefractionStaticConversionRequest = Field(
        default_factory=RefractionStaticConversionRequest,
    )
    reduced_time_qc: RefractionStaticReducedTimeQcRequest = Field(
        default_factory=RefractionStaticReducedTimeQcRequest,
    )
    field_corrections: RefractionStaticFieldCorrectionsRequest = Field(
        default_factory=RefractionStaticFieldCorrectionsRequest,
    )
    export: RefractionStaticExportRequest = Field(
        default_factory=RefractionStaticExportRequest,
    )
    apply: RefractionStaticApplyOptions = Field(
        default_factory=RefractionStaticApplyOptions,
    )

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

    @model_validator(mode='after')
    def _check_supported_refraction_apply_model(
        self,
    ) -> 'RefractionStaticApplyRequest':
        if self.conversion.mode == 't1lsst_multilayer':
            enabled_count = self.model.enabled_refraction_layer_count
            if self.conversion.layer_count != enabled_count:
                enabled_kinds = [
                    layer.kind
                    for layer in self.model.layers or []
                    if layer.enabled
                ]
                enabled_text = ', '.join(enabled_kinds) if enabled_kinds else 'none'
                raise ValueError(
                    'conversion.layer_count must match enabled refraction layers; '
                    f'conversion.layer_count={self.conversion.layer_count!r}, '
                    f'enabled layer kinds={enabled_text}'
                )
        if (
            self.field_corrections.source_depth.mode == 'weathering_velocity_time'
            and self.field_corrections.source_depth.source_depth_byte is None
            and self.geometry.source_depth_byte is None
        ):
            raise ValueError(
                'field_corrections.source_depth.source_depth_byte or '
                'geometry.source_depth_byte is required when '
                'field_corrections.source_depth.mode is weathering_velocity_time'
            )
        return self


class RefractionStaticApplyResponse(BaseModel):
    """Response model for creating a refraction static apply job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str
    requested_formats: list[RefractionStaticExportFormat] | None = None
