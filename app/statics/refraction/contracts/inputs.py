"""Input reference contracts for refraction static requests."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _validate_artifact_basename,
    require_trace_header_byte,
)


class RefractionStaticPickSourceRequest(BaseModel):
    """First-break pick source reference for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal[
        'uploaded_npz',
        'batch_predicted_npz',
        'manual_npz_artifact',
        'manual_memmap',
    ] = Field(
        description=(
            'Use uploaded_npz for Static Correction UI direct .npz uploads; '
            'artifact-backed kinds are retained for legacy job-artifact flows.'
        )
    )
    job_id: str | None = None
    artifact_name: str | None = None

    @model_validator(mode='after')
    def _check_ref(self) -> 'RefractionStaticPickSourceRequest':
        if self.kind == 'uploaded_npz':
            if self.job_id is not None:
                raise ValueError(
                    'pick_source.job_id must be omitted for uploaded_npz'
                )
            if self.artifact_name is not None:
                raise ValueError(
                    'pick_source.artifact_name must be omitted for uploaded_npz'
                )
            return self

        if self.kind == 'manual_memmap':
            if self.job_id is not None or self.artifact_name is not None:
                raise ValueError(
                    'pick_source.job_id/artifact_name must be omitted for manual_memmap'
                )
            return self

        if not self.job_id:
            raise ValueError('pick_source.job_id is required for artifact sources')
        if self.kind == 'batch_predicted_npz' and self.artifact_name is None:
            self.artifact_name = 'predicted_picks_time_s.npz'
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


class RefractionStaticGeometryRequest(BaseModel):
    """Trace header configuration for refraction static inversion."""

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
    def _check_distinct_endpoint_ids(self) -> 'RefractionStaticGeometryRequest':
        if self.source_id_byte == self.receiver_id_byte:
            raise ValueError('geometry.source_id_byte and receiver_id_byte must differ')
        return self


class RefractionStaticLinkageRequest(BaseModel):
    """Source/receiver endpoint linkage artifact reference."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['required', 'optional', 'none'] = 'none'
    job_id: str | None = None
    artifact_name: str = 'geometry_linkage.npz'

    @field_validator('artifact_name', mode='before')
    @classmethod
    def _check_artifact_name(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError('linkage.artifact_name must be a plain file name')
        return _validate_artifact_basename(value, 'linkage.artifact_name')

    @model_validator(mode='after')
    def _check_ref(self) -> 'RefractionStaticLinkageRequest':
        if self.mode == 'required' and not self.job_id:
            raise ValueError('linkage.job_id is required when linkage.mode is required')
        if self.mode == 'none' and self.job_id is not None:
            raise ValueError('linkage.job_id must be omitted when linkage.mode is none')
        if self.job_id is not None and not self.job_id:
            raise ValueError('linkage.job_id must be a non-empty string')
        return self
