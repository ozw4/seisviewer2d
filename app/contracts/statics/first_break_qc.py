"""Contracts for first-break QC requests and responses."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.contracts._validation import _validate_artifact_basename
from app.utils.validation import require_positive_int


class FirstBreakQcDatumSolutionRequest(BaseModel):
    """Datum static solution artifact reference for first-break QC."""

    job_id: str
    name: str = 'datum_static_solution.npz'

    @model_validator(mode='after')
    def _check_values(self) -> 'FirstBreakQcDatumSolutionRequest':
        if not self.job_id:
            raise ValueError('datum_solution.job_id must be a non-empty string')
        _validate_artifact_basename(self.name, 'datum_solution.name')
        return self


class FirstBreakQcPickSourceRequest(BaseModel):
    """First-break pick source reference for first-break QC."""

    kind: Literal['batch_job_artifact', 'manual_npz_artifact', 'manual_memmap']
    job_id: str | None = None
    name: str | None = None

    @model_validator(mode='after')
    def _check_ref(self) -> 'FirstBreakQcPickSourceRequest':
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


class FirstBreakQcOffsetRequest(BaseModel):
    """Offset header configuration for first-break QC."""

    offset_byte: int = 37

    @model_validator(mode='after')
    def _check_values(self) -> 'FirstBreakQcOffsetRequest':
        require_positive_int(self.offset_byte, 'offset.offset_byte')
        return self


class FirstBreakQcOptionsRequest(BaseModel):
    """QC options for first-break QC."""

    require_linear_offset_model: bool = False


class FirstBreakQcRequest(BaseModel):
    """Request model for ``/statics/first-break/qc``."""

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    datum_solution: FirstBreakQcDatumSolutionRequest
    pick_source: FirstBreakQcPickSourceRequest
    offset: FirstBreakQcOffsetRequest = Field(default_factory=FirstBreakQcOffsetRequest)
    qc: FirstBreakQcOptionsRequest = Field(default_factory=FirstBreakQcOptionsRequest)

    @model_validator(mode='after')
    def _check_values(self) -> 'FirstBreakQcRequest':
        if not self.file_id:
            raise ValueError('file_id must be a non-empty string')
        require_positive_int(self.key1_byte, 'key1_byte')
        require_positive_int(self.key2_byte, 'key2_byte')
        return self


class FirstBreakQcJobResponse(BaseModel):
    """Response model for creating a first-break QC job."""

    job_id: str
    state: str
