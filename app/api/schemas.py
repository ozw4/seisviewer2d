"""Pydantic models for describing pipeline operations."""

from typing import Any, Literal

from pydantic import BaseModel, Field, root_validator

TransformName = Literal['bandpass', 'denoise']
AnalyzerName = Literal['fbpick']


class BandpassParams(BaseModel):
        """Parameters for the band-pass filter."""

        low_hz: float = Field(..., ge=0.0)
        high_hz: float = Field(..., ge=0.0)
        dt: float = Field(0.002, gt=0.0)
        taper: float = Field(0.0, ge=0.0)

        @root_validator
        def _check_bounds(cls, values: dict[str, Any]) -> dict[str, Any]:
                low = values.get('low_hz')
                high = values.get('high_hz')
                dt = values.get('dt')
                if low is None or high is None or dt is None:
                        return values
                if low >= high:
                        raise ValueError('low_hz must be less than high_hz')
                nyq = 0.5 / dt
                if high > nyq:
                        raise ValueError('high_hz must be <= Nyquist (0.5/dt)')
                return values


class PipelineOp(BaseModel):
        """Specification for a single pipeline operation."""

        kind: Literal['transform', 'analyzer']
        name: TransformName | AnalyzerName
        params: dict[str, Any] = Field(default_factory=dict)
        label: str | None = None

        @root_validator
        def _validate_params(cls, values: dict[str, Any]) -> dict[str, Any]:
                name = values.get('name')
                params = values.get('params') or {}
                if name == 'bandpass':
                        BandpassParams(**params)
                return values


class PipelineSpec(BaseModel):
        """Sequence of pipeline operations."""

        steps: list[PipelineOp]


class PipelineSectionResponse(BaseModel):
        """Response model for ``/pipeline/section``."""

        taps: dict[str, Any]
        pipeline_key: str


class PipelineAllResponse(BaseModel):
        """Response model for ``/pipeline/all``."""

        job_id: str
        state: str


class PipelineJobStatusResponse(BaseModel):
        """Response model for pipeline job status."""

        state: str
        progress: float
        message: str
