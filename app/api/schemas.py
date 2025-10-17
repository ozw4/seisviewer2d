"""Pydantic models for describing pipeline operations."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TransformName = Literal['bandpass', 'denoise']
AnalyzerName = Literal['fbpick']


class BandpassParams(BaseModel):
	"""Parameters for the band-pass filter."""

	low_hz: float = Field(..., ge=0.0)
	high_hz: float = Field(..., ge=0.0)
	taper: float = Field(0.0, ge=0.0)

	@model_validator(mode='before')
	@classmethod
	def _ensure_no_dt(cls, data: Any) -> Any:
		if isinstance(data, dict) and 'dt' in data:
			raise ValueError(
				'dt is derived from the data and can no longer be specified'
			)
		return data

	@model_validator(mode='after')
	def _check_bounds(self) -> 'BandpassParams':
		# フィールド制約（ge/gt）は Field で既に検証済み。
		if self.low_hz >= self.high_hz:
			raise ValueError('low_hz must be less than high_hz')
		return self


class PipelineOp(BaseModel):
	"""Specification for a single pipeline operation."""

	kind: Literal['transform', 'analyzer']
	name: TransformName | AnalyzerName
	params: dict[str, Any] = Field(default_factory=dict)
	label: str | None = None

	@model_validator(mode='after')
	def _validate_params(self) -> 'PipelineOp':
		# name に応じて params をサブモデルで検証
		if self.name == 'bandpass':
			BandpassParams(**(self.params or {}))
		return self


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
