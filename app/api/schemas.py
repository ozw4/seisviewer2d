"""Pydantic models for describing pipeline operations."""

from typing import Any, Literal

from pydantic import BaseModel, Field

TransformName = Literal["bandpass", "denoise"]
AnalyzerName = Literal["fbpick"]


class PipelineOp(BaseModel):
    """Specification for a single pipeline operation."""

    kind: Literal["transform", "analyzer"]
    name: TransformName | AnalyzerName
    params: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None


class PipelineSpec(BaseModel):
    """Sequence of pipeline operations."""

    steps: list[PipelineOp]
