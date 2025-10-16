"""Pydantic models for describing pipeline operations."""

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, PrivateAttr, model_validator, root_validator

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
                        raise ValueError('dt is derived from the data and can no longer be specified')
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


class _Key1GatherRequest(BaseModel):
        """Base class for requests that target a specific key1 gather."""

        file_id: str
        key1_value: Union[int, float, str]

        # Hidden field, carried through the model (readable by routers), excluded from OpenAPI
        used_deprecated_idx: bool = Field(default=False, exclude=True)

        # Optional micro-cache for computed index; PrivateAttr is fine here
        _key1_idx_cache: int | None = PrivateAttr(default=None)

        @root_validator(pre=True)
        def _migrate_key1_idx(cls, values: dict[str, Any]) -> dict[str, Any]:
                """Translate legacy key1_idx -> key1_value exactly once; fail fast on missing data."""
                if not isinstance(values, dict):
                        return values
                if 'key1_value' in values or 'key1_idx' not in values:
                        return values

                # basic inputs
                idx = values.get('key1_idx')
                file_id = values.get('file_id')
                if not isinstance(file_id, str) or not file_id:
                        raise ValueError('file_id is required when using key1_idx')
                if not isinstance(idx, int):
                        raise ValueError('key1_idx must be an integer when provided')

                # resolve key list from registry; no silent fallback
                from app.utils.segy_meta import FILE_REGISTRY  # lazy import to avoid cycles
                entry = FILE_REGISTRY.get(file_id)
                if not isinstance(entry, dict) or 'key1_values' not in entry:
                        raise KeyError(f'key1_values unavailable for file_id={file_id}')

                raw = entry['key1_values']
                if hasattr(raw, 'tolist'):
                        raw = raw.tolist()
                if not isinstance(raw, (list, tuple)):
                        raise TypeError('key1_values registry entry must be list-like')

                key_list = [cls._normalize_key1_value(v) for v in raw]
                if idx < 0 or idx >= len(key_list):
                        raise ValueError(f'key1_idx out of range: {idx}')

                values = dict(values)  # copy before mutation
                values['key1_value'] = key_list[idx]
                values['used_deprecated_idx'] = True
                values.pop('key1_idx', None)  # keep legacy out of model_extra
                return values

        def _resolve_key1_values(self) -> list[Union[int, float, str]]:
                """Fetch the normalized key1_values list for this file_id, or raise."""
                from app.utils.segy_meta import FILE_REGISTRY  # lazy import
                entry = FILE_REGISTRY.get(self.file_id)
                if not isinstance(entry, dict) or 'key1_values' not in entry:
                        raise KeyError(f'key1_values unavailable for file_id {self.file_id}')
                raw = entry['key1_values']
                if hasattr(raw, 'tolist'):
                        raw = raw.tolist()
                if not isinstance(raw, (list, tuple)):
                        raise TypeError('key1_values registry entry must be list-like')
                return [self._normalize_key1_value(v) for v in raw]

        @property
        def key1_idx(self) -> int:
                """Compute the index of key1_value within the normalized key1_values list (cached)."""
                if self._key1_idx_cache is not None:
                        return self._key1_idx_cache
                values = self._resolve_key1_values()
                try:
                        idx = values.index(self.key1_value)
                except ValueError as exc:
                        raise KeyError(
                                f'unknown key1_value: {self.key1_value!r} for file_id {self.file_id}'
                        ) from exc
                self._key1_idx_cache = idx
                return idx

        @staticmethod
        def _normalize_key1_value(value: object) -> Union[int, float, str]:
                """Coerce numpy scalars and odd types into plain int/float/str deterministically."""
                if isinstance(value, (int, float, str)):
                        return value
                if hasattr(value, '__int__'):
                        try:
                                return int(value)  # type: ignore[arg-type]
                        except Exception:  # noqa: BLE001
                                pass
                if hasattr(value, '__float__'):
                        try:
                                return float(value)  # type: ignore[arg-type]
                        except Exception:  # noqa: BLE001
                                pass
                return str(value)


class PickPostModel(_Key1GatherRequest):
        """Request payload for posting manual picks."""

        trace: int
        time: float
        key1_byte: int


class FbpickRequest(_Key1GatherRequest):
        """Request payload for fbpick inference."""

        key1_byte: int = 189
        key2_byte: int = 193
        offset_byte: int | None = None
        tile_h: int = 128
        tile_w: int = 6016
        overlap: int = 32
        amp: bool = True
        pipeline_key: str | None = None
        tap_label: str | None = None
        start: int = Field(0, ge=0)
        length: int | None = None


class SectionQuery(_Key1GatherRequest):
        """Query parameters for section retrieval endpoints."""

        key1_byte: int = 189
        key2_byte: int = 193
        start: int = Field(0, ge=0)
        length: int | None = None

        @model_validator(mode='after')
        def _validate_length(self) -> 'SectionQuery':
                if self.length is not None and self.length < 1:
                        raise ValueError('length must be >= 1')
                return self


class SectionBinQuery(SectionQuery):
        """Binary section retrieval query."""


class SectionWindowBinQuery(SectionQuery):
        """Query parameters for ``get_section_window_bin``."""

        offset_byte: int | None = None
        y0: int = Field(0, ge=0)
        y1: int | None = None
        step_x: int = Field(1, ge=1)
        step_y: int = Field(1, ge=1)
        pipeline_key: str | None = None
        tap_label: str | None = None

        @model_validator(mode='before')
        @classmethod
        def _migrate_trace_window(cls, data: Any) -> Any:
                if not isinstance(data, dict):
                        return data
                if 'length' not in data:
                        x0 = data.get('x0')
                        x1 = data.get('x1')
                        if x0 is not None and x1 is not None:
                                start = int(x0)
                                end = int(x1)
                                data = dict(data)
                                data['start'] = start
                                data['length'] = max(1, end - start + 1)
                return data

        @model_validator(mode='after')
        def _validate_ranges(self) -> 'SectionWindowBinQuery':
                if self.length is not None and self.length < 1:
                        raise ValueError('length must be >= 1')
                if self.y1 is not None and self.y1 < self.y0:
                        raise ValueError('y1 must be >= y0')
                return self


class PipelineSectionQuery(_Key1GatherRequest):
        """Query parameters for ``/pipeline/section``."""

        key1_byte: int = 189
        key2_byte: int = 193
        offset_byte: int | None = None
        start: int = Field(0, ge=0)
        length: int | None = None
