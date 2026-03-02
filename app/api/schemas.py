"""Pydantic models for describing pipeline operations."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class DenoiseParams(BaseModel):
    """Parameters for the denoise transform."""

    model_config = ConfigDict(extra='forbid')

    chunk_h: int = Field(128, ge=1)
    overlap: int | tuple[int, int] | list[int] = 32
    mask_ratio: float = Field(0.5, ge=0.0, le=1.0)
    noise_std: float = Field(1.0, ge=0.0)
    mask_noise_mode: Literal['replace', 'add'] = 'replace'
    passes_batch: int = Field(4, ge=1)
    seed: int = 12345
    tile: int | tuple[int, int] | list[int] | None = None
    amp: bool | None = None
    use_amp: bool | None = None
    tiles_per_batch: int | None = Field(default=None, ge=1)
    use_ema: bool | None = None
    ckpt_path: str | None = None
    device: str | None = None

    @staticmethod
    def _is_int_value(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    @classmethod
    def _validate_positive_int_like(cls, value: object, *, name: str) -> None:
        if not cls._is_int_value(value):
            raise ValueError(f'{name} must be an int, got {value!r}')
        if int(value) <= 0:
            raise ValueError(f'{name} must be positive, got {int(value)}')

    @classmethod
    def _validate_non_negative_int_like(cls, value: object, *, name: str) -> None:
        if not cls._is_int_value(value):
            raise ValueError(f'{name} must be an int, got {value!r}')
        if int(value) < 0:
            raise ValueError(f'{name} must be non-negative, got {int(value)}')

    @classmethod
    def _validate_positive_tile_value(cls, value: object) -> None:
        if cls._is_int_value(value):
            cls._validate_positive_int_like(value, name='tile')
            return
        if isinstance(value, tuple | list) and len(value) == 2:
            cls._validate_positive_int_like(value[0], name='tile[0]')
            cls._validate_positive_int_like(value[1], name='tile[1]')
            return
        raise ValueError(f'tile must be int or pair of ints, got {value!r}')

    @classmethod
    def _validate_non_negative_overlap_value(cls, value: object) -> None:
        if cls._is_int_value(value):
            cls._validate_non_negative_int_like(value, name='overlap')
            return
        if isinstance(value, tuple | list) and len(value) == 2:
            cls._validate_non_negative_int_like(value[0], name='overlap[0]')
            cls._validate_non_negative_int_like(value[1], name='overlap[1]')
            return
        raise ValueError(f'overlap must be int or pair of ints, got {value!r}')

    @model_validator(mode='after')
    def _check_bounds_and_canonicalize(self) -> 'DenoiseParams':
        if self.tile is None:
            self._validate_non_negative_int_like(self.overlap, name='overlap')
            if int(self.overlap) >= self.chunk_h:
                raise ValueError('overlap must be less than chunk_h')
        else:
            self._validate_positive_tile_value(self.tile)
            self._validate_non_negative_overlap_value(self.overlap)
        if self.mask_ratio == 0.0:
            self.noise_std = 1.0
            self.mask_noise_mode = 'replace'
            self.seed = 12345
            self.passes_batch = 4
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
        if self.name == 'denoise':
            self.params = DenoiseParams(**(self.params or {})).model_dump(
                exclude_none=True
            )
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


class SnapOptions(BaseModel):
    """Configuration for optional raw-waveform snap refinement."""

    enabled: bool = False
    mode: Literal['peak', 'trough', 'rise'] = 'peak'
    refine: Literal['none', 'parabolic', 'zc'] = 'parabolic'
    window_ms: float = 20.0


class PickOptions(BaseModel):
    """Configuration for predicted-pick generation from probability maps."""

    method: Literal['expectation', 'argmax'] = 'expectation'
    subsample: bool = False
    sigma_ms_max: float | None = None
    snap: SnapOptions = Field(default_factory=SnapOptions)


class BatchApplyRequest(BaseModel):
    """Request model for ``/batch/apply``."""

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    pipeline_spec: PipelineSpec
    pick_options: PickOptions = Field(default_factory=PickOptions)
    save_picks: bool = False


class BatchApplyResponse(BaseModel):
    """Response model for creating a batch apply job."""

    job_id: str
    state: str


class BatchJobStatusResponse(BaseModel):
    """Response model for batch job status."""

    state: str
    progress: float
    message: str


class BatchJobFile(BaseModel):
    """One file entry generated by a batch job."""

    name: str
    size_bytes: int


class BatchJobFilesResponse(BaseModel):
    """Response model for batch job files listing."""

    files: list[BatchJobFile]
