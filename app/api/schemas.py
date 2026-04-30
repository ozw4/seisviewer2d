"""Pydantic models for describing pipeline operations."""

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.utils.validation import require_non_negative_int, require_positive_int

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

    @classmethod
    def _validate_positive_tile_value(cls, value: object) -> None:
        if isinstance(value, int) and not isinstance(value, bool):
            require_positive_int(value, 'tile')
            return
        if isinstance(value, tuple | list) and len(value) == 2:
            require_positive_int(value[0], 'tile[0]')
            require_positive_int(value[1], 'tile[1]')
            return
        raise ValueError(f'tile must be int or pair of ints, got {value!r}')

    @classmethod
    def _validate_non_negative_overlap_value(cls, value: object) -> None:
        if isinstance(value, int) and not isinstance(value, bool):
            require_non_negative_int(value, 'overlap')
            return
        if isinstance(value, tuple | list) and len(value) == 2:
            require_non_negative_int(value[0], 'overlap[0]')
            require_non_negative_int(value[1], 'overlap[1]')
            return
        raise ValueError(f'overlap must be int or pair of ints, got {value!r}')

    @model_validator(mode='after')
    def _check_bounds_and_canonicalize(self) -> 'DenoiseParams':
        if self.tile is None:
            require_non_negative_int(self.overlap, 'overlap')
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


class FbpickParams(BaseModel):
    """Parameters for the fbpick analyzer."""

    model_config = ConfigDict(extra='forbid')

    amp: bool | None = None
    use_amp: bool | None = None
    overlap: int | tuple[int, int] | list[int] | None = None
    tau: float | None = Field(default=None, ge=0.0)
    tile: tuple[int, int] | list[int] | None = None
    channel: str | int | None = None
    tiles_per_batch: int | None = Field(default=None, ge=1)
    model_id: str | None = None
    offsets: list[float] | None = None

    @model_validator(mode='after')
    def _check_values(self) -> 'FbpickParams':
        overlap = self.overlap
        if overlap is not None:
            if isinstance(overlap, int) and not isinstance(overlap, bool):
                require_positive_int(overlap, 'overlap')
            elif isinstance(overlap, (tuple, list)):
                if len(overlap) != 2:
                    raise ValueError(f'overlap must be a pair of ints, got {overlap!r}')
                require_positive_int(overlap[0], 'overlap[0]')
                require_positive_int(overlap[1], 'overlap[1]')
            else:
                raise ValueError(
                    f'overlap must be int or pair of ints, got {overlap!r}'
                )

        tile = self.tile
        if tile is not None:
            if not isinstance(tile, (tuple, list)):
                raise ValueError(f'tile must be a pair of ints, got {tile!r}')
            if len(tile) != 2:
                raise ValueError(f'tile must be a pair of ints, got {tile!r}')
            require_positive_int(tile[0], 'tile[0]')
            require_positive_int(tile[1], 'tile[1]')

        model_id = self.model_id
        if model_id is not None:
            if Path(model_id).name != model_id:
                raise ValueError('model_id must be a plain file name')
            if not (model_id.startswith('fbpick_') and model_id.endswith('.pt')):
                raise ValueError(
                    "model_id must start with 'fbpick_' and end with '.pt'"
                )
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
        if self.name == 'fbpick':
            self.params = FbpickParams(**(self.params or {})).model_dump(
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


class StaticJobStatusResponse(BaseModel):
    """Response model for static correction job status."""

    state: str
    progress: float
    message: str


class StaticJobFile(BaseModel):
    """One file entry generated by a static correction job."""

    name: str
    size_bytes: int


class StaticJobFilesResponse(BaseModel):
    """Response model for static correction job files listing."""

    files: list[StaticJobFile]


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
