"""Pydantic models for describing pipeline operations."""

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


def _validate_artifact_basename(name: str, field_name: str) -> str:
    if not name:
        raise ValueError(f'{field_name} must be a non-empty file name')
    if name in {'.', '..'}:
        raise ValueError(f'{field_name} must be a plain file name')
    if Path(name).name != name:
        raise ValueError(f'{field_name} must be a plain file name')
    return name


def require_trace_header_byte(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer SEG-Y trace header byte')
    if not isinstance(value, int):
        raise ValueError(f'{name} must be an integer SEG-Y trace header byte')
    if value < 1 or value > 240:
        raise ValueError(f'{name} must be in the range 1..240')
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer')
    if not isinstance(value, int):
        raise ValueError(f'{name} must be a positive integer')
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be a bool')
    return value


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


def _require_nonnegative_finite_float(value: object, name: str) -> float:
    out = _require_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f'{name} must be finite and >= 0')
    return out


def _require_positive_finite_float(value: object, name: str) -> float:
    out = _require_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f'{name} must be finite and > 0')
    return out


def _velocity_values_match(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=1.0e-9)


class StaticLinkageGeometryRequest(BaseModel):
    """Geometry header configuration for static linkage."""

    model_config = ConfigDict(extra='forbid')

    source_x_byte: int = 73
    source_y_byte: int = 77
    receiver_x_byte: int = 81
    receiver_y_byte: int = 85
    coordinate_scalar_byte: int = 71

    @field_validator(
        'source_x_byte',
        'source_y_byte',
        'receiver_x_byte',
        'receiver_y_byte',
        'coordinate_scalar_byte',
        mode='before',
    )
    @classmethod
    def _check_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @model_validator(mode='after')
    def _check_unique_headers(self) -> 'StaticLinkageGeometryRequest':
        header_bytes = (
            self.source_x_byte,
            self.source_y_byte,
            self.receiver_x_byte,
            self.receiver_y_byte,
            self.coordinate_scalar_byte,
        )
        if len(set(header_bytes)) != len(header_bytes):
            raise ValueError('geometry header bytes must be unique')
        return self


class StaticLinkageOptionsRequest(BaseModel):
    """Linkage options for static linkage geometry building."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['none', 'auto_threshold']
    threshold_m: float | None = None
    receiver_location_interval_m: float | None = None
    prefer_receiver_anchor: bool = True

    @field_validator('threshold_m', mode='before')
    @classmethod
    def _check_threshold_m(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(value, 'linkage.threshold_m')

    @field_validator('receiver_location_interval_m', mode='before')
    @classmethod
    def _check_receiver_location_interval_m(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'linkage.receiver_location_interval_m',
        )

    @field_validator('prefer_receiver_anchor', mode='before')
    @classmethod
    def _check_prefer_receiver_anchor(cls, value: object) -> bool:
        return _require_bool(value, 'linkage.prefer_receiver_anchor')

    @model_validator(mode='after')
    def _check_mode_options(self) -> 'StaticLinkageOptionsRequest':
        if self.mode == 'auto_threshold' and self.threshold_m is None:
            raise ValueError('linkage.threshold_m is required for auto_threshold')
        if self.mode == 'none':
            if self.threshold_m is not None:
                raise ValueError('linkage.threshold_m must be null for none mode')
            if self.receiver_location_interval_m is not None:
                raise ValueError(
                    'linkage.receiver_location_interval_m must be null for none mode'
                )
        return self


class StaticLinkageBuildRequest(BaseModel):
    """Request model for future ``/statics/linkage/build`` jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    geometry: StaticLinkageGeometryRequest = Field(
        default_factory=StaticLinkageGeometryRequest,
    )
    linkage: StaticLinkageOptionsRequest

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


class StaticLinkageBuildResponse(BaseModel):
    """Response model for creating a static linkage build job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str


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


class ResidualStaticDatumSolutionRequest(BaseModel):
    """Datum static solution artifact reference for residual statics."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    name: str = 'datum_static_solution.npz'

    @model_validator(mode='after')
    def _check_values(self) -> 'ResidualStaticDatumSolutionRequest':
        if not self.job_id:
            raise ValueError('datum_solution.job_id must be a non-empty string')
        _validate_artifact_basename(self.name, 'datum_solution.name')
        return self


class ResidualStaticPickSourceRequest(BaseModel):
    """First-break pick source reference for residual static estimation."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['batch_job_artifact', 'manual_npz_artifact', 'manual_memmap']
    job_id: str | None = None
    name: str | None = None

    @model_validator(mode='after')
    def _check_ref(self) -> 'ResidualStaticPickSourceRequest':
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


class ResidualStaticGeometryRequest(BaseModel):
    """Source and receiver header configuration for residual statics."""

    model_config = ConfigDict(extra='forbid')

    source_id_byte: int
    receiver_id_byte: int

    @field_validator('source_id_byte', 'receiver_id_byte', mode='before')
    @classmethod
    def _check_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @model_validator(mode='after')
    def _check_distinct_headers(self) -> 'ResidualStaticGeometryRequest':
        if self.source_id_byte == self.receiver_id_byte:
            raise ValueError('source_id_byte and receiver_id_byte must differ')
        return self


class ResidualStaticOffsetRequest(BaseModel):
    """Offset header configuration for residual statics."""

    model_config = ConfigDict(extra='forbid')

    offset_byte: int | None = 37

    @field_validator('offset_byte', mode='before')
    @classmethod
    def _check_offset_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(value, 'offset.offset_byte')


class ResidualStaticMoveoutRequest(BaseModel):
    """Moveout model configuration for residual static estimation."""

    model_config = ConfigDict(extra='forbid')

    model: Literal['linear_abs_offset', 'none'] = 'linear_abs_offset'


class ResidualStaticSolverRequest(BaseModel):
    """Sparse solver stabilization options for residual statics."""

    model_config = ConfigDict(extra='forbid')

    gauge: Literal['zero_mean_source_receiver'] = 'zero_mean_source_receiver'
    damping_lambda: float = 0.0
    min_valid_picks: int = 10
    min_picks_per_source: int = 1
    min_picks_per_receiver: int = 1
    max_abs_estimated_delay_ms: float = 250.0

    @field_validator('damping_lambda', mode='before')
    @classmethod
    def _check_damping_lambda(cls, value: object) -> float:
        return _require_nonnegative_finite_float(value, 'solver.damping_lambda')

    @field_validator(
        'min_valid_picks',
        'min_picks_per_source',
        'min_picks_per_receiver',
        mode='before',
    )
    @classmethod
    def _check_positive_int(cls, value: object, info: Any) -> int:
        return _require_positive_int(value, f'solver.{info.field_name}')

    @field_validator('max_abs_estimated_delay_ms', mode='before')
    @classmethod
    def _check_max_abs_estimated_delay_ms(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'solver.max_abs_estimated_delay_ms',
        )


class ResidualStaticRobustRequest(BaseModel):
    """Robust outlier-rejection options for residual statics."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    method: Literal['mad', 'sigma'] = 'mad'
    max_iterations: int = 3
    threshold: float = 4.0
    min_used_fraction: float = 0.5

    @field_validator('enabled', mode='before')
    @classmethod
    def _check_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'robust.enabled')

    @field_validator('max_iterations', mode='before')
    @classmethod
    def _check_max_iterations(cls, value: object) -> int:
        return _require_positive_int(value, 'robust.max_iterations')

    @field_validator('threshold', mode='before')
    @classmethod
    def _check_threshold(cls, value: object) -> float:
        return _require_positive_finite_float(value, 'robust.threshold')

    @field_validator('min_used_fraction', mode='before')
    @classmethod
    def _check_min_used_fraction(cls, value: object) -> float:
        fraction = _require_positive_finite_float(value, 'robust.min_used_fraction')
        if fraction > 1.0:
            raise ValueError('robust.min_used_fraction must be <= 1')
        return fraction


class ResidualStaticApplyOptions(BaseModel):
    """Options for applying residual statics to a corrected TraceStore."""

    model_config = ConfigDict(extra='forbid')

    interpolation: Literal['linear'] = 'linear'
    fill_value: float = 0.0
    max_abs_shift_ms: float = 250.0
    output_dtype: Literal['float32'] = 'float32'
    register_corrected_file: bool = True

    @field_validator('fill_value', mode='before')
    @classmethod
    def _check_fill_value(cls, value: object) -> float:
        return _require_finite_float(value, 'apply.fill_value')

    @field_validator('max_abs_shift_ms', mode='before')
    @classmethod
    def _check_max_abs_shift_ms(cls, value: object) -> float:
        return _require_positive_finite_float(value, 'apply.max_abs_shift_ms')

    @field_validator('register_corrected_file', mode='before')
    @classmethod
    def _check_register_corrected_file_bool(cls, value: object) -> bool:
        return _require_bool(value, 'apply.register_corrected_file')

    @model_validator(mode='after')
    def _check_values(self) -> 'ResidualStaticApplyOptions':
        if self.register_corrected_file is not True:
            raise ValueError('apply.register_corrected_file must be true')
        return self


class ResidualStaticApplyRequest(BaseModel):
    """Request model for future ``/statics/residual/apply`` jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    datum_solution: ResidualStaticDatumSolutionRequest
    pick_source: ResidualStaticPickSourceRequest
    geometry: ResidualStaticGeometryRequest
    offset: ResidualStaticOffsetRequest = Field(
        default_factory=ResidualStaticOffsetRequest,
    )
    moveout: ResidualStaticMoveoutRequest = Field(
        default_factory=ResidualStaticMoveoutRequest,
    )
    solver: ResidualStaticSolverRequest = Field(
        default_factory=ResidualStaticSolverRequest,
    )
    robust: ResidualStaticRobustRequest = Field(
        default_factory=ResidualStaticRobustRequest,
    )
    apply: ResidualStaticApplyOptions = Field(
        default_factory=ResidualStaticApplyOptions,
    )

    @field_validator('key1_byte', 'key2_byte', mode='before')
    @classmethod
    def _check_key_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @model_validator(mode='after')
    def _check_values(self) -> 'ResidualStaticApplyRequest':
        if not self.file_id:
            raise ValueError('file_id must be a non-empty string')
        if self.moveout.model == 'linear_abs_offset' and self.offset.offset_byte is None:
            raise ValueError(
                'offset.offset_byte is required for linear_abs_offset moveout'
            )
        if self.moveout.model == 'none' and self.offset.offset_byte is not None:
            raise ValueError('offset.offset_byte must be null for none moveout')
        return self

    @property
    def source_id_byte(self) -> int:
        return self.geometry.source_id_byte

    @property
    def receiver_id_byte(self) -> int:
        return self.geometry.receiver_id_byte

    @property
    def offset_byte(self) -> int | None:
        return self.offset.offset_byte


class ResidualStaticApplyResponse(BaseModel):
    """Response model for creating a residual static apply job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str


class TimeTermStaticPickSourceRequest(BaseModel):
    """First-break pick source reference for time-term static inversion."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['batch_predicted_npz', 'manual_npz_artifact', 'manual_memmap']
    job_id: str | None = None
    artifact_name: str | None = None

    @model_validator(mode='after')
    def _check_ref(self) -> 'TimeTermStaticPickSourceRequest':
        if self.kind == 'manual_memmap':
            if self.job_id is not None or self.artifact_name is not None:
                raise ValueError(
                    'pick_source.job_id/artifact_name must be omitted for manual_memmap'
                )
            return self

        if not self.job_id:
            raise ValueError('pick_source.job_id is required for artifact sources')
        if self.kind == 'batch_predicted_npz':
            self.artifact_name = self.artifact_name or 'predicted_picks_time_s.npz'
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


class TimeTermStaticGeometryRequest(BaseModel):
    """Trace header configuration for time-term static inversion."""

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
    def _check_distinct_endpoint_ids(self) -> 'TimeTermStaticGeometryRequest':
        if self.source_id_byte == self.receiver_id_byte:
            raise ValueError('geometry.source_id_byte and receiver_id_byte must differ')
        return self


class TimeTermStaticLinkageRequest(BaseModel):
    """Source/receiver endpoint linkage artifact reference."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['required', 'optional', 'none'] = 'required'
    job_id: str | None = None
    artifact_name: str = 'geometry_linkage.npz'

    @field_validator('artifact_name', mode='before')
    @classmethod
    def _check_artifact_name(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError('linkage.artifact_name must be a plain file name')
        return _validate_artifact_basename(value, 'linkage.artifact_name')

    @model_validator(mode='after')
    def _check_ref(self) -> 'TimeTermStaticLinkageRequest':
        if self.mode == 'required' and not self.job_id:
            raise ValueError('linkage.job_id is required when linkage.mode is required')
        if self.mode == 'none' and self.job_id is not None:
            raise ValueError('linkage.job_id must be omitted when linkage.mode is none')
        if self.job_id is not None and not self.job_id:
            raise ValueError('linkage.job_id must be a non-empty string')
        return self


class TimeTermStaticVelocityRequest(BaseModel):
    """Velocity parameters for time-term static inversion."""

    model_config = ConfigDict(extra='forbid')

    replacement_velocity_m_s: float
    refractor_velocity_m_s: float
    weathering_velocity_m_s: float | None = None

    @field_validator(
        'replacement_velocity_m_s',
        'refractor_velocity_m_s',
        mode='before',
    )
    @classmethod
    def _check_positive_velocity(cls, value: object, info: Any) -> float:
        return _require_positive_finite_float(value, f'velocity.{info.field_name}')

    @field_validator('weathering_velocity_m_s', mode='before')
    @classmethod
    def _check_optional_positive_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'velocity.weathering_velocity_m_s',
        )

    @model_validator(mode='after')
    def _check_head_wave_velocity_order(self) -> 'TimeTermStaticVelocityRequest':
        if self.refractor_velocity_m_s <= self.replacement_velocity_m_s:
            raise ValueError(
                'velocity.refractor_velocity_m_s must be greater than '
                'velocity.replacement_velocity_m_s'
            )
        return self


class TimeTermStaticMoveoutRequest(BaseModel):
    """Moveout model configuration for time-term static inversion."""

    model_config = ConfigDict(extra='forbid')

    model: Literal[
        'head_wave_linear_offset',
        'reciprocal_head_wave',
        'linear_offset',
        'none',
    ] = 'head_wave_linear_offset'
    distance_source: Literal['geometry', 'offset_header', 'auto'] = 'geometry'
    offset_byte: int | None = 37
    allow_missing_offset: bool = False
    max_geometry_offset_mismatch_m: float | None = None

    @field_validator('offset_byte', mode='before')
    @classmethod
    def _check_offset_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(value, 'moveout.offset_byte')

    @field_validator('allow_missing_offset', mode='before')
    @classmethod
    def _check_allow_missing_offset(cls, value: object) -> bool:
        return _require_bool(value, 'moveout.allow_missing_offset')

    @field_validator('max_geometry_offset_mismatch_m', mode='before')
    @classmethod
    def _check_max_geometry_offset_mismatch_m(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            'moveout.max_geometry_offset_mismatch_m',
        )

    @model_validator(mode='after')
    def _check_offset_requirement(self) -> 'TimeTermStaticMoveoutRequest':
        if self.distance_source == 'offset_header' and self.offset_byte is None:
            raise ValueError(
                'moveout.offset_byte is required when '
                'moveout.distance_source is offset_header'
            )
        return self


class TimeTermStaticRobustRequest(BaseModel):
    """Robust outlier-rejection options for time-term inversion."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    method: Literal['mad', 'sigma'] = 'mad'
    threshold: float = 3.5
    max_iterations: int = 5
    min_used_fraction: float = 0.5
    min_used_observations: int = 1

    @field_validator('enabled', mode='before')
    @classmethod
    def _check_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'solver.robust.enabled')

    @field_validator('threshold', mode='before')
    @classmethod
    def _check_threshold(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'solver.robust.threshold',
        )

    @field_validator('max_iterations', mode='before')
    @classmethod
    def _check_max_iterations(cls, value: object) -> int:
        return _require_positive_int(value, 'solver.robust.max_iterations')

    @field_validator('min_used_fraction', mode='before')
    @classmethod
    def _check_min_used_fraction(cls, value: object) -> float:
        fraction = _require_positive_finite_float(
            value,
            'solver.robust.min_used_fraction',
        )
        if fraction > 1.0:
            raise ValueError('solver.robust.min_used_fraction must be <= 1')
        return fraction

    @field_validator('min_used_observations', mode='before')
    @classmethod
    def _check_min_used_observations(cls, value: object) -> int:
        return _require_positive_int(value, 'solver.robust.min_used_observations')


class TimeTermStaticSolverRequest(BaseModel):
    """Solver options for time-term static inversion."""

    model_config = ConfigDict(extra='forbid')

    damping: float = 0.01
    gauge: Literal['mean_zero', 'reference_node'] = 'mean_zero'
    reference_node_id: int | None = None
    robust: TimeTermStaticRobustRequest = Field(
        default_factory=TimeTermStaticRobustRequest,
    )

    @field_validator('damping', mode='before')
    @classmethod
    def _check_damping(cls, value: object) -> float:
        return _require_nonnegative_finite_float(value, 'solver.damping')

    @field_validator('reference_node_id', mode='before')
    @classmethod
    def _check_reference_node_id(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('solver.reference_node_id must be a non-negative integer')
        if value < 0:
            raise ValueError('solver.reference_node_id must be a non-negative integer')
        return value

    @model_validator(mode='after')
    def _check_reference_node_gauge(self) -> 'TimeTermStaticSolverRequest':
        if self.gauge == 'reference_node' and self.reference_node_id is None:
            raise ValueError(
                'solver.reference_node_id is required when '
                "solver.gauge is 'reference_node'"
            )
        if self.gauge != 'reference_node' and self.reference_node_id is not None:
            raise ValueError(
                'solver.reference_node_id is only allowed when '
                "solver.gauge is 'reference_node'"
            )
        return self


class TimeTermStaticApplyOptions(BaseModel):
    """Options for time-term artifact output and corrected TraceStore registration."""

    model_config = ConfigDict(extra='forbid')

    interpolation: Literal['linear'] = 'linear'
    fill_value: float = 0.0
    mode: Literal['weathering_only'] = 'weathering_only'
    register_corrected_file: bool = False
    max_abs_shift_ms: float = 250.0
    output_dtype: Literal['float32'] = 'float32'

    @field_validator('fill_value', mode='before')
    @classmethod
    def _check_fill_value(cls, value: object) -> float:
        return _require_finite_float(value, 'apply.fill_value')

    @field_validator('register_corrected_file', mode='before')
    @classmethod
    def _check_register_corrected_file_bool(cls, value: object) -> bool:
        return _require_bool(value, 'apply.register_corrected_file')

    @field_validator('max_abs_shift_ms', mode='before')
    @classmethod
    def _check_max_abs_shift_ms(cls, value: object) -> float:
        return _require_positive_finite_float(value, 'apply.max_abs_shift_ms')


class TimeTermStaticApplyRequest(BaseModel):
    """Request model for future ``/statics/time-term/apply`` jobs."""

    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1_byte: int = 189
    key2_byte: int = 193
    pick_source: TimeTermStaticPickSourceRequest
    geometry: TimeTermStaticGeometryRequest = Field(
        default_factory=TimeTermStaticGeometryRequest,
    )
    linkage: TimeTermStaticLinkageRequest = Field(
        default_factory=TimeTermStaticLinkageRequest,
    )
    velocity: TimeTermStaticVelocityRequest
    moveout: TimeTermStaticMoveoutRequest = Field(
        default_factory=TimeTermStaticMoveoutRequest,
    )
    solver: TimeTermStaticSolverRequest = Field(
        default_factory=TimeTermStaticSolverRequest,
    )
    apply: TimeTermStaticApplyOptions = Field(
        default_factory=TimeTermStaticApplyOptions,
    )

    @field_validator('key1_byte', 'key2_byte', mode='before')
    @classmethod
    def _check_key_header_byte(cls, value: object, info: Any) -> int:
        return require_trace_header_byte(value, info.field_name)

    @model_validator(mode='after')
    def _check_values(self) -> 'TimeTermStaticApplyRequest':
        if not self.file_id:
            raise ValueError('file_id must be a non-empty string')
        return self


class TimeTermStaticApplyResponse(BaseModel):
    """Response model for creating a time-term static apply job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str


class RefractionStaticPickSourceRequest(BaseModel):
    """First-break pick source reference for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['batch_predicted_npz', 'manual_npz_artifact', 'manual_memmap']
    job_id: str | None = None
    artifact_name: str | None = None

    @model_validator(mode='after')
    def _check_ref(self) -> 'RefractionStaticPickSourceRequest':
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

    mode: Literal['required', 'optional', 'none'] = 'required'
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


class RefractionStaticFirstLayerRequest(BaseModel):
    """First-layer / V1 configuration for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['constant', 'estimate_direct_arrival'] = 'constant'
    weathering_velocity_m_s: float | None = None

    min_weathering_velocity_m_s: float = 250.0
    max_weathering_velocity_m_s: float = 1800.0

    min_direct_offset_m: float | None = None
    max_direct_offset_m: float | None = None

    min_picks_per_fit: int = 5
    min_groups: int = 3

    robust_enabled: bool = True
    robust_threshold: float = 3.5

    @field_validator('weathering_velocity_m_s', mode='before')
    @classmethod
    def _check_optional_weathering_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'model.first_layer.weathering_velocity_m_s',
        )

    @field_validator(
        'min_weathering_velocity_m_s',
        'max_weathering_velocity_m_s',
        mode='before',
    )
    @classmethod
    def _check_weathering_velocity_bound(cls, value: object, info: Any) -> float:
        return _require_positive_finite_float(
            value,
            f'model.first_layer.{info.field_name}',
        )

    @field_validator('min_direct_offset_m', 'max_direct_offset_m', mode='before')
    @classmethod
    def _check_direct_offset(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            f'model.first_layer.{info.field_name}',
        )

    @field_validator('min_picks_per_fit', 'min_groups', mode='before')
    @classmethod
    def _check_positive_count(cls, value: object, info: Any) -> int:
        return _require_positive_int(value, f'model.first_layer.{info.field_name}')

    @field_validator('robust_enabled', mode='before')
    @classmethod
    def _check_robust_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'model.first_layer.robust_enabled')

    @field_validator('robust_threshold', mode='before')
    @classmethod
    def _check_robust_threshold(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'model.first_layer.robust_threshold',
        )

    @model_validator(mode='after')
    def _check_first_layer_values(self) -> 'RefractionStaticFirstLayerRequest':
        if self.min_weathering_velocity_m_s >= self.max_weathering_velocity_m_s:
            raise ValueError(
                'model.first_layer.min_weathering_velocity_m_s must be less than '
                'model.first_layer.max_weathering_velocity_m_s'
            )
        if (
            self.mode == 'estimate_direct_arrival'
            and self.weathering_velocity_m_s is not None
        ):
            raise ValueError(
                'model.first_layer.weathering_velocity_m_s must be omitted when '
                'model.first_layer.mode is estimate_direct_arrival'
            )
        if self.mode == 'estimate_direct_arrival' and (
            self.min_direct_offset_m is None or self.max_direct_offset_m is None
        ):
            raise ValueError(
                'model.first_layer.min_direct_offset_m and '
                'model.first_layer.max_direct_offset_m are required when '
                'model.first_layer.mode is estimate_direct_arrival'
            )
        if (
            self.min_direct_offset_m is not None
            and self.max_direct_offset_m is not None
            and self.min_direct_offset_m >= self.max_direct_offset_m
        ):
            raise ValueError(
                'model.first_layer.min_direct_offset_m must be less than '
                'model.first_layer.max_direct_offset_m'
            )
        return self


class RefractionStaticRefractorCellRequest(BaseModel):
    """Spatial refractor V2 cell configuration for Phase 2 request contracts."""

    model_config = ConfigDict(extra='forbid')

    number_of_cell_x: int
    size_of_cell_x_m: float
    x_coordinate_origin_m: float

    number_of_cell_y: int = 1
    size_of_cell_y_m: float | None = None
    y_coordinate_origin_m: float = 0.0

    assignment_mode: Literal['midpoint'] = 'midpoint'
    outside_grid_policy: Literal['reject'] = 'reject'
    coordinate_mode: Literal['grid_3d', 'line_2d_projected'] = 'grid_3d'
    line_origin_x_m: float | None = None
    line_origin_y_m: float | None = None
    line_azimuth_deg: float | None = None

    min_observations_per_cell: int = 5
    velocity_smoothing_weight: float = 0.0
    smoothing_reference_distance_m: float | None = None

    @field_validator(
        'number_of_cell_x',
        'number_of_cell_y',
        'min_observations_per_cell',
        mode='before',
    )
    @classmethod
    def _check_positive_count(cls, value: object, info: Any) -> int:
        return _require_positive_int(
            value,
            f'model.refractor_cell.{info.field_name}',
        )

    @field_validator('size_of_cell_x_m', mode='before')
    @classmethod
    def _check_size_of_cell_x(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'model.refractor_cell.size_of_cell_x_m',
        )

    @field_validator('size_of_cell_y_m', mode='before')
    @classmethod
    def _check_size_of_cell_y(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'model.refractor_cell.size_of_cell_y_m',
        )

    @field_validator(
        'x_coordinate_origin_m',
        'y_coordinate_origin_m',
        mode='before',
    )
    @classmethod
    def _check_origin(cls, value: object, info: Any) -> float:
        return _require_finite_float(
            value,
            f'model.refractor_cell.{info.field_name}',
        )

    @field_validator('assignment_mode', mode='before')
    @classmethod
    def _check_assignment_mode(cls, value: object) -> Literal['midpoint']:
        if value != 'midpoint':
            raise ValueError('model.refractor_cell.assignment_mode must be midpoint')
        return 'midpoint'

    @field_validator('outside_grid_policy', mode='before')
    @classmethod
    def _check_outside_grid_policy(cls, value: object) -> Literal['reject']:
        if value != 'reject':
            raise ValueError(
                'model.refractor_cell.outside_grid_policy must be reject'
            )
        return 'reject'

    @field_validator(
        'line_origin_x_m',
        'line_origin_y_m',
        'line_azimuth_deg',
        mode='before',
    )
    @classmethod
    def _check_optional_line_coordinate(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_finite_float(
            value,
            f'model.refractor_cell.{info.field_name}',
        )

    @field_validator('velocity_smoothing_weight', mode='before')
    @classmethod
    def _check_velocity_smoothing_weight(cls, value: object) -> float:
        return _require_nonnegative_finite_float(
            value,
            'model.refractor_cell.velocity_smoothing_weight',
        )

    @field_validator('smoothing_reference_distance_m', mode='before')
    @classmethod
    def _check_smoothing_reference_distance(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'model.refractor_cell.smoothing_reference_distance_m',
        )

    @model_validator(mode='after')
    def _check_cell_values(self) -> 'RefractionStaticRefractorCellRequest':
        if self.number_of_cell_y > 1 and self.size_of_cell_y_m is None:
            raise ValueError(
                'model.refractor_cell.size_of_cell_y_m is required when '
                'model.refractor_cell.number_of_cell_y > 1'
            )
        if self.coordinate_mode == 'line_2d_projected':
            if (
                self.line_origin_x_m is None
                or self.line_origin_y_m is None
                or self.line_azimuth_deg is None
            ):
                raise ValueError(
                    'model.refractor_cell.line_origin_x_m, '
                    'model.refractor_cell.line_origin_y_m, and '
                    'model.refractor_cell.line_azimuth_deg are required when '
                    'model.refractor_cell.coordinate_mode is line_2d_projected'
                )
            if self.number_of_cell_y != 1:
                raise ValueError(
                    'model.refractor_cell.number_of_cell_y must be 1 when '
                    'model.refractor_cell.coordinate_mode is line_2d_projected'
                )
        return self


RefractionStaticLayerKind = Literal['v2_t1', 'v3_t2', 'vsub_t3']
RefractionStaticLayerVelocityMode = Literal[
    'fixed_global',
    'solve_global',
    'solve_cell',
]
_REFRACTION_STATIC_LAYER_ORDER: dict[RefractionStaticLayerKind, int] = {
    'v2_t1': 0,
    'v3_t2': 1,
    'vsub_t3': 2,
}


class RefractionStaticLayerRequest(BaseModel):
    """Layer-specific time-term configuration for multi-layer refraction statics."""

    model_config = ConfigDict(extra='forbid')

    kind: RefractionStaticLayerKind
    enabled: bool = True
    min_offset_m: float | None = None
    max_offset_m: float | None = None
    velocity_mode: RefractionStaticLayerVelocityMode = 'solve_global'
    initial_velocity_m_s: float | None = None
    fixed_velocity_m_s: float | None = None
    min_velocity_m_s: float | None = None
    max_velocity_m_s: float | None = None
    min_observations_per_cell: int | None = None
    smoothing_weight: float | None = None

    @field_validator('enabled', mode='before')
    @classmethod
    def _check_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'model.layers.enabled')

    @field_validator('min_offset_m', 'max_offset_m', mode='before')
    @classmethod
    def _check_offset_gate(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            f'model.layers.{info.field_name}',
        )

    @field_validator(
        'initial_velocity_m_s',
        'fixed_velocity_m_s',
        'min_velocity_m_s',
        'max_velocity_m_s',
        mode='before',
    )
    @classmethod
    def _check_optional_velocity(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            f'model.layers.{info.field_name}',
        )

    @field_validator('min_observations_per_cell', mode='before')
    @classmethod
    def _check_min_observations_per_cell(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_positive_int(value, 'model.layers.min_observations_per_cell')

    @field_validator('smoothing_weight', mode='before')
    @classmethod
    def _check_smoothing_weight(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            'model.layers.smoothing_weight',
        )

    @model_validator(mode='after')
    def _check_layer_values(self) -> 'RefractionStaticLayerRequest':
        if (
            self.min_offset_m is not None
            and self.max_offset_m is not None
            and self.min_offset_m >= self.max_offset_m
        ):
            raise ValueError(
                'model.layers.min_offset_m must be less than '
                'model.layers.max_offset_m'
            )
        if (
            self.min_velocity_m_s is not None
            and self.max_velocity_m_s is not None
            and self.min_velocity_m_s >= self.max_velocity_m_s
        ):
            raise ValueError(
                'model.layers.min_velocity_m_s must be less than '
                'model.layers.max_velocity_m_s'
            )
        return self


class RefractionStaticModelRequest(BaseModel):
    """Near-surface model options for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    method: Literal['gli_variable_thickness', 'multilayer_time_term'] = (
        'gli_variable_thickness'
    )
    weathering_velocity_m_s: float | None = None
    first_layer: RefractionStaticFirstLayerRequest | None = None
    bedrock_velocity_mode: Literal[
        'solve_global',
        'fixed_global',
        'solve_cell',
    ] = 'solve_global'
    bedrock_velocity_m_s: float | None = None
    initial_bedrock_velocity_m_s: float | None = None
    min_bedrock_velocity_m_s: float = 1200.0
    max_bedrock_velocity_m_s: float = 6000.0
    max_weathering_thickness_m: float | None = None
    refractor_cell: RefractionStaticRefractorCellRequest | None = None
    layers: list[RefractionStaticLayerRequest] | None = None
    allow_overlapping_layer_gates: bool = False

    @field_validator('weathering_velocity_m_s', mode='before')
    @classmethod
    def _check_optional_weathering_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(value, 'model.weathering_velocity_m_s')

    @field_validator('min_bedrock_velocity_m_s', 'max_bedrock_velocity_m_s', mode='before')
    @classmethod
    def _check_positive_velocity(cls, value: object, info: Any) -> float:
        return _require_positive_finite_float(value, f'model.{info.field_name}')

    @field_validator(
        'bedrock_velocity_m_s',
        'initial_bedrock_velocity_m_s',
        'max_weathering_thickness_m',
        mode='before',
    )
    @classmethod
    def _check_optional_positive_velocity(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(value, f'model.{info.field_name}')

    @field_validator('allow_overlapping_layer_gates', mode='before')
    @classmethod
    def _check_allow_overlapping_layer_gates(cls, value: object) -> bool:
        return _require_bool(value, 'model.allow_overlapping_layer_gates')

    @model_validator(mode='after')
    def _check_velocity_values(self) -> 'RefractionStaticModelRequest':
        resolved_weathering_velocity = self._constant_weathering_velocity_or_none()
        if self.method == 'multilayer_time_term':
            self._check_multilayer_values(resolved_weathering_velocity)
            return self

        if self.layers is not None:
            raise ValueError(
                'model.layers is only allowed when '
                'model.method is multilayer_time_term'
            )
        if self.min_bedrock_velocity_m_s >= self.max_bedrock_velocity_m_s:
            raise ValueError(
                'model.min_bedrock_velocity_m_s must be less than '
                'model.max_bedrock_velocity_m_s'
            )
        if resolved_weathering_velocity is not None:
            if self.min_bedrock_velocity_m_s <= resolved_weathering_velocity:
                raise ValueError(
                    'model.min_bedrock_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
            if self.max_bedrock_velocity_m_s <= resolved_weathering_velocity:
                raise ValueError(
                    'model.max_bedrock_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
        if (
            self.initial_bedrock_velocity_m_s is not None
            and resolved_weathering_velocity is not None
            and self.initial_bedrock_velocity_m_s <= resolved_weathering_velocity
        ):
            raise ValueError(
                'model.initial_bedrock_velocity_m_s must be greater than '
                'model.resolved_weathering_velocity_m_s'
            )
        if self.initial_bedrock_velocity_m_s is not None and not (
            self.min_bedrock_velocity_m_s
            <= self.initial_bedrock_velocity_m_s
            <= self.max_bedrock_velocity_m_s
        ):
            raise ValueError(
                'model.initial_bedrock_velocity_m_s must be within '
                'bedrock velocity bounds'
            )
        if self.bedrock_velocity_mode == 'fixed_global':
            if self.bedrock_velocity_m_s is None:
                raise ValueError(
                    'model.bedrock_velocity_m_s is required when '
                    'model.bedrock_velocity_mode is fixed_global'
                )
            if (
                resolved_weathering_velocity is not None
                and self.bedrock_velocity_m_s <= resolved_weathering_velocity
            ):
                raise ValueError(
                    'model.bedrock_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
            if not (
                self.min_bedrock_velocity_m_s
                <= self.bedrock_velocity_m_s
                <= self.max_bedrock_velocity_m_s
            ):
                raise ValueError(
                    'model.bedrock_velocity_m_s must be within bedrock velocity bounds'
                )
        elif self.bedrock_velocity_m_s is not None:
            raise ValueError(
                'model.bedrock_velocity_m_s is only allowed when '
                'model.bedrock_velocity_mode is fixed_global'
            )
        if self.bedrock_velocity_mode == 'solve_cell':
            if self.refractor_cell is None:
                raise ValueError(
                    'model.refractor_cell is required when '
                    'model.bedrock_velocity_mode is solve_cell'
                )
        elif self.refractor_cell is not None:
            raise ValueError(
                'model.refractor_cell is only allowed when '
                'model.bedrock_velocity_mode is solve_cell'
            )
        return self

    def _check_multilayer_values(
        self,
        resolved_weathering_velocity: float | None,
    ) -> None:
        layers = self.layers
        if not layers:
            raise ValueError(
                'model.layers must include enabled v2_t1 when '
                'model.method is multilayer_time_term'
            )

        seen_kinds: set[RefractionStaticLayerKind] = set()
        previous_order = -1
        for layer in layers:
            if layer.kind in seen_kinds:
                raise ValueError('model.layers must not contain duplicate layer kinds')
            order = _REFRACTION_STATIC_LAYER_ORDER[layer.kind]
            if order < previous_order:
                raise ValueError(
                    'model.layers must be ordered v2_t1, v3_t2, vsub_t3'
                )
            seen_kinds.add(layer.kind)
            previous_order = order

        enabled_layers = [layer for layer in layers if layer.enabled]
        enabled_kinds = {layer.kind for layer in enabled_layers}
        if 'v2_t1' not in enabled_kinds:
            raise ValueError(
                'model.layers must include an enabled v2_t1 layer when '
                'model.method is multilayer_time_term'
            )
        if 'vsub_t3' in enabled_kinds and 'v3_t2' not in enabled_kinds:
            raise ValueError(
                'model.layers cannot enable vsub_t3 unless v3_t2 is enabled'
            )

        deepest_enabled_order = max(
            _REFRACTION_STATIC_LAYER_ORDER[layer.kind] for layer in enabled_layers
        )
        for layer in enabled_layers:
            if layer.min_offset_m is None and layer.max_offset_m is None:
                raise ValueError(
                    'model.layers.min_offset_m or model.layers.max_offset_m is '
                    'required for each enabled layer'
                )
            if (
                layer.max_offset_m is None
                and _REFRACTION_STATIC_LAYER_ORDER[layer.kind] != deepest_enabled_order
            ):
                raise ValueError(
                    'model.layers.max_offset_m may be null only for the deepest '
                    'enabled layer'
                )
            self._check_multilayer_velocity_layer(
                layer,
                resolved_weathering_velocity=resolved_weathering_velocity,
            )

        if not self.allow_overlapping_layer_gates:
            self._check_multilayer_layer_gate_overlap(enabled_layers)
        self._check_multilayer_legacy_aliases()
        self._check_multilayer_velocity_sequence(enabled_layers)

    def _check_multilayer_legacy_aliases(self) -> None:
        v2_layer = self._layer_by_kind('v2_t1')
        if (
            self.bedrock_velocity_m_s is not None
            and v2_layer is not None
            and v2_layer.velocity_mode != 'fixed_global'
        ):
            raise ValueError(
                'model.bedrock_velocity_m_s is only allowed as a v2_t1 fixed '
                'velocity when model.method is multilayer_time_term'
            )
        if (
            self.bedrock_velocity_m_s is not None
            and v2_layer is not None
            and v2_layer.fixed_velocity_m_s is not None
            and not _velocity_values_match(
                self.bedrock_velocity_m_s,
                v2_layer.fixed_velocity_m_s,
            )
        ):
            raise ValueError(
                'model.bedrock_velocity_m_s and '
                'model.layers.fixed_velocity_m_s must match for v2_t1'
            )
        if (
            self.initial_bedrock_velocity_m_s is not None
            and v2_layer is not None
            and v2_layer.initial_velocity_m_s is not None
            and not _velocity_values_match(
                self.initial_bedrock_velocity_m_s,
                v2_layer.initial_velocity_m_s,
            )
        ):
            raise ValueError(
                'model.initial_bedrock_velocity_m_s and '
                'model.layers.initial_velocity_m_s must match for v2_t1'
            )
        has_enabled_solve_cell_layer = any(
            layer.enabled and layer.velocity_mode == 'solve_cell'
            for layer in self.layers or []
        )
        if has_enabled_solve_cell_layer and self.refractor_cell is None:
            raise ValueError(
                'model.refractor_cell is required when an enabled '
                'multi-layer refraction layer uses solve_cell'
            )
        if self.refractor_cell is not None and not has_enabled_solve_cell_layer:
            raise ValueError(
                'model.refractor_cell is only allowed when an enabled '
                'multi-layer refraction layer uses solve_cell'
            )

    def _check_multilayer_velocity_layer(
        self,
        layer: RefractionStaticLayerRequest,
        *,
        resolved_weathering_velocity: float | None,
    ) -> None:
        min_velocity = self._layer_min_velocity_m_s(layer)
        max_velocity = self._layer_max_velocity_m_s(layer)
        if (
            min_velocity is not None
            and max_velocity is not None
            and min_velocity >= max_velocity
        ):
            raise ValueError(
                'model.layers.min_velocity_m_s must be less than '
                'model.layers.max_velocity_m_s'
            )
        if resolved_weathering_velocity is not None:
            if min_velocity is not None and min_velocity <= resolved_weathering_velocity:
                raise ValueError(
                    'model.layers.min_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
            if max_velocity is not None and max_velocity <= resolved_weathering_velocity:
                raise ValueError(
                    'model.layers.max_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )

        if layer.velocity_mode == 'fixed_global':
            fixed_velocity = self._layer_fixed_velocity_m_s(layer)
            if fixed_velocity is None:
                raise ValueError(
                    'model.layers.fixed_velocity_m_s is required when '
                    'model.layers.velocity_mode is fixed_global'
                )
            self._check_layer_velocity_in_bounds(
                fixed_velocity,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                field_name='fixed_velocity_m_s',
            )
            if (
                resolved_weathering_velocity is not None
                and fixed_velocity <= resolved_weathering_velocity
            ):
                raise ValueError(
                    'model.layers.fixed_velocity_m_s must be greater than '
                    'model.resolved_weathering_velocity_m_s'
                )
            return

        initial_velocity = self._layer_initial_velocity_m_s(layer)
        if initial_velocity is None:
            raise ValueError(
                'model.layers.initial_velocity_m_s or '
                'model.initial_bedrock_velocity_m_s is required when '
                'model.layers.velocity_mode is solve_global or solve_cell'
            )
        self._check_layer_velocity_in_bounds(
            initial_velocity,
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            field_name='initial_velocity_m_s',
        )
        if (
            resolved_weathering_velocity is not None
            and initial_velocity <= resolved_weathering_velocity
        ):
            raise ValueError(
                'model.layers.initial_velocity_m_s must be greater than '
                'model.resolved_weathering_velocity_m_s'
            )

    def _check_layer_velocity_in_bounds(
        self,
        velocity: float,
        *,
        min_velocity: float | None,
        max_velocity: float | None,
        field_name: str,
    ) -> None:
        if min_velocity is not None and velocity < min_velocity:
            raise ValueError(f'model.layers.{field_name} must be within velocity bounds')
        if max_velocity is not None and velocity > max_velocity:
            raise ValueError(f'model.layers.{field_name} must be within velocity bounds')

    def _check_multilayer_velocity_sequence(
        self,
        enabled_layers: list[RefractionStaticLayerRequest],
    ) -> None:
        enabled_by_kind = {layer.kind: layer for layer in enabled_layers}
        for shallow_kind, deep_kind in (
            ('v2_t1', 'v3_t2'),
            ('v3_t2', 'vsub_t3'),
        ):
            shallow = enabled_by_kind.get(shallow_kind)
            deep = enabled_by_kind.get(deep_kind)
            if shallow is None or deep is None:
                continue
            shallow_min = self._layer_min_velocity_m_s(shallow)
            if shallow_min is None:
                continue
            deep_min = self._layer_min_velocity_m_s(deep)
            if deep_min is not None and deep_min <= shallow_min:
                raise ValueError(
                    f'model.layers {deep_kind} velocity bounds must be greater '
                    f'than {shallow_kind} minimum velocity'
                )
            deep_max = self._layer_max_velocity_m_s(deep)
            if deep_max is not None and deep_max <= shallow_min:
                raise ValueError(
                    f'model.layers {deep_kind} velocity bounds must allow '
                    f'velocities greater than {shallow_kind} minimum velocity'
                )
            configured_velocity = (
                self._layer_fixed_velocity_m_s(deep)
                if deep.velocity_mode == 'fixed_global'
                else self._layer_initial_velocity_m_s(deep)
            )
            if configured_velocity is not None and configured_velocity <= shallow_min:
                raise ValueError(
                    f'model.layers {deep_kind} configured velocity must be '
                    f'greater than {shallow_kind} minimum velocity'
                )

    def _check_multilayer_layer_gate_overlap(
        self,
        enabled_layers: list[RefractionStaticLayerRequest],
    ) -> None:
        for index, layer in enumerate(enabled_layers):
            layer_min = self._layer_gate_min_offset(layer)
            layer_max = self._layer_gate_max_offset(layer)
            for other in enabled_layers[index + 1 :]:
                other_min = self._layer_gate_min_offset(other)
                other_max = self._layer_gate_max_offset(other)
                if max(layer_min, other_min) < min(layer_max, other_max):
                    raise ValueError(
                        'model.layers offset gates must not overlap unless '
                        'model.allow_overlapping_layer_gates is true'
                    )

    def _layer_gate_min_offset(self, layer: RefractionStaticLayerRequest) -> float:
        if layer.min_offset_m is None:
            return float('-inf')
        return float(layer.min_offset_m)

    def _layer_gate_max_offset(self, layer: RefractionStaticLayerRequest) -> float:
        if layer.max_offset_m is None:
            return float('inf')
        return float(layer.max_offset_m)

    def _layer_by_kind(
        self,
        kind: RefractionStaticLayerKind,
    ) -> RefractionStaticLayerRequest | None:
        for layer in self.layers or []:
            if layer.kind == kind:
                return layer
        return None

    def _layer_initial_velocity_m_s(
        self,
        layer: RefractionStaticLayerRequest,
    ) -> float | None:
        if layer.initial_velocity_m_s is not None:
            return layer.initial_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.initial_bedrock_velocity_m_s
        return None

    def _layer_fixed_velocity_m_s(
        self,
        layer: RefractionStaticLayerRequest,
    ) -> float | None:
        if layer.fixed_velocity_m_s is not None:
            return layer.fixed_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.bedrock_velocity_m_s
        return None

    def _layer_min_velocity_m_s(
        self,
        layer: RefractionStaticLayerRequest,
    ) -> float | None:
        if layer.min_velocity_m_s is not None:
            return layer.min_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.min_bedrock_velocity_m_s
        return None

    def _layer_max_velocity_m_s(
        self,
        layer: RefractionStaticLayerRequest,
    ) -> float | None:
        if layer.max_velocity_m_s is not None:
            return layer.max_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.max_bedrock_velocity_m_s
        return None

    @property
    def enabled_refraction_layer_count(self) -> int:
        if self.layers is None:
            return 1
        return sum(1 for layer in self.layers if layer.enabled)

    @property
    def first_layer_mode(self) -> Literal['constant', 'estimate_direct_arrival']:
        first_layer = self.first_layer
        if first_layer is None:
            return 'constant'
        return first_layer.mode

    @property
    def resolved_weathering_velocity_m_s(self) -> float:
        value = self._constant_weathering_velocity_or_none()
        if value is None:
            raise ValueError(
                'model.first_layer.mode="estimate_direct_arrival" requires a '
                'resolved weathering velocity before downstream processing'
            )
        return value

    def _constant_weathering_velocity_or_none(self) -> float | None:
        legacy_velocity = self.weathering_velocity_m_s
        first_layer = self.first_layer
        if first_layer is None:
            if legacy_velocity is None:
                raise ValueError(
                    'model.weathering_velocity_m_s is required when '
                    'model.first_layer is omitted'
                )
            return legacy_velocity

        first_layer_velocity = first_layer.weathering_velocity_m_s
        if first_layer.mode == 'estimate_direct_arrival':
            if legacy_velocity is not None:
                raise ValueError(
                    'model.weathering_velocity_m_s must be omitted when '
                    'model.first_layer.mode is estimate_direct_arrival'
                )
            if first_layer_velocity is not None:
                raise ValueError(
                    'model.first_layer.weathering_velocity_m_s must be omitted when '
                    'model.first_layer.mode is estimate_direct_arrival'
                )
            return None
        if (
            legacy_velocity is not None
            and first_layer_velocity is not None
            and not _velocity_values_match(legacy_velocity, first_layer_velocity)
        ):
            raise ValueError(
                'model.weathering_velocity_m_s and '
                'model.first_layer.weathering_velocity_m_s must match when both '
                'are specified'
            )
        if first_layer_velocity is None:
            raise ValueError(
                'model.first_layer.weathering_velocity_m_s is required when '
                'model.first_layer.mode is constant'
            )
        return first_layer_velocity


class RefractionStaticMoveoutRequest(BaseModel):
    """Moveout distance source and filtering options for refraction statics."""

    model_config = ConfigDict(extra='forbid')

    model: Literal['head_wave_linear_offset'] = 'head_wave_linear_offset'
    distance_source: Literal['geometry', 'offset_header', 'auto'] = 'geometry'
    offset_byte: int | None = 37
    min_offset_m: float | None = None
    max_offset_m: float | None = None
    allow_missing_offset: bool = False
    max_geometry_offset_mismatch_m: float | None = None

    @field_validator('offset_byte', mode='before')
    @classmethod
    def _check_offset_byte(cls, value: object) -> int | None:
        if value is None:
            return None
        return require_trace_header_byte(value, 'moveout.offset_byte')

    @field_validator('min_offset_m', 'max_offset_m', mode='before')
    @classmethod
    def _check_offset_gate(cls, value: object, info: Any) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(value, f'moveout.{info.field_name}')

    @field_validator('allow_missing_offset', mode='before')
    @classmethod
    def _check_allow_missing_offset(cls, value: object) -> bool:
        return _require_bool(value, 'moveout.allow_missing_offset')

    @field_validator('max_geometry_offset_mismatch_m', mode='before')
    @classmethod
    def _check_max_geometry_offset_mismatch_m(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_nonnegative_finite_float(
            value,
            'moveout.max_geometry_offset_mismatch_m',
        )

    @model_validator(mode='after')
    def _check_offset_values(self) -> 'RefractionStaticMoveoutRequest':
        if self.distance_source == 'offset_header' and self.offset_byte is None:
            raise ValueError(
                'moveout.offset_byte is required when '
                'moveout.distance_source is offset_header'
            )
        if (
            self.min_offset_m is not None
            and self.max_offset_m is not None
            and self.min_offset_m >= self.max_offset_m
        ):
            raise ValueError(
                'moveout.min_offset_m must be less than moveout.max_offset_m'
            )
        return self


class RefractionStaticRobustRequest(BaseModel):
    """Robust outlier-rejection options for refraction inversion."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    method: Literal['mad', 'sigma'] = 'mad'
    threshold: float = 3.5
    max_iterations: int = 5
    min_used_fraction: float = 0.5
    min_used_observations: int = 1

    @field_validator('enabled', mode='before')
    @classmethod
    def _check_enabled(cls, value: object) -> bool:
        return _require_bool(value, 'solver.robust.enabled')

    @field_validator('threshold', mode='before')
    @classmethod
    def _check_threshold(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'solver.robust.threshold',
        )

    @field_validator('max_iterations', mode='before')
    @classmethod
    def _check_max_iterations(cls, value: object) -> int:
        return _require_positive_int(value, 'solver.robust.max_iterations')

    @field_validator('min_used_fraction', mode='before')
    @classmethod
    def _check_min_used_fraction(cls, value: object) -> float:
        fraction = _require_positive_finite_float(
            value,
            'solver.robust.min_used_fraction',
        )
        if fraction > 1.0:
            raise ValueError('solver.robust.min_used_fraction must be <= 1')
        return fraction

    @field_validator('min_used_observations', mode='before')
    @classmethod
    def _check_min_used_observations(cls, value: object) -> int:
        return _require_positive_int(value, 'solver.robust.min_used_observations')


class RefractionStaticSolverRequest(BaseModel):
    """Solver options for refraction static inversion."""

    model_config = ConfigDict(extra='forbid')

    damping: float = 0.01
    min_picks_per_node: int = 1
    max_abs_half_intercept_time_ms: float = 500.0
    robust: RefractionStaticRobustRequest = Field(
        default_factory=RefractionStaticRobustRequest,
    )

    @field_validator('damping', mode='before')
    @classmethod
    def _check_damping(cls, value: object) -> float:
        return _require_nonnegative_finite_float(value, 'solver.damping')

    @field_validator('min_picks_per_node', mode='before')
    @classmethod
    def _check_min_picks_per_node(cls, value: object) -> int:
        return _require_positive_int(value, 'solver.min_picks_per_node')

    @field_validator('max_abs_half_intercept_time_ms', mode='before')
    @classmethod
    def _check_max_abs_half_intercept_time_ms(cls, value: object) -> float:
        return _require_positive_finite_float(
            value,
            'solver.max_abs_half_intercept_time_ms',
        )


class RefractionStaticDatumRequest(BaseModel):
    """Datum options for GLI refraction static composition."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal[
        'floating_and_flat',
        'floating_only',
        'flat_only',
        'none',
    ] = 'none'
    floating_datum_mode: Literal[
        'smoothed_topography',
        'constant',
        'surface',
        'from_artifact',
    ] = 'smoothed_topography'
    flat_datum_elevation_m: float | None = None
    floating_datum_elevation_m: float | None = None
    smoothing_radius_m: float | None = None
    smoothing_window_nodes: int | None = 11
    smoothing_method: Literal['moving_average', 'median'] = 'moving_average'
    floating_datum_job_id: str | None = None
    floating_datum_artifact_name: str | None = None
    allow_flat_datum_above_topography: bool = True
    allow_flat_datum_below_refractor: bool = False

    @field_validator(
        'flat_datum_elevation_m',
        'floating_datum_elevation_m',
        mode='before',
    )
    @classmethod
    def _check_optional_elevation(
        cls,
        value: object,
        info: Any,
    ) -> float | None:
        if value is None:
            return None
        return _require_finite_float(value, f'datum.{info.field_name}')

    @field_validator('smoothing_radius_m', mode='before')
    @classmethod
    def _check_smoothing_radius(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(value, 'datum.smoothing_radius_m')

    @field_validator('smoothing_window_nodes', mode='before')
    @classmethod
    def _check_smoothing_window_nodes(cls, value: object) -> int | None:
        if value is None:
            return None
        window = _require_positive_int(value, 'datum.smoothing_window_nodes')
        if window % 2 == 0:
            raise ValueError('datum.smoothing_window_nodes must be odd')
        return window

    @field_validator('floating_datum_artifact_name', mode='before')
    @classmethod
    def _check_floating_datum_artifact_name(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                'datum.floating_datum_artifact_name must be a plain file name'
            )
        return _validate_artifact_basename(
            value,
            'datum.floating_datum_artifact_name',
        )

    @field_validator(
        'allow_flat_datum_above_topography',
        'allow_flat_datum_below_refractor',
        mode='before',
    )
    @classmethod
    def _check_bool(cls, value: object, info: Any) -> bool:
        return _require_bool(value, f'datum.{info.field_name}')

    @model_validator(mode='after')
    def _check_datum_config(self) -> 'RefractionStaticDatumRequest':
        if self.mode in {'flat_only', 'floating_and_flat'}:
            if self.flat_datum_elevation_m is None:
                raise ValueError(
                    'datum.flat_datum_elevation_m is required for flat datum modes'
                )
        if self.floating_datum_mode == 'constant':
            if self.floating_datum_elevation_m is None:
                raise ValueError(
                    'datum.floating_datum_elevation_m is required when '
                    'floating_datum_mode is constant'
                )
        if self.floating_datum_mode == 'from_artifact':
            if not self.floating_datum_job_id:
                raise ValueError(
                    'datum.floating_datum_job_id is required when '
                    'floating_datum_mode is from_artifact'
                )
            if not self.floating_datum_artifact_name:
                raise ValueError(
                    'datum.floating_datum_artifact_name is required when '
                    'floating_datum_mode is from_artifact'
                )
        elif self.floating_datum_job_id is not None:
            raise ValueError(
                'datum.floating_datum_job_id is only allowed when '
                'floating_datum_mode is from_artifact'
            )
        elif self.floating_datum_artifact_name is not None:
            raise ValueError(
                'datum.floating_datum_artifact_name is only allowed when '
                'floating_datum_mode is from_artifact'
            )
        return self


class RefractionStaticApplyOptions(BaseModel):
    """Options for eventual refraction static TraceStore application."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['refraction_from_raw'] = 'refraction_from_raw'
    interpolation: Literal['linear'] = 'linear'
    fill_value: float = 0.0
    max_abs_shift_ms: float = 250.0
    output_dtype: Literal['float32'] = 'float32'
    register_corrected_file: bool = False

    @field_validator('fill_value', mode='before')
    @classmethod
    def _check_fill_value(cls, value: object) -> float:
        return _require_finite_float(value, 'apply.fill_value')

    @field_validator('max_abs_shift_ms', mode='before')
    @classmethod
    def _check_max_abs_shift_ms(cls, value: object) -> float:
        return _require_positive_finite_float(value, 'apply.max_abs_shift_ms')

    @field_validator('register_corrected_file', mode='before')
    @classmethod
    def _check_register_corrected_file_bool(cls, value: object) -> bool:
        return _require_bool(value, 'apply.register_corrected_file')


class RefractionStaticConversionRequest(BaseModel):
    """Conversion/output mode for refraction static component artifacts."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['existing', 't1lsst_1layer', 't1lsst_multilayer'] = 'existing'
    layer_count: int | None = None

    @field_validator('layer_count', mode='before')
    @classmethod
    def _check_layer_count(cls, value: object) -> int | None:
        if value is None:
            return None
        count = _require_positive_int(value, 'conversion.layer_count')
        if count > 3:
            raise ValueError('conversion.layer_count must be 1, 2, or 3')
        return count

    @model_validator(mode='after')
    def _check_multilayer_count(self) -> 'RefractionStaticConversionRequest':
        if self.mode == 't1lsst_multilayer':
            if self.layer_count is None:
                raise ValueError(
                    'conversion.layer_count is required when '
                    'conversion.mode is t1lsst_multilayer'
                )
            return self
        if self.layer_count is not None:
            raise ValueError(
                'conversion.layer_count is only allowed when '
                'conversion.mode is t1lsst_multilayer'
            )
        return self


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
                raise ValueError(
                    'conversion.layer_count must match enabled refraction layers'
                )
        return self


class RefractionStaticApplyResponse(BaseModel):
    """Response model for creating a refraction static apply job."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    state: str
