"""Option request contracts for refraction static workflows."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts._validation import (
    _require_bool,
    _require_finite_float,
    _require_nonnegative_finite_float,
    _require_positive_finite_float,
    _require_positive_int,
    _validate_artifact_basename,
    require_trace_header_byte,
)


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
    scale_floor_ms: float = 0.05
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

    @field_validator('scale_floor_ms', mode='before')
    @classmethod
    def _check_scale_floor_ms(cls, value: object) -> float:
        return _require_nonnegative_finite_float(
            value,
            'solver.robust.scale_floor_ms',
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


class RefractionStaticReducedTimeQcRequest(BaseModel):
    """Reduced-time QC velocity selection for refraction first-break artifacts."""

    model_config = ConfigDict(extra='forbid')

    reduction_velocity_mode: Literal[
        'layer_velocity',
        'fixed',
        'initial_velocity',
    ] = 'layer_velocity'
    fixed_velocity_m_s: float | None = None

    @field_validator('fixed_velocity_m_s', mode='before')
    @classmethod
    def _check_fixed_velocity(cls, value: object) -> float | None:
        if value is None:
            return None
        return _require_positive_finite_float(
            value,
            'reduced_time_qc.fixed_velocity_m_s',
        )

    @model_validator(mode='after')
    def _check_reduction_velocity_values(
        self,
    ) -> 'RefractionStaticReducedTimeQcRequest':
        if self.reduction_velocity_mode == 'fixed':
            if self.fixed_velocity_m_s is None:
                raise ValueError(
                    'reduced_time_qc.fixed_velocity_m_s is required when '
                    'reduced_time_qc.reduction_velocity_mode is fixed'
                )
            return self
        if self.fixed_velocity_m_s is not None:
            raise ValueError(
                'reduced_time_qc.fixed_velocity_m_s is only allowed when '
                'reduced_time_qc.reduction_velocity_mode is fixed'
            )
        return self
