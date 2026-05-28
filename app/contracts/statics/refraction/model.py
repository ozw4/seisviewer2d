"""Geologic and velocity model request contracts for refraction statics."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.contracts._validation import (
    _require_bool,
    _require_finite_float,
    _require_nonnegative_finite_float,
    _require_positive_finite_float,
    _require_positive_int,
    _velocity_values_match,
)
from app.contracts.statics.refraction.common import (
    _REFRACTION_STATIC_LAYER_ORDER,
    RefractionStaticLayerKind,
    RefractionStaticLayerVelocityMode,
)


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
