"""Dependency-light normalized layer config for refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.api.schemas import (
    RefractionStaticLayerKind,
    RefractionStaticLayerRequest,
    RefractionStaticLayerVelocityMode,
)


@dataclass(frozen=True)
class RefractionStaticLayerConfig:
    """Resolved layer settings for downstream multi-layer time-term services."""

    kind: RefractionStaticLayerKind
    enabled: bool
    min_offset_m: float | None
    max_offset_m: float | None
    velocity_mode: RefractionStaticLayerVelocityMode
    initial_velocity_m_s: float | None
    fixed_velocity_m_s: float | None
    min_velocity_m_s: float | None
    max_velocity_m_s: float | None
    min_observations_per_cell: int | None
    smoothing_weight: float | None


def normalize_refraction_static_layers(
    model: Any,
    *,
    enabled_only: bool = True,
) -> tuple[RefractionStaticLayerConfig, ...]:
    """Return normalized layer configs, mapping legacy one-layer fields to v2_t1."""
    layers = getattr(model, 'layers', None)
    if layers is None:
        configs = (_legacy_v2_layer_config(model),)
    else:
        configs = tuple(_layer_config(model, layer) for layer in layers)
    if enabled_only:
        return tuple(config for config in configs if config.enabled)
    return configs


def _legacy_v2_layer_config(model: Any) -> RefractionStaticLayerConfig:
    mode = getattr(model, 'bedrock_velocity_mode', 'solve_global')
    refractor_cell = getattr(model, 'refractor_cell', None)
    return RefractionStaticLayerConfig(
        kind='v2_t1',
        enabled=True,
        min_offset_m=None,
        max_offset_m=None,
        velocity_mode=mode,
        initial_velocity_m_s=getattr(model, 'initial_bedrock_velocity_m_s', None),
        fixed_velocity_m_s=(
            getattr(model, 'bedrock_velocity_m_s', None)
            if mode == 'fixed_global'
            else None
        ),
        min_velocity_m_s=getattr(model, 'min_bedrock_velocity_m_s', None),
        max_velocity_m_s=getattr(model, 'max_bedrock_velocity_m_s', None),
        min_observations_per_cell=(
            getattr(refractor_cell, 'min_observations_per_cell', None)
            if mode == 'solve_cell'
            else None
        ),
        smoothing_weight=(
            getattr(refractor_cell, 'velocity_smoothing_weight', None)
            if mode == 'solve_cell'
            else None
        ),
    )


def _layer_config(
    model: Any,
    layer: RefractionStaticLayerRequest,
) -> RefractionStaticLayerConfig:
    refractor_cell = getattr(model, 'refractor_cell', None)
    is_legacy_v2 = layer.kind == 'v2_t1'
    return RefractionStaticLayerConfig(
        kind=layer.kind,
        enabled=layer.enabled,
        min_offset_m=layer.min_offset_m,
        max_offset_m=layer.max_offset_m,
        velocity_mode=layer.velocity_mode,
        initial_velocity_m_s=(
            layer.initial_velocity_m_s
            if layer.initial_velocity_m_s is not None or not is_legacy_v2
            else getattr(model, 'initial_bedrock_velocity_m_s', None)
        ),
        fixed_velocity_m_s=(
            layer.fixed_velocity_m_s
            if layer.fixed_velocity_m_s is not None or not is_legacy_v2
            else getattr(model, 'bedrock_velocity_m_s', None)
        ),
        min_velocity_m_s=(
            layer.min_velocity_m_s
            if layer.min_velocity_m_s is not None or not is_legacy_v2
            else getattr(model, 'min_bedrock_velocity_m_s', None)
        ),
        max_velocity_m_s=(
            layer.max_velocity_m_s
            if layer.max_velocity_m_s is not None or not is_legacy_v2
            else getattr(model, 'max_bedrock_velocity_m_s', None)
        ),
        min_observations_per_cell=(
            layer.min_observations_per_cell
            if layer.min_observations_per_cell is not None
            else _legacy_min_observations_per_cell(layer, refractor_cell)
        ),
        smoothing_weight=(
            layer.smoothing_weight
            if layer.smoothing_weight is not None
            else _legacy_smoothing_weight(layer, refractor_cell)
        ),
    )


def _legacy_min_observations_per_cell(
    layer: RefractionStaticLayerRequest,
    refractor_cell: Any,
) -> int | None:
    if layer.kind != 'v2_t1' or layer.velocity_mode != 'solve_cell':
        return None
    return getattr(refractor_cell, 'min_observations_per_cell', None)


def _legacy_smoothing_weight(
    layer: RefractionStaticLayerRequest,
    refractor_cell: Any,
) -> float | None:
    if layer.kind != 'v2_t1' or layer.velocity_mode != 'solve_cell':
        return None
    return getattr(refractor_cell, 'velocity_smoothing_weight', None)


__all__ = [
    'RefractionStaticLayerConfig',
    'normalize_refraction_static_layers',
]
