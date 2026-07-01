"""Pure request model for section-window payload semantics."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_FBPICK_MODEL_ID = 'fbpick_edgenext_small.pt'
DEFAULT_FIXED_OFFSET_BYTE = 37
_SCALING_MODES = frozenset({'amax', 'tracewise'})


@dataclass(frozen=True)
class RawBaselineRequest:
    """Raw baseline preflight target for a section-window request."""

    file_id: str
    key1_byte: int
    key2_byte: int


@dataclass(frozen=True)
class SectionWindowRequest:
    """HTTP-independent model for section-window request semantics."""

    file_id: str
    key1: int
    key1_byte: int
    key2_byte: int
    normalization_file_id: str | None
    offset_byte: int | None
    x0: int
    x1: int
    y0: int
    y1: int
    step_x: int
    step_y: int
    transpose: bool
    pipeline_key: str | None
    tap_label: str | None
    reference_pipeline_key: str | None
    reference_tap_label: str | None
    scaling: str | None
    lmo_enabled: bool
    lmo_velocity_mps: float | None
    lmo_offset_byte: int
    lmo_offset_scale: float
    lmo_offset_mode: str
    lmo_ref_mode: str
    lmo_ref_trace: int | None
    lmo_polarity: int
    default_fbpick_model_id: str = DEFAULT_FBPICK_MODEL_ID
    fixed_offset_byte: int = DEFAULT_FIXED_OFFSET_BYTE

    def __post_init__(self) -> None:
        _normalize_scaling_mode(self.scaling)

    @property
    def scaling_mode(self) -> str:
        return _normalize_scaling_mode(self.scaling)

    @property
    def uses_pipeline_source(self) -> bool:
        return bool(self.pipeline_key) or bool(self.tap_label)

    @property
    def uses_reference_source(self) -> bool:
        return bool(self.reference_pipeline_key) or bool(self.reference_tap_label)

    @property
    def normalization_applies_to_raw(self) -> bool:
        return not self.uses_pipeline_source and not self.uses_reference_source

    @property
    def resolved_normalization_file_id(self) -> str:
        return self.normalization_file_id or self.file_id

    @property
    def raw_normalization_file_id(self) -> str:
        if self.normalization_applies_to_raw:
            return self.resolved_normalization_file_id
        return self.file_id

    @property
    def offset_byte_for_payload(self) -> int | None:
        if 'offset' in self.default_fbpick_model_id.lower():
            return self.fixed_offset_byte
        return self.offset_byte

    def raw_baseline_request(self) -> RawBaselineRequest | None:
        """Return raw baseline preflight target, or None for pipeline/reference sources."""
        if not self.normalization_applies_to_raw:
            return None
        return RawBaselineRequest(
            file_id=self.raw_normalization_file_id,
            key1_byte=self.key1_byte,
            key2_byte=self.key2_byte,
        )

    def payload_kwargs(self) -> dict[str, object]:
        """Return section-window payload kwargs excluding runtime dependencies."""
        return {
            'file_id': self.file_id,
            'normalization_file_id': self.raw_normalization_file_id,
            'key1': self.key1,
            'key1_byte': self.key1_byte,
            'key2_byte': self.key2_byte,
            'offset_byte': self.offset_byte_for_payload,
            'x0': self.x0,
            'x1': self.x1,
            'y0': self.y0,
            'y1': self.y1,
            'step_x': self.step_x,
            'step_y': self.step_y,
            'transpose': self.transpose,
            'pipeline_key': self.pipeline_key,
            'tap_label': self.tap_label,
            'reference_pipeline_key': self.reference_pipeline_key,
            'reference_tap_label': self.reference_tap_label,
            'scaling_mode': self.scaling_mode,
            'lmo_enabled': self.lmo_enabled,
            'lmo_velocity_mps': self.lmo_velocity_mps,
            'lmo_offset_byte': self.lmo_offset_byte,
            'lmo_offset_scale': self.lmo_offset_scale,
            'lmo_offset_mode': self.lmo_offset_mode,
            'lmo_ref_mode': self.lmo_ref_mode,
            'lmo_ref_trace': self.lmo_ref_trace,
            'lmo_polarity': self.lmo_polarity,
        }

    def cache_key(self) -> tuple[object, ...]:
        """Return the current canonical section-window cache key."""
        base_key: tuple[object, ...] = (
            self.file_id,
            self.raw_normalization_file_id,
            int(self.key1),
            int(self.key1_byte),
            int(self.key2_byte),
            self.offset_byte_for_payload,
            int(self.x0),
            int(self.x1),
            int(self.y0),
            int(self.y1),
            int(self.step_x),
            int(self.step_y),
            bool(self.transpose),
            self.pipeline_key,
            self.tap_label,
            self.reference_pipeline_key,
            self.reference_tap_label,
            str(self.scaling_mode),
        )
        if not bool(self.lmo_enabled):
            return base_key
        return (
            *base_key,
            'lmo',
            True,
            None if self.lmo_velocity_mps is None else float(self.lmo_velocity_mps),
            int(self.lmo_offset_byte),
            float(self.lmo_offset_scale),
            str(self.lmo_offset_mode),
            str(self.lmo_ref_mode),
            None if self.lmo_ref_trace is None else int(self.lmo_ref_trace),
            int(self.lmo_polarity),
        )


def _normalize_scaling_mode(scaling: str | None) -> str:
    mode = 'amax' if scaling is None else scaling.lower()
    if mode not in _SCALING_MODES:
        raise ValueError('Unsupported scaling mode')
    return mode


__all__ = [
    'DEFAULT_FBPICK_MODEL_ID',
    'DEFAULT_FIXED_OFFSET_BYTE',
    'RawBaselineRequest',
    'SectionWindowRequest',
]
