"""Shared status vocabulary for GLI refraction statics outputs."""

from __future__ import annotations

REFRACTION_STATIC_STATUSES: frozenset[str] = frozenset(
    {
        'ok',
        'solved',
        'not_observed',
        'inactive',
        'low_fold',
        'clipped_lower',
        'clipped_upper',
        'clipped_half_intercept_lower',
        'clipped_half_intercept_upper',
        'zero_thickness',
        'exceeds_max_abs_shift',
        'exceeds_max_thickness',
        'invalid_shift',
        'invalid_solution',
        'missing_solution',
        'invalid_velocity',
        'invalid_bedrock_velocity',
        'invalid_surface_elevation',
        'invalid_floating_datum_elevation',
        'invalid_flat_datum_elevation',
        'invalid_weathering_replacement',
        'invalid_weathering_thickness',
        'negative_weathering_thickness',
        'negative_thickness',
        'invalid_half_intercept',
        'invalid_refractor_elevation',
        'invalid_datum_shift',
        'floating_datum_below_refractor',
        'flat_datum_below_refractor',
        'missing_endpoint',
        'missing_node',
    }
)

__all__ = ['REFRACTION_STATIC_STATUSES']
