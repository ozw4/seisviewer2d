"""Compatibility imports for common static trace-shift helpers."""

from __future__ import annotations

from seis_statics.trace_shift import (
    StaticTraceShiftValidationResult,
    apply_trace_shifts_to_array,
    require_1d_bool,
    require_1d_float64,
    require_1d_string,
    status_counts_by_value,
    validate_trace_shifts_for_application,
)


__all__ = [
    'StaticTraceShiftValidationResult',
    'apply_trace_shifts_to_array',
    'require_1d_bool',
    'require_1d_float64',
    'require_1d_string',
    'status_counts_by_value',
    'validate_trace_shifts_for_application',
]
