"""Application-owned result types for refraction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RefractionTraceShiftValidationResult:
    trace_shift_s_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_status_counts: dict[str, int]
    max_abs_shift_ms: float
    max_abs_applied_shift_ms: float
    exceeds_max_abs_shift_count: int
    n_valid_trace_shifts: int
    n_invalid_trace_shifts: int
    n_zero_trace_shifts: int
    n_positive_trace_shifts: int
    n_negative_trace_shifts: int


@dataclass(frozen=True)
class RefractionStaticApplyTraceStoreResult:
    source_file_id: str
    corrected_file_id: str | None

    source_trace_store_path: Path
    corrected_trace_store_path: Path | None

    n_traces: int
    n_samples: int
    sample_interval_s: float

    interpolation: str
    fill_value: float
    output_dtype: str

    applied_shift_s_sorted: np.ndarray
    applied_shift_ms_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray

    max_abs_applied_shift_ms: float
    n_valid_trace_shifts: int
    n_invalid_trace_shifts: int
    n_zero_trace_shifts: int
    n_positive_trace_shifts: int
    n_negative_trace_shifts: int

    corrected_file_json: Path | None
    qc_json: Path | None
    qc: dict[str, Any]


__all__ = [
    'RefractionStaticApplyTraceStoreResult',
    'RefractionTraceShiftValidationResult',
]
