"""Compatibility exports for time-term design matrix core computations."""

from __future__ import annotations

from seis_statics.time_term.design_matrix import (
    TimeTermDesignMatrix,
    TimeTermDesignMatrixOptions,
    build_time_term_design_matrix,
    summarize_time_term_design_matrix,
)

__all__ = [
    'TimeTermDesignMatrix',
    'TimeTermDesignMatrixOptions',
    'build_time_term_design_matrix',
    'summarize_time_term_design_matrix',
]
