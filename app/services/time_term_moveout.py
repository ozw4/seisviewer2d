"""Compatibility exports for time-term moveout core computations."""

from __future__ import annotations

from seis_statics.time_term.moveout import (
    MoveoutDistanceSource,
    TimeTermMoveoutConfig,
    TimeTermMoveoutModel,
    TimeTermMoveoutResult,
    build_reciprocal_pair_index,
    compute_geometry_distance_m,
    compute_time_term_moveout,
    summarize_time_term_moveout,
)

__all__ = [
    'MoveoutDistanceSource',
    'TimeTermMoveoutConfig',
    'TimeTermMoveoutModel',
    'TimeTermMoveoutResult',
    'build_reciprocal_pair_index',
    'compute_geometry_distance_m',
    'compute_time_term_moveout',
    'summarize_time_term_moveout',
]
