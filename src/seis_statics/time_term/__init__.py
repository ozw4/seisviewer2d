"""Public API for time-term input and moveout core computations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

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
from seis_statics.time_term.types import (
    ORDER,
    SIGN_CONVENTION,
    TimeTermInversionInputs,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    'APPLY_SHIFT_SIGN_CONVENTION': ('seis_statics.time_term.apply_shift', 'SIGN_CONVENTION'),
    'DELAY_TO_SHIFT_CONVENTION': (
        'seis_statics.time_term.apply_shift',
        'DELAY_TO_SHIFT_CONVENTION',
    ),
    'FINAL_SHIFT_CONVENTION': (
        'seis_statics.time_term.apply_shift',
        'FINAL_SHIFT_CONVENTION',
    ),
    'INPUT_SIGN_CONVENTION': ('seis_statics.time_term.types', 'SIGN_CONVENTION'),
    'ROBUST_SCALE_FLOOR_S': ('seis_statics.time_term.robust_solver', 'ROBUST_SCALE_FLOOR_S'),
    'TimeTermAppliedShiftOptions': (
        'seis_statics.time_term.apply_shift',
        'TimeTermAppliedShiftOptions',
    ),
    'TimeTermAppliedShiftResult': (
        'seis_statics.time_term.apply_shift',
        'TimeTermAppliedShiftResult',
    ),
    'TimeTermDesignMatrix': ('seis_statics.time_term.design_matrix', 'TimeTermDesignMatrix'),
    'TimeTermDesignMatrixOptions': (
        'seis_statics.time_term.design_matrix',
        'TimeTermDesignMatrixOptions',
    ),
    'TimeTermGaugeMode': ('seis_statics.time_term.sparse_solver', 'TimeTermGaugeMode'),
    'TimeTermRobustIteration': (
        'seis_statics.time_term.robust_solver',
        'TimeTermRobustIteration',
    ),
    'TimeTermRobustMethod': ('seis_statics.time_term.robust_solver', 'TimeTermRobustMethod'),
    'TimeTermRobustSolverOptions': (
        'seis_statics.time_term.robust_solver',
        'TimeTermRobustSolverOptions',
    ),
    'TimeTermRobustSolverResult': (
        'seis_statics.time_term.robust_solver',
        'TimeTermRobustSolverResult',
    ),
    'TimeTermRobustStopReason': (
        'seis_statics.time_term.robust_solver',
        'TimeTermRobustStopReason',
    ),
    'TimeTermRejectedTracePolicy': (
        'seis_statics.time_term.apply_shift',
        'TimeTermRejectedTracePolicy',
    ),
    'TimeTermSolverSystem': ('seis_statics.time_term.sparse_solver', 'TimeTermSolverSystem'),
    'TimeTermSparseSolverName': (
        'seis_statics.time_term.sparse_solver',
        'TimeTermSparseSolverName',
    ),
    'TimeTermSparseSolverOptions': (
        'seis_statics.time_term.sparse_solver',
        'TimeTermSparseSolverOptions',
    ),
    'TimeTermSparseSolverResult': (
        'seis_statics.time_term.sparse_solver',
        'TimeTermSparseSolverResult',
    ),
    'build_gauge_matrix': ('seis_statics.time_term.sparse_solver', 'build_gauge_matrix'),
    'build_node_components': (
        'seis_statics.time_term.sparse_solver',
        'build_node_components',
    ),
    'build_time_term_applied_shift_result': (
        'seis_statics.time_term.apply_shift',
        'build_time_term_applied_shift_result',
    ),
    'build_time_term_design_matrix': (
        'seis_statics.time_term.design_matrix',
        'build_time_term_design_matrix',
    ),
    'build_time_term_solver_system': (
        'seis_statics.time_term.sparse_solver',
        'build_time_term_solver_system',
    ),
    'compute_estimated_trace_time_term_delay_s': (
        'seis_statics.time_term.apply_shift',
        'compute_estimated_trace_time_term_delay_s',
    ),
    'compute_time_term_robust_scores': (
        'seis_statics.time_term.robust_solver',
        'compute_time_term_robust_scores',
    ),
    'delay_to_applied_shift_s': (
        'seis_statics.time_term.apply_shift',
        'delay_to_applied_shift_s',
    ),
    'solve_time_term_robust_least_squares': (
        'seis_statics.time_term.robust_solver',
        'solve_time_term_robust_least_squares',
    ),
    'solve_time_term_sparse_least_squares': (
        'seis_statics.time_term.sparse_solver',
        'solve_time_term_sparse_least_squares',
    ),
    'subset_time_term_design_matrix_rows': (
        'seis_statics.time_term.robust_solver',
        'subset_time_term_design_matrix_rows',
    ),
    'summarize_time_term_applied_shift_result': (
        'seis_statics.time_term.apply_shift',
        'summarize_time_term_applied_shift_result',
    ),
    'summarize_time_term_design_matrix': (
        'seis_statics.time_term.design_matrix',
        'summarize_time_term_design_matrix',
    ),
    'summarize_time_term_robust_solver_result': (
        'seis_statics.time_term.robust_solver',
        'summarize_time_term_robust_solver_result',
    ),
    'summarize_time_term_sparse_solver_result': (
        'seis_statics.time_term.sparse_solver',
        'summarize_time_term_sparse_solver_result',
    ),
    'validate_time_term_robust_solver_options': (
        'seis_statics.time_term.robust_solver',
        'validate_time_term_robust_solver_options',
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve later time-term submodule exports without loading them on import."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    'ORDER',
    'SIGN_CONVENTION',
    'MoveoutDistanceSource',
    'TimeTermInversionInputs',
    'TimeTermMoveoutConfig',
    'TimeTermMoveoutModel',
    'TimeTermMoveoutResult',
    'build_reciprocal_pair_index',
    'compute_geometry_distance_m',
    'compute_time_term_moveout',
    'summarize_time_term_moveout',
]
