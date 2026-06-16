"""Public API for time-term static core computations."""

from __future__ import annotations

from seis_statics.time_term.design_matrix import (
    TimeTermDesignMatrix,
    TimeTermDesignMatrixOptions,
    build_time_term_design_matrix,
    summarize_time_term_design_matrix,
)
from seis_statics.time_term.moveout import (
    MoveoutDistanceSource,
    TimeTermMoveoutConfig,
    TimeTermMoveoutModel,
    TimeTermMoveoutResult,
    compute_time_term_moveout,
)
from seis_statics.time_term.robust_solver import (
    ROBUST_SCALE_FLOOR_S,
    TimeTermRobustIteration,
    TimeTermRobustMethod,
    TimeTermRobustSolverOptions,
    TimeTermRobustSolverResult,
    TimeTermRobustStopReason,
    compute_time_term_robust_scores,
    solve_time_term_robust_least_squares,
    subset_time_term_design_matrix_rows,
    summarize_time_term_robust_solver_result,
    validate_time_term_robust_solver_options,
)
from seis_statics.time_term.sparse_solver import (
    TimeTermGaugeMode,
    TimeTermSolverSystem,
    TimeTermSparseSolverName,
    TimeTermSparseSolverOptions,
    TimeTermSparseSolverResult,
    build_gauge_matrix,
    build_node_components,
    build_time_term_solver_system,
    solve_time_term_sparse_least_squares,
    summarize_time_term_sparse_solver_result,
)
from seis_statics.time_term.types import (
    ORDER,
    SIGN_CONVENTION,
    TimeTermInversionInputs,
)

__all__ = [
    'ORDER',
    'SIGN_CONVENTION',
    'MoveoutDistanceSource',
    'ROBUST_SCALE_FLOOR_S',
    'TimeTermDesignMatrix',
    'TimeTermDesignMatrixOptions',
    'TimeTermGaugeMode',
    'TimeTermInversionInputs',
    'TimeTermMoveoutConfig',
    'TimeTermMoveoutModel',
    'TimeTermMoveoutResult',
    'TimeTermRobustIteration',
    'TimeTermRobustMethod',
    'TimeTermRobustSolverOptions',
    'TimeTermRobustSolverResult',
    'TimeTermRobustStopReason',
    'TimeTermSolverSystem',
    'TimeTermSparseSolverName',
    'TimeTermSparseSolverOptions',
    'TimeTermSparseSolverResult',
    'build_gauge_matrix',
    'build_node_components',
    'build_time_term_design_matrix',
    'build_time_term_solver_system',
    'compute_time_term_robust_scores',
    'compute_time_term_moveout',
    'solve_time_term_robust_least_squares',
    'solve_time_term_sparse_least_squares',
    'subset_time_term_design_matrix_rows',
    'summarize_time_term_design_matrix',
    'summarize_time_term_robust_solver_result',
    'summarize_time_term_sparse_solver_result',
    'validate_time_term_robust_solver_options',
]
