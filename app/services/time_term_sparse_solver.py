"""Compatibility exports for time-term sparse solver core computations."""

from __future__ import annotations

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

__all__ = [
    'TimeTermGaugeMode',
    'TimeTermSolverSystem',
    'TimeTermSparseSolverName',
    'TimeTermSparseSolverOptions',
    'TimeTermSparseSolverResult',
    'build_gauge_matrix',
    'build_node_components',
    'build_time_term_solver_system',
    'solve_time_term_sparse_least_squares',
    'summarize_time_term_sparse_solver_result',
]
