from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


TIME_TERM_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'seis_statics' / 'time_term'
FORBIDDEN_RUNTIME_MODULES = {
    'app.core.state',
    'app.services.reader',
    'app.trace_store.reader',
    'fastapi',
    'pydantic',
    'scipy',
    'segyio',
}


def test_seis_statics_time_term_imports_without_app_dependency() -> None:
    for module_name in list(sys.modules):
        if module_name == 'seis_statics.time_term' or module_name.startswith(
            'seis_statics.time_term.'
        ):
            sys.modules.pop(module_name, None)
    for module_name in FORBIDDEN_RUNTIME_MODULES:
        sys.modules.pop(module_name, None)

    time_term = importlib.import_module('seis_statics.time_term')

    assert time_term.__name__ == 'seis_statics.time_term'
    assert hasattr(time_term, 'TimeTermInversionInputs')
    assert hasattr(time_term, 'TimeTermMoveoutConfig')
    assert hasattr(time_term, 'TimeTermMoveoutResult')
    assert hasattr(time_term, 'compute_time_term_moveout')
    assert hasattr(time_term, 'compute_geometry_distance_m')
    assert hasattr(time_term, 'build_reciprocal_pair_index')
    assert hasattr(time_term, 'summarize_time_term_moveout')

    for module_name in FORBIDDEN_RUNTIME_MODULES:
        assert module_name not in sys.modules


def test_seis_statics_time_term_all_public_api_importable() -> None:
    time_term = importlib.import_module('seis_statics.time_term')

    expected_symbols = {
        'APPLY_SHIFT_SIGN_CONVENTION',
        'DELAY_TO_SHIFT_CONVENTION',
        'FINAL_SHIFT_CONVENTION',
        'ROBUST_SCALE_FLOOR_S',
        'TimeTermAppliedShiftOptions',
        'TimeTermAppliedShiftResult',
        'TimeTermRobustIteration',
        'TimeTermRobustMethod',
        'TimeTermRobustSolverOptions',
        'TimeTermRobustSolverResult',
        'TimeTermRobustStopReason',
        'TimeTermRejectedTracePolicy',
        'build_time_term_applied_shift_result',
        'compute_estimated_trace_time_term_delay_s',
        'compute_time_term_robust_scores',
        'delay_to_applied_shift_s',
        'solve_time_term_robust_least_squares',
        'subset_time_term_design_matrix_rows',
        'summarize_time_term_applied_shift_result',
        'summarize_time_term_robust_solver_result',
        'validate_time_term_robust_solver_options',
    }

    assert expected_symbols <= set(time_term.__all__)
    for symbol in expected_symbols:
        assert getattr(time_term, symbol) is not None


def test_seis_statics_time_term_source_does_not_import_app() -> None:
    offenders: list[str] = []
    for path in sorted(TIME_TERM_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'app' or alias.name.startswith('app.'):
                        offenders.append(f'{path.relative_to(TIME_TERM_ROOT)}:{node.lineno}')
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if module == 'app' or module.startswith('app.'):
                    offenders.append(f'{path.relative_to(TIME_TERM_ROOT)}:{node.lineno}')

    assert offenders == []
