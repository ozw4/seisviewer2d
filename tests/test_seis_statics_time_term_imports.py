from __future__ import annotations

import importlib
import sys


def test_seis_statics_time_term_imports_without_app_dependency() -> None:
    before_app_modules = {
        name for name in sys.modules if name == 'app' or name.startswith('app.')
    }

    time_term = importlib.import_module('seis_statics.time_term')

    assert time_term.__name__ == 'seis_statics.time_term'
    assert hasattr(time_term, 'TimeTermInversionInputs')
    assert hasattr(time_term, 'compute_time_term_moveout')
    assert hasattr(time_term, 'TimeTermDesignMatrix')
    assert hasattr(time_term, 'solve_time_term_sparse_least_squares')
    assert hasattr(time_term, 'solve_time_term_robust_least_squares')
    assert hasattr(time_term, 'TimeTermAppliedShiftResult')
    assert hasattr(time_term, 'build_time_term_applied_shift_result')

    after_app_modules = {
        name for name in sys.modules if name == 'app' or name.startswith('app.')
    }
    assert after_app_modules == before_app_modules
