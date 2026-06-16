from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


_NUMERIC_MODULE_PATHS = [
    'app/services/time_term_types.py',
    'app/services/time_term_moveout.py',
    'app/services/time_term_design_matrix.py',
    'app/services/time_term_sparse_solver.py',
    'app/services/time_term_robust_solver.py',
    'app/services/time_term_apply_shift.py',
]


def _source(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')


@pytest.mark.parametrize('path', _NUMERIC_MODULE_PATHS)
def test_time_term_numeric_services_do_not_import_api_reader_or_segyio(
    path: str,
) -> None:
    source = _source(path)

    assert 'app.api.schemas' not in source
    assert 'app.services.reader' not in source
    assert 'TraceStoreSectionReader' not in source
    assert 'app.trace_store.reader' not in source
    assert 'fastapi' not in source
    assert 'segyio' not in source


@pytest.mark.parametrize(
    'path',
    [
        'app/services/time_term_moveout.py',
        'app/services/time_term_design_matrix.py',
        'app/services/time_term_sparse_solver.py',
        'app/services/time_term_robust_solver.py',
        'app/services/time_term_apply_shift.py',
    ],
)
def test_time_term_numeric_services_do_not_import_static_inputs(path: str) -> None:
    assert 'time_term_static_inputs' not in _source(path)


def test_time_term_numeric_services_import_without_segyio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in (
        'app.services.time_term_types',
        'app.services.time_term_moveout',
        'app.services.time_term_design_matrix',
        'app.services.time_term_sparse_solver',
        'app.services.time_term_robust_solver',
        'app.services.time_term_apply_shift',
        'app.services.time_term_static_inputs',
        'app.trace_store.reader',
    ):
        sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, 'segyio', None)

    for module_name in (
        'app.services.time_term_types',
        'app.services.time_term_moveout',
        'app.services.time_term_design_matrix',
        'app.services.time_term_sparse_solver',
        'app.services.time_term_robust_solver',
        'app.services.time_term_apply_shift',
    ):
        importlib.import_module(module_name)

    assert 'app.services.time_term_static_inputs' not in sys.modules


def test_time_term_solver_shims_reexport_core_objects() -> None:
    apply_shim = importlib.import_module('app.services.time_term_apply_shift')
    design_shim = importlib.import_module('app.services.time_term_design_matrix')
    sparse_shim = importlib.import_module('app.services.time_term_sparse_solver')
    robust_shim = importlib.import_module('app.services.time_term_robust_solver')
    core = importlib.import_module('seis_statics.time_term')

    assert apply_shim.TimeTermAppliedShiftOptions is core.TimeTermAppliedShiftOptions
    assert (
        apply_shim.build_time_term_applied_shift_result
        is core.build_time_term_applied_shift_result
    )
    assert design_shim.TimeTermDesignMatrix is core.TimeTermDesignMatrix
    assert design_shim.build_time_term_design_matrix is core.build_time_term_design_matrix
    assert sparse_shim.TimeTermSparseSolverOptions is core.TimeTermSparseSolverOptions
    assert (
        sparse_shim.solve_time_term_sparse_least_squares
        is core.solve_time_term_sparse_least_squares
    )
    assert robust_shim.TimeTermRobustSolverOptions is core.TimeTermRobustSolverOptions
    assert (
        robust_shim.solve_time_term_robust_least_squares
        is core.solve_time_term_robust_least_squares
    )


@pytest.mark.parametrize(
    'path',
    [
        'app/services/time_term_design_matrix.py',
        'app/services/time_term_sparse_solver.py',
        'app/services/time_term_robust_solver.py',
        'app/services/time_term_apply_shift.py',
    ],
)
def test_time_term_solver_shims_only_import_core_package(path: str) -> None:
    source = _source(path)

    assert 'from seis_statics.time_term' in source
    assert 'from app.' not in source
    assert 'import app.' not in source
