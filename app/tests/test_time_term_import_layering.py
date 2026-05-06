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
