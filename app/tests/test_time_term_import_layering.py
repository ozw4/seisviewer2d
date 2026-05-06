from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from types import ModuleType

NUMERIC_MODULES = (
    'app.services.time_term_types',
    'app.services.time_term_moveout',
    'app.services.time_term_design_matrix',
    'app.services.time_term_sparse_solver',
    'app.services.time_term_robust_solver',
)

FORBIDDEN_IMPORT_MARKERS = (
    'app.api.schemas',
    'TraceStoreSectionReader',
    'app.trace_store.reader',
    'segyio',
)


def _source(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')


def _assert_no_forbidden_imports(source: str) -> None:
    for marker in FORBIDDEN_IMPORT_MARKERS:
        assert marker not in source


def test_time_term_types_does_not_import_api_reader_or_segyio() -> None:
    _assert_no_forbidden_imports(_source('app/services/time_term_types.py'))


def test_time_term_moveout_does_not_import_static_inputs_or_reader() -> None:
    source = _source('app/services/time_term_moveout.py')

    _assert_no_forbidden_imports(source)
    assert 'time_term_static_inputs' not in source


def test_time_term_design_matrix_does_not_import_static_inputs_or_reader() -> None:
    source = _source('app/services/time_term_design_matrix.py')

    _assert_no_forbidden_imports(source)
    assert 'time_term_static_inputs' not in source


def test_time_term_sparse_solver_does_not_import_api_reader_or_segyio() -> None:
    _assert_no_forbidden_imports(_source('app/services/time_term_sparse_solver.py'))


def test_time_term_robust_solver_does_not_import_api_reader_or_segyio() -> None:
    _assert_no_forbidden_imports(_source('app/services/time_term_robust_solver.py'))


def test_time_term_numeric_services_import_without_segyio(monkeypatch) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == 'segyio' or name.startswith('segyio.'):
            raise ModuleNotFoundError("No module named 'segyio'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    for module_name in reversed(NUMERIC_MODULES):
        sys.modules.pop(module_name, None)

    for module_name in NUMERIC_MODULES:
        importlib.import_module(module_name)
