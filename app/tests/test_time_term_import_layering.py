from __future__ import annotations

import ast
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

_SHIM_MODULES = [
    'app.services.time_term_types',
    'app.services.time_term_moveout',
    'app.services.time_term_design_matrix',
    'app.services.time_term_sparse_solver',
    'app.services.time_term_robust_solver',
    'app.services.time_term_apply_shift',
]


def _source(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')


def _shim_path(module_name: str) -> Path:
    return Path(module_name.replace('.', '/') + '.py')


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

    for module_name in _SHIM_MODULES:
        importlib.import_module(module_name)

    assert 'app.services.time_term_static_inputs' not in sys.modules


@pytest.mark.parametrize('module_name', _SHIM_MODULES)
def test_time_term_shims_are_only_core_reexports(module_name: str) -> None:
    path = _shim_path(module_name)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    imported_names: set[str] = set()
    has_all = False

    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            assert [alias.name for alias in node.names] == ['annotations']
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module == 'seis_statics.time_term' or node.module.startswith(
                'seis_statics.time_term.'
            )
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
            continue
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1
            target = node.targets[0]
            assert isinstance(target, ast.Name)
            assert target.id == '__all__'
            assert isinstance(node.value, ast.List)
            has_all = True
            continue
        raise AssertionError(f'unexpected shim statement in {path}: {ast.dump(node)}')

    assert has_all

    shim = importlib.import_module(module_name)
    assert isinstance(shim.__all__, list)
    assert set(shim.__all__) == imported_names


@pytest.mark.parametrize('module_name', _SHIM_MODULES)
def test_time_term_shim_public_objects_are_core_objects(module_name: str) -> None:
    path = _shim_path(module_name)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    shim = importlib.import_module(module_name)

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module == '__future__':
            continue
        assert node.module is not None
        core_module = importlib.import_module(node.module)
        for alias in node.names:
            public_name = alias.asname or alias.name
            assert getattr(shim, public_name) is getattr(core_module, alias.name)
