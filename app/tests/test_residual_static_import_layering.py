from __future__ import annotations

from pathlib import Path
from types import ModuleType

import app.services.residual_static_design_matrix as design_matrix
import app.services.residual_static_artifacts as residual_static_artifacts
import app.services.residual_static_robust_solver as robust_solver
import app.services.residual_static_sparse_solver as sparse_solver
import app.services.residual_static_types as residual_static_types


def _module_source(module: ModuleType) -> str:
    path = Path(module.__file__ or '')
    return path.read_text(encoding='utf-8')


def test_residual_static_lightweight_modules_import() -> None:
    assert residual_static_types.MoveoutModel is not None
    assert design_matrix.ResidualStaticColumnLayout is not None
    assert sparse_solver.ResidualStaticLsmrOptions is not None
    assert robust_solver.ResidualStaticRobustOptions is not None


def test_residual_static_design_matrix_has_no_api_reader_or_input_dependency() -> None:
    source = _module_source(design_matrix)

    assert 'app.api.schemas' not in source
    assert 'app.trace_store.reader' not in source
    assert 'segyio' not in source
    assert 'residual_static_inputs' not in source


def test_residual_static_sparse_solver_has_no_api_reader_or_input_dependency() -> None:
    source = _module_source(sparse_solver)

    assert 'app.api.schemas' not in source
    assert 'app.trace_store.reader' not in source
    assert 'segyio' not in source
    assert 'residual_static_inputs' not in source


def test_residual_static_robust_solver_does_not_import_api_schema_reader_or_segyio() -> None:
    source = _module_source(robust_solver)

    assert 'app.api.schemas' not in source
    assert 'app.trace_store.reader' not in source
    assert 'segyio' not in source


def test_residual_static_artifacts_does_not_import_api_schema_reader_or_segyio() -> None:
    source = _module_source(residual_static_artifacts)

    assert 'app.api.schemas' not in source
    assert 'app.trace_store.reader' not in source
    assert 'TraceStoreSectionReader' not in source
    assert 'segyio' not in source
