from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import app.services.refraction_static_bedrock as bedrock
import app.services.refraction_static_design_matrix as design_matrix
import app.services.refraction_static_half_intercept as half_intercept
import app.services.refraction_static_solver as solver
import app.services.refraction_static_types as refraction_types
import app.services.refraction_static_weathering as weathering


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORTS = {
    'app.services.reader',
    'app.services.refraction_static_inputs',
    'app.trace_store.reader',
    'segyio',
}


def _module_source(module: ModuleType) -> str:
    path = Path(module.__file__ or '')
    return path.read_text(encoding='utf-8')


def _forbidden_modules_imported_by(module_name: str) -> set[str]:
    code = f"""
from __future__ import annotations

import importlib
import json
import sys

for name in {sorted(_FORBIDDEN_IMPORTS)!r}:
    sys.modules.pop(name, None)

importlib.import_module({module_name!r})

print(json.dumps(sorted(name for name in {sorted(_FORBIDDEN_IMPORTS)!r} if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(result.stdout))


def test_refraction_static_types_is_dependency_light() -> None:
    assert refraction_types.RefractionStaticInputModel is not None
    assert refraction_types.RefractionStaticDesignMatrix is not None
    assert refraction_types.RefractionDatumStaticsResult is not None

    source = _module_source(refraction_types)

    assert 'app.api.schemas' not in source
    assert 'app.services.reader' not in source
    assert 'app.trace_store.reader' not in source
    assert 'refraction_static_inputs' not in source
    assert 'segyio' not in source
    assert 'scipy' not in source


def test_numeric_refraction_modules_import_without_tracestore_readers() -> None:
    assert design_matrix.RefractionStaticDesignMatrix is not None
    assert solver.RefractionStaticSolverResult is not None
    assert bedrock.RefractionBedrockSlownessResult is not None
    assert half_intercept.RefractionHalfInterceptTimeResult is not None
    assert weathering.RefractionWeatheringThicknessResult is not None

    for module_name in (
        'app.services.refraction_static_design_matrix',
        'app.services.refraction_static_solver',
        'app.services.refraction_static_bedrock',
        'app.services.refraction_static_half_intercept',
        'app.services.refraction_static_weathering',
    ):
        assert _forbidden_modules_imported_by(module_name) == set()
