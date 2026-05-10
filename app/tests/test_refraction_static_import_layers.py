from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

import app.services.refraction_static_bedrock as bedrock
import app.services.refraction_static_design_matrix as design_matrix
import app.services.refraction_static_half_intercept as half_intercept
import app.services.refraction_static_solver as solver
import app.services.refraction_static_t1lsst as t1lsst
import app.services.refraction_static_types as refraction_types
import app.services.refraction_static_v1 as v1
import app.services.refraction_static_weathering as weathering


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORTS = {
    'app.api.routers',
    'app.main',
    'app.services.reader',
    'app.services.refraction_static_service',
    'app.services.refraction_static_inputs',
    'app.trace_store.reader',
    'segyio',
}
_FORBIDDEN_IMPORT_PREFIXES = ('app.api.routers.',)
_IMPORT_LAYER_CHECK_TIMEOUT_S = 10.0


def _module_source(module: ModuleType) -> str:
    path = Path(module.__file__ or '')
    return path.read_text(encoding='utf-8')


def _subprocess_output_text(value: bytes | str | None) -> str:
    if value is None:
        return '<none>'
    if isinstance(value, bytes):
        value = value.decode(errors='replace')
    return value if value else '<empty>'


def _forbidden_modules_imported_by(module_name: str) -> set[str]:
    code = f"""
from __future__ import annotations

import importlib
import json
import sys

for name in {sorted(_FORBIDDEN_IMPORTS)!r}:
    sys.modules.pop(name, None)

importlib.import_module({module_name!r})

forbidden = set({sorted(_FORBIDDEN_IMPORTS)!r})
imported = []
for name in sys.modules:
    if name in forbidden or name.startswith({_FORBIDDEN_IMPORT_PREFIXES!r}):
        imported.append(name)
print(json.dumps(sorted(imported)))
"""
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=_IMPORT_LAYER_CHECK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            'Timed out while checking dependency-light import layer\n'
            f'module under test: {module_name}\n'
            f'forbidden modules: {sorted(_FORBIDDEN_IMPORTS)!r}\n'
            f'forbidden module prefixes: {_FORBIDDEN_IMPORT_PREFIXES!r}\n'
            f'timeout_s: {_IMPORT_LAYER_CHECK_TIMEOUT_S}\n'
            f'stdout:\n{_subprocess_output_text(exc.stdout)}\n'
            f'stderr:\n{_subprocess_output_text(exc.stderr)}'
        ) from exc
    return set(json.loads(result.stdout))


def test_refraction_static_import_layer_checks_use_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured['timeout'] = kwargs.get('timeout')
        return subprocess.CompletedProcess(args, 0, stdout='[]', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    assert _forbidden_modules_imported_by('app.services.refraction_static_v1') == set()
    assert captured['timeout'] == _IMPORT_LAYER_CHECK_TIMEOUT_S


def test_import_layer_timeout_failure_message_is_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd=args,
            timeout=kwargs.get('timeout'),
            output='partial stdout',
            stderr='partial stderr',
        )

    monkeypatch.setattr(subprocess, 'run', fake_run)

    with pytest.raises(AssertionError) as exc_info:
        _forbidden_modules_imported_by('app.services.refraction_static_v1')

    message = str(exc_info.value)
    assert 'module under test: app.services.refraction_static_v1' in message
    assert 'forbidden modules:' in message
    assert 'app.main' in message
    assert 'segyio' in message
    assert 'stdout:\npartial stdout' in message
    assert 'stderr:\npartial stderr' in message


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


def test_refraction_static_v1_imports_without_reader_or_segyio() -> None:
    assert v1.RefractionV1EstimateResult is not None

    assert _forbidden_modules_imported_by('app.services.refraction_static_v1') == set()


def test_refraction_static_t1lsst_imports_without_reader_or_segyio() -> None:
    assert t1lsst.RefractionT1LSSTError is not None

    assert (
        _forbidden_modules_imported_by('app.services.refraction_static_t1lsst')
        == set()
    )


def test_refraction_static_schema_tests_do_not_import_service_or_reader() -> None:
    assert (
        _forbidden_modules_imported_by('app.tests.test_refraction_static_schema')
        == set()
    )


def test_refraction_static_artifact_helpers_do_not_import_app_main() -> None:
    assert (
        _forbidden_modules_imported_by(
            'app.tests._refraction_static_artifact_helpers'
        )
        == set()
    )
