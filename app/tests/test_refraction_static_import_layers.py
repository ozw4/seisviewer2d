from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
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
_IMPORT_LAYER_CHILD_ENV_OVERRIDES = {
    'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
    'DD_TRACE_ENABLED': 'false',
}


def _module_source(module: ModuleType) -> str:
    path = Path(module.__file__ or '')
    return path.read_text(encoding='utf-8')


def _subprocess_output_text(value: bytes | str | None) -> str:
    if value is None:
        return '<none>'
    if isinstance(value, bytes):
        value = value.decode(errors='replace')
    return value if value else '<empty>'


def _import_check_failure_message(
    title: str,
    *,
    module_name: str,
    forbidden_modules: set[str],
    forbidden_module_prefixes: tuple[str, ...],
    timeout_s: float,
    stdout: bytes | str | None,
    stderr: bytes | str | None,
    returncode: int | None = None,
) -> str:
    lines = [
        title,
        f'module under test: {module_name}',
        f'forbidden modules: {sorted(forbidden_modules)!r}',
        f'forbidden module prefixes: {forbidden_module_prefixes!r}',
        f'timeout_s: {timeout_s}',
    ]
    if returncode is not None:
        lines.append(f'returncode: {returncode}')
    lines.extend(
        [
            f'stdout:\n{_subprocess_output_text(stdout)}',
            f'stderr:\n{_subprocess_output_text(stderr)}',
        ]
    )
    return '\n'.join(lines)


def _run_import_check_subprocess(
    code: str,
    timeout_s: float,
    *,
    module_name: str,
    forbidden_modules: set[str],
    forbidden_module_prefixes: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, **_IMPORT_LAYER_CHILD_ENV_OVERRIDES},
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired as drain_exc:
            stdout = drain_exc.stdout or exc.stdout
            stderr = drain_exc.stderr or exc.stderr
        raise AssertionError(
            _import_check_failure_message(
                'Timed out while checking dependency-light import layer',
                module_name=module_name,
                forbidden_modules=forbidden_modules,
                forbidden_module_prefixes=forbidden_module_prefixes,
                timeout_s=timeout_s,
                stdout=stdout,
                stderr=stderr,
            )
        ) from exc

    if proc.returncode != 0:
        raise AssertionError(
            _import_check_failure_message(
                'Dependency-light import subprocess failed',
                module_name=module_name,
                forbidden_modules=forbidden_modules,
                forbidden_module_prefixes=forbidden_module_prefixes,
                timeout_s=timeout_s,
                stdout=stdout,
                stderr=stderr,
                returncode=proc.returncode,
            )
        )
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


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
    result = _run_import_check_subprocess(
        code,
        _IMPORT_LAYER_CHECK_TIMEOUT_S,
        module_name=module_name,
        forbidden_modules=_FORBIDDEN_IMPORTS,
        forbidden_module_prefixes=_FORBIDDEN_IMPORT_PREFIXES,
    )
    return set(json.loads(result.stdout))


def test_refraction_static_import_layer_checks_use_process_group_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'parent-value')
    monkeypatch.setenv('DD_TRACE_ENABLED', 'true')

    class FakePopen:
        def __init__(self, args: list[str], **kwargs: object) -> None:
            captured['args'] = args
            captured['kwargs'] = kwargs
            self.args = args
            self.pid = 12345
            self.returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            captured['timeout'] = timeout
            return '[]', ''

    monkeypatch.setattr(subprocess, 'Popen', FakePopen)

    assert _forbidden_modules_imported_by('app.services.refraction_static_v1') == set()

    kwargs = captured['kwargs']
    assert kwargs['cwd'] == _REPO_ROOT
    assert kwargs['stdout'] == subprocess.PIPE
    assert kwargs['stderr'] == subprocess.PIPE
    assert kwargs['text'] is True
    assert kwargs['start_new_session'] is True
    assert captured['timeout'] == _IMPORT_LAYER_CHECK_TIMEOUT_S
    env = kwargs['env']
    assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
    assert env['DD_TRACE_ENABLED'] == 'false'
    assert os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == 'parent-value'
    assert os.environ['DD_TRACE_ENABLED'] == 'true'


def test_import_layer_timeout_kills_child_process_group() -> None:
    code = """
from __future__ import annotations

import subprocess
import sys
import time

grandchild_code = (
    "import sys, time; "
    "sys.stdout.write('grandchild stdout before timeout\\\\n'); "
    "sys.stdout.flush(); "
    "sys.stderr.write('grandchild stderr before timeout\\\\n'); "
    "sys.stderr.flush(); "
    "time.sleep(60)"
)
grandchild = subprocess.Popen([sys.executable, '-c', grandchild_code])
sys.stdout.write(f'spawned grandchild {grandchild.pid}\\n')
sys.stdout.flush()
sys.stderr.write('child stderr before timeout\\n')
sys.stderr.flush()
time.sleep(60)
"""

    started = time.monotonic()
    with pytest.raises(AssertionError) as exc_info:
        _run_import_check_subprocess(
            code,
            0.5,
            module_name='synthetic.import_timeout',
            forbidden_modules=_FORBIDDEN_IMPORTS,
            forbidden_module_prefixes=_FORBIDDEN_IMPORT_PREFIXES,
        )

    assert time.monotonic() - started < 5.0
    message = str(exc_info.value)
    assert 'Timed out while checking dependency-light import layer' in message
    assert 'module under test: synthetic.import_timeout' in message
    assert 'forbidden modules:' in message
    assert 'app.main' in message
    assert 'segyio' in message
    assert 'timeout_s: 0.5' in message
    assert 'stdout:\n' in message
    assert 'spawned grandchild' in message
    assert 'stderr:\n' in message
    assert 'child stderr before timeout' in message


def test_import_layer_timeout_failure_message_is_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeTimeoutPopen:
        def __init__(self, args: list[str], **kwargs: object) -> None:
            self.args = args
            self.pid = 67890
            self.returncode = None
            self.communicate_calls = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(
                    cmd=self.args,
                    timeout=timeout,
                    output='partial stdout',
                    stderr='partial stderr',
                )
            return 'partial stdout', 'partial stderr'

    def fake_killpg(pid: int, sig: int) -> None:
        captured['killpg'] = (pid, sig)

    monkeypatch.setattr(subprocess, 'Popen', FakeTimeoutPopen)
    monkeypatch.setattr(os, 'killpg', fake_killpg)

    with pytest.raises(AssertionError) as exc_info:
        _forbidden_modules_imported_by('app.services.refraction_static_v1')

    message = str(exc_info.value)
    assert 'module under test: app.services.refraction_static_v1' in message
    assert 'forbidden modules:' in message
    assert 'app.main' in message
    assert 'segyio' in message
    assert f'timeout_s: {_IMPORT_LAYER_CHECK_TIMEOUT_S}' in message
    assert 'stdout:\npartial stdout' in message
    assert 'stderr:\npartial stderr' in message
    assert captured['killpg'] == (67890, signal.SIGKILL)


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
