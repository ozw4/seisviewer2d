from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import TextIO

import pytest

import app.statics.refraction.application.bedrock as bedrock
import app.statics.refraction.application.design_matrix as design_matrix
import app.statics.refraction.domain.export_types as export_types
import app.statics.refraction.application.half_intercept as half_intercept
import app.statics.refraction.domain.layer_config as layer_config
import app.statics.refraction.domain.solver as solver
import app.statics.refraction.domain.source_depth as source_depth
import app.statics.refraction.domain.t1lsst as t1lsst
import app.statics.refraction.domain.types as refraction_types
import app.statics.refraction.domain.uphole as uphole
import app.statics.refraction.domain.v1 as v1
import app.statics.refraction.application.weathering as weathering


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORTS = {
    'app.api.routers',
    'app.main',
    'app.services.reader',
    'app.statics.refraction.application.workflow',
    'app.statics.refraction.application.input_model',
    'app.trace_store.reader',
    'segyio',
}
_TYPE_MODULE_FORBIDDEN_IMPORTS = _FORBIDDEN_IMPORTS | {'app.api.schemas'}
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


def _read_tmp_text(file_obj: TextIO) -> str:
    file_obj.flush()
    file_obj.seek(0)
    return file_obj.read()


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
    with (
        tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8') as stdout_file,
        tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8') as stderr_file,
    ):
        proc = subprocess.Popen(
            [sys.executable, '-c', code],
            cwd=_REPO_ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
            env={**os.environ, **_IMPORT_LAYER_CHILD_ENV_OVERRIDES},
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            stdout = _read_tmp_text(stdout_file)
            stderr = _read_tmp_text(stderr_file)
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

        stdout = _read_tmp_text(stdout_file)
        stderr = _read_tmp_text(stderr_file)
        if returncode != 0:
            raise AssertionError(
                _import_check_failure_message(
                    'Dependency-light import subprocess failed',
                    module_name=module_name,
                    forbidden_modules=forbidden_modules,
                    forbidden_module_prefixes=forbidden_module_prefixes,
                    timeout_s=timeout_s,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                )
            )
        return subprocess.CompletedProcess(proc.args, returncode, stdout, stderr)


def _forbidden_modules_imported_by(
    module_name: str,
    *,
    forbidden_imports: set[str] | None = None,
    forbidden_import_prefixes: tuple[str, ...] = _FORBIDDEN_IMPORT_PREFIXES,
) -> set[str]:
    forbidden_imports = forbidden_imports or _FORBIDDEN_IMPORTS
    code = f"""
from __future__ import annotations

import importlib
import json
import sys

for name in {sorted(forbidden_imports)!r}:
    sys.modules.pop(name, None)

importlib.import_module({module_name!r})

forbidden = set({sorted(forbidden_imports)!r})
imported = []
for name in sys.modules:
    if name in forbidden or name.startswith({forbidden_import_prefixes!r}):
        imported.append(name)
print(json.dumps(sorted(imported)))
"""
    result = _run_import_check_subprocess(
        code,
        _IMPORT_LAYER_CHECK_TIMEOUT_S,
        module_name=module_name,
        forbidden_modules=forbidden_imports,
        forbidden_module_prefixes=forbidden_import_prefixes,
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
            stdout = kwargs['stdout']
            assert hasattr(stdout, 'write')
            stdout.write('[]')

        def wait(self, timeout: float | None = None) -> int:
            captured['timeout'] = timeout
            return self.returncode

    monkeypatch.setattr(subprocess, 'Popen', FakePopen)

    assert _forbidden_modules_imported_by('app.statics.refraction.domain.v1') == set()

    kwargs = captured['kwargs']
    assert kwargs['cwd'] == _REPO_ROOT
    assert kwargs['stdout'] != subprocess.PIPE
    assert kwargs['stderr'] != subprocess.PIPE
    assert hasattr(kwargs['stdout'], 'write')
    assert hasattr(kwargs['stderr'], 'write')
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
    assert 'stderr:\n' in message


def test_import_layer_timeout_failure_message_is_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeTimeoutPopen:
        def __init__(self, args: list[str], **kwargs: object) -> None:
            self.args = args
            self.pid = 67890
            self.returncode = None
            self.wait_calls = 0
            stdout = kwargs['stdout']
            stderr = kwargs['stderr']
            assert hasattr(stdout, 'write')
            assert hasattr(stderr, 'write')
            stdout.write('partial stdout')
            stderr.write('partial stderr')

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(
                    cmd=self.args,
                    timeout=timeout,
                )
            self.returncode = -signal.SIGKILL
            return self.returncode

    def fake_killpg(pid: int, sig: int) -> None:
        captured['killpg'] = (pid, sig)

    monkeypatch.setattr(subprocess, 'Popen', FakeTimeoutPopen)
    monkeypatch.setattr(os, 'killpg', fake_killpg)

    with pytest.raises(AssertionError) as exc_info:
        _forbidden_modules_imported_by('app.statics.refraction.domain.v1')

    message = str(exc_info.value)
    assert 'module under test: app.statics.refraction.domain.v1' in message
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
    assert refraction_types.RefractionLayerSolveResult is not None
    assert refraction_types.RefractionMultiLayerSolveResult is not None
    assert refraction_types.RefractionMultiLayerStaticComponents is not None
    assert refraction_types.RefractionEndpointFieldCorrectionResult is not None
    assert refraction_types.RefractionTraceFieldCorrectionResult is not None
    assert refraction_types.RefractionSourceDepthResult is not None
    assert refraction_types.RefractionUpholeResult is not None
    assert refraction_types.REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES == (
        'source_depth_shift_s',
        'uphole_shift_s',
        'manual_static_shift_s',
    )
    assert refraction_types.RefractionDatumStaticsResult is not None

    assert (
        _forbidden_modules_imported_by(
            'app.statics.refraction.domain.types',
            forbidden_imports=_TYPE_MODULE_FORBIDDEN_IMPORTS,
        )
        == set()
    )

    source = _module_source(refraction_types)

    assert 'app.api.schemas' not in source
    assert 'app.api.routers' not in source
    assert 'app.main' not in source
    assert 'app.services.reader' not in source
    assert 'app.statics.refraction.application.workflow' not in source
    assert 'app.trace_store.reader' not in source
    assert 'refraction_static_inputs' not in source
    assert 'segyio' not in source
    assert 'scipy' not in source


def test_refraction_static_export_types_is_dependency_light() -> None:
    assert export_types.RefractionStaticEndpointExportRow is not None
    assert export_types.RefractionStaticExportBundle is not None
    assert export_types.RefractionStaticCanonicalTableRow is not None

    assert (
        _forbidden_modules_imported_by(
            'app.statics.refraction.domain.export_types',
            forbidden_imports=_TYPE_MODULE_FORBIDDEN_IMPORTS,
        )
        == set()
    )

    source = _module_source(export_types)
    assert 'app.api.schemas' not in source
    assert 'app.api.routers' not in source
    assert 'app.main' not in source
    assert 'app.statics.refraction.application.workflow' not in source
    assert 'app.trace_store.reader' not in source
    assert 'segyio' not in source
    assert 'numpy' not in source


def test_refraction_static_source_depth_is_dependency_light() -> None:
    assert source_depth.resolve_refraction_source_depth is not None

    assert (
        _forbidden_modules_imported_by(
            'app.statics.refraction.domain.source_depth',
            forbidden_imports=_TYPE_MODULE_FORBIDDEN_IMPORTS,
        )
        == set()
    )

    source = _module_source(source_depth)
    assert 'app.api.schemas' not in source
    assert 'app.api.routers' not in source
    assert 'app.main' not in source
    assert 'app.services.reader' not in source
    assert 'app.statics.refraction.application.workflow' not in source
    assert 'app.trace_store.reader' not in source
    assert 'refraction_static_inputs' not in source
    assert 'segyio' not in source
    assert 'scipy' not in source


def test_refraction_static_uphole_is_dependency_light() -> None:
    assert uphole.resolve_refraction_uphole is not None

    assert (
        _forbidden_modules_imported_by(
            'app.statics.refraction.domain.uphole',
            forbidden_imports=_TYPE_MODULE_FORBIDDEN_IMPORTS,
        )
        == set()
    )

    source = _module_source(uphole)
    assert 'app.api.schemas' not in source
    assert 'app.api.routers' not in source
    assert 'app.main' not in source
    assert 'app.services.reader' not in source
    assert 'app.statics.refraction.application.workflow' not in source
    assert 'app.trace_store.reader' not in source
    assert 'refraction_static_inputs' not in source
    assert 'segyio' not in source
    assert 'scipy' not in source


def test_refraction_static_layer_config_is_dependency_light() -> None:
    assert layer_config.RefractionStaticLayerConfig is not None

    assert (
        _forbidden_modules_imported_by(
            'app.statics.refraction.domain.layer_config',
            forbidden_imports=_TYPE_MODULE_FORBIDDEN_IMPORTS,
        )
        == set()
    )

    source = _module_source(layer_config)
    assert 'app.api.schemas' not in source
    assert 'app.statics.refraction.application.workflow' not in source
    assert 'app.statics.refraction.application.input_model' not in source


def test_numeric_refraction_modules_import_without_tracestore_readers() -> None:
    assert design_matrix.RefractionStaticDesignMatrix is not None
    assert solver.RefractionStaticSolverResult is not None
    assert bedrock.RefractionBedrockSlownessResult is not None
    assert half_intercept.RefractionHalfInterceptTimeResult is not None
    assert weathering.RefractionWeatheringThicknessResult is not None

    for module_name in (
        'app.statics.refraction.application.design_matrix',
        'app.statics.refraction.domain.solver',
        'app.statics.refraction.application.bedrock',
        'app.statics.refraction.application.half_intercept',
        'app.statics.refraction.application.weathering',
    ):
        assert _forbidden_modules_imported_by(module_name) == set()


def test_refraction_static_v1_imports_without_reader_or_segyio() -> None:
    assert v1.RefractionV1EstimateResult is not None

    assert _forbidden_modules_imported_by('app.statics.refraction.domain.v1') == set()


def test_refraction_static_t1lsst_imports_without_reader_or_segyio() -> None:
    assert t1lsst.RefractionT1LSSTError is not None

    assert (
        _forbidden_modules_imported_by('app.statics.refraction.domain.t1lsst')
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
