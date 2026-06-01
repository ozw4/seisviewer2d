from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.state import create_app_state
from app.services.job_runner import JobCancelledError
from app.statics.refraction.adapters.seisviewer2d import (
    SeisViewer2DRefractionArtifactResolver,
    SeisViewer2DRefractionJobContext,
    SeisViewer2DRefractionRuntime,
    SeisViewer2DRefractionTraceStoreProvider,
)


PORT_MODULES = (
    'app.statics.refraction.ports',
    'app.statics.refraction.ports.artifact_resolver',
    'app.statics.refraction.ports.job_context',
    'app.statics.refraction.ports.pick_source',
    'app.statics.refraction.ports.runtime',
    'app.statics.refraction.ports.trace_store',
)

ADAPTER_MODULES = (
    'app.statics.refraction.adapters.seisviewer2d',
    'app.statics.refraction.adapters.seisviewer2d.artifact_resolver',
    'app.statics.refraction.adapters.seisviewer2d.job_context',
    'app.statics.refraction.adapters.seisviewer2d.runtime',
    'app.statics.refraction.adapters.seisviewer2d.trace_store',
)


def test_refraction_port_modules_import() -> None:
    for module_name in PORT_MODULES:
        importlib.import_module(module_name)


def test_refraction_adapter_modules_import() -> None:
    for module_name in ADAPTER_MODULES:
        importlib.import_module(module_name)


def test_refraction_ports_do_not_import_application_boundaries() -> None:
    ports_dir = Path(__file__).resolve().parents[1] / 'statics' / 'refraction' / 'ports'
    forbidden_imports = ('app.core.state', 'fastapi')

    for source_path in sorted(ports_dir.glob('*.py')):
        source = source_path.read_text()
        for forbidden_import in forbidden_imports:
            assert forbidden_import not in source, source_path.name


def test_trace_store_adapter_uses_state_registry_and_cached_reader(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    state.file_registry.update('file-a', store_path=tmp_path, dt=0.004)
    cached_reader = SimpleNamespace(meta={})
    state.cached_readers['file-a_189_193'] = cached_reader

    adapter = SeisViewer2DRefractionTraceStoreProvider(state)

    assert adapter.get_store_path('file-a') == tmp_path
    assert adapter.get_dt('file-a') == pytest.approx(0.004)
    assert adapter.filename('file-a') == tmp_path.name
    assert adapter.get_reader('file-a', 189, 193) is cached_reader
    assert cached_reader.meta['dt'] == pytest.approx(0.004)


def test_artifact_resolver_adapter_delegates_to_job_artifact_refs(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    artifacts_dir = tmp_path / 'job'
    artifacts_dir.mkdir()
    artifact_path = artifacts_dir / 'picks.npz'
    artifact_path.write_bytes(b'data')
    state.jobs.create_batch_apply_job(
        'job-a',
        file_id='file-a',
        key1_byte=189,
        key2_byte=193,
        artifacts_dir=str(artifacts_dir),
    )

    adapter = SeisViewer2DRefractionArtifactResolver(state)

    resolved = adapter.resolve_artifact(
        job_id='job-a',
        name='picks.npz',
        allowed_job_types={'batch_apply'},
        expected_file_id='file-a',
        expected_key1_byte=189,
        expected_key2_byte=193,
        reference_label='pick_source',
    )

    assert resolved == artifact_path


def test_job_context_adapter_updates_progress_and_checks_cancel(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    state.jobs.create_static_job(
        'job-a',
        file_id='file-a',
        key1_byte=189,
        key2_byte=193,
        statics_kind='refraction',
        artifacts_dir=str(tmp_path),
    )
    state.jobs.set_status('job-a', 'running')

    context = SeisViewer2DRefractionJobContext(
        state=state,
        job_id='job-a',
        artifacts_dir=tmp_path,
    )
    context.set_progress(0.25, 'loading')

    job = state.jobs['job-a']
    assert job['progress'] == pytest.approx(0.25)
    assert job['message'] == 'loading'

    context.set_message('solving')
    assert state.jobs['job-a']['message'] == 'solving'

    state.jobs.request_cancel('job-a')
    with pytest.raises(JobCancelledError):
        context.ensure_not_cancelled()


def test_runtime_adapter_builds_refraction_dependencies(tmp_path: Path) -> None:
    state = create_app_state()
    runtime = SeisViewer2DRefractionRuntime(state)

    assert isinstance(runtime.trace_store, SeisViewer2DRefractionTraceStoreProvider)
    assert isinstance(runtime.artifacts, SeisViewer2DRefractionArtifactResolver)

    context = runtime.job_context(job_id='job-a', artifacts_dir=tmp_path)

    assert isinstance(context, SeisViewer2DRefractionJobContext)
    assert context.job_id == 'job-a'
    assert context.artifacts_dir == tmp_path
