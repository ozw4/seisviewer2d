from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import app.services.first_break_qc_service as service
from app.api.schemas import FirstBreakQcRequest
from app.core.state import AppState, create_app_state


class _Reader:
    key1_byte = 189
    key2_byte = 193

    def __init__(self) -> None:
        self.traces = np.zeros((4, 8), dtype=np.float32)

    def get_n_samples(self) -> int:
        return 8


def _request(
    *,
    datum_job_id: str = 'datum-job',
    pick_source: dict[str, object] | None = None,
) -> FirstBreakQcRequest:
    if pick_source is None:
        pick_source = {
            'kind': 'batch_job_artifact',
            'job_id': 'batch-job',
            'name': 'predicted_picks_time_s.npz',
        }
    return FirstBreakQcRequest(
        file_id='source-file-id',
        key1_byte=189,
        key2_byte=193,
        datum_solution={
            'job_id': datum_job_id,
            'name': 'datum_static_solution.npz',
        },
        pick_source=pick_source,
        offset={'offset_byte': 37},
        qc={'require_linear_offset_model': False},
    )


def _setup_state(tmp_path: Path) -> tuple[AppState, Path]:
    state = create_app_state()
    state.file_registry.set_record(
        'source-file-id',
        {
            'path': str(tmp_path / 'line.sgy'),
            'store_path': str(tmp_path / 'trace-store'),
            'dt': 0.004,
        },
    )
    job_dir = tmp_path / 'qc-job'
    with state.lock:
        state.jobs.create_static_job(
            'qc-job',
            file_id='source-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='first_break_qc',
            artifacts_dir=str(job_dir),
        )
    return state, job_dir


def _add_datum_job(state: AppState, tmp_path: Path, *, kind: str = 'datum') -> Path:
    datum_dir = tmp_path / 'datum-job'
    datum_dir.mkdir(parents=True, exist_ok=True)
    solution = datum_dir / 'datum_static_solution.npz'
    solution.write_bytes(b'solution')
    with state.lock:
        state.jobs.create_static_job(
            'datum-job',
            file_id='source-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind=kind,
            artifacts_dir=str(datum_dir),
        )
    return solution


def _add_batch_job(state: AppState, tmp_path: Path, *, write_artifact: bool = True) -> Path:
    batch_dir = tmp_path / 'batch-job'
    batch_dir.mkdir(parents=True, exist_ok=True)
    artifact = batch_dir / 'predicted_picks_time_s.npz'
    if write_artifact:
        artifact.write_bytes(b'picks')
    with state.lock:
        state.jobs.create_batch_apply_job(
            'batch-job',
            file_id='source-file-id',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(batch_dir),
        )
    return artifact


def _add_manual_npz_job(state: AppState, tmp_path: Path) -> Path:
    manual_dir = tmp_path / 'manual-job'
    manual_dir.mkdir(parents=True, exist_ok=True)
    artifact = manual_dir / 'manual_picks.npz'
    artifact.write_bytes(b'manual')
    with state.lock:
        state.jobs.create_static_job(
            'manual-job',
            file_id='source-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='first_break_qc',
            artifacts_dir=str(manual_dir),
        )
    return artifact


def _patch_success_path(monkeypatch, *, captured: dict[str, Any] | None = None) -> None:
    if captured is None:
        captured = {}
    reader = _Reader()
    pick_source = SimpleNamespace(name='pick-source')
    inputs = SimpleNamespace(name='inputs')
    metrics = SimpleNamespace(name='metrics')

    monkeypatch.setattr(service, 'get_reader', lambda *args, **kwargs: reader)

    def _load_npz(path: Path, **kwargs: Any) -> object:
        captured['npz_path'] = Path(path)
        captured['npz_kwargs'] = kwargs
        return pick_source

    def _load_memmap(**kwargs: Any) -> object:
        captured['memmap_kwargs'] = kwargs
        return pick_source

    def _build_inputs(**kwargs: Any) -> object:
        captured['build_kwargs'] = kwargs
        return inputs

    def _compute(actual_inputs: object, **kwargs: Any) -> object:
        captured['compute_inputs'] = actual_inputs
        captured['compute_kwargs'] = kwargs
        return metrics

    def _write_artifacts(**kwargs: Any) -> object:
        captured['write_kwargs'] = kwargs
        job_dir = Path(kwargs['job_dir'])
        (job_dir / 'first_break_qc.json').write_text('{"ok":true}', encoding='utf-8')
        (job_dir / 'first_break_qc.csv').write_text('a,b\n', encoding='utf-8')
        (job_dir / 'residual_by_key1.csv').write_text('key1\n', encoding='utf-8')
        return SimpleNamespace(
            qc_json=job_dir / 'first_break_qc.json',
            qc_csv=job_dir / 'first_break_qc.csv',
            residual_by_key1_csv=job_dir / 'residual_by_key1.csv',
        )

    monkeypatch.setattr(service, 'load_npz_pick_source', _load_npz)
    monkeypatch.setattr(service, 'load_manual_memmap_pick_source', _load_memmap)
    monkeypatch.setattr(service, 'build_first_break_qc_inputs', _build_inputs)
    monkeypatch.setattr(service, 'compute_first_break_qc_metrics', _compute)
    monkeypatch.setattr(service, 'write_first_break_qc_artifacts', _write_artifacts)


def test_first_break_qc_job_writes_job_meta_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    _add_batch_job(state, tmp_path)
    _patch_success_path(monkeypatch)

    service.run_first_break_qc_job('qc-job', _request(), state)

    meta = json.loads((job_dir / 'job_meta.json').read_text(encoding='utf-8'))
    assert meta['job_id'] == 'qc-job'
    assert meta['job_type'] == 'statics'
    assert meta['statics_kind'] == 'first_break_qc'
    assert meta['source_file_id'] == 'source-file-id'
    assert meta['inputs'] == {
        'datum_solution': {
            'job_id': 'datum-job',
            'name': 'datum_static_solution.npz',
        },
        'pick_source': {
            'kind': 'batch_job_artifact',
            'job_id': 'batch-job',
            'name': 'predicted_picks_time_s.npz',
        },
        'offset_byte': 37,
    }
    assert meta['artifacts'] == {
        'qc_json': 'first_break_qc.json',
        'qc_csv': 'first_break_qc.csv',
        'residual_by_key1_csv': 'residual_by_key1.csv',
    }


def test_first_break_qc_job_uses_batch_predicted_picks_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    pick_path = _add_batch_job(state, tmp_path)
    captured: dict[str, Any] = {}
    _patch_success_path(monkeypatch, captured=captured)

    service.run_first_break_qc_job('qc-job', _request(), state)

    assert captured['npz_path'] == pick_path
    assert captured['npz_kwargs']['source_kind'] == 'batch_npz'
    assert captured['write_kwargs']['pick_source_artifact_name'] == (
        'predicted_picks_time_s.npz'
    )


def test_first_break_qc_job_uses_manual_memmap_pick_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    captured: dict[str, Any] = {}
    _patch_success_path(monkeypatch, captured=captured)
    req = _request(pick_source={'kind': 'manual_memmap'})

    service.run_first_break_qc_job('qc-job', req, state)

    assert captured['memmap_kwargs']['file_id'] == 'source-file-id'
    assert captured['write_kwargs']['pick_source_artifact_name'] is None
    assert 'npz_path' not in captured


def test_first_break_qc_job_uses_manual_npz_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    pick_path = _add_manual_npz_job(state, tmp_path)
    captured: dict[str, Any] = {}
    _patch_success_path(monkeypatch, captured=captured)
    req = _request(
        pick_source={
            'kind': 'manual_npz_artifact',
            'job_id': 'manual-job',
            'name': 'manual_picks.npz',
        }
    )

    service.run_first_break_qc_job('qc-job', req, state)

    assert captured['npz_path'] == pick_path
    assert captured['npz_kwargs']['source_kind'] == 'manual_npz'
    assert captured['write_kwargs']['pick_source_artifact_name'] == 'manual_picks.npz'


def test_first_break_qc_job_writes_expected_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    _add_batch_job(state, tmp_path)
    _patch_success_path(monkeypatch)

    service.run_first_break_qc_job('qc-job', _request(), state)

    assert (job_dir / 'job_meta.json').is_file()
    assert (job_dir / 'first_break_qc.json').is_file()
    assert (job_dir / 'first_break_qc.csv').is_file()
    assert (job_dir / 'residual_by_key1.csv').is_file()
    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert job['status'] == 'done'
    assert job['progress'] == 1.0


def test_first_break_qc_job_errors_when_datum_job_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_batch_job(state, tmp_path)
    _patch_success_path(monkeypatch)

    service.run_first_break_qc_job('qc-job', _request(), state)

    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert job['status'] == 'error'
    assert 'job_id not found: datum-job' in str(job['message'])


def test_first_break_qc_job_errors_when_datum_job_is_not_datum_static(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path, kind='first_break_qc')
    _add_batch_job(state, tmp_path)
    _patch_success_path(monkeypatch)

    service.run_first_break_qc_job('qc-job', _request(), state)

    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert job['status'] == 'error'
    assert 'unsupported statics_kind' in str(job['message'])


def test_first_break_qc_job_errors_when_solution_artifact_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    solution = _add_datum_job(state, tmp_path)
    solution.unlink()
    _add_batch_job(state, tmp_path)
    _patch_success_path(monkeypatch)

    service.run_first_break_qc_job('qc-job', _request(), state)

    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert job['status'] == 'error'
    assert 'job artifact not found: datum_static_solution.npz' in str(job['message'])


def test_first_break_qc_job_errors_when_pick_source_artifact_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    _add_batch_job(state, tmp_path, write_artifact=False)
    _patch_success_path(monkeypatch)

    service.run_first_break_qc_job('qc-job', _request(), state)

    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert job['status'] == 'error'
    assert 'job artifact not found: predicted_picks_time_s.npz' in str(job['message'])


def test_first_break_qc_job_errors_when_batch_pick_source_job_is_not_batch_apply(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    wrong_dir = tmp_path / 'batch-job'
    wrong_dir.mkdir()
    (wrong_dir / 'predicted_picks_time_s.npz').write_bytes(b'picks')
    with state.lock:
        state.jobs.create_static_job(
            'batch-job',
            file_id='source-file-id',
            key1_byte=189,
            key2_byte=193,
            statics_kind='datum',
            artifacts_dir=str(wrong_dir),
        )
    _patch_success_path(monkeypatch)

    service.run_first_break_qc_job('qc-job', _request(), state)

    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert job['status'] == 'error'
    assert 'unsupported job_type' in str(job['message'])


def test_first_break_qc_job_errors_when_input_shapes_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    _add_batch_job(state, tmp_path)
    _patch_success_path(monkeypatch)

    def _fail_build(**_kwargs: Any) -> object:
        raise ValueError('n_traces mismatch')

    monkeypatch.setattr(service, 'build_first_break_qc_inputs', _fail_build)

    service.run_first_break_qc_job('qc-job', _request(), state)

    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert job['status'] == 'error'
    assert job['message'] == 'n_traces mismatch'


def test_first_break_qc_job_cancel_before_artifact_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    _add_batch_job(state, tmp_path)
    captured: dict[str, Any] = {}
    _patch_success_path(monkeypatch, captured=captured)

    def _cancel_after_compute(actual_inputs: object, **kwargs: Any) -> object:
        with state.lock:
            state.jobs.request_cancel('qc-job')
        return SimpleNamespace(name='metrics')

    monkeypatch.setattr(service, 'compute_first_break_qc_metrics', _cancel_after_compute)

    service.run_first_break_qc_job('qc-job', _request(), state)

    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert job['status'] == 'cancelled'
    assert 'write_kwargs' not in captured
    assert not list(job_dir.glob('first_break_qc*.tmp'))
    assert not list(job_dir.glob('residual_by_key1*.tmp'))


def test_first_break_qc_job_does_not_register_corrected_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state, _job_dir = _setup_state(tmp_path)
    _add_datum_job(state, tmp_path)
    _add_batch_job(state, tmp_path)
    _patch_success_path(monkeypatch)

    service.run_first_break_qc_job('qc-job', _request(), state)

    assert set(state.file_registry.records) == {'source-file-id'}
    with state.lock:
        job = dict(state.jobs['qc-job'])
    assert 'corrected_file_id' not in job
    assert 'corrected_store_path' not in job
