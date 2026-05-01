from __future__ import annotations

from pathlib import Path

import pytest

from app.core.state import create_app_state
from app.services.job_artifact_refs import resolve_job_artifact_path


def test_resolve_job_artifact_path_returns_named_file(tmp_path: Path) -> None:
    state = create_app_state()
    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    artifact = job_dir / 'datum_static_solution.npz'
    artifact.write_bytes(b'data')
    with state.lock:
        state.jobs.create_static_job(
            'datum-job',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            statics_kind='datum',
            artifacts_dir=str(job_dir),
        )

    resolved = resolve_job_artifact_path(
        state,
        job_id='datum-job',
        name='datum_static_solution.npz',
        allowed_job_types={'statics'},
        allowed_statics_kinds={'datum'},
    )

    assert resolved == artifact


@pytest.mark.parametrize('name', ['', '.', '..', '../x.npz', 'nested/x.npz', '/x.npz'])
def test_resolve_job_artifact_path_rejects_path_names(
    tmp_path: Path,
    name: str,
) -> None:
    state = create_app_state()
    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    with state.lock:
        state.jobs.create_static_job(
            'job-a',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            statics_kind='datum',
            artifacts_dir=str(job_dir),
        )

    with pytest.raises(ValueError, match='plain file name'):
        resolve_job_artifact_path(state, job_id='job-a', name=name)


def test_resolve_job_artifact_path_rejects_wrong_job_type(tmp_path: Path) -> None:
    state = create_app_state()
    job_dir = tmp_path / 'batch'
    job_dir.mkdir()
    (job_dir / 'artifact.npz').write_bytes(b'data')
    with state.lock:
        state.jobs.create_batch_apply_job(
            'batch-job',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir=str(job_dir),
        )

    with pytest.raises(ValueError, match='unsupported job_type'):
        resolve_job_artifact_path(
            state,
            job_id='batch-job',
            name='artifact.npz',
            allowed_job_types={'statics'},
        )


def test_resolve_job_artifact_path_rejects_wrong_statics_kind(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    job_dir = tmp_path / 'statics'
    job_dir.mkdir()
    (job_dir / 'artifact.npz').write_bytes(b'data')
    with state.lock:
        state.jobs.create_static_job(
            'qc-job',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            statics_kind='first_break_qc',
            artifacts_dir=str(job_dir),
        )

    with pytest.raises(ValueError, match='unsupported statics_kind'):
        resolve_job_artifact_path(
            state,
            job_id='qc-job',
            name='artifact.npz',
            allowed_statics_kinds={'datum'},
        )
