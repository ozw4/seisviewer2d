from __future__ import annotations

from pathlib import Path

import pytest

from app.core.state import create_app_state
from app.services.refraction_static_artifacts import (
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
    REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_V1_ESTIMATES_CSV_NAME,
    REFRACTION_V1_QC_JSON_NAME,
    REFRACTION_STATIC_REQUEST_JSON_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.job_artifact_refs import resolve_job_artifact_path
from app.services.refraction_static_source_depth import (
    REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
    REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
)
from app.services.refraction_static_uphole import (
    REFRACTION_UPHOLE_QC_JSON_NAME,
    REFRACTION_UPHOLE_SOURCES_CSV_NAME,
)


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


@pytest.mark.parametrize(
    'artifact_name',
    [
        REFRACTION_V1_QC_JSON_NAME,
        REFRACTION_V1_ESTIMATES_CSV_NAME,
        REFRACTION_T1LSST_1LAYER_COMPONENTS_CSV_NAME,
        REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
        REFRACTION_V3_REFRACTOR_VELOCITY_QC_JSON_NAME,
        REFRACTION_VSUB_REFRACTOR_VELOCITY_QC_JSON_NAME,
        REFRACTION_SOURCE_DEPTH_QC_JSON_NAME,
        REFRACTION_SOURCE_DEPTH_SOURCES_CSV_NAME,
        REFRACTION_UPHOLE_QC_JSON_NAME,
        REFRACTION_UPHOLE_SOURCES_CSV_NAME,
        REFRACTION_STATIC_REQUEST_JSON_NAME,
        SOURCE_STATIC_TABLE_CSV_NAME,
        RECEIVER_STATIC_TABLE_CSV_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    ],
)
def test_resolve_job_artifact_path_accepts_refraction_registered_artifacts(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    state = create_app_state()
    job_dir = tmp_path / 'refraction'
    job_dir.mkdir()
    artifact = job_dir / artifact_name
    artifact.write_bytes(b'data')
    with state.lock:
        state.jobs.create_static_job(
            'refraction-job',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )

    resolved = resolve_job_artifact_path(
        state,
        job_id='refraction-job',
        name=artifact_name,
        allowed_job_types={'statics'},
        allowed_statics_kinds={'refraction'},
        expected_file_id='file-a',
        expected_key1_byte=189,
        expected_key2_byte=193,
    )

    assert resolved == artifact


def test_resolve_job_artifact_path_rejects_unregistered_refraction_artifact(
    tmp_path: Path,
) -> None:
    state = create_app_state()
    job_dir = tmp_path / 'refraction'
    job_dir.mkdir()
    (job_dir / 'debug-only.npz').write_bytes(b'data')
    with state.lock:
        state.jobs.create_static_job(
            'refraction-job',
            file_id='file-a',
            key1_byte=189,
            key2_byte=193,
            statics_kind='refraction',
            artifacts_dir=str(job_dir),
        )

    with pytest.raises(ValueError, match='not registered for statics_kind'):
        resolve_job_artifact_path(
            state,
            job_id='refraction-job',
            name='debug-only.npz',
            allowed_job_types={'statics'},
            allowed_statics_kinds={'refraction'},
        )
