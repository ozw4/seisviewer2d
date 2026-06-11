from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.job_routes import (
    cancel_job_and_get_status_payload,
    get_job_or_404,
    infer_job_type,
    job_status_payload,
)
from app.core.state import create_app_state


def test_infer_job_type_uses_explicit_job_type() -> None:
    assert infer_job_type({'job_type': 'statics', 'pipeline_key': 'pipe-1'}) == 'statics'


def test_infer_job_type_uses_pipeline_key_convention() -> None:
    assert infer_job_type({'pipeline_key': 'pipe-1'}) == 'pipeline'


def test_get_job_or_404_rejects_unknown_job() -> None:
    state = create_app_state()

    with pytest.raises(HTTPException) as exc_info:
        get_job_or_404(state, 'missing')

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == 'Job ID not found'


def test_get_job_or_404_rejects_wrong_job_type() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1',
            created_ts=0.0,
        )

    with pytest.raises(HTTPException) as exc_info:
        get_job_or_404(state, 'job-1', allowed_job_types={'statics'})

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == 'Job ID not found'


def test_get_job_or_404_returns_copy() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1',
            created_ts=0.0,
        )

    job = get_job_or_404(state, 'job-1', allowed_job_types={'batch_apply'})
    job['status'] = 'error'

    with state.lock:
        assert state.jobs['job-1']['status'] == 'queued'


@pytest.mark.parametrize(
    ('raw_status', 'expected_state'),
    [
        ('completed', 'done'),
        ('failed', 'error'),
    ],
)
def test_job_status_payload_normalizes_completed_failed_aliases(
    raw_status: str,
    expected_state: str,
) -> None:
    payload = job_status_payload(
        {'status': raw_status, 'progress': 0.75, 'message': 'Working'}
    )

    assert payload == {
        'state': expected_state,
        'progress': 0.75,
        'message': 'Working',
    }


def test_job_status_payload_uses_zero_for_non_numeric_progress() -> None:
    payload = job_status_payload(
        {'status': 'running', 'progress': 'half', 'message': 'Working'}
    )

    assert payload == {
        'state': 'running',
        'progress': 0.0,
        'message': 'Working',
    }


def test_job_status_payload_uses_empty_string_for_non_string_message() -> None:
    payload = job_status_payload({'status': 'running', 'progress': 0.5, 'message': 7})

    assert payload == {
        'state': 'running',
        'progress': 0.5,
        'message': '',
    }


def test_cancel_job_and_get_status_payload_validates_type_before_cancel() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1',
            created_ts=0.0,
        )

    with pytest.raises(HTTPException) as exc_info:
        cancel_job_and_get_status_payload(
            state,
            'job-1',
            allowed_job_types={'statics'},
        )

    assert exc_info.value.status_code == 404
    with state.lock:
        assert state.jobs['job-1']['cancel_requested'] is False


def test_cancel_job_and_get_status_payload_returns_post_cancel_status() -> None:
    state = create_app_state()
    with state.lock:
        state.jobs.create_batch_apply_job(
            'job-1',
            file_id='file-1',
            key1_byte=189,
            key2_byte=193,
            artifacts_dir='/tmp/job-1',
            created_ts=0.0,
        )

    payload = cancel_job_and_get_status_payload(
        state,
        'job-1',
        allowed_job_types={'batch_apply'},
    )

    assert payload == {
        'state': 'cancelled',
        'progress': 0.0,
        'message': 'The job was cancelled by the user before it started.',
    }
