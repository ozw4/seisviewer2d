"""Typed in-memory job table manager."""

from __future__ import annotations

import time
from typing import Literal, TypedDict

from app.core.settings import Settings
from app.services.pipeline_artifacts import pipeline_jobs_ttl_seconds

JobStatus = Literal[
    'queued',
    'running',
    'cancel_requested',
    'done',
    'error',
    'cancelled',
    'expired',
    'unknown',
]


class PipelineAllJobState(TypedDict):
    status: JobStatus
    progress: float
    message: str
    cancel_requested: bool
    created_ts: float
    finished_ts: float | None
    file_id: str
    key1_byte: int
    key2_byte: int
    pipeline_key: str
    offset_byte: int | None
    artifacts_dir: str


class BatchApplyJobState(TypedDict):
    status: JobStatus
    progress: float
    message: str
    cancel_requested: bool
    created_ts: float
    finished_ts: float | None
    file_id: str
    key1_byte: int
    key2_byte: int
    artifacts_dir: str
    job_type: Literal['batch_apply']


class StaticJobState(TypedDict):
    status: JobStatus
    progress: float
    message: str
    cancel_requested: bool
    created_ts: float
    finished_ts: float | None
    file_id: str
    key1_byte: int
    key2_byte: int
    artifacts_dir: str
    job_type: Literal['statics']
    statics_kind: str


class FbpickJobState(TypedDict):
    status: JobStatus
    message: str
    cancel_requested: bool
    created_ts: float
    finished_ts: float | None
    cache_key: object
    file_id: str
    key1: int
    key1_byte: int
    key2_byte: int
    pipeline_key: str | None
    tap_label: str | None
    offset_byte: int | None
    model_id: str
    channel: str


_MISSING = object()


class JobManager:
    """Mutable in-memory jobs table with dict-compatible accessors."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, object]] = {}

    def __getitem__(self, job_id: str) -> dict[str, object]:
        return self._jobs[job_id]

    def __setitem__(self, job_id: str, value: dict[str, object]) -> None:
        if not isinstance(value, dict):
            raise TypeError('job state must be a dict')
        self._jobs[job_id] = value

    def __iter__(self):
        return iter(self._jobs)

    def __len__(self) -> int:
        return len(self._jobs)

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._jobs

    def get(
        self,
        job_id: str,
        default: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        return self._jobs.get(job_id, default)

    def pop(
        self,
        job_id: str,
        default: object = _MISSING,
    ) -> dict[str, object] | object:
        if default is _MISSING:
            return self._jobs.pop(job_id)
        return self._jobs.pop(job_id, default)

    def items(self):
        return self._jobs.items()

    def values(self):
        return self._jobs.values()

    def clear(self) -> None:
        self._jobs.clear()

    @staticmethod
    def normalize_status_value(status: object) -> JobStatus:
        if not isinstance(status, str):
            return 'unknown'
        normalized = status.strip().lower()
        if normalized == 'completed':
            return 'done'
        if normalized == 'failed':
            return 'error'
        if normalized in {
            'queued',
            'running',
            'cancel_requested',
            'done',
            'error',
            'cancelled',
            'expired',
            'unknown',
        }:
            return normalized
        return 'unknown'

    @classmethod
    def is_terminal_status_value(cls, status: object) -> bool:
        return cls.normalize_status_value(status) in {
            'done',
            'error',
            'cancelled',
            'expired',
        }

    @classmethod
    def is_ready_status_value(cls, status: object) -> bool:
        return cls.normalize_status_value(status) == 'done'

    def create_pipeline_all_job(
        self,
        job_id: str,
        *,
        file_id: str,
        key1_byte: int,
        key2_byte: int,
        pipeline_key: str,
        offset_byte: int | None,
        artifacts_dir: str,
        created_ts: float | None = None,
    ) -> PipelineAllJobState:
        created = time.time() if created_ts is None else float(created_ts)
        job: PipelineAllJobState = {
            'status': 'queued',
            'progress': 0.0,
            'message': '',
            'cancel_requested': False,
            'created_ts': created,
            'finished_ts': None,
            'file_id': file_id,
            'key1_byte': int(key1_byte),
            'key2_byte': int(key2_byte),
            'pipeline_key': pipeline_key,
            'offset_byte': offset_byte,
            'artifacts_dir': artifacts_dir,
        }
        self._jobs[job_id] = job
        return job

    def create_batch_apply_job(
        self,
        job_id: str,
        *,
        file_id: str,
        key1_byte: int,
        key2_byte: int,
        artifacts_dir: str,
        created_ts: float | None = None,
    ) -> BatchApplyJobState:
        created = time.time() if created_ts is None else float(created_ts)
        job: BatchApplyJobState = {
            'status': 'queued',
            'progress': 0.0,
            'message': '',
            'cancel_requested': False,
            'created_ts': created,
            'finished_ts': None,
            'file_id': file_id,
            'key1_byte': int(key1_byte),
            'key2_byte': int(key2_byte),
            'artifacts_dir': artifacts_dir,
            'job_type': 'batch_apply',
        }
        self._jobs[job_id] = job
        return job

    def create_static_job(
        self,
        job_id: str,
        *,
        file_id: str,
        key1_byte: int,
        key2_byte: int,
        statics_kind: str,
        artifacts_dir: str,
        created_ts: float | None = None,
    ) -> StaticJobState:
        if not isinstance(file_id, str) or not file_id:
            raise ValueError('file_id must be a non-empty string')
        if not isinstance(statics_kind, str) or not statics_kind:
            raise ValueError('statics_kind must be a non-empty string')
        if not isinstance(artifacts_dir, str) or not artifacts_dir:
            raise ValueError('artifacts_dir must be a non-empty string')

        created = time.time() if created_ts is None else float(created_ts)
        job: StaticJobState = {
            'status': 'queued',
            'progress': 0.0,
            'message': '',
            'cancel_requested': False,
            'created_ts': created,
            'finished_ts': None,
            'file_id': file_id,
            'key1_byte': int(key1_byte),
            'key2_byte': int(key2_byte),
            'artifacts_dir': artifacts_dir,
            'job_type': 'statics',
            'statics_kind': statics_kind,
        }
        self._jobs[job_id] = job
        return job

    def create_fbpick_job(
        self,
        job_id: str,
        *,
        cache_key: object,
        file_id: str,
        key1: int,
        key1_byte: int,
        key2_byte: int,
        pipeline_key: str | None,
        tap_label: str | None,
        offset_byte: int | None,
        model_id: str,
        channel: str,
        created_ts: float | None = None,
    ) -> FbpickJobState:
        created = time.time() if created_ts is None else float(created_ts)
        job: FbpickJobState = {
            'status': 'queued',
            'message': '',
            'cancel_requested': False,
            'created_ts': created,
            'finished_ts': None,
            'cache_key': cache_key,
            'file_id': file_id,
            'key1': int(key1),
            'key1_byte': int(key1_byte),
            'key2_byte': int(key2_byte),
            'pipeline_key': pipeline_key,
            'tap_label': tap_label,
            'offset_byte': offset_byte,
            'model_id': model_id,
            'channel': channel,
        }
        self._jobs[job_id] = job
        return job

    def set_status(self, job_id: str, status: JobStatus) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job['status'] = status

    def set_progress(self, job_id: str, progress: float) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job['progress'] = float(progress)

    def set_message(self, job_id: str, message: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job['message'] = message

    def set_static_corrected_file(
        self,
        job_id: str,
        *,
        corrected_file_id: str,
        corrected_store_path: str,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job['corrected_file_id'] = corrected_file_id
        job['corrected_store_path'] = corrected_store_path

    def request_cancel(
        self,
        job_id: str,
        *,
        message: str = 'Cancel requested. The job will stop at the next safe point.',
        finished_ts: float | None = None,
    ) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        status_value = self.normalize_status_value(job.get('status'))
        if self.is_terminal_status_value(status_value):
            return False
        job['cancel_requested'] = True
        if status_value == 'queued':
            job['status'] = 'cancelled'
            job['message'] = 'The job was cancelled by the user before it started.'
            job['finished_ts'] = (
                time.time() if finished_ts is None else float(finished_ts)
            )
            return True
        job['status'] = 'cancel_requested'
        job['message'] = message
        return True

    def is_cancel_requested(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        return bool(job.get('cancel_requested'))

    def mark_done(
        self,
        job_id: str,
        *,
        finished_ts: float | None = None,
        progress_1: bool = False,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job['status'] = 'done'
        job['cancel_requested'] = False
        job['message'] = ''
        if progress_1:
            job['progress'] = 1.0
        job['finished_ts'] = time.time() if finished_ts is None else float(finished_ts)

    def mark_error(
        self,
        job_id: str,
        message: str,
        *,
        finished_ts: float | None = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job['status'] = 'error'
        job['cancel_requested'] = False
        job['message'] = message
        job['finished_ts'] = time.time() if finished_ts is None else float(finished_ts)

    def mark_cancelled(
        self,
        job_id: str,
        message: str,
        *,
        finished_ts: float | None = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job['status'] = 'cancelled'
        job['cancel_requested'] = True
        job['message'] = message
        job['finished_ts'] = time.time() if finished_ts is None else float(finished_ts)

    def mark_expired(
        self,
        job_id: str,
        *,
        finished_ts: float | None = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job['status'] = 'expired'
        job['cancel_requested'] = False
        job['finished_ts'] = time.time() if finished_ts is None else float(finished_ts)

    @staticmethod
    def _job_timestamp(job: dict[str, object], field: str) -> float | None:
        value = job.get(field)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @classmethod
    def _job_created_ts(cls, job: dict[str, object]) -> float:
        created_ts = cls._job_timestamp(job, 'created_ts')
        if created_ts is None:
            return 0.0
        return created_ts

    @classmethod
    def _job_expiry_ts(cls, job: dict[str, object], now_ts: float) -> float:
        finished_ts = cls._job_timestamp(job, 'finished_ts')
        if finished_ts is not None:
            return finished_ts
        created_ts = cls._job_timestamp(job, 'created_ts')
        if created_ts is not None:
            return created_ts
        return now_ts

    @staticmethod
    def _job_kind(job: dict[str, object]) -> str:
        if job.get('job_type') == 'batch_apply':
            return 'pipeline'
        if job.get('job_type') == 'statics':
            return 'pipeline'
        if 'cache_key' in job:
            return 'fbpick'
        if 'pipeline_key' in job:
            return 'pipeline'
        return 'other'

    def cleanup_in_memory(self, *, now_ts: float, settings: Settings) -> None:
        fbpick_ttl_sec = settings.sv_fbpick_job_ttl_sec
        jobs_max = settings.sv_jobs_max
        pipeline_ttl_sec = pipeline_jobs_ttl_seconds(settings=settings)

        expired_job_ids: list[str] = []
        for job_id, raw_job in self._jobs.items():
            if not isinstance(raw_job, dict):
                continue
            status_value = self.normalize_status_value(raw_job.get('status'))
            kind = self._job_kind(raw_job)

            ttl_sec: int | None = None
            if kind == 'fbpick' and status_value in {
                'done',
                'error',
                'cancelled',
                'expired',
            }:
                ttl_sec = fbpick_ttl_sec
            elif kind == 'pipeline' and status_value in {
                'done',
                'error',
                'cancelled',
            }:
                ttl_sec = pipeline_ttl_sec

            if ttl_sec is None:
                continue

            expiry_base_ts = self._job_expiry_ts(raw_job, now_ts)
            if now_ts - expiry_base_ts > float(ttl_sec):
                expired_job_ids.append(job_id)

        for job_id in expired_job_ids:
            self._jobs.pop(job_id, None)

        overflow = len(self._jobs) - jobs_max
        if overflow <= 0:
            return

        oldest_first = sorted(
            self._jobs.items(),
            key=lambda item: self._job_created_ts(item[1]),
        )
        for job_id, _ in oldest_first[:overflow]:
            self._jobs.pop(job_id, None)


__all__ = [
    'BatchApplyJobState',
    'FbpickJobState',
    'JobManager',
    'JobStatus',
    'PipelineAllJobState',
    'StaticJobState',
]
