"""Background runner for batch apply jobs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from app.api.schemas import BatchApplyRequest, PickOptions
from app.core.state import AppState
from app.services.fbpick_predict_math import (
    apply_sigma_gate,
    expectation_idx_and_sigma_ms,
    pick_index_from_prob,
    sigma_ms_from_prob,
)
from app.services import reader as _reader_service
from app.services.job_runner import (
    JobCompletion,
    JobFailure,
    run_job_with_lifecycle,
    set_job_progress,
)
from app.services.pipeline_artifacts import get_job_dir
from app.services.pipeline_execution import (
    SectionSourceSpec,
    extract_pipeline_outputs,
    prepare_pipeline_execution,
    resolve_effective_offset_byte,
    resolve_execution_dt,
)
from app.services.reader import get_reader
from app.utils.manual_pick_csr import empty_csr, picks_time_s_to_csr
from app.utils.pick_snap import parabolic_refine, snap_pick_time_s
from app.utils.pipeline import apply_pipeline
from app.trace_store.reader import TraceStoreSectionReader

coerce_section_f32 = _reader_service.coerce_section_f32


def _write_job_meta(*, job_dir: Path, payload: dict[str, object]) -> Path:
    """Write ``job_meta.json`` directly under the job directory."""
    job_meta_path = job_dir / 'job_meta.json'
    job_meta_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True),
        encoding='utf-8',
    )
    return job_meta_path


def _denoise_tap_labels(spec) -> list[str]:
    lineage: list[str] = []
    taps: list[str] = []
    for step in spec.steps:
        if step.kind != 'transform':
            continue
        lineage.append(step.label or step.name)
        if step.name == 'denoise':
            taps.append('+'.join(lineage))
    return taps


def _fbpick_label(spec) -> str | None:
    for step in spec.steps:
        if step.kind == 'analyzer' and step.name == 'fbpick':
            return step.label or step.name
    return None


def _load_key2_values(
    *, reader: TraceStoreSectionReader, key1: int, key2_byte: int
) -> np.ndarray:
    trace_seq = reader.get_trace_seq_for_value(int(key1), align_to='display')
    key2s_all = reader.get_header(int(key2_byte))
    return np.asarray(key2s_all[trace_seq], dtype=np.int32)


def _load_section_and_key2(
    *, reader: TraceStoreSectionReader, key1: int, key2_byte: int
) -> tuple[np.ndarray, np.ndarray]:
    view = reader.get_section(int(key1))
    section = coerce_section_f32(view.arr, view.scale)
    key2_vals = _load_key2_values(reader=reader, key1=key1, key2_byte=key2_byte)
    return section, key2_vals


_ORIGINAL_LOAD_SECTION_AND_KEY2 = _load_section_and_key2


def _resolve_section_and_key2(
    *,
    context,
    reader: TraceStoreSectionReader,
    key1: int,
    key2_byte: int,
) -> tuple[np.ndarray, np.ndarray]:
    if _load_section_and_key2 is _ORIGINAL_LOAD_SECTION_AND_KEY2:
        return context.section, _load_key2_values(
            reader=reader,
            key1=key1,
            key2_byte=key2_byte,
        )
    return _load_section_and_key2(reader=reader, key1=key1, key2_byte=key2_byte)


def _run_pipeline_outputs(
    *,
    section: np.ndarray,
    meta: dict[str, object],
    spec,
    denoise_taps: list[str],
    fbpick_label: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    outputs = apply_pipeline(section, spec=spec, meta=meta, taps=denoise_taps)
    return extract_pipeline_outputs(
        outputs,
        denoise_taps=denoise_taps,
        fbpick_label=fbpick_label,
    )


def _predict_section_picks_time_s(
    *,
    prob: np.ndarray,
    raw_section: np.ndarray,
    dt: float,
    pick_options: PickOptions,
    chunk: int = 4096,
) -> np.ndarray:
    if prob.ndim != 2:
        raise ValueError('Probability map must be 2D')
    if raw_section.shape != prob.shape:
        raise ValueError('Raw section and probability shape mismatch')
    if dt <= 0:
        raise ValueError('dt must be > 0')

    if pick_options.method == 'expectation':
        idx, sigma_ms = expectation_idx_and_sigma_ms(prob, dt=dt, chunk=chunk)
    elif pick_options.method == 'argmax':
        idx = pick_index_from_prob(prob, method='argmax', chunk=chunk)
        if pick_options.subsample:
            n_samples = int(prob.shape[1])
            idx = np.asarray(idx, dtype=np.float64).copy()
            for t in range(int(prob.shape[0])):
                if not np.isfinite(idx[t]):
                    continue
                ii = int(idx[t])
                if 1 <= ii <= (n_samples - 2):
                    idx[t] = parabolic_refine(prob[t], ii)
        sigma_ms = sigma_ms_from_prob(prob, dt=dt, chunk=chunk)
    else:
        raise ValueError(f'Unsupported pick method: {pick_options.method}')

    gated_idx = apply_sigma_gate(
        idx,
        sigma_ms,
        sigma_ms_max=pick_options.sigma_ms_max,
    )
    times_s = np.asarray(gated_idx, dtype=np.float64) * float(dt)

    if pick_options.snap.enabled:
        for t in range(int(times_s.shape[0])):
            if not np.isfinite(times_s[t]):
                continue
            times_s[t] = snap_pick_time_s(
                raw_section[t],
                float(times_s[t]),
                dt=float(dt),
                mode=pick_options.snap.mode,
                refine=pick_options.snap.refine,
                window_ms=float(pick_options.snap.window_ms),
            )

    return times_s


def _manual_pick_npz_payload(
    *,
    picks_time_s: np.ndarray,
    n_samples: int,
    dt: float,
    source_hint: str,
) -> dict[str, object]:
    picks_time_s_f32 = np.asarray(picks_time_s, dtype=np.float32)
    n_traces = int(picks_time_s_f32.shape[0])
    p_indptr, p_data = picks_time_s_to_csr(
        picks_time_s_f32,
        dt=float(dt),
        n_samples=int(n_samples),
    )
    s_indptr, s_data = empty_csr(n_traces)
    return {
        'manual_pick_format': np.asarray('seisai_csr'),
        'picks_time_s': picks_time_s_f32,
        'n_traces': np.int64(n_traces),
        'p_indptr': p_indptr,
        'p_data': p_data,
        's_indptr': s_indptr,
        's_data': s_data,
        'n_samples': np.int64(n_samples),
        'dt': np.float64(dt),
        'format_version': np.int64(1),
        'exported_at': np.asarray(datetime.now(timezone.utc).isoformat()),
        'export_app': np.asarray('seisviewer2d'),
        'source_hint': np.asarray(source_hint),
    }


@dataclass
class _BatchApplyLifecycle:
    created_ts: float
    job_dir: Path | None = None


def _resolve_created_ts(state: AppState, job_id: str) -> float:
    with state.lock:
        job = state.jobs.get(job_id)
        if job is None:
            return time.time()
        created_ts_obj = job.get('created_ts')
    return (
        float(created_ts_obj)
        if isinstance(created_ts_obj, (int, float))
        else time.time()
    )


def _run_batch_apply_job_body(
    job_id: str,
    req: BatchApplyRequest,
    state: AppState,
    *,
    lifecycle: _BatchApplyLifecycle,
) -> JobCompletion | None:
    denoise_taps = _denoise_tap_labels(req.pipeline_spec)
    fbpick_label = _fbpick_label(req.pipeline_spec)
    if not denoise_taps and fbpick_label is None:
        raise ValueError('No supported outputs in pipeline spec')

    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.job_dir = job_dir
    started_meta: dict[str, object] = {
        'job_id': job_id,
        'status': 'running',
        'created_ts': lifecycle.created_ts,
        'request': req.model_dump(mode='json'),
    }
    _write_job_meta(job_dir=job_dir, payload=started_meta)

    reader = get_reader(req.file_id, req.key1_byte, req.key2_byte, state=state)
    key1_values = np.asarray(reader.get_key1_values(), dtype=np.int32)
    n_key1 = int(key1_values.shape[0])
    total = n_key1 if n_key1 > 0 else 1
    n_samples = int(reader.get_n_samples())
    save_picks_enabled = bool(req.save_picks and fbpick_label is not None)
    sorted_to_original: np.ndarray | None = None
    n_traces_total = 0
    picks_sorted: np.ndarray | None = None
    if save_picks_enabled:
        sorted_to_original = np.asarray(reader.get_sorted_to_original(), dtype=np.int64)
        n_traces_total = int(sorted_to_original.shape[0])
        if n_traces_total > 0:
            if (
                int(sorted_to_original.min()) < 0
                or int(sorted_to_original.max()) >= n_traces_total
            ):
                raise ValueError('sorted_to_original out of range')
        picks_sorted = np.full((n_traces_total,), np.nan, dtype=np.float64)

    max_traces = 0
    for key1 in key1_values:
        view = reader.get_section(int(key1))
        n_traces = int(view.arr.shape[0])
        if n_traces > max_traces:
            max_traces = n_traces

    np.save(job_dir / 'key1_values.npy', key1_values.astype(np.int32, copy=False))

    key2_values_padded = open_memmap(
        job_dir / 'key2_values_padded.npy',
        mode='w+',
        dtype=np.int32,
        shape=(n_key1, max_traces),
    )
    key2_values_padded.fill(0)

    denoise_padded = None
    if denoise_taps:
        denoise_padded = open_memmap(
            job_dir / 'denoise_f32_padded.npy',
            mode='w+',
            dtype=np.float32,
            shape=(n_key1, max_traces, n_samples),
        )
        denoise_padded.fill(0)

    prob_padded = None
    if fbpick_label is not None:
        prob_padded = open_memmap(
            job_dir / 'fbpick_prob_f16_padded.npy',
            mode='w+',
            dtype=np.float16,
            shape=(n_key1, max_traces, n_samples),
        )
        prob_padded.fill(0)

    dt = resolve_execution_dt(req.file_id, None, state=state)
    forced_offset_byte = resolve_effective_offset_byte(req.pipeline_spec, None)

    for i, key1 in enumerate(key1_values):
        key1_int = int(key1)
        context = prepare_pipeline_execution(
            spec=req.pipeline_spec,
            source=SectionSourceSpec(
                file_id=req.file_id,
                key1=key1_int,
                key1_byte=req.key1_byte,
                key2_byte=req.key2_byte,
                offset_byte=forced_offset_byte,
            ),
            state=state,
            reader=reader,
        )
        section, key2_vals = _resolve_section_and_key2(
            context=context,
            reader=reader,
            key1=key1_int,
            key2_byte=req.key2_byte,
        )
        n_traces = int(section.shape[0])
        if int(section.shape[1]) != n_samples:
            raise ValueError('Section sample count mismatch')
        if int(key2_vals.shape[0]) != n_traces:
            raise ValueError('key2 length does not match section trace count')

        key2_values_padded[i, :n_traces] = key2_vals

        dt = context.dt
        denoise_out, prob_out = _run_pipeline_outputs(
            section=section,
            meta=context.meta,
            spec=req.pipeline_spec,
            denoise_taps=denoise_taps,
            fbpick_label=fbpick_label,
        )

        if denoise_padded is not None:
            if denoise_out is None:
                raise ValueError('Denoise output is missing')
            if denoise_out.shape != section.shape:
                raise ValueError('Denoise output shape mismatch')
            denoise_padded[i, :n_traces, :] = denoise_out

        if prob_padded is not None:
            if prob_out is None:
                raise ValueError('fbpick prob output is missing')
            if prob_out.shape != section.shape:
                raise ValueError('fbpick prob shape mismatch')
            prob_padded[i, :n_traces, :] = prob_out.astype(np.float16, copy=False)
        if save_picks_enabled:
            if prob_out is None:
                raise ValueError('fbpick prob output is missing for pick export')
            if picks_sorted is None:
                raise ValueError('picks_sorted is not initialized')
            trace_seq = np.asarray(
                reader.get_trace_seq_for_value(key1_int, align_to='display'),
                dtype=np.int64,
            )
            if trace_seq.shape != (n_traces,):
                raise ValueError('trace_seq length mismatch')
            if trace_seq.size:
                if int(trace_seq.min()) < 0 or int(trace_seq.max()) >= n_traces_total:
                    raise ValueError('trace_seq out of range')
            times_s = _predict_section_picks_time_s(
                prob=prob_out,
                raw_section=section,
                dt=dt,
                pick_options=req.pick_options,
            )
            if times_s.shape != (n_traces,):
                raise ValueError('Predicted picks length mismatch')
            picks_sorted[trace_seq] = times_s

        if not set_job_progress(state, job_id, float(i + 1) / float(total)):
            return None

    key2_values_padded.flush()
    if denoise_padded is not None:
        denoise_padded.flush()
    if prob_padded is not None:
        prob_padded.flush()
    if save_picks_enabled:
        if sorted_to_original is None or picks_sorted is None:
            raise ValueError('Predicted picks buffers are missing')
        picks_original = np.full((n_traces_total,), np.nan, dtype=np.float32)
        picks_original[sorted_to_original] = picks_sorted.astype(np.float32, copy=False)
        np.savez(
            job_dir / 'predicted_picks_time_s.npz',
            **_manual_pick_npz_payload(
                picks_time_s=picks_original,
                n_samples=n_samples,
                dt=dt,
                source_hint='batch_predicted_picks',
            ),
        )

    finished_ts = time.time()
    output_files: list[str] = ['key1_values.npy', 'key2_values_padded.npy']
    if denoise_taps:
        output_files.append('denoise_f32_padded.npy')
    if fbpick_label is not None:
        output_files.append('fbpick_prob_f16_padded.npy')
    if save_picks_enabled:
        output_files.append('predicted_picks_time_s.npz')
    finished_meta: dict[str, object] = {
        'job_id': job_id,
        'status': 'done',
        'created_ts': lifecycle.created_ts,
        'finished_ts': finished_ts,
        'request': req.model_dump(mode='json'),
        'n_key1': n_key1,
        'max_traces': max_traces,
        'n_samples': n_samples,
        'outputs': output_files,
    }
    _write_job_meta(job_dir=job_dir, payload=finished_meta)
    return JobCompletion(finished_ts=finished_ts)


def _handle_batch_apply_job_error(
    job_id: str,
    req: BatchApplyRequest,
    *,
    lifecycle: _BatchApplyLifecycle,
    exc: Exception,
) -> JobFailure:
    finished_ts = time.time()
    if lifecycle.job_dir is not None:
        error_meta: dict[str, object] = {
            'job_id': job_id,
            'status': 'error',
            'created_ts': lifecycle.created_ts,
            'finished_ts': finished_ts,
            'request': req.model_dump(mode='json'),
            'error': str(exc),
        }
        _write_job_meta(job_dir=lifecycle.job_dir, payload=error_meta)
    return JobFailure(finished_ts=finished_ts)


def run_batch_apply_job(job_id: str, req: BatchApplyRequest, state: AppState) -> None:
    """Run one batch job over all key1 sections and persist padded outputs."""
    lifecycle = _BatchApplyLifecycle(created_ts=_resolve_created_ts(state, job_id))
    run_job_with_lifecycle(
        state=state,
        job_id=job_id,
        worker=lambda: _run_batch_apply_job_body(
            job_id,
            req,
            state,
            lifecycle=lifecycle,
        ),
        progress_1_on_done=True,
        start_progress=0.0,
        clear_message_on_start=True,
        on_error=lambda exc: _handle_batch_apply_job_error(
            job_id,
            req,
            lifecycle=lifecycle,
            exc=exc,
        ),
    )


__all__ = ['run_batch_apply_job']
