"""Server-side FB picking endpoint returning pick positions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api._helpers import (
	OFFSET_BYTE_FIXED,
	USE_FBPICK_OFFSET,
	PipelineTapNotFoundError,
	_maybe_attach_fbpick_offsets,
	coerce_section_f32,
	get_reader,
	get_section_and_meta_from_pipeline_tap,
)
from app.api.schemas import PipelineSpec
from app.utils.fbpick import _MODEL_PATH as FBPICK_MODEL_PATH
from app.utils.pipeline import apply_pipeline
from app.utils.segy_meta import get_dt_for_file

logger = logging.getLogger(__name__)

router = APIRouter()

_CHUNK_SIZE = 4096
_DEFAULT_TILE = (128, 6016)
_DEFAULT_OVERLAP = 32
_DEFAULT_AMP = True


@dataclass
class _LastProbabilityState:
	key: tuple | None = None
	value: np.ndarray | None = None
	dt: float | None = None
	source: str | None = None


_last_prob_state = _LastProbabilityState()


class FbpickPredictRequest(BaseModel):
	file_id: str
	key1_val: int
	key1_byte: int = 189
	key2_byte: int = 193
	pipeline_key: str | None = None
	tap_label: str | None = None
	method: Literal['argmax', 'expectation'] = 'argmax'
	sigma_ms_max: float = Field(
		gt=0, description='Standard deviation gate in milliseconds'
	)


def _model_version() -> str:
	path = Path(FBPICK_MODEL_PATH)
	if not path.exists():
		return 'missing'
	try:
		stat = path.stat()
	except OSError:
		return path.name
	return f'{path.name}:{stat.st_mtime_ns}'


@dataclass
class _ProbabilityPayload:
	prob: np.ndarray
	dt: float
	source: str


def _resolve_dt(file_id: str, meta: dict[str, Any] | None) -> float:
	dt_file = float(get_dt_for_file(file_id))
	dt = dt_file
	if isinstance(meta, dict) and 'dt' in meta:
		dt_meta = meta.get('dt')
		if not isinstance(dt_meta, (int, float)):
			raise HTTPException(status_code=422, detail='Invalid dt metadata')
		if dt_meta <= 0:
			raise HTTPException(status_code=422, detail='Non-positive dt metadata')
		dt_meta_f = float(dt_meta)
		if abs(dt_meta_f - dt_file) > 1e-9:
			raise HTTPException(
				status_code=409, detail='dt mismatch between tap and source'
			)
		dt = dt_meta_f
	if dt <= 0:
		raise HTTPException(status_code=422, detail='Non-positive dt value')
	return float(dt)


def _compute_probability_map(req: FbpickPredictRequest) -> _ProbabilityPayload:
	if not Path(FBPICK_MODEL_PATH).exists():
		raise HTTPException(status_code=409, detail='FB pick model weights not found')

	pipeline_key = req.pipeline_key
	tap_label = req.tap_label
	if (pipeline_key is None) ^ (tap_label is None):
		raise HTTPException(
			status_code=422,
			detail='pipeline_key and tap_label must be provided together',
		)

	forced_offset_byte = OFFSET_BYTE_FIXED if USE_FBPICK_OFFSET else None
	reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)

	meta_source = getattr(reader, 'meta', None)
	dt = _resolve_dt(
		req.file_id, meta_source if isinstance(meta_source, dict) else None
	)

	if pipeline_key and tap_label:
		try:
			section, tap_meta = get_section_and_meta_from_pipeline_tap(
				file_id=req.file_id,
				key1_val=req.key1_val,
				key1_byte=req.key1_byte,
				pipeline_key=pipeline_key,
				tap_label=tap_label,
				offset_byte=forced_offset_byte if USE_FBPICK_OFFSET else None,
			)
		except PipelineTapNotFoundError as exc:
			raise HTTPException(status_code=404, detail=str(exc)) from exc
		meta_for_dt = tap_meta if isinstance(tap_meta, dict) else None
		dt = _resolve_dt(req.file_id, meta_for_dt)
		source = f'pipeline:{tap_label}'
	else:
		view = reader.get_section(req.key1_val)
		section = coerce_section_f32(view.arr, view.scale)
		source = 'raw'

	section = np.ascontiguousarray(section, dtype=np.float32)
	if section.ndim != 2:
		raise HTTPException(status_code=422, detail='Section must be 2D')

	spec = PipelineSpec(
		steps=[
			{
				'kind': 'analyzer',
				'name': 'fbpick',
				'params': {
					'tile': _DEFAULT_TILE,
					'overlap': _DEFAULT_OVERLAP,
					'amp': _DEFAULT_AMP,
				},
			}
		]
	)

	meta: dict[str, Any] = {}
	if reader is not None:
		meta = _maybe_attach_fbpick_offsets(
			meta,
			spec=spec,
			reader=reader,
			key1_val=req.key1_val,
			offset_byte=forced_offset_byte if USE_FBPICK_OFFSET else None,
		)

	out = apply_pipeline(section, spec=spec, meta=meta, taps=None)
	fbpick_out = out.get('fbpick') or out.get('final')
	if not isinstance(fbpick_out, dict):
		raise HTTPException(status_code=500, detail='fbpick analyzer output missing')
	prob = np.asarray(fbpick_out.get('prob'), dtype=np.float32)
	if prob.ndim != 2:
		raise HTTPException(status_code=422, detail='Probability map must be 2D')

	return _ProbabilityPayload(prob=prob, dt=dt, source=source)


def _load_probability_map(req: FbpickPredictRequest) -> _ProbabilityPayload:
	key = (req.file_id, req.key1_val, req.pipeline_key, req.tap_label, _model_version())
	if _last_prob_state.key == key and _last_prob_state.value is not None:
		return _ProbabilityPayload(
			prob=_last_prob_state.value,
			dt=float(_last_prob_state.dt),
			source=_last_prob_state.source or 'unknown',
		)
	payload = _compute_probability_map(req)
	_last_prob_state.key = key
	_last_prob_state.value = payload.prob
	_last_prob_state.dt = payload.dt
	_last_prob_state.source = payload.source
	return payload


def _chunked_expectations(
	prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	n_traces, n_samples = prob.shape
	indices = np.arange(n_samples, dtype=np.float64)
	indices_sq = indices * indices
	sums = np.empty(n_traces, dtype=np.float64)
	sum_i = np.empty(n_traces, dtype=np.float64)
	sum_i2 = np.empty(n_traces, dtype=np.float64)
	for start in range(0, n_traces, _CHUNK_SIZE):
		end = min(start + _CHUNK_SIZE, n_traces)
		chunk = prob[start:end]
		if not np.all(np.isfinite(chunk)):
			raise HTTPException(
				status_code=422, detail='Probability map contains NaN or Inf'
			)
		sums[start:end] = np.sum(chunk, axis=1, dtype=np.float64)
		sum_i[start:end] = np.einsum(
			'ij,j->i', chunk, indices, dtype=np.float64, optimize=True
		)
		sum_i2[start:end] = np.einsum(
			'ij,j->i', chunk, indices_sq, dtype=np.float64, optimize=True
		)
	return sums, sum_i, sum_i2


def _compute_picks(
	prob: np.ndarray,
	dt: float,
	method: str,
	sigma_ms_max: float,
) -> tuple[list[dict[str, float]], float]:
	if prob.ndim != 2:
		raise HTTPException(status_code=422, detail='Probability map must be 2D')
	n_traces, n_samples = prob.shape
	if n_traces == 0 or n_samples == 0:
		return [], 0.0
	method_norm = method.lower()
	if method_norm not in {'argmax', 'expectation'}:
		raise HTTPException(status_code=422, detail='Unsupported method')
	sums, sum_i, sum_i2 = _chunked_expectations(prob)
	if not np.all(np.isfinite(sums)):
		raise HTTPException(status_code=422, detail='Probability sum invalid')
	if np.any(sums <= 0):
		raise HTTPException(
			status_code=422, detail='Probability mass is zero for a trace'
		)
	mu = sum_i / sums
	second_moment = sum_i2 / sums
	var = np.maximum(second_moment - mu * mu, 0.0)
	if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(var)):
		raise HTTPException(status_code=422, detail='Expectation calculation failed')
	sigma = np.sqrt(var)
	dt_ms = dt * 1000.0
	sigma_ms = sigma * dt_ms
	mask = sigma_ms <= sigma_ms_max
	accepted = int(np.count_nonzero(mask))
	if method_norm == 'expectation':
		idx = mu
	else:
		idx = np.empty(n_traces, dtype=np.float64)
		for start in range(0, n_traces, _CHUNK_SIZE):
			end = min(start + _CHUNK_SIZE, n_traces)
			chunk = prob[start:end]
			idx[start:end] = np.argmax(chunk, axis=1, keepdims=False)
	times = idx * dt
	picks: list[dict[str, float]] = []
	for trace_idx, keep in enumerate(mask):
		if keep:
			picks.append({'trace': int(trace_idx), 'time': float(times[trace_idx])})
	accepted_ratio = accepted / n_traces if n_traces else 0.0
	return picks, accepted_ratio


@router.post('/fbpick_predict')
def fbpick_predict(req: FbpickPredictRequest) -> dict[str, Any]:
	payload = _load_probability_map(req)
	picks, accepted_ratio = _compute_picks(
		payload.prob,
		payload.dt,
		req.method,
		req.sigma_ms_max,
	)
	logger.info(
		'fbpick_predict file_id=%s key1=%d method=%s sigma_ms_max=%.2f dt=%.6f accepted_ratio=%.3f source=%s',
		req.file_id,
		req.key1_val,
		req.method,
		req.sigma_ms_max,
		payload.dt,
		accepted_ratio,
		payload.source,
	)
	return {'dt': payload.dt, 'picks': picks}
