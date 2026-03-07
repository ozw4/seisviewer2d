"""Server-side FB picking endpoint returning pick positions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api._helpers import get_state
from app.core.state import AppState
from app.services.fbpick_predict_math import (
    apply_sigma_gate,
    expectation_idx_and_sigma_ms,
    pick_index_from_prob,
    sigma_ms_from_prob,
)
from app.services.fbpick_support import (
    DEFAULT_FBPICK_MODEL_ID,
    _maybe_attach_fbpick_offsets,
)
from app.services.pipeline_execution import (
    SectionSourceSpec,
    build_fbpick_spec,
    extract_fbpick_probability_map,
    prepare_pipeline_execution,
    resolve_effective_offset_byte,
    run_pipeline_execution,
)
from app.services.reader import get_reader
from app.utils.fbpick_models import model_version, resolve_model_path
from app.utils.pipeline import apply_pipeline

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
    model_config = ConfigDict(extra='forbid')

    file_id: str
    key1: int
    key1_byte: int = 189
    key2_byte: int = 193
    pipeline_key: str | None = None
    tap_label: str | None = None
    model_id: str | None = None
    channel: str | None = None
    method: Literal['argmax', 'expectation'] = 'argmax'
    sigma_ms_max: float = Field(
        gt=0, description='Standard deviation gate in milliseconds'
    )


@dataclass
class _ModelSelection:
    model_id: str
    model_path: Path
    model_ver: str
    uses_offset: bool


def _resolve_model_selection(model_id: str | None) -> _ModelSelection:
    path = resolve_model_path(model_id, require_exists=True)
    chosen_id = DEFAULT_FBPICK_MODEL_ID if model_id is None else model_id
    return _ModelSelection(
        model_id=chosen_id,
        model_path=path,
        model_ver=model_version(path),
        uses_offset='offset' in chosen_id.lower(),
    )


@dataclass
class _ProbabilityPayload:
    prob: np.ndarray
    dt: float
    source: str


def _effective_channel(channel: str | None) -> str:
    return 'P' if channel is None else channel


def _compute_probability_map(
    req: FbpickPredictRequest,
    *,
    state: AppState,
    model_sel: _ModelSelection,
) -> _ProbabilityPayload:
    pipeline_key = req.pipeline_key
    tap_label = req.tap_label
    channel = _effective_channel(req.channel)
    if (pipeline_key is None) ^ (tap_label is None):
        raise HTTPException(
            status_code=422,
            detail='pipeline_key and tap_label must be provided together',
        )

    spec = build_fbpick_spec(
        model_id=model_sel.model_id,
        channel=channel,
        tile=_DEFAULT_TILE,
        overlap=_DEFAULT_OVERLAP,
        amp=_DEFAULT_AMP,
    )
    context = prepare_pipeline_execution(
        spec=spec,
        source=SectionSourceSpec(
            file_id=req.file_id,
            key1=req.key1,
            key1_byte=req.key1_byte,
            key2_byte=req.key2_byte,
            pipeline_key=pipeline_key,
            tap_label=tap_label,
        ),
        state=state,
        validate_source_dt=True,
        reader_getter=get_reader,
        offset_attacher=_maybe_attach_fbpick_offsets,
    )
    out = run_pipeline_execution(context, taps=None, apply_fn=apply_pipeline)
    prob = extract_fbpick_probability_map(out, label='fbpick')
    source = (
        f'pipeline:{tap_label}'
        if context.source_kind == 'pipeline_tap' and tap_label is not None
        else 'raw'
    )
    return _ProbabilityPayload(prob=prob, dt=context.dt, source=source)


def _load_probability_map(
    req: FbpickPredictRequest, *, state: AppState
) -> _ProbabilityPayload:
    model_sel = _resolve_model_selection(req.model_id)
    channel = _effective_channel(req.channel)
    spec = build_fbpick_spec(
        model_id=model_sel.model_id,
        channel=channel,
        tile=_DEFAULT_TILE,
        overlap=_DEFAULT_OVERLAP,
        amp=_DEFAULT_AMP,
    )
    forced_offset_byte = resolve_effective_offset_byte(spec, None)
    key = (
        req.file_id,
        req.key1,
        req.pipeline_key,
        req.tap_label,
        model_sel.model_id,
        model_sel.model_ver,
        int(_DEFAULT_TILE[0]),
        int(_DEFAULT_TILE[1]),
        int(_DEFAULT_OVERLAP),
        bool(_DEFAULT_AMP),
        forced_offset_byte,
        channel,
    )
    if _last_prob_state.key == key and _last_prob_state.value is not None:
        return _ProbabilityPayload(
            prob=_last_prob_state.value,
            dt=float(_last_prob_state.dt),
            source=_last_prob_state.source or 'unknown',
        )
    payload = _compute_probability_map(req, state=state, model_sel=model_sel)
    _last_prob_state.key = key
    _last_prob_state.value = payload.prob
    _last_prob_state.dt = payload.dt
    _last_prob_state.source = payload.source
    return payload


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
    sums = np.sum(prob, axis=1, dtype=np.float64)
    # 全トレースが「質量ゼロ」なら API として無効扱いで 422
    if np.all(sums <= 0.0):
        raise HTTPException(
            status_code=422, detail='Probability mass is zero for a trace'
        )
    try:
        if method.lower() == 'expectation':
            idx, sigma_ms = expectation_idx_and_sigma_ms(prob, dt=dt, chunk=_CHUNK_SIZE)
        else:
            idx = pick_index_from_prob(prob, method=method, chunk=_CHUNK_SIZE)
            sigma_ms = sigma_ms_from_prob(prob, dt=dt, chunk=_CHUNK_SIZE)
        gated_idx = apply_sigma_gate(idx, sigma_ms, sigma_ms_max=sigma_ms_max)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    accepted = int(np.count_nonzero(np.isfinite(gated_idx)))
    times = gated_idx * dt
    picks: list[dict[str, float]] = []
    for trace_idx in range(n_traces):
        if np.isfinite(gated_idx[trace_idx]):
            picks.append({'trace': int(trace_idx), 'time': float(times[trace_idx])})
    accepted_ratio = accepted / n_traces if n_traces else 0.0
    return picks, accepted_ratio


@router.post('/fbpick_predict')
def fbpick_predict(req: FbpickPredictRequest, request: Request) -> dict[str, Any]:
    state = get_state(request.app)
    payload = _load_probability_map(req, state=state)
    picks, accepted_ratio = _compute_picks(
        payload.prob,
        payload.dt,
        req.method,
        req.sigma_ms_max,
    )
    logger.info(
        'fbpick_predict file_id=%s key1=%d method=%s sigma_ms_max=%.2f dt=%.6f accepted_ratio=%.3f source=%s',
        req.file_id,
        req.key1,
        req.method,
        req.sigma_ms_max,
        payload.dt,
        accepted_ratio,
        payload.source,
    )
    return {'dt': payload.dt, 'picks': picks}
