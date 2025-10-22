from __future__ import annotations

import importlib
import logging
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api._helpers import (
	coerce_section_f32,
	get_reader,
	get_section_from_pipeline_tap,
)
from app.api.schemas import PipelineSpec
from app.utils.ops import TRANSFORMS
from app.utils.raw_stats import per_trace_normalization
from app.utils.segy_meta import get_dt_for_file
from app.utils.utils import to_builtin

logger = logging.getLogger(__name__)

router = APIRouter()

INFERENCE_DDOF = 0


class InferencePredictRequest(BaseModel):
	file_id: str
	key1_val: int
	key1_byte: int = 189
	key2_byte: int = 193
	pipeline: PipelineSpec | None = None
	pipeline_key: str | None = None
	tap_label: str | None = None
	model: str = 'passthrough'
	step_x: int = 1
	step_y: int = 1


def _run_transforms(
	x: np.ndarray, *, spec: PipelineSpec | None, meta: dict[str, object]
) -> np.ndarray:
	if spec is None:
		return x

	y = x
	for step in spec.steps:
		if step.kind != 'transform':
			msg = 'Inference preprocessing only supports transform steps'
			raise HTTPException(status_code=422, detail=msg)
		op = TRANSFORMS.get(step.name)
		if op is None:
			msg = f'Unknown transform: {step.name}'
			raise HTTPException(status_code=422, detail=msg)
		y = op(y, params=step.params, meta=meta)

	y = np.ascontiguousarray(y, dtype=np.float32)
	if y.ndim != 2:
		msg = 'Transform output must remain 2D'
		raise HTTPException(status_code=500, detail=msg)
	return y


def _load_denoise_modules() -> tuple[object, object]:
	torch_spec = importlib.util.find_spec('torch')
	if torch_spec is None:
		raise HTTPException(status_code=503, detail='denoise model unavailable')

	torch = importlib.import_module('torch')
	denoise_mod_spec = importlib.util.find_spec('app.utils.denoise')
	if denoise_mod_spec is None:
		raise HTTPException(status_code=503, detail='denoise model unavailable')

	denoise_mod = importlib.import_module('app.utils.denoise')
	return torch, denoise_mod.denoise_tensor


def _run_model(name: str, data: np.ndarray) -> np.ndarray:
	if name == 'passthrough':
		return data
	if name == 'denoise':
		torch, denoise_tensor = _load_denoise_modules()
		tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
		out = denoise_tensor(tensor)
		return out.squeeze(0).squeeze(0).cpu().numpy()

	msg = f'Unsupported model: {name}'
	raise HTTPException(status_code=422, detail=msg)


def _resolve_section(
	*,
	req: InferencePredictRequest,
	reader,
) -> np.ndarray:
	if req.pipeline_key and req.tap_label:
		section = get_section_from_pipeline_tap(
			file_id=req.file_id,
			key1_val=req.key1_val,
			key1_byte=req.key1_byte,
			pipeline_key=req.pipeline_key,
			tap_label=req.tap_label,
		)
		return np.ascontiguousarray(section, dtype=np.float32)

	if (req.pipeline_key is None) ^ (req.tap_label is None):
		raise HTTPException(
			status_code=422,
			detail='pipeline_key and tap_label must be provided together',
		)

	view = reader.get_section(req.key1_val)
	return coerce_section_f32(view.arr, view.scale)


@router.post('/inference/predict')
def inference_predict(
	*,
	req: Annotated[InferencePredictRequest, Body(...)],
) -> JSONResponse:
	if req.step_x != 1 or req.step_y != 1:
		msg = 'step_x and step_y must be 1'
		raise HTTPException(status_code=400, detail=msg)

	reader = get_reader(req.file_id, req.key1_byte, req.key2_byte)
	dt_val = float(get_dt_for_file(req.file_id))
	meta = {'dt': dt_val}

	section = _resolve_section(req=req, reader=reader)
	section = np.ascontiguousarray(section, dtype=np.float32)
	if section.ndim != 2:
		msg = 'Inference section must be 2D'
		raise HTTPException(status_code=500, detail=msg)

	preprocessed = _run_transforms(section, spec=req.pipeline, meta=meta)

	normed, mu, sigma, zero_mask = per_trace_normalization(
		preprocessed,
		ddof=INFERENCE_DDOF,
	)

	pred_norm = _run_model(req.model, normed)
	if pred_norm.shape != normed.shape:
		msg = 'Model output shape mismatch'
		raise HTTPException(status_code=500, detail=msg)

	restored = mu[:, None] + sigma[:, None] * pred_norm.astype(np.float32)

	logger.info(
		'inference.predict model=%s key1=%s ddof=%s',
		req.model,
		req.key1_val,
		INFERENCE_DDOF,
	)

	payload = {
		'model': req.model,
		'dt': dt_val,
		'normalization': {'method': 'mean_std', 'ddof': INFERENCE_DDOF},
		'mu_traces': to_builtin(mu),
		'sigma_traces': to_builtin(sigma),
		'zero_var_mask': to_builtin(zero_mask.astype(np.bool_)),
		'prediction': to_builtin(restored.astype(np.float32)),
	}
	return JSONResponse(content=payload)


__all__ = ['router', 'InferencePredictRequest']
