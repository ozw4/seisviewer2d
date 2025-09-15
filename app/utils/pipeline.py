"""Pipeline execution utilities."""

import hashlib
import json
from typing import Any

import numpy as np

from app.api.schemas import PipelineSpec

from .ops import ANALYZERS, TRANSFORMS


def _hash_params(obj: object) -> str:
	"""Return a short, stable hash for parameter dictionaries."""
	raw = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
	return hashlib.sha256(raw).hexdigest()[:12]


def pipeline_key(spec: PipelineSpec) -> str:
	"""Generate a stable key representing the pipeline specification."""
	return '.'.join(f'{s.kind}:{s.name}:{_hash_params(s.params)}' for s in spec.steps)


def apply_pipeline(
	x: np.ndarray,
	*,
	spec: PipelineSpec,
	meta: dict[str, Any],
	taps: list[str] | None = None,
) -> dict[str, Any]:
	"""Run the pipeline over ``x`` and collect requested taps/results."""
	y = x
	results: dict[str, Any] = {}
	tap_set = set(taps or [])
	lineage: list[str] = []
	for step in spec.steps:
		if step.kind == 'transform':
			op = TRANSFORMS[step.name]
			y = op(y, params=step.params, meta=meta)
			lineage.append(step.label or step.name)
			tap_name = '+'.join(lineage)
			if tap_name in tap_set:
				vmin, vmax = np.percentile(y, [1, 99])
				results[tap_name] = {
					'data': y,
					'meta': {'vmin': float(vmin), 'vmax': float(vmax)},
				}
		elif step.kind == 'analyzer':
			op = ANALYZERS[step.name]
			res = op(y, params=step.params, meta=meta)
			label = step.label or step.name
			results[label] = res
			results['final'] = res
		else:
			msg = f'Unsupported step kind: {step.kind}'
			raise ValueError(msg)
	return results
