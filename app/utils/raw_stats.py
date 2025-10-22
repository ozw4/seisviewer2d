"""Raw baseline statistics and inference normalization utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

from app.utils.utils import to_builtin

logger = logging.getLogger(__name__)


_EPS = 1e-12


@dataclass(slots=True)
class RawBaselineStats:
	"""Container for persisted raw baseline statistics."""

	stage: str
	ddof: int
	method: str
	dtype_base: str | None
	dt: float
	mu_section: np.float32
	sigma_section: np.float32
	mu_traces: np.ndarray
	sigma_traces: np.ndarray
	zero_var_mask: np.ndarray
	source_sha256: str
	computed_at: str

	def to_payload(self) -> dict[str, object]:
		"""Return a JSON-serialisable payload."""

		return {
			'stage': self.stage,
			'ddof': int(self.ddof),
			'method': self.method,
			'dtype_base': self.dtype_base,
			'dt': float(self.dt),
			'mu_section': float(self.mu_section),
			'sigma_section': float(self.sigma_section),
			'mu_traces': to_builtin(self.mu_traces.astype(np.float32)),
			'sigma_traces': to_builtin(self.sigma_traces.astype(np.float32)),
			'zero_var_mask': to_builtin(self.zero_var_mask.astype(np.bool_)),
			'source_sha256': self.source_sha256,
			'computed_at': self.computed_at,
		}


def _welford_per_trace(arr: np.ndarray, *, ddof: int) -> tuple[np.ndarray, np.ndarray]:
	"""Return per-trace mean and M2 using Welford's algorithm."""

	if arr.ndim != 2:
		msg = f'Expected 2D array for per-trace stats, got {arr.ndim}D'
		raise ValueError(msg)

	traces, samples = arr.shape
	if samples == 0:
		raise ValueError('Cannot compute statistics on empty traces')

	means = np.zeros(traces, dtype=np.float64)
	m2 = np.zeros(traces, dtype=np.float64)
	for idx in range(samples):
		column = np.asarray(arr[:, idx], dtype=np.float64)
		delta = column - means
		means += delta / float(idx + 1)
		delta2 = column - means
		m2 += delta * delta2
	return means, m2


def _finalize_sigma(m2: np.ndarray, count: int, *, ddof: int) -> np.ndarray:
	"""Convert accumulated M2 into standard deviations."""

	den = count - ddof
	if den <= 0:
		raise ValueError('Degrees of freedom lead to non-positive denominator')

	var = m2 / float(den)
	var[var < 0] = 0.0
	sigma = np.sqrt(var, dtype=np.float64)
	return sigma


def _resolve_sha256_from_file(path: Path) -> str:
	"""Return the SHA-256 hash of ``path``."""

	hash_obj = sha256()
	with path.open('rb') as fh:
		for chunk in iter(lambda: fh.read(1024 * 1024), b''):
			hash_obj.update(chunk)
	return hash_obj.hexdigest()


def _resolve_sha256_from_array(arr: np.ndarray) -> str:
	"""Return SHA-256 for ``arr`` without persisting to disk."""

	hash_obj = sha256()
	view = np.ascontiguousarray(arr).view(np.uint8)
	hash_obj.update(view.tobytes())
	return hash_obj.hexdigest()


def resolve_source_sha256(*, reader: object | None, section: np.ndarray | None) -> str:
	"""Resolve the source checksum from TraceStore artifacts or fallback."""

	if reader is not None:
		traces = getattr(reader, 'traces', None)
		filename = getattr(traces, 'filename', None)
		if filename:
			path = Path(filename)
			if path.exists():
				return _resolve_sha256_from_file(path)

		sha_attr = getattr(reader, 'source_sha256', None)
		if isinstance(sha_attr, str) and len(sha_attr) == 64:
			return sha_attr

	if section is None:
		raise ValueError('Unable to resolve source checksum without section data')

	return _resolve_sha256_from_array(section)


def compute_raw_baseline(
	arr: np.ndarray,
	*,
	dtype_base: str | None,
	dt: float,
	source_sha256: str,
	ddof: int,
) -> RawBaselineStats:
	"""Compute baseline stats for ``arr`` (traces x samples)."""

	arr = np.ascontiguousarray(arr, dtype=np.float32)
	mu_traces64, m2_traces = _welford_per_trace(arr, ddof=ddof)
	samples = arr.shape[1]
	sigma_traces64 = _finalize_sigma(m2_traces, samples, ddof=ddof)
	mu_traces32 = mu_traces64.astype(np.float32)
	sigma_traces32 = sigma_traces64.astype(np.float32)
	zero_mask = np.less_equal(np.abs(sigma_traces32), _EPS)
	sigma_traces32[zero_mask] = 1.0
	total_samples = arr.shape[0] * samples
	weighted_sum = float(mu_traces64.sum(dtype=np.float64) * samples)
	mu_section = weighted_sum / float(total_samples)
	delta = mu_traces64 - mu_section
	total_m2 = float(m2_traces.sum()) + float(samples) * float(np.dot(delta, delta))
	sigma_section = np.sqrt(max(total_m2, 0.0) / float(total_samples))
	if sigma_section <= _EPS:
		sigma_section = 1.0

	computed_at = datetime.now(timezone.utc).isoformat()
	stats = RawBaselineStats(
		stage='raw',
		ddof=ddof,
		method='mean_std',
		dtype_base=dtype_base,
		dt=float(dt),
		mu_section=np.float32(mu_section),
		sigma_section=np.float32(sigma_section),
		mu_traces=mu_traces32,
		sigma_traces=sigma_traces32,
		zero_var_mask=zero_mask.astype(np.bool_),
		source_sha256=source_sha256,
		computed_at=computed_at,
	)
	return stats


def _baseline_path(base_dir: Path, stage: str, key1_val: int) -> Path:
	return base_dir / stage / f'key1_{key1_val}.json'


def load_raw_baseline(*, store_dir: Path, key1_val: int) -> RawBaselineStats | None:
	"""Load baseline stats from disk if present."""

	path = _baseline_path(Path(store_dir), 'raw', key1_val)
	if not path.exists():
		return None

	data = json.loads(path.read_text())
	mu_traces = np.asarray(data['mu_traces'], dtype=np.float32)
	sigma_traces = np.asarray(data['sigma_traces'], dtype=np.float32)
	zero_mask = np.asarray(data['zero_var_mask'], dtype=np.bool_)
	return RawBaselineStats(
		stage=data.get('stage', 'raw'),
		ddof=int(data.get('ddof', 0)),
		method=data.get('method', 'mean_std'),
		dtype_base=data.get('dtype_base'),
		dt=float(data.get('dt', 0.0)),
		mu_section=np.float32(data['mu_section']),
		sigma_section=np.float32(data['sigma_section']),
		mu_traces=mu_traces,
		sigma_traces=sigma_traces,
		zero_var_mask=zero_mask,
		source_sha256=str(data.get('source_sha256', '')),
		computed_at=str(data.get('computed_at', '')),
	)


def save_raw_baseline(
	*, store_dir: Path, key1_val: int, stats: RawBaselineStats
) -> None:
	"""Persist baseline stats to disk."""

	base_dir = Path(store_dir)
	stage_dir = base_dir / stats.stage
	stage_dir.mkdir(parents=True, exist_ok=True)
	path = _baseline_path(base_dir, stats.stage, key1_val)
	tmp_path = path.with_suffix('.json.tmp')
	tmp_path.write_text(json.dumps(stats.to_payload()))
	tmp_path.replace(path)
	logger.info(
		'Persisted baseline stats stage=%s key1=%s path=%s',
		stats.stage,
		key1_val,
		path,
	)


def ensure_raw_baseline(
	*,
	reader: object,
	key1_val: int,
	section: np.ndarray,
	dtype_base: str | None,
	dt: float,
	ddof: int,
) -> RawBaselineStats:
	"""Return cached baseline or compute a new one when source changes."""

	store_dir = Path(getattr(reader, 'store_dir', '.')) / 'baseline_stats'
	store_dir.mkdir(parents=True, exist_ok=True)
	current_sha = resolve_source_sha256(reader=reader, section=section)
	cached = load_raw_baseline(store_dir=store_dir, key1_val=key1_val)
	if cached and cached.source_sha256 == current_sha:
		logger.info(
			'Loaded cached baseline stage=%s key1=%s ddof=%s',
			cached.stage,
			key1_val,
			cached.ddof,
		)
		return cached

	logger.info('Computing baseline stage=raw key1=%s ddof=%s', key1_val, ddof)
	stats = compute_raw_baseline(
		section,
		dtype_base=dtype_base,
		dt=dt,
		source_sha256=current_sha,
		ddof=ddof,
	)
	save_raw_baseline(store_dir=store_dir, key1_val=key1_val, stats=stats)
	return stats


def per_trace_normalization(
	arr: np.ndarray,
	*,
	ddof: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Return z-scored array along with mean/std per trace."""

	arr = np.ascontiguousarray(arr, dtype=np.float32)
	mu_traces64, m2_traces = _welford_per_trace(arr, ddof=ddof)
	samples = arr.shape[1]
	sigma64 = _finalize_sigma(m2_traces, samples, ddof=ddof)
	mu = mu_traces64.astype(np.float32)
	sigma = sigma64.astype(np.float32)
	zero_mask = np.less_equal(np.abs(sigma), _EPS)
	sigma_safe = sigma.copy()
	sigma_safe[zero_mask] = 1.0
	mu_broadcast = mu[:, None]
	sigma_broadcast = sigma_safe[:, None]
	z = (arr - mu_broadcast) / sigma_broadcast
	return z.astype(np.float32), mu, sigma_safe, zero_mask


__all__ = [
	'RawBaselineStats',
	'compute_raw_baseline',
	'ensure_raw_baseline',
	'load_raw_baseline',
	'per_trace_normalization',
	'resolve_source_sha256',
	'save_raw_baseline',
]
