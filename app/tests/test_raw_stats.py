from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.raw_stats import compute_raw_baseline, ensure_raw_baseline
from app.utils.utils import SectionView


class _StubReader:
	def __init__(self, store_dir: Path, arr: np.ndarray) -> None:
		self.store_dir = Path(store_dir)
		self.store_dir.mkdir(parents=True, exist_ok=True)
		self._arr = np.asarray(arr, dtype=np.float32)
		self.traces = type('T', (), {'filename': str(self.store_dir / 'traces.npy')})()
		np.save(self.store_dir / 'traces.npy', self._arr.astype(np.float32))

	def get_section(self, key1_val: int) -> SectionView:
		return SectionView(arr=self._arr, dtype=self._arr.dtype, scale=None)

	def update_array(self, arr: np.ndarray) -> None:
		self._arr = np.asarray(arr, dtype=np.float32)
		np.save(self.store_dir / 'traces.npy', self._arr.astype(np.float32))


def test_compute_raw_baseline_stats() -> None:
	arr = np.array([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=np.float32)
	stats = compute_raw_baseline(
		arr,
		dtype_base='int8',
		dt=0.004,
		source_sha256='abc123',
		ddof=0,
	)
	assert pytest.approx(float(stats.mu_section), rel=1e-6) == 4.5
	assert pytest.approx(float(stats.sigma_section), rel=1e-6) == 2.291288
	assert stats.mu_traces.shape == (2,)
	assert stats.sigma_traces.shape == (2,)
	assert pytest.approx(stats.mu_traces[0], rel=1e-6) == 4.0
	assert pytest.approx(stats.mu_traces[1], rel=1e-6) == 5.0
	assert pytest.approx(stats.sigma_traces[0], rel=1e-6) == np.sqrt(5.0)
	assert pytest.approx(stats.sigma_traces[1], rel=1e-6) == np.sqrt(5.0)
	assert not np.any(stats.zero_var_mask)


def test_compute_raw_baseline_zero_variance() -> None:
	arr = np.array([[4, 4, 4, 4], [1, 2, 3, 4]], dtype=np.float32)
	stats = compute_raw_baseline(
		arr,
		dtype_base='float32',
		dt=0.002,
		source_sha256='hash',
		ddof=0,
	)
	assert stats.zero_var_mask.shape == (2,)
	assert bool(stats.zero_var_mask[0])
	assert pytest.approx(stats.sigma_traces[0], rel=1e-6) == 1.0
	assert not bool(stats.zero_var_mask[1])


def test_ensure_raw_baseline_cache_and_recompute(tmp_path: Path) -> None:
	store_dir = tmp_path / 'store'
	reader = _StubReader(store_dir, np.array([[0, 1], [2, 3]], dtype=np.float32))
	section = reader.get_section(1).arr
	stats1 = ensure_raw_baseline(
		reader=reader,
		key1_val=1,
		section=section,
		dtype_base=str(section.dtype),
		dt=0.004,
		ddof=0,
	)
	baseline_path = store_dir / 'baseline_stats' / 'raw' / 'key1_1.json'
	assert baseline_path.exists()
	stats2 = ensure_raw_baseline(
		reader=reader,
		key1_val=1,
		section=reader.get_section(1).arr,
		dtype_base=str(section.dtype),
		dt=0.004,
		ddof=0,
	)
	assert stats1.source_sha256 == stats2.source_sha256
	assert stats1.computed_at == stats2.computed_at
	payload = json.loads(baseline_path.read_text())
	assert payload['source_sha256'] == stats1.source_sha256

	reader.update_array(np.array([[10, 10], [20, 20]], dtype=np.float32))
	stats3 = ensure_raw_baseline(
		reader=reader,
		key1_val=1,
		section=reader.get_section(1).arr,
		dtype_base=str(section.dtype),
		dt=0.004,
		ddof=0,
	)
	assert stats3.source_sha256 != stats1.source_sha256
	assert stats3.mu_section != pytest.approx(float(stats1.mu_section))


def test_section_stats_endpoint_uses_cache(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	client = TestClient(app)
	store_dir = tmp_path / 'store'
	initial = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
	reader = _StubReader(store_dir, initial)

	def _get_reader(*_args, **_kwargs):
		return reader

	monkeypatch.setattr('app.api.routers.section.get_reader', _get_reader)
	monkeypatch.setattr(
		'app.api.routers.section.get_dt_for_file', lambda _file_id: 0.002
	)

	params = {'file_id': 'demo', 'key1_val': 1, 'baseline': 'raw'}
	resp1 = client.get('/section/stats', params=params)
	assert resp1.status_code == 200
	data1 = resp1.json()
	path = store_dir / 'baseline_stats' / 'raw' / 'key1_1.json'
	assert path.exists()
	resp2 = client.get('/section/stats', params=params)
	assert resp2.status_code == 200
	assert resp2.json() == data1

	reader.update_array(np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32))
	resp3 = client.get('/section/stats', params=params)
	assert resp3.status_code == 200
	data3 = resp3.json()
	assert data3['source_sha256'] != data1['source_sha256']
	assert data3['mu_section'] != pytest.approx(data1['mu_section'])
