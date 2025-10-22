import json

import numpy as np
import pytest

pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from app.api.baselines import BASELINE_FILENAME_RAW, get_or_create_raw_baseline
from app.api._helpers import cached_readers
from app.main import app
from app.utils.segy_meta import FILE_REGISTRY


@pytest.fixture
def sample_store(tmp_path):
	store = tmp_path / 'store'
	store.mkdir()
	traces = np.array(
		[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]],
		dtype=np.float32,
	)
	n_samples = traces.shape[1]
	np.save(store / 'traces.npy', traces)
	np.save(store / 'headers_byte_189.npy', np.array([10, 10, 20], dtype=np.int32))
	np.save(store / 'headers_byte_200.npy', np.array([10, 10, 20], dtype=np.int32))
	np.save(store / 'headers_byte_193.npy', np.array([1, 2, 1], dtype=np.int32))
	np.savez(
		store / 'index.npz',
		key1_values=np.array([10, 20], dtype=np.int32),
		key1_offsets=np.array([0, 2], dtype=np.int64),
		key1_counts=np.array([2, 1], dtype=np.int64),
		sorted_to_original=np.arange(3, dtype=np.int64),
	)
	meta = {
		'schema_version': 1,
		'dtype': 'float32',
		'n_traces': traces.shape[0],
		'n_samples': n_samples,
		'key_bytes': {'key1': 189, 'key2': 193},
		'sorted_by': ['key1', 'key2'],
		'dt': 0.004,
		'original_segy_path': 'dummy.sgy',
		'original_mtime': 0.0,
		'original_size': 0,
		'source_sha256': 'sha1',
	}
	(store / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
	file_id = 'file_test'
	FILE_REGISTRY[file_id] = {'store_path': str(store), 'dt': 0.004}
	cached_readers.clear()
	yield file_id, store
	cached_readers.clear()
	FILE_REGISTRY.pop(file_id, None)
	baseline_path = store / BASELINE_FILENAME_RAW
	if baseline_path.exists():
		baseline_path.unlink()


def test_get_or_create_raw_baseline(sample_store):
	file_id, store = sample_store
	baseline = get_or_create_raw_baseline(
		file_id=file_id,
		key1_byte=189,
		key2_byte=193,
	)
	assert baseline['stage'] == 'raw'
	assert baseline['ddof'] == 0
	assert baseline['method'] == 'mean_std'
	assert baseline['source_sha256'] == 'sha1'
	assert baseline['key1_values'] == [10, 20]
	assert baseline['trace_index_map'] == {'10': [0, 2], '20': [2, 3]}
	mu_traces = baseline['mu_traces']
	sigma_traces = baseline['sigma_traces']
	zero_mask = baseline['zero_var_mask']
	assert mu_traces == pytest.approx([2.0, 1.0, -1.0 / 3.0])
	assert sigma_traces == pytest.approx(
		[
			float(np.sqrt(2.0 / 3.0)),
			1.0,
			float(np.sqrt(8.0 / 9.0)),
		],
	)
	assert zero_mask == [False, True, False]
	mu_sections = baseline['mu_section_by_key1']
	sigma_sections = baseline['sigma_section_by_key1']
	assert mu_sections == pytest.approx([1.5, -1.0 / 3.0])
	assert sigma_sections == pytest.approx(
		[
			float(np.sqrt(7.0 / 12.0)),
			float(np.sqrt(8.0 / 9.0)),
		],
	)
	baseline_path = store / BASELINE_FILENAME_RAW
	assert baseline_path.is_file()


def test_baseline_recompute_on_source_change(sample_store):
	file_id, store = sample_store
	first = get_or_create_raw_baseline(file_id=file_id, key1_byte=189, key2_byte=193)
	meta_path = store / 'meta.json'
	meta = json.loads(meta_path.read_text(encoding='utf-8'))
	meta['source_sha256'] = 'sha2'
	meta_path.write_text(json.dumps(meta), encoding='utf-8')
	cached_readers.clear()
	second = get_or_create_raw_baseline(file_id=file_id, key1_byte=189, key2_byte=193)
	assert second['source_sha256'] == 'sha2'
	assert second['computed_at'] != first['computed_at']


def test_baseline_recompute_on_key1_byte_change(sample_store):
	file_id, _ = sample_store
	first = get_or_create_raw_baseline(file_id=file_id, key1_byte=189, key2_byte=193)
	second = get_or_create_raw_baseline(file_id=file_id, key1_byte=200, key2_byte=193)
	assert second['key1_byte'] == 200
	assert second['computed_at'] != first['computed_at']


def test_baseline_partition_matches_requested_key1(sample_store):
	file_id, store = sample_store
	# Override the cached headers for byte 200 with a different grouping than the store default.
	new_headers = np.array([5, 15, 15], dtype=np.int32)
	with open(store / 'headers_byte_200.npy', 'wb') as fh:
		np.save(fh, new_headers)
	cached_readers.clear()
	baseline = get_or_create_raw_baseline(file_id=file_id, key1_byte=200, key2_byte=193)
	assert baseline['key1_values'] == [5, 15]
	assert baseline['trace_index_map'] == {'5': [0, 1], '15': [1, 3]}
	mu_sections = baseline['mu_section_by_key1']
	sigma_sections = baseline['sigma_section_by_key1']
	assert mu_sections == pytest.approx([2.0, 1.0 / 3.0])
	assert sigma_sections == pytest.approx(
		[
			float(np.sqrt(2.0 / 3.0)),
			float(np.sqrt(8.0 / 9.0)),
		]
	)


def test_section_stats_endpoint(sample_store):
	file_id, _ = sample_store
	client = TestClient(app)
	resp = client.get(
		'/section/stats',
		params={'file_id': file_id, 'baseline': 'raw', 'key1_idx': 20},
	)
	assert resp.status_code == 200
	payload = resp.json()
	assert payload['key1_values'] == [10, 20]
	selected = payload['selected_key1']
	assert selected['key1_value'] == 20
	assert selected['trace_range'] == [2, 3]
	assert selected['mu_section'] == pytest.approx(-1.0 / 3.0)
	assert selected['sigma_section'] == pytest.approx(float(np.sqrt(8.0 / 9.0)))
	bad_baseline = client.get(
		'/section/stats',
		params={'file_id': file_id, 'baseline': 'processed'},
	)
	assert bad_baseline.status_code == 400
	bad_step = client.get(
		'/section/stats',
		params={'file_id': file_id, 'baseline': 'raw', 'step_x': 2},
	)
	assert bad_step.status_code == 400
	not_found = client.get(
		'/section/stats',
		params={'file_id': file_id, 'baseline': 'raw', 'key1_idx': 999},
	)
	assert not_found.status_code == 404
