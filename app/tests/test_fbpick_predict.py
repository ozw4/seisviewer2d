import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routers import fbpick_predict as fbpick_predict_mod
from app.main import app


@pytest.fixture()
def client(monkeypatch):
	monkeypatch.setattr(
		fbpick_predict_mod,
		'_last_prob_state',
		fbpick_predict_mod._LastProbabilityState(),
	)
	return TestClient(app)


def _set_prob(monkeypatch, prob, dt=0.004, source='raw'):
	payload = fbpick_predict_mod._ProbabilityPayload(
		prob=np.asarray(prob, dtype=np.float32), dt=dt, source=source
	)

	def _fake_loader(_req):
		return payload

	monkeypatch.setattr(fbpick_predict_mod, '_load_probability_map', _fake_loader)


def test_expectation_method_returns_times(client, monkeypatch):
	prob = np.array([[0.0, 0.5, 0.5], [0.1, 0.0, 0.9]], dtype=np.float32)
	_set_prob(monkeypatch, prob)

	res = client.post(
		'/fbpick_predict',
		json={
			'file_id': 'abc',
			'key1_val': 1,
			'key1_byte': 189,
			'key2_byte': 193,
			'method': 'expectation',
			'sigma_ms_max': 5.0,
		},
	)
	assert res.status_code == 200
	data = res.json()
	assert pytest.approx(data['dt'], rel=0, abs=1e-12) == 0.004
	assert data['picks'] == [
		{'trace': 0, 'time': pytest.approx(0.006)},
		{'trace': 1, 'time': pytest.approx(0.0072)},
	]


def test_argmax_with_sigma_gate_filters_rows(client, monkeypatch):
	prob = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
	_set_prob(monkeypatch, prob)

	res = client.post(
		'/fbpick_predict',
		json={
			'file_id': 'abc',
			'key1_val': 2,
			'method': 'argmax',
			'sigma_ms_max': 1.0,
		},
	)
	assert res.status_code == 200
	assert res.json()['picks'] == []


@pytest.mark.parametrize(
	'prob',
	[
		np.zeros((2, 3), dtype=np.float32),
		np.array([[np.nan, 1.0, 0.0]], dtype=np.float32),
	],
)
def test_invalid_probability_map_rejected(client, monkeypatch, prob):
	_set_prob(monkeypatch, prob)
	res = client.post(
		'/fbpick_predict',
		json={
			'file_id': 'bad',
			'key1_val': 0,
			'method': 'expectation',
			'sigma_ms_max': 10.0,
		},
	)
	assert res.status_code == 422
