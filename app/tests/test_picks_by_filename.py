import numpy as np
from fastapi.testclient import TestClient

from app.api.routers import picks
from app.main import app


def test_picks_memmap_roundtrip(monkeypatch):
	file_id = "test-file-id"

	monkeypatch.setattr(
		picks,
		"_filename_for_file_id",
		lambda fid: "LineA.sgy" if fid == file_id else None,
	)
	monkeypatch.setattr(picks, "get_ntraces_for", lambda fid: 20)
	monkeypatch.setattr(
		picks,
		"get_trace_seq_for_value",
		lambda fid, key1_val, key1_byte: np.arange(0, 12, dtype=np.int64),
	)

	store: dict[int, float] = {}

	def fake_set(file_name, ntraces, trace_seq, time_s):
		assert file_name == "LineA.sgy"
		assert ntraces == 20
		store[int(trace_seq)] = float(time_s)

	def fake_pairs(file_name, ntraces, sec_map):
		assert file_name == "LineA.sgy"
		data = []
		for local_idx, trace_seq in enumerate(sec_map):
			val = store.get(int(trace_seq))
			if val is not None:
				data.append({"trace": local_idx, "time": val})
		return data

	def fake_clear_section(file_name, ntraces, sec_map):
		for trace_seq in sec_map:
			store.pop(int(trace_seq), None)

	def fake_clear_by(file_name, ntraces, trace_seq):
		store.pop(int(trace_seq), None)

	monkeypatch.setattr(picks, "set_by_traceseq", fake_set)
	monkeypatch.setattr(picks, "to_pairs_for_section", fake_pairs)
	monkeypatch.setattr(picks, "clear_section", fake_clear_section)
	monkeypatch.setattr(picks, "clear_by_traceseq", fake_clear_by)

	with TestClient(app) as client:
		response = client.post(
			"/picks",
			json={
				"file_id": file_id,
				"trace": 10,
				"time": 0.12,
				"key1_val": 0,
				"key1_byte": 189,
			},
		)
		assert response.status_code == 200

		response = client.get(
			"/picks",
			params={"file_id": file_id, "key1_val": 0, "key1_byte": 189},
		)
		assert response.status_code == 200
		assert response.json() == {
			"picks": [
				{
					"trace": 10,
					"time": 0.12,
				}
			]
		}

		response = client.delete(
			"/picks",
			params={
				"file_id": file_id,
				"trace": 10,
				"key1_val": 0,
				"key1_byte": 189,
			},
		)
		assert response.status_code == 200

		response = client.get(
			"/picks",
			params={"file_id": file_id, "key1_val": 0, "key1_byte": 189},
		)
		assert response.status_code == 200
		assert response.json() == {"picks": []}
