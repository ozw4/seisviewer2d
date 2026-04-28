# app/tests/test_missing_endpoints_section.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    """HTTP client with clean file registry and AppState caches per test."""
    app.state.sv.file_registry.clear()
    state = app.state.sv
    state.cached_readers.clear()
    state.window_section_cache.clear()
    state.section_offsets_cache.clear()
    state.trace_stats_cache.clear()
    with TestClient(app) as c:
        yield c
    app.state.sv.file_registry.clear()
    state.cached_readers.clear()
    state.window_section_cache.clear()
    state.section_offsets_cache.clear()
    state.trace_stats_cache.clear()


def test_get_key1_values_returns_unique_values_and_passes_bytes(
    client: TestClient, monkeypatch
):
    from app.api.routers import section as sec

    captured: dict[str, object] = {}

    class _StubReader:
        def get_key1_values(self):
            # Router is expected to return unique values (real reader does this).
            return np.asarray([3, 1, 2], dtype=np.int32)

    def _stub_get_reader(file_id: str, key1_byte: int, key2_byte: int, *, state=None):
        captured["file_id"] = file_id
        captured["key1_byte"] = int(key1_byte)
        captured["key2_byte"] = int(key2_byte)
        return _StubReader()

    monkeypatch.setattr(sec, "get_reader", _stub_get_reader, raising=True)

    r = client.get(
        "/get_key1_values",
        params={"file_id": "fid", "key1_byte": 191, "key2_byte": 195},
    )
    assert r.status_code == 200
    out = r.json()
    assert "values" in out
    vals = out["values"]
    assert isinstance(vals, list)
    assert set(vals) == {1, 2, 3}
    assert len(vals) == len(set(vals))
    assert captured == {"file_id": "fid", "key1_byte": 191, "key2_byte": 195}


def test_get_key1_values_unknown_file_id_returns_404(client: TestClient):
    # Use the real get_reader: missing file_id must be rejected.
    r = client.get("/get_key1_values", params={"file_id": "no-such-id"})
    assert r.status_code == 404
    assert r.json().get("detail") == "File ID not found"


def test_get_section_meta_unknown_file_id_returns_404(client: TestClient):
    # Use the real get_reader: missing file_id must be rejected.
    r = client.get("/get_section_meta", params={"file_id": "no-such-id"})
    assert r.status_code == 404
    assert r.json().get("detail") == "File ID not found"


def test_get_section_meta_contract_and_dt_from_meta_json(
    client: TestClient, monkeypatch, tmp_path: Path
):
    from app.api.routers import section as sec

    file_id = "fid-meta"
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "meta.json").write_text(
        json.dumps({"dt": 0.004, "original_segy_path": "/tmp/does-not-matter.sgy"}),
        encoding="utf-8",
    )
    app.state.sv.file_registry.set_record(file_id, {"store_path": str(store_dir)})

    class _StubReader:
        dtype = np.dtype("float32")
        scale = 2.5

        def get_key1_values(self):
            return np.asarray([10, 20], dtype=np.int32)

        def get_trace_seq_for_value(self, key1: int, align_to: str = "display"):
            assert align_to == "display"
            assert int(key1) == 10
            return np.arange(4, dtype=np.int64)

        def get_n_samples(self) -> int:
            return 128

    def _stub_get_reader(_file_id: str, _kb1: int, _kb2: int, *, state=None):
        assert _file_id == file_id
        assert int(_kb1) == 189
        assert int(_kb2) == 193
        return _StubReader()

    monkeypatch.setattr(sec, "get_reader", _stub_get_reader, raising=True)
    # baseline 作成は重い/外部依存になりやすいので、契約テストでは呼べることだけ担保して stub
    def _stub_baseline(**kwargs):
        status = kwargs.get("status")
        if isinstance(status, dict):
            status["source"] = "precomputed"
        return {}

    monkeypatch.setattr(sec, "get_or_create_raw_baseline", _stub_baseline, raising=True)

    r = client.get("/get_section_meta", params={"file_id": file_id})
    assert r.status_code == 200
    out = r.json()
    assert set(out.keys()) >= {"shape", "dt", "dtype", "scale"}
    assert out["shape"] == [4, 128]
    assert out["dt"] == pytest.approx(0.004)
    assert out["dtype"] == "float32"
    assert out["scale"] == pytest.approx(2.5)
    assert r.headers["x-sv-baseline-source"] == "precomputed"
    assert float(r.headers["x-sv-baseline-ms"]) >= 0.0
    assert float(r.headers["x-sv-server-ms"]) >= 0.0
    assert "sv_baseline;dur=" in r.headers["server-timing"]


def test_get_section_meta_dt_from_segy_header(
    client: TestClient, monkeypatch, tmp_path: Path
):
    from app.api.routers import section as sec

    file_id = "fid-segy"
    segy_path = tmp_path / "LineA.sgy"
    us = 4000  # 4000 microseconds => 0.004 seconds
    offset = 3200 + 16
    buf = bytearray(offset + 2)
    buf[offset : offset + 2] = int(us).to_bytes(2, byteorder="big", signed=False)
    segy_path.write_bytes(bytes(buf))
    app.state.sv.file_registry.set_record(file_id, {"path": str(segy_path)})

    captured: dict[str, object] = {}

    class _StubReader:
        dtype = np.dtype("int16")
        scale = None

        def get_key1_values(self):
            return np.asarray([7], dtype=np.int32)

        def get_trace_seq_for_value(self, key1: int, align_to: str = "display"):
            captured["key1"] = int(key1)
            captured["align_to"] = align_to
            return np.arange(3, dtype=np.int64)

        def get_n_samples(self) -> int:
            return 11

    def _stub_get_reader(_file_id: str, _kb1: int, _kb2: int, *, state=None):
        captured["file_id"] = _file_id
        captured["kb1"] = int(_kb1)
        captured["kb2"] = int(_kb2)
        return _StubReader()

    monkeypatch.setattr(sec, "get_reader", _stub_get_reader, raising=True)
    def _stub_baseline(**kwargs):
        status = kwargs.get("status")
        if isinstance(status, dict):
            status["source"] = "precomputed"
        return {}

    monkeypatch.setattr(sec, "get_or_create_raw_baseline", _stub_baseline, raising=True)

    r = client.get(
        "/get_section_meta",
        params={"file_id": file_id, "key1_byte": 191, "key2_byte": 195},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["shape"] == [3, 11]
    assert out["dt"] == pytest.approx(us / 1_000_000.0)
    assert out["dtype"] == "int16"
    assert out["scale"] is None
    assert r.headers["x-sv-baseline-source"] == "precomputed"
    assert float(r.headers["x-sv-baseline-ms"]) >= 0.0
    assert float(r.headers["x-sv-server-ms"]) >= 0.0

    assert captured == {
        "key1": 7,
        "align_to": "display",
        "file_id": file_id,
        "kb1": 191,
        "kb2": 195,
    }
