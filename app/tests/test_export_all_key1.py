import io

import numpy as np
from fastapi.testclient import TestClient

from app.api.routers import picks as ep
from app.main import app


def test_export_all_key1_basic(monkeypatch):
    """Ensure export uses memmap-backed picks for each section."""

    assert any(
        getattr(r, "path", "") == "/export_manual_picks_all_npy"
        for r in app.router.routes
    ), (
        f"route not found. routes={[getattr(r, 'path', None) for r in app.router.routes]}"
    )

    class FakeReader:
        key1_byte = 189

        def get_key1_values(self):
            return [100, 200]

        @property
        def traces(self):
            # provide n_samples for clamping
            return np.zeros((5, 1000), dtype=np.float32)

    # ---- monkeypatch を全部先に当てる ----
    monkeypatch.setattr(
        ep, "get_reader", lambda file_id, key1_byte, key2_byte: FakeReader()
    )
    monkeypatch.setattr(ep, "_filename_for_file_id", lambda file_id: "lineA.sgy")
    monkeypatch.setattr(ep, "get_dt_for_file", lambda file_id: 0.004)  # 4 ms
    monkeypatch.setattr(ep, "get_ntraces_for", lambda file_id: 5)

    def fake_get_trace_seq(file_id, key1_idx, key1_byte):
        if key1_idx == 0:
            return np.array([0, 1, 2], dtype=np.int64)
        if key1_idx == 1:
            return np.array([3, 4], dtype=np.int64)
        raise AssertionError(f"unexpected key1_idx {key1_idx}")

    monkeypatch.setattr(ep, "get_trace_seq_for", fake_get_trace_seq)

    calls = []

    def fake_to_pairs(file_name, ntraces, sec_map):
        calls.append((file_name, int(ntraces), tuple(int(v) for v in sec_map)))
        if tuple(sec_map) == (0, 1, 2):
            return [
                {"trace": 0, "time": 0.012},
                {"trace": 2, "time": 0.020},
            ]
        if tuple(sec_map) == (3, 4):
            return [{"trace": 1, "time": 0.0}]
        return []

    monkeypatch.setattr(ep, "to_pairs_for_section", fake_to_pairs)

    # ✨ ここがキモ：isinstance チェックを通すためにシンボルを差し替える
    monkeypatch.setattr(ep, "TraceStoreSectionReader", FakeReader)

    # ---- その後で TestClient を生成 ----
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get(
        "/export_manual_picks_all_npy",
        params={"file_id": "X", "key1_byte": 189, "key2_byte": 193},
    )
    assert r.status_code == 200

    arr = np.load(io.BytesIO(r.content))
    # width=max(3,2)=3
    assert arr.shape == (2, 3)
    # dt=0.004 ⇒ 0.012/0.004=3, 0.020/0.004=5, 0.0/0.004=0
    assert arr[0].tolist() == [3, -1, 5]  # key1=100 row
    assert arr[1].tolist() == [-1, 0, -1]  # key1=200 row
    assert calls == [
        ("lineA.sgy", 5, (0, 1, 2)),
        ("lineA.sgy", 5, (3, 4)),
    ]


def test_export_all_key1_empty_is_all_minus1(monkeypatch):
    """Empty memmap rows stay -1 with computed section widths."""

    assert any(
        getattr(r, "path", "") == "/export_manual_picks_all_npy"
        for r in app.router.routes
    ), (
        f"route not found. routes={[getattr(r, 'path', None) for r in app.router.routes]}"
    )

    class FakeReader:
        key1_byte = 189

        def get_key1_values(self):
            return [10]

        @property
        def traces(self):
            return np.zeros((2, 100), dtype=np.float32)

    # ---- monkeypatch を全部先に当てる ----
    monkeypatch.setattr(
        ep, "get_reader", lambda file_id, key1_byte, key2_byte: FakeReader()
    )
    monkeypatch.setattr(ep, "_filename_for_file_id", lambda file_id: "lineB.sgy")
    monkeypatch.setattr(ep, "get_dt_for_file", lambda file_id: 0.002)
    monkeypatch.setattr(ep, "get_ntraces_for", lambda file_id: 2)
    monkeypatch.setattr(
        ep,
        "get_trace_seq_for",
        lambda file_id, key1_idx, key1_byte: np.array([0, 1], dtype=np.int64),
    )
    monkeypatch.setattr(ep, "to_pairs_for_section", lambda *args, **kwargs: [])

    # ✨ ここも同じく isinstance を通す
    monkeypatch.setattr(ep, "TraceStoreSectionReader", FakeReader)

    # ---- その後で TestClient を生成 ----
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get(
        "/export_manual_picks_all_npy",
        params={"file_id": "Y", "key1_byte": 189, "key2_byte": 193},
    )
    assert r.status_code == 200

    arr = np.load(io.BytesIO(r.content))
    assert arr.shape == (1, 2)
    assert arr.tolist() == [[-1, -1]]
