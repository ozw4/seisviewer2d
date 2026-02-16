import numpy as np
from fastapi.testclient import TestClient

from app.api.routers import picks as ep
from app.main import app


def test_picks_are_isolated_per_key1_val(tmp_path, monkeypatch):
    # memmapの保存先を一時ディレクトリへ
    monkeypatch.setenv('PICKS_NPY_DIR', str(tmp_path / 'picks_npy'))

    # /picks が参照するヘルパをモック（軽量に）
    monkeypatch.setattr(ep, '_filename_for_file_id', lambda fid: 'LineX.sgy')
    monkeypatch.setattr(
        ep, 'get_ntraces_for', lambda fid, key1_byte, key2_byte=193, state=None: 5
    )

    # key1=0 -> sec_map=[0,1,2], key1=1 -> sec_map=[3,4]
    def fake_get_trace_seq(file_id, key1, key1_byte, key2_byte=193, state=None):
        if key1 == 0:
            return np.array([0, 1, 2], dtype=np.int64)
        if key1 == 1:
            return np.array([3, 4], dtype=np.int64)
        raise AssertionError(f'unexpected key1 {key1}')

    monkeypatch.setattr(ep, 'get_trace_seq_for_value', fake_get_trace_seq)

    client = TestClient(app, raise_server_exceptions=False)

    # key1=0 のセクションにピックを1本
    r = client.post(
        '/picks',
        json={
            'file_id': 'F',
            'trace': 1,  # sec-local index (=> global 1)
            'time': 1.23,
            'key1': 0,
            'key1_byte': 189,
        },
    )
    assert r.status_code == 200

    # そのセクションをGET → 1本だけ
    r = client.get('/picks', params={'file_id': 'F', 'key1': 0, 'key1_byte': 189})
    assert r.status_code == 200
    picks0 = r.json()['picks']
    assert len(picks0) == 1
    assert picks0[0]['trace'] == 1 and abs(picks0[0]['time'] - 1.23) < 1e-6

    # 別セクション(key1=1)は空のはず（独立性の確認）
    r = client.get('/picks', params={'file_id': 'F', 'key1': 1, 'key1_byte': 189})
    assert r.status_code == 200
    picks1 = r.json()['picks']
    assert picks1 == []

    # key1=1 にも1本追加
    r = client.post(
        '/picks',
        json={
            'file_id': 'F',
            'trace': 0,  # sec-local index (=> global 3)
            'time': 2.5,
            'key1': 1,
            'key1_byte': 189,
        },
    )
    assert r.status_code == 200

    # それぞれのセクションで期待通りに見えるか
    r = client.get('/picks', params={'file_id': 'F', 'key1': 1, 'key1_byte': 189})
    assert [p['trace'] for p in r.json()['picks']] == [0]

    r = client.get('/picks', params={'file_id': 'F', 'key1': 0, 'key1_byte': 189})
    assert [p['trace'] for p in r.json()['picks']] == [1]


def test_delete_whole_section_only_affects_that_section(tmp_path, monkeypatch):
    monkeypatch.setenv('PICKS_NPY_DIR', str(tmp_path / 'picks_npy'))
    monkeypatch.setattr(ep, '_filename_for_file_id', lambda fid: 'LineY.sgy')
    monkeypatch.setattr(
        ep, 'get_ntraces_for', lambda fid, key1_byte, key2_byte=193, state=None: 5
    )

    def fake_get_trace_seq(file_id, key1, key1_byte, key2_byte=193, state=None):
        return (
            np.array([0, 1, 2], dtype=np.int64)
            if key1 == 0
            else np.array([3, 4], dtype=np.int64)
        )

    monkeypatch.setattr(ep, 'get_trace_seq_for_value', fake_get_trace_seq)

    client = TestClient(app, raise_server_exceptions=False)

    # 両セクションにピックを投入
    client.post(
        '/picks',
        json={'file_id': 'F', 'trace': 0, 'time': 1.0, 'key1': 0, 'key1_byte': 189},
    )
    client.post(
        '/picks',
        json={'file_id': 'F', 'trace': 1, 'time': 2.0, 'key1': 1, 'key1_byte': 189},
    )

    # key1=0 を丸ごと消す（trace=None）
    r = client.delete(
        '/picks',
        params={'file_id': 'F', 'key1': 0, 'key1_byte': 189},
    )
    assert r.status_code == 200

    # 0側は空に、1側は残る
    r0 = client.get('/picks', params={'file_id': 'F', 'key1': 0, 'key1_byte': 189})
    r1 = client.get('/picks', params={'file_id': 'F', 'key1': 1, 'key1_byte': 189})
    assert r0.json()['picks'] == []
    assert [p['trace'] for p in r1.json()['picks']] == [1]


def test_post_trace_out_of_range_returns_400(tmp_path, monkeypatch):
    monkeypatch.setenv('PICKS_NPY_DIR', str(tmp_path / 'picks_npy'))
    monkeypatch.setattr(ep, '_filename_for_file_id', lambda fid: 'LineZ.sgy')
    monkeypatch.setattr(
        ep, 'get_ntraces_for', lambda fid, key1_byte, key2_byte=193, state=None: 5
    )
    monkeypatch.setattr(
        ep,
        'get_trace_seq_for_value',
        lambda fid, val, b, key2_byte=193, state=None: np.array(
            [10, 11], dtype=np.int64
        ),
    )  # セクション幅2

    client = TestClient(app, raise_server_exceptions=False)
    # trace=2 はセクション幅(2)に対して範囲外 → 400
    r = client.post(
        '/picks',
        json={'file_id': 'F', 'trace': 2, 'time': 0.5, 'key1': 0, 'key1_byte': 189},
    )
    assert r.status_code == 400


def test_picks_key2_byte_default_and_override_are_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv('PICKS_NPY_DIR', str(tmp_path / 'picks_npy'))
    monkeypatch.setattr(ep, '_filename_for_file_id', lambda fid: 'LineK.sgy')

    seen_ntr: list[tuple[int, int]] = []
    seen_seq: list[tuple[int, int]] = []

    def fake_ntr(file_id, key1_byte, key2_byte=193, state=None):
        _ = file_id, state
        seen_ntr.append((int(key1_byte), int(key2_byte)))
        return 2

    def fake_seq(file_id, key1, key1_byte, key2_byte=193, state=None):
        _ = file_id, key1, state
        seen_seq.append((int(key1_byte), int(key2_byte)))
        return np.array([0, 1], dtype=np.int64)

    monkeypatch.setattr(ep, 'get_ntraces_for', fake_ntr)
    monkeypatch.setattr(ep, 'get_trace_seq_for_value', fake_seq)
    monkeypatch.setattr(ep, 'to_pairs_for_section', lambda *args, **kwargs: [])
    monkeypatch.setattr(ep, 'set_by_traceseq', lambda *args, **kwargs: None)
    monkeypatch.setattr(ep, 'clear_by_traceseq', lambda *args, **kwargs: None)

    client = TestClient(app, raise_server_exceptions=False)

    r = client.get('/picks', params={'file_id': 'F', 'key1': 0, 'key1_byte': 189})
    assert r.status_code == 200
    r = client.get(
        '/picks',
        params={'file_id': 'F', 'key1': 0, 'key1_byte': 189, 'key2_byte': 321},
    )
    assert r.status_code == 200

    r = client.post(
        '/picks',
        json={
            'file_id': 'F',
            'trace': 1,
            'time': 0.25,
            'key1': 0,
            'key1_byte': 189,
            'key2_byte': 322,
        },
    )
    assert r.status_code == 200

    r = client.delete(
        '/picks',
        params={
            'file_id': 'F',
            'key1': 0,
            'key1_byte': 189,
            'key2_byte': 323,
            'trace': 1,
        },
    )
    assert r.status_code == 200

    assert seen_ntr == [(189, 193), (189, 321), (189, 322), (189, 323)]
    assert seen_seq == [(189, 193), (189, 321), (189, 322), (189, 323)]
