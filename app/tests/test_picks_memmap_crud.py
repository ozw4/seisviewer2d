import numpy as np
from fastapi.testclient import TestClient

from app.api.routers import picks as ep
from app.main import app


def test_picks_are_isolated_per_key1(tmp_path, monkeypatch):
        # memmapの保存先を一時ディレクトリへ
        monkeypatch.setenv('PICKS_NPY_DIR', str(tmp_path / 'picks_npy'))

        # /picks が参照するヘルパをモック（軽量に）
        monkeypatch.setattr(ep, '_filename_for_file_id', lambda fid: 'LineX.sgy')
        monkeypatch.setattr(ep, 'get_ntraces_for', lambda fid: 5)

        # key1 header values 0 and 1 correspond to two different sections.  Provide
        # deterministic trace maps for each value so picks can be isolated per key1.
        def fake_get_trace_seq(file_id, key1, key1_byte):
                if key1 == 0:
                        return np.array([0, 1, 2], dtype=np.int64)
                if key1 == 1:
                        return np.array([3, 4], dtype=np.int64)
                raise AssertionError(f'unexpected key1 {key1}')

        monkeypatch.setattr(ep, 'get_trace_seq_for', fake_get_trace_seq)

        client = TestClient(app, raise_server_exceptions=False)

        # key1=0 のセクションにピックを1本登録
        r = client.post(
                '/picks',
                json={
                        'file_id': 'F',
                        'trace': 1,  # sec‑local index (=> global 1)
                        'time': 1.23,
                        'key1': 0,
                        'key1_byte': 189,
                },
        )
        assert r.status_code == 200

        # そのセクションを GET → 1本だけ
        r = client.get('/picks', params={'file_id': 'F', 'key1': 0, 'key1_byte': 189})
        assert r.status_code == 200
        picks0 = r.json()['picks']
        assert len(picks0) == 1
        assert picks0[0]['trace'] == 1 and abs(picks0[0]['time'] - 1.23) < 1e-6

        # 別セクション (key1=1) は空のはず（独立性の確認）
        r = client.get('/picks', params={'file_id': 'F', 'key1': 1, 'key1_byte': 189})
        assert r.status_code == 200
        picks1 = r.json()['picks']
        assert picks1 == []

        # key1=1 にも1本追加
        r = client.post(
                '/picks',
                json={
                        'file_id': 'F',
                        'trace': 0,  # sec‑local index (=> global 3)
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
        monkeypatch.setattr(ep, 'get_ntraces_for', lambda fid: 5)

        def fake_get_trace_seq(file_id, key1, key1_byte):
                return (
                        np.array([0, 1, 2], dtype=np.int64)
                        if key1 == 0
                        else np.array([3, 4], dtype=np.int64)
                )

        monkeypatch.setattr(ep, 'get_trace_seq_for', fake_get_trace_seq)

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
        monkeypatch.setattr(ep, 'get_ntraces_for', lambda fid: 5)
        # Provide a trace mapping of width 2 for key1=0
        monkeypatch.setattr(
                ep,
                'get_trace_seq_for',
                lambda fid, key1, b: np.array([10, 11], dtype=np.int64),
        )

        client = TestClient(app, raise_server_exceptions=False)
        # trace=2 はセクション幅(2)に対して範囲外 → 400
        r = client.post(
                '/picks',
                json={'file_id': 'F', 'trace': 2, 'time': 0.5, 'key1': 0, 'key1_byte': 189},
        )
        assert r.status_code == 400
