import numpy as np
from fastapi.testclient import TestClient

import app.api.routers.picks as picks_router
from app.main import app


class _FakeReader:
    def __init__(self):
        self._map = {
            100: np.asarray([0, 1, 3], dtype=np.int64),
            200: np.asarray([2, 4], dtype=np.int64),
        }

    def get_key1_values(self):
        return np.asarray([100, 200], dtype=np.int64)

    def get_trace_seq_for_value(self, key1_val: int, align_to: str = 'display'):
        if align_to != 'display':
            raise ValueError('align_to must be display')
        return self._map[int(key1_val)]


def test_export_manual_picks_grstat_txt_route_exists():
    assert any(
        getattr(r, 'path', '') == '/export_manual_picks_grstat_txt'
        for r in app.router.routes
    )


def test_export_manual_picks_grstat_txt_response_body_has_ffid_and_sentinel(
    monkeypatch,
):
    captured = {}
    fake_reader = _FakeReader()
    prepared = picks_router._PreparedManualPickExport(
        file_name='lineA.sgy',
        reader=fake_reader,
        n_traces=5,
        n_samples=1000,
        dt=0.002,
        p_sorted=np.asarray([0.004, np.nan, 3.0, 2.0, 0.002], dtype=np.float32),
        sorted_to_original=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        p_orig=np.asarray([0.004, np.nan, 3.0, 2.0, 0.002], dtype=np.float32),
    )

    def fake_prepare(request, *, file_id: str, key1_byte: int, key2_byte: int):
        captured['file_id'] = file_id
        captured['key1_byte'] = key1_byte
        captured['key2_byte'] = key2_byte
        return prepared

    def fake_numpy2fbcrd(
        *, dt, fbnum, gather_range, output_name, header_comment, **kwargs
    ):
        del kwargs
        captured['dt'] = float(dt)
        captured['gather_range'] = [int(v) for v in gather_range]
        captured['header_comment'] = header_comment
        arr = np.asarray(fbnum, dtype=np.float32)
        captured['fbnum'] = arr.copy()

        fb_time = np.rint(arr * float(dt)).astype(np.int32)
        nopick = arr == 0
        fb_time[nopick] = -9999
        fb_time[fb_time <= 0] = -9999

        with open(output_name, 'w', encoding='utf-8') as handle:
            for rec_no, row in zip(gather_range, fb_time, strict=False):
                handle.write(f'* rec.no.={int(rec_no):5d}\n')
                start = 1
                end = row.size
                handle.write(f'fb{start:13d}{end:5d}')
                handle.write(''.join(f'{int(v):5d}' for v in row))
                handle.write('\n')
            handle.write('*\n')
        return fb_time

    monkeypatch.setattr(picks_router, '_prepare_manual_pick_export', fake_prepare)
    monkeypatch.setattr(picks_router, '_load_numpy2fbcrd', lambda: fake_numpy2fbcrd)

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(picks_router.asyncio, 'to_thread', fake_to_thread)

    with TestClient(app) as client:
        response = client.get(
            '/export_manual_picks_grstat_txt',
            params={'file_id': 'file_1', 'key2_byte': 193},
        )

    assert response.status_code == 200
    assert captured['file_id'] == 'file_1'
    assert captured['key1_byte'] == 9
    assert captured['key2_byte'] == 193
    assert captured['gather_range'] == [100, 200]
    assert captured['dt'] == 2.0

    # sample=1000 (==n_samples) and 1500 (>n_samples) must become no-pick.
    expected_fbnum = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(captured['fbnum'], expected_fbnum)

    body = response.text
    assert '* rec.no.=  100' in body
    assert '* rec.no.=  200' in body
    assert '-9999' in body
