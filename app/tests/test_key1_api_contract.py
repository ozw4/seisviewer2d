from fastapi.testclient import TestClient

from app.main import app


def test_picks_rejects_legacy_body_param() -> None:
    legacy_key = 'key1' + '_val'
    with TestClient(app) as client:
        res = client.post(
            '/picks',
            json={
                'file_id': 'dummy',
                'trace': 0,
                'time': 0.0,
                'key1': 10,
                legacy_key: 10,
                'key1_byte': 189,
            },
        )
    assert res.status_code == 422


def test_fbpick_section_bin_rejects_legacy_body_param() -> None:
    legacy_key = 'key1' + '_val'
    with TestClient(app) as client:
        res = client.post(
            '/fbpick_section_bin',
            json={
                'file_id': 'dummy',
                'key1': 10,
                legacy_key: 10,
            },
        )
    assert res.status_code == 422


def test_fbpick_predict_rejects_legacy_body_param() -> None:
    legacy_key = 'key1' + '_val'
    with TestClient(app) as client:
        res = client.post(
            '/fbpick_predict',
            json={
                'file_id': 'dummy',
                'key1': 10,
                legacy_key: 10,
                'sigma_ms_max': 10.0,
            },
        )
    assert res.status_code == 422


def test_get_section_rejects_legacy_query_param() -> None:
    legacy_key = 'key1' + '_val'
    with TestClient(app) as client:
        res = client.get(
            '/get_section',
            params={'file_id': 'dummy', 'key1': 10, legacy_key: 10},
        )
    assert res.status_code == 422


def test_get_section_window_bin_rejects_legacy_query_param() -> None:
    legacy_key = 'key1' + '_val'
    with TestClient(app) as client:
        res = client.get(
            '/get_section_window_bin',
            params={
                'file_id': 'dummy',
                'key1': 10,
                legacy_key: 10,
                'x0': 0,
                'x1': 0,
                'y0': 0,
                'y1': 0,
            },
        )
    assert res.status_code == 422


def test_pipeline_section_rejects_legacy_query_param() -> None:
    legacy_key = 'key1' + '_val'
    with TestClient(app) as client:
        res = client.post(
            '/pipeline/section',
            params={'file_id': 'dummy', 'key1': 10, legacy_key: 10},
            json={'spec': {'steps': []}},
        )
    assert res.status_code == 422


def test_pipeline_job_artifact_rejects_legacy_query_param() -> None:
    legacy_key = 'key1' + '_val'
    with TestClient(app) as client:
        res = client.get(
            '/pipeline/job/missing/artifact',
            params={'key1': 10, legacy_key: 10, 'tap': 'final'},
        )
    assert res.status_code == 422


def test_picks_get_rejects_legacy_query_param() -> None:
    legacy_key = 'key1' + '_val'
    with TestClient(app) as client:
        res = client.get(
            '/picks',
            params={'file_id': 'dummy', 'key1': 10, legacy_key: 10, 'key1_byte': 189},
        )
    assert res.status_code == 422


def test_section_stats_rejects_legacy_query_param() -> None:
    legacy_idx = 'key1' + '_idx'
    with TestClient(app) as client:
        res = client.get(
            '/section/stats',
            params={
                'file_id': 'dummy',
                'baseline': 'raw',
                'key1': 10,
                legacy_idx: 10,
            },
        )
    assert res.status_code == 422
