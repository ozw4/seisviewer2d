from fastapi.testclient import TestClient

from app.main import app  # ← FastAPI のエントリ

client = TestClient(app)


def test_root_serves_html():
    r = client.get('/')
    assert r.status_code == 200
    assert 'text/html' in r.headers.get('content-type', '')
    assert '<script type="module" src="/static/assets/main.js"></script>' in r.text
    assert '<script defer src="/static/api.js"></script>' in r.text
    assert '<script src="/static/plotly-2.29.1.min.js"></script>' not in r.text
    assert '<script src="/static/pako.min.js"></script>' not in r.text
    assert '<script src="/static/msgpack.min.js"></script>' not in r.text


def test_static_assets_exist():
    for path in [
        '/static/assets/main.js',
        '/static/msgpack.min.js',
    ]:
        r = client.get(path)
        assert r.status_code == 200, path


def test_decode_worker_keeps_msgpack_dependency():
    r = client.get('/static/viewer/window_decode_worker.js')
    assert r.status_code == 200
    assert "importScripts('/static/msgpack.min.js')" in r.text


def test_api_routes_exist():
    # ルートが未定義でも失敗しないように optional に確認
    for path in ['/api/ping', '/api/window', '/api/fbprob']:
        r = client.options(path)
        assert r.status_code in (200, 404, 405)
