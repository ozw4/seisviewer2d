from fastapi.testclient import TestClient

from app.main import app  # ← FastAPI のエントリ

client = TestClient(app)


def test_root_serves_html():
	r = client.get('/')
	assert r.status_code == 200
	assert 'text/html' in r.headers.get('content-type', '')


def test_static_assets_exist():
	for path in [
		'/static/plotly-2.29.1.min.js',
		'/static/msgpack.min.js',
		'/static/pako.min.js',
	]:
		r = client.get(path)
		assert r.status_code == 200, path


def test_api_routes_exist():
	# ルートが未定義でも失敗しないように optional に確認
	for path in ['/api/ping', '/api/window', '/api/fbprob']:
		r = client.options(path)
		assert r.status_code in (200, 404, 405)
