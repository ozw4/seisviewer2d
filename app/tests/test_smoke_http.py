# tests/test_smoke_http.py
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _collect_safe_get_paths() -> list[str]:
	r = client.get('/openapi.json')
	assert r.status_code == 200
	spec = r.json()
	paths = spec.get('paths', {})
	testables: list[str] = []
	for path, ops in paths.items():
		if '{' in path:  # パスパラメータが要るのはスキップ
			continue
		get_op = ops.get('get')
		if not get_op:
			continue
		params = get_op.get('parameters', [])
		# 必須クエリがあるものはスキップ（最小スモーク）
		if any(p.get('required', False) for p in params):
			continue
		testables.append(path)
	return sorted(set(testables))


def test_docs_and_top_pages():
	# ドキュメント＆基本ページ
	assert client.get('/openapi.json').status_code == 200
	assert client.get('/docs').status_code == 200
	# 以下は本リポで提供されている前提（なければ削除/調整）
	assert client.get('/').status_code == 200
	assert client.get('/upload').status_code == 200


def test_all_safe_gets_return_200():
	for p in _collect_safe_get_paths():
		r = client.get(p)
		assert r.status_code == 200, f'{p} -> {r.status_code}'


def test_router_modules_present_and_no_leftovers():
	# ルータ分割の健全性チェック
	routes = [r for r in app.routes if isinstance(r, APIRoute)]
	mods = {r.endpoint.__module__ for r in routes}
	expected = {
		'app.api.routers.upload',
		'app.api.routers.section',
		'app.api.routers.pipeline',
		'app.api.routers.fbpick',
		'app.api.routers.picks',
	}
	missing = expected - mods
	assert not missing, f'Missing router modules: {missing}'

	leftovers = [r for r in routes if r.endpoint.__module__ == 'app.api.endpoints']
	assert not leftovers, (
		f'Leftover routes in app.api.endpoints: {[r.path for r in leftovers]}'
	)
