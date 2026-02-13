# app/tests/test_file_info.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.segy_meta import FILE_REGISTRY


@pytest.fixture()
def client() -> TestClient:
    FILE_REGISTRY.clear()
    with TestClient(app) as c:
        yield c
    FILE_REGISTRY.clear()


def test_file_info_returns_basename_for_path_field(client: TestClient):
    FILE_REGISTRY['fid-path'] = {'path': '/tmp/some/LineA.sgy'}

    res = client.get('/file_info', params={'file_id': 'fid-path'})
    assert res.status_code == 200
    assert res.json() == {'file_name': 'LineA.sgy'}


def test_file_info_returns_basename_for_store_path_field(client: TestClient):
    FILE_REGISTRY['fid-store'] = {'store_path': '/var/cache/traces/LineB'}

    res = client.get('/file_info', params={'file_id': 'fid-store'})
    assert res.status_code == 200
    assert res.json() == {'file_name': 'LineB'}


def test_file_info_prefers_path_over_store_path_when_both_present(client: TestClient):
    FILE_REGISTRY['fid-both'] = {
        'path': '/tmp/primary/Primary.sgy',
        'store_path': '/tmp/secondary/Secondary',
    }

    res = client.get('/file_info', params={'file_id': 'fid-both'})
    assert res.status_code == 200
    assert res.json() == {'file_name': 'Primary.sgy'}


def test_file_info_unknown_file_id_returns_404(client: TestClient):
    res = client.get('/file_info', params={'file_id': 'no-such-id'})
    assert res.status_code == 404
    assert res.json().get('detail') == 'Unknown file_id'


def test_file_info_path_traversal_like_store_path_is_safely_reduced_to_basename(
    client: TestClient,
):
    FILE_REGISTRY['fid-trav'] = {'store_path': '../../etc/passwd'}

    res = client.get('/file_info', params={'file_id': 'fid-trav'})
    assert res.status_code == 200
    assert res.json() == {'file_name': 'passwd'}

    name = res.json()['file_name']
    assert '/' not in name
    assert '\\' not in name
    assert name != ''


def test_file_info_windows_style_path_traversal_like_store_path_is_safely_reduced_to_basename(
    client: TestClient,
):
    FILE_REGISTRY['fid-win-trav'] = {'store_path': '..\\..\\Windows\\System32\\cmd.exe'}

    res = client.get('/file_info', params={'file_id': 'fid-win-trav'})
    assert res.status_code == 200
    assert res.json() == {'file_name': 'cmd.exe'}

    name = res.json()['file_name']
    assert '/' not in name
    assert '\\' not in name
    assert name != ''
