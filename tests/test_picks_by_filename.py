from fastapi.testclient import TestClient

from app.main import app
from app.utils import picks_by_name
from app.utils.segy_meta import FILE_REGISTRY


def test_picks_by_filename_roundtrip():
        file_id = 'test-file-id'
        FILE_REGISTRY[file_id] = {'path': '/data/LineA.sgy'}
        picks_by_name.load()
        picks_by_name.delete_pick('LineA.sgy', None, 0, 189)
        picks_by_name.save()

        with TestClient(app) as client:
                response = client.post(
                        '/picks',
                        json={
                                'file_id': file_id,
                                'trace': 10,
                                'time': 0.12,
                                'key1_idx': 0,
                                'key1_byte': 189,
                        },
                )
                assert response.status_code == 200

                response = client.get(
                        '/picks/by-filename',
                        params={'file_name': 'LineA.sgy', 'key1_idx': 0, 'key1_byte': 189},
                )
                assert response.status_code == 200
                assert response.json() == {
                        'picks': [
                                {
                                        'trace': 10,
                                        'time': 0.12,
                                }
                        ]
                }

                response = client.delete(
                        '/picks',
                        params={
                                'file_id': file_id,
                                'trace': 10,
                                'key1_idx': 0,
                                'key1_byte': 189,
                        },
                )
                assert response.status_code == 200

                response = client.get(
                        '/picks/by-filename',
                        params={'file_name': 'LineA.sgy', 'key1_idx': 0, 'key1_byte': 189},
                )
                assert response.status_code == 200
                assert response.json() == {'picks': []}
        FILE_REGISTRY.pop(file_id, None)
        picks_by_name.delete_pick('LineA.sgy', None, 0, 189)
        picks_by_name.save()
