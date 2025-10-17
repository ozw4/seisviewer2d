"""Lightweight smoke checks for API route availability."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


ROUTES: list[dict[str, object]] = [
        {
                'method': 'POST',
                'path': '/open_segy',
                'data': {
                        'original_name': 'missing.sgy',
                        'key1_byte': 189,
                        'key2_byte': 193,
                },
                'expected': {404},
        },
        {
                'method': 'POST',
                'path': '/upload_segy',
                'expected': {422},
        },
        {
                'method': 'GET',
                'path': '/file_info',
                'params': {'file_id': 'missing'},
                'expected': {404},
        },
        {
                'method': 'GET',
                'path': '/get_key1_values',
                'params': {'file_id': 'missing', 'key1_byte': 189, 'key2_byte': 193},
                'expected': {404},
        },
        {
                'method': 'GET',
                'path': '/get_section',
                'params': {
                        'file_id': 'missing',
                        'key1_val': 0,
                        'key1_byte': 189,
                        'key2_byte': 193,
                },
                'expected': {404},
        },
        {
                'method': 'GET',
                'path': '/get_section_bin',
                'params': {
                        'file_id': 'missing',
                        'key1_val': 0,
                        'key1_byte': 189,
                        'key2_byte': 193,
                },
                'expected': {404},
        },
        {
                'method': 'GET',
                'path': '/get_section_window_bin',
                'params': {
                        'file_id': 'missing',
                        'key1_val': 0,
                        'key1_byte': 189,
                        'key2_byte': 193,
                        'x0': 0,
                        'x1': 0,
                        'y0': 0,
                        'y1': 0,
                        'step_x': 1,
                        'step_y': 1,
                },
                'expected': {404},
        },
        {
                'method': 'POST',
                'path': '/fbpick_section_bin',
                'json': {
                        'file_id': 'missing',
                        'key1_val': 0,
                        'key1_byte': 189,
                        'key2_byte': 193,
                        'tile_h': 128,
                        'tile_w': 128,
                        'overlap': 32,
                        'amp': True,
                },
                'expected': {409},
        },
        {
                'method': 'GET',
                'path': '/fbpick_job_status',
                'params': {'job_id': 'missing'},
                'expected': {404},
        },
        {
                'method': 'GET',
                'path': '/get_fbpick_section_bin',
                'params': {'job_id': 'missing'},
                'expected': {404},
        },
        {
                'method': 'POST',
                'path': '/pipeline/section',
                'params': {
                        'file_id': 'missing',
                        'key1_val': 0,
                        'key1_byte': 189,
                        'key2_byte': 193,
                },
                'json': {'spec': {'steps': []}},
                'expected': {404},
        },
        {
                'method': 'POST',
                'path': '/pipeline/all',
                'params': {'file_id': 'missing', 'key1_byte': 189, 'key2_byte': 193},
                'json': {'spec': {'steps': []}, 'taps': []},
                'expected': {200},
        },
        {
                'method': 'GET',
                'path': '/pipeline/job/missing/status',
                'expected': {404},
        },
        {
                'method': 'GET',
                'path': '/pipeline/job/missing/artifact',
                'params': {'key1_val': 0, 'tap': 'fbpick'},
                'expected': {404},
        },
        {
                'method': 'POST',
                'path': '/picks',
                'json': {
                        'file_id': 'missing',
                        'trace': 0,
                        'time': 0.0,
                        'key1_val': 0,
                        'key1_byte': 189,
                },
                'expected': {200},
        },
        {
                'method': 'GET',
                'path': '/picks',
                'params': {'file_id': 'missing', 'key1_val': 0, 'key1_byte': 189},
                'expected': {200},
        },
        {
                'method': 'GET',
                'path': '/export_manual_picks_all_npy',
                'params': {'file_id': 'missing', 'key1_byte': 189, 'key2_byte': 193},
                'expected': {404},
        },
        {
                'method': 'DELETE',
                'path': '/picks',
                'params': {
                        'file_id': 'missing',
                        'trace': 0,
                        'key1_val': 0,
                        'key1_byte': 189,
                },
                'expected': {200},
        },
]


def main() -> None:
        """Execute smoke checks and exit with non-zero on failure."""
        for spec in ROUTES:
                method = spec['method']
                path = spec['path']
                expected = set(spec['expected'])
                kwargs = {
                        key: value
                        for key, value in spec.items()
                        if key in {'params', 'json', 'data', 'files'}
                }
                response = client.request(method, path, **kwargs)
                status = response.status_code
                ok = status in expected
                print(f'{method} {path} -> {status} (expected {sorted(expected)})')
                if not ok:
                        raise SystemExit(1)
        print('Smoke route checks completed.')


if __name__ == '__main__':
        main()

