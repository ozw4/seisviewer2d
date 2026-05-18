from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.create_refraction_static_ui_fixture import (
    GEOMETRY_HEADERS,
    SCENARIO_ONE_LAYER_2D_CLEAN,
    TRACE_STORE_SORT_HEADERS,
    FixtureConfig,
    SyntheticFixture,
    build_pick_arrays,
    build_synthetic_fixture,
)

FILE_ID = 'synthetic-static-ui-fixture'


def default_ui_fixture_config(output_dir: Path) -> FixtureConfig:
    return FixtureConfig(
        output_dir=output_dir,
        scenario=SCENARIO_ONE_LAYER_2D_CLEAN,
        seed=7,
        n_shots=24,
        n_receivers=96,
        shot_interval_m=100.0,
        receiver_interval_m=25.0,
        dt_s=0.002,
        n_samples=1000,
        v1_m_s=800.0,
        v2_m_s=2400.0,
        noise_std=0.02,
        overwrite=True,
    )


def build_ui_fixture_trace_store(root: Path) -> tuple[FixtureConfig, SyntheticFixture, Path]:
    config = default_ui_fixture_config(root)
    fixture = build_synthetic_fixture(config)
    store_dir = root / 'trace_store'
    store_dir.mkdir(parents=True)

    np.save(store_dir / 'traces.npy', fixture.traces)
    np.savez(
        store_dir / 'index.npz',
        key1_values=np.arange(1, config.n_shots + 1, dtype=np.int64),
        key1_offsets=(
            np.arange(config.n_shots, dtype=np.int64) * int(config.n_receivers)
        ),
        key1_counts=np.full(config.n_shots, config.n_receivers, dtype=np.int64),
        sorted_to_original=np.arange(fixture.traces.shape[0], dtype=np.int64),
    )
    _write_headers(store_dir, fixture)
    (store_dir / 'meta.json').write_text(
        json.dumps(
            {
                'schema_version': 1,
                'dtype': 'float32',
                'n_traces': int(fixture.traces.shape[0]),
                'n_samples': int(fixture.traces.shape[1]),
                'key_bytes': {
                    'key1': TRACE_STORE_SORT_HEADERS['key1_byte'],
                    'key2': TRACE_STORE_SORT_HEADERS['key2_byte'],
                },
                'sorted_by': ['key1', 'key2'],
                'dt': float(config.dt_s),
                'original_segy_path': '',
                'source_sha256': None,
                'original_name': 'synthetic_static_2d_one_layer.sgy',
            }
        ),
        encoding='utf-8',
    )
    return config, fixture, store_dir


def ui_fixture_pick_npz_bytes(config: FixtureConfig, fixture: SyntheticFixture) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **build_pick_arrays(config, fixture))
    return buffer.getvalue()


def ui_fixture_static_correction_payload() -> dict[str, Any]:
    return {
        'file_id': FILE_ID,
        'key1_byte': TRACE_STORE_SORT_HEADERS['key1_byte'],
        'key2_byte': TRACE_STORE_SORT_HEADERS['key2_byte'],
        'pick_source': {'kind': 'uploaded_npz'},
        'geometry': {
            **{
                key: value
                for key, value in GEOMETRY_HEADERS.items()
                if key != 'offset_byte'
            },
            'source_depth_byte': None,
            'coordinate_unit': 'm',
            'elevation_unit': 'm',
        },
        'linkage': {'mode': 'none'},
        'model': {
            'method': 'gli_variable_thickness',
            'weathering_velocity_m_s': 800.0,
            'bedrock_velocity_mode': 'fixed_global',
            'bedrock_velocity_m_s': 2400.0,
            'initial_bedrock_velocity_m_s': None,
            'min_bedrock_velocity_m_s': 1200.0,
            'max_bedrock_velocity_m_s': 6000.0,
            'max_weathering_thickness_m': None,
        },
        'moveout': {
            'model': 'head_wave_linear_offset',
            'distance_source': 'geometry',
            'offset_byte': GEOMETRY_HEADERS['offset_byte'],
            'min_offset_m': 300.0,
            'max_offset_m': 1800.0,
            'allow_missing_offset': False,
            'max_geometry_offset_mismatch_m': None,
        },
        'solver': {
            'damping': 0.01,
            'min_picks_per_node': 1,
            'max_abs_half_intercept_time_ms': 500.0,
            'robust': {
                'enabled': True,
                'method': 'mad',
                'threshold': 3.5,
                'max_iterations': 5,
                'min_used_fraction': 0.5,
                'min_used_observations': 1,
            },
        },
        'datum': {'mode': 'none'},
        'conversion': {'mode': 't1lsst_1layer'},
        'apply': {
            'mode': 'refraction_from_raw',
            'interpolation': 'linear',
            'fill_value': 0.0,
            'max_abs_shift_ms': 250.0,
            'output_dtype': 'float32',
            'register_corrected_file': False,
        },
    }


def _write_headers(store_dir: Path, fixture: SyntheticFixture) -> None:
    headers = {
        GEOMETRY_HEADERS['source_id_byte']: fixture.source_id,
        GEOMETRY_HEADERS['receiver_id_byte']: fixture.receiver_id,
        GEOMETRY_HEADERS['source_x_byte']: np.rint(fixture.source_x_m),
        GEOMETRY_HEADERS['source_y_byte']: np.rint(fixture.source_y_m),
        GEOMETRY_HEADERS['receiver_x_byte']: np.rint(fixture.receiver_x_m),
        GEOMETRY_HEADERS['receiver_y_byte']: np.rint(fixture.receiver_y_m),
        GEOMETRY_HEADERS['source_elevation_byte']: np.rint(
            fixture.source_elevation_m
        ),
        GEOMETRY_HEADERS['receiver_elevation_byte']: np.rint(
            fixture.receiver_elevation_m
        ),
        GEOMETRY_HEADERS['offset_byte']: np.rint(fixture.offset_m),
        GEOMETRY_HEADERS['coordinate_scalar_byte']: np.ones(
            fixture.traces.shape[0],
            dtype=np.int32,
        ),
        GEOMETRY_HEADERS['elevation_scalar_byte']: np.ones(
            fixture.traces.shape[0],
            dtype=np.int32,
        ),
    }
    for byte, values in headers.items():
        np.save(store_dir / f'headers_byte_{byte}.npy', np.asarray(values))
