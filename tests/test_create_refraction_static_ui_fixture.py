from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / 'scripts' / 'create_refraction_static_ui_fixture.py'
_EXPECTED_FILES = {
    'synthetic_static_2d_one_layer.sgy',
    'predicted_picks_time_s.npz',
    'fixture_metadata.json',
    'expected_static_summary.json',
    'README.md',
}


def _module():
    return importlib.import_module('scripts.create_refraction_static_ui_fixture')


def _config(tmp_path: Path, **overrides):
    module = _module()
    values = {
        'output_dir': tmp_path / 'fixture',
        'scenario': 'one_layer_2d_clean',
        'seed': 7,
        'n_shots': 24,
        'n_receivers': 96,
        'shot_interval_m': 100.0,
        'receiver_interval_m': 25.0,
        'dt_s': 0.002,
        'n_samples': 1000,
        'v1_m_s': 800.0,
        'v2_m_s': 2400.0,
        'noise_std': 0.02,
        'overwrite': False,
    }
    values.update(overrides)
    return module.FixtureConfig(**values)


def _skip_without_segyio() -> None:
    pytest.importorskip('segyio')


def test_refraction_static_ui_fixture_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), '--help'],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert '--output-dir' in result.stdout
    assert '--scenario' in result.stdout
    assert 'one_layer_2d_clean' in result.stdout


def test_refraction_static_ui_fixture_writes_output_layout(tmp_path: Path) -> None:
    _skip_without_segyio()
    output_dir = tmp_path / 'fixture'

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            '--output-dir',
            str(output_dir),
            '--scenario',
            'one_layer_2d_clean',
            '--seed',
            '7',
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert str(output_dir) in result.stdout
    assert {path.name for path in output_dir.iterdir()} == _EXPECTED_FILES
    assert (output_dir / 'synthetic_static_2d_one_layer.sgy').read_bytes()
    with np.load(output_dir / 'predicted_picks_time_s.npz', allow_pickle=False) as data:
        assert data['artifact_kind'].item() == 'synthetic_first_break_picks'
        assert data['scenario'].item() == 'one_layer_2d_clean'
        assert data['pick_time_s'].shape == (24, 96)


def test_refraction_static_ui_fixture_refuses_overwrite_without_flag(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / 'fixture'
    output_dir.mkdir()
    (output_dir / 'existing.txt').write_text('keep me', encoding='utf-8')

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), '--output-dir', str(output_dir)],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert result.returncode != 0
    assert 'output directory is not empty' in result.stderr
    assert (output_dir / 'existing.txt').read_text(encoding='utf-8') == 'keep me'


def test_refraction_static_ui_fixture_overwrite_replaces_layout(
    tmp_path: Path,
) -> None:
    _skip_without_segyio()
    output_dir = tmp_path / 'fixture'
    output_dir.mkdir()
    (output_dir / 'existing.txt').write_text('replace me', encoding='utf-8')

    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            '--output-dir',
            str(output_dir),
            '--overwrite',
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert {path.name for path in output_dir.iterdir()} == _EXPECTED_FILES


def test_refraction_static_ui_fixture_metadata_is_valid_json(tmp_path: Path) -> None:
    _skip_without_segyio()
    output_dir = tmp_path / 'fixture'

    subprocess.run(
        [sys.executable, str(_SCRIPT), '--output-dir', str(output_dir)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    metadata = json.loads(
        (output_dir / 'fixture_metadata.json').read_text(encoding='utf-8')
    )
    assert metadata['schema_version'] == 1
    assert metadata['scenario'] == 'one_layer_2d_clean'
    assert metadata['files'] == {
        'sgy': 'synthetic_static_2d_one_layer.sgy',
        'pick_artifact': 'predicted_picks_time_s.npz',
    }
    assert metadata['ui_workflow']['upload_sgy_via_normal_ui'] is True
    assert metadata['ui_workflow']['static_correction_tab'] == 'Static Correction'
    assert metadata['ui_workflow']['linkage_default'] == 'none'
    assert metadata['recommended_static_correction']['linkage'] == {'mode': 'none'}
    assert metadata['recommended_static_correction']['exports'] == [
        'canonical_static_table',
        'lsst_plus',
    ]
    assert metadata['geometry_headers'] == {
        'source_id_byte': 17,
        'receiver_id_byte': 13,
        'source_x_byte': 73,
        'source_y_byte': 77,
        'receiver_x_byte': 81,
        'receiver_y_byte': 85,
        'source_elevation_byte': 41,
        'receiver_elevation_byte': 45,
        'offset_byte': 37,
        'coordinate_scalar_byte': 71,
        'elevation_scalar_byte': 69,
    }
    assert metadata['synthetic_model'] == {
        'v1_m_s': 800.0,
        'v2_m_s': 2400.0,
        'dt_s': 0.002,
        'n_samples': 1000,
    }

    summary = json.loads(
        (output_dir / 'expected_static_summary.json').read_text(encoding='utf-8')
    )
    assert summary['schema_version'] == 1
    assert summary['scenario'] == 'one_layer_2d_clean'
    assert summary['status'] == 'placeholder'


def test_refraction_static_ui_fixture_readme_is_created(tmp_path: Path) -> None:
    _skip_without_segyio()
    output_dir = tmp_path / 'fixture'

    subprocess.run(
        [sys.executable, str(_SCRIPT), '--output-dir', str(output_dir)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    readme = (output_dir / 'README.md').read_text(encoding='utf-8')
    assert 'Static Correction UI refraction workflow' in readme
    assert 'one_layer_2d_clean' in readme


def test_refraction_static_ui_fixture_import_has_no_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module('scripts.create_refraction_static_ui_fixture')
    assert module.SGY_NAME == 'synthetic_static_2d_one_layer.sgy'
    assert list(tmp_path.iterdir()) == []


def test_synthetic_ui_fixture_geometry_shapes(tmp_path: Path) -> None:
    module = _module()
    config = _config(tmp_path)

    fixture = module.build_synthetic_fixture(config)

    assert fixture.traces.shape == (24 * 96, 1000)
    assert fixture.traces.dtype == np.float32
    assert fixture.source_id.shape == (24 * 96,)
    assert fixture.receiver_id.shape == (24 * 96,)
    assert fixture.source_x_m.shape == (24 * 96,)
    assert fixture.receiver_x_m.shape == (24 * 96,)
    assert fixture.offset_m.shape == (24 * 96,)
    assert fixture.source_y_m.tolist() == [0.0] * (24 * 96)
    assert fixture.receiver_y_m.tolist() == [0.0] * (24 * 96)
    assert fixture.source_id[:96].tolist() == [1] * 96
    assert fixture.receiver_id[:5].tolist() == [1, 2, 3, 4, 5]
    assert fixture.offset_m[0] == pytest.approx(0.0)
    assert fixture.offset_m[1] == pytest.approx(25.0)


def test_synthetic_ui_fixture_pick_times_are_finite_and_in_record_window(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(tmp_path)

    fixture = module.build_synthetic_fixture(config)
    expected = (
        fixture.source_t1_s
        + fixture.receiver_t1_s
        + fixture.offset_m / config.v2_m_s
    )

    assert np.all(np.isfinite(fixture.pick_time_s))
    assert np.all(fixture.pick_time_s >= 0.0)
    assert np.max(fixture.pick_time_s) < config.dt_s * config.n_samples
    np.testing.assert_allclose(fixture.pick_time_s, expected)


def test_synthetic_ui_fixture_waveforms_have_first_break_energy(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(tmp_path, noise_std=0.0)

    fixture = module.build_synthetic_fixture(config)
    pick_samples = np.rint(fixture.pick_time_s / config.dt_s).astype(np.int64)
    trace_indices = np.asarray([0, 75, 511, fixture.traces.shape[0] - 1])

    for trace_index in trace_indices:
        pick_sample = pick_samples[trace_index]
        window = fixture.traces[trace_index, pick_sample - 2 : pick_sample + 3]
        assert np.max(np.abs(window)) > 0.75
        assert fixture.traces[trace_index, pick_sample] == pytest.approx(1.0)


def test_synthetic_ui_fixture_generation_is_deterministic(tmp_path: Path) -> None:
    module = _module()
    config = _config(tmp_path)

    first = module.build_synthetic_fixture(config)
    second = module.build_synthetic_fixture(config)

    np.testing.assert_array_equal(first.traces, second.traces)
    np.testing.assert_array_equal(first.pick_time_s, second.pick_time_s)


def test_synthetic_ui_fixture_sgy_writer_skips_without_segyio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config = _config(tmp_path, n_shots=1, n_receivers=1, n_samples=16)
    fixture = module.build_synthetic_fixture(config)
    real_import = __import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'segyio':
            raise ImportError('blocked segyio import')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr('builtins.__import__', _blocked_import)

    with pytest.raises(RuntimeError, match='This script requires segyio'):
        module.write_synthetic_segy(tmp_path / 'out.sgy', config=config, fixture=fixture)


def test_synthetic_ui_fixture_sgy_headers_round_trip_when_segyio_available(
    tmp_path: Path,
) -> None:
    segyio = pytest.importorskip('segyio')
    module = _module()
    config = _config(
        tmp_path,
        n_shots=2,
        n_receivers=3,
        n_samples=256,
        noise_std=0.0,
    )
    fixture = module.build_synthetic_fixture(config)
    path = tmp_path / 'synthetic_static_2d_one_layer.sgy'

    module.write_synthetic_segy(path, config=config, fixture=fixture)

    with segyio.open(str(path), 'r', ignore_geometry=True) as segy_file:
        assert segy_file.tracecount == 6
        assert segy_file.bin[segyio.BinField.Interval] == 2000
        traces = np.asarray([np.array(segy_file.trace[i]) for i in range(6)])
        assert np.all(np.isfinite(traces))
        np.testing.assert_allclose(traces, fixture.traces)

        for index in range(6):
            header = segy_file.header[index]
            assert header[module.GEOMETRY_HEADERS['source_id_byte']] == int(
                fixture.source_id[index]
            )
            assert header[module.GEOMETRY_HEADERS['receiver_id_byte']] == int(
                fixture.receiver_id[index]
            )
            assert header[module.GEOMETRY_HEADERS['source_x_byte']] == int(
                round(fixture.source_x_m[index])
            )
            assert header[module.GEOMETRY_HEADERS['source_y_byte']] == 0
            assert header[module.GEOMETRY_HEADERS['receiver_x_byte']] == int(
                round(fixture.receiver_x_m[index])
            )
            assert header[module.GEOMETRY_HEADERS['receiver_y_byte']] == 0
            assert header[module.GEOMETRY_HEADERS['source_elevation_byte']] == int(
                round(fixture.source_elevation_m[index])
            )
            assert header[module.GEOMETRY_HEADERS['receiver_elevation_byte']] == int(
                round(fixture.receiver_elevation_m[index])
            )
            assert header[module.GEOMETRY_HEADERS['offset_byte']] == int(
                round(fixture.offset_m[index])
            )
            assert header[module.GEOMETRY_HEADERS['coordinate_scalar_byte']] == 1
            assert header[module.GEOMETRY_HEADERS['elevation_scalar_byte']] == 1
