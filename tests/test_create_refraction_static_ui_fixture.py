from __future__ import annotations

import importlib
import json
import math
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


def _single_spaced(text: str) -> str:
    return ' '.join(text.split())


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
        assert data['pick_time_s'].shape == (24 * 96,)
        assert data['pick_time_s'].dtype == np.float32
        assert data['n_traces'].item() == 24 * 96
        assert data['n_samples'].item() == 1000
        assert data['dt'].item() == pytest.approx(0.002)
        np.testing.assert_array_equal(
            data['sorted_to_original'],
            np.arange(24 * 96, dtype=np.int64),
        )


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
    assert metadata['pick_source'] == {
        'kind': 'uploaded_npz',
        'file_field': 'pick_npz',
        'file_name': 'predicted_picks_time_s.npz',
    }
    assert metadata['recommended_static_correction']['linkage'] == {'mode': 'none'}
    assert metadata['recommended_static_correction']['v1_m_s'] == 800.0
    assert metadata['recommended_static_correction']['v2_initial_m_s'] == 2400.0
    assert metadata['recommended_static_correction']['min_offset_m'] == 300.0
    assert metadata['recommended_static_correction']['max_offset_m'] == 1800.0
    assert metadata['recommended_static_correction']['key1_byte'] == 9
    assert metadata['recommended_static_correction']['key2_byte'] == 13
    assert metadata['recommended_static_correction']['exports'] == [
        'canonical_static_table',
        'lsst_plus',
    ]
    assert metadata['recommended_static_correction']['register_corrected_file'] is True
    assert metadata['geometry_headers'] == {
        'source_id_byte': 9,
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
    assert metadata['trace_store_sort_headers'] == {
        'key1_byte': 9,
        'key1_label': 'source_id',
        'key2_byte': 13,
        'key2_label': 'receiver_id',
        'ordering': 'source_major_receiver_minor',
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
    assert summary['truth']['v1_m_s'] == 800.0
    assert summary['truth']['v2_m_s'] == 2400.0
    assert summary['truth']['n_traces'] == 24 * 96
    assert summary['truth']['pick_time_min_s'] >= 0.0
    assert summary['truth']['pick_time_max_s'] > summary['truth']['pick_time_min_s']
    assert summary['truth']['source_t1_min_ms'] > 0.0
    assert summary['truth']['source_t1_max_ms'] > summary['truth']['source_t1_min_ms']
    assert summary['truth']['receiver_t1_min_ms'] > 0.0
    assert summary['truth']['receiver_t1_max_ms'] > summary['truth']['receiver_t1_min_ms']
    assert summary['truth']['weathering_thickness_min_m'] > 0.0
    assert (
        summary['truth']['weathering_thickness_max_m']
        > summary['truth']['weathering_thickness_min_m']
    )
    assert summary['truth']['weathering_correction_min_ms'] < 0.0
    assert summary['truth']['weathering_correction_max_ms'] < 0.0
    assert (
        summary['truth']['weathering_correction_min_ms']
        < summary['truth']['weathering_correction_max_ms']
    )


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
    assert (
        'Open `synthetic_static_2d_one_layer.sgy` in the viewer Loader with '
        '`key1=9`, `key2=13`'
    ) in readme
    assert (
        'Select `predicted_picks_time_s.npz` directly in `First-break pick NPZ`'
    ) in readme
    assert 'You do not need a `file_id` input' in readme
    assert 'create_batch_apply_job' in readme
    assert '/batch/job/<job_id>/files' in readme
    assert 'Open the `Static Correction` tab' in readme
    assert 'Trace sorting for Static Correction UI' in readme
    assert 'key1_byte = 9  # source_id' in readme
    assert 'key2_byte = 13  # receiver_id' in readme
    assert 'Refraction QC' in readme


def test_generated_fixture_readme_mentions_manual_registration_workflow() -> None:
    module = _module()
    readme = module.build_readme(_config(Path('/tmp')))

    assert 'legacy developer-only manual registration workflow' in readme
    assert 'get_job_dir(job_id)' in readme
    assert 'create_batch_apply_job' in readme
    assert '`file_id`/`key1_byte`/`key2_byte` metadata' in readme
    assert '/batch/job/<job_id>/files' in readme
    assert (
        'refraction_static_ui_fixture.md'
        '#legacy-developer-appendix-job-artifact-registration'
    ) in readme


def test_refraction_static_ui_fixture_docs_list_sort_headers() -> None:
    docs = (
        _REPO_ROOT / 'docs' / 'statics' / 'refraction_static_ui_fixture.md'
    ).read_text(encoding='utf-8')

    assert '`key1_byte` | 9 | `source_id_byte`' in docs
    assert '`key2_byte` | 13 | `receiver_id_byte`' in docs
    assert 'recommended_static_correction.key1_byte = 9' in docs
    assert 'recommended_static_correction.key2_byte = 13' in docs
    assert 'Sorted order mismatch' in docs
    assert '`key1=9` and `key2=13`' in docs


def test_fixture_docs_mention_batch_job_files_verification() -> None:
    docs = (
        _REPO_ROOT / 'docs' / 'statics' / 'refraction_static_ui_fixture.md'
    ).read_text(encoding='utf-8')

    assert 'Legacy Developer Appendix: Job Artifact Registration' in docs
    assert 'curl http://localhost:8000/batch/job/ui-fixture-picks-001/files' in docs
    assert '"name": "predicted_picks_time_s.npz"' in docs
    assert '"size_bytes": 12345' in docs
    assert 'no longer the recommended UI workflow' in docs


def test_fixture_docs_mention_existing_job_artifact_mechanism() -> None:
    docs = (
        _REPO_ROOT / 'docs' / 'statics' / 'refraction_static_ui_fixture.md'
    ).read_text(encoding='utf-8')
    normalized = _single_spaced(docs)

    assert 'legacy job-artifact plumbing' in normalized
    assert 'from app.services.pipeline_artifacts import get_job_dir' in docs
    assert 'state.jobs.create_batch_apply_job' in docs
    assert 'job_type = batch_apply' in docs
    assert 'file_id = <file_id returned by import>' in docs
    assert 'key1_byte = 9' in docs
    assert 'key2_byte = 13' in docs
    assert 'artifacts_dir = <path returned by get_job_dir(job_id)>' in docs


def test_fixture_docs_mention_in_memory_ttl_limitation() -> None:
    docs = (
        _REPO_ROOT / 'docs' / 'statics' / 'refraction_static_ui_fixture.md'
    ).read_text(encoding='utf-8')
    normalized = _single_spaced(docs)

    assert 'not a persistent artifact registration API' in normalized
    assert (
        'state` is not automatically available in a normal Python shell'
    ) in normalized
    assert 'live `AppState` object' in docs
    assert 'registration disappears after app restart' in docs
    assert 'TTL-managed' in docs
    assert 'PIPELINE_JOBS_TTL_HOURS' in docs


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


def test_ui_fixture_generation_is_deterministic_for_fixed_seed(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(tmp_path)

    first_fixture = module.build_synthetic_fixture(config)
    second_fixture = module.build_synthetic_fixture(config)
    first_arrays = module.build_pick_arrays(config, first_fixture)
    second_arrays = module.build_pick_arrays(config, second_fixture)

    assert module.build_fixture_metadata(config) == module.build_fixture_metadata(config)
    assert module.build_expected_static_summary(
        config,
        first_fixture,
    ) == module.build_expected_static_summary(config, second_fixture)

    for key, first_value in first_arrays.items():
        np.testing.assert_array_equal(first_value, second_arrays[key])


def test_synthetic_ui_fixture_pick_npz_has_accepted_pick_key(tmp_path: Path) -> None:
    module = _module()
    config = _config(tmp_path)
    fixture = module.build_synthetic_fixture(config)

    arrays = module.build_pick_arrays(config, fixture)

    assert 'pick_time_s' in arrays
    assert arrays['pick_time_s'].shape == (24 * 96,)
    assert arrays['pick_time_s'].dtype == np.float32
    assert arrays['n_traces'].shape == ()
    assert arrays['n_traces'].item() == 24 * 96
    assert arrays['n_samples'].shape == ()
    assert arrays['n_samples'].item() == 1000
    assert arrays['dt'].shape == ()
    assert arrays['dt'].item() == pytest.approx(0.002)
    np.testing.assert_array_equal(
        arrays['sorted_to_original'],
        np.arange(24 * 96, dtype=np.int64),
    )


def test_synthetic_ui_fixture_pick_npz_values_within_record_window(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(tmp_path)
    fixture = module.build_synthetic_fixture(config)

    arrays = module.build_pick_arrays(config, fixture)
    picks = arrays['pick_time_s']

    assert np.all(np.isfinite(picks))
    assert np.all(picks >= 0.0)
    assert np.max(picks) < config.dt_s * config.n_samples


def test_synthetic_ui_fixture_pick_aliases_match(tmp_path: Path) -> None:
    module = _module()
    config = _config(tmp_path)
    fixture = module.build_synthetic_fixture(config)

    arrays = module.build_pick_arrays(config, fixture)

    np.testing.assert_array_equal(arrays['picks_time_s'], arrays['pick_time_s'])
    np.testing.assert_array_equal(
        arrays['predicted_picks_time_s'],
        arrays['pick_time_s'],
    )
    np.testing.assert_array_equal(
        arrays['first_break_time_s'],
        arrays['pick_time_s'],
    )


def test_ui_fixture_pick_npz_is_refraction_loader_compatible(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(tmp_path)
    fixture = module.build_synthetic_fixture(config)
    metadata = module.build_fixture_metadata(config)
    path = tmp_path / 'predicted_picks_time_s.npz'

    np.savez(path, **module.build_pick_arrays(config, fixture))

    accepted_pick_keys = (
        'pick_time_s',
        'picks_time_s',
        'predicted_picks_time_s',
        'first_break_time_s',
    )
    with np.load(path, allow_pickle=False) as data:
        present_keys = [key for key in accepted_pick_keys if key in data.files]
        assert present_keys

        picks = np.asarray(data[present_keys[0]])
        assert picks.shape == (config.n_shots * config.n_receivers,)
        assert np.all(np.isfinite(picks))
        assert np.all(picks >= 0.0)
        assert np.max(picks) < config.n_samples * config.dt_s
        assert int(data['n_traces'].item()) == config.n_shots * config.n_receivers
        assert int(data['n_samples'].item()) == metadata['synthetic_model']['n_samples']
        assert float(data['dt'].item()) == pytest.approx(
            metadata['synthetic_model']['dt_s'],
        )

        for alias in present_keys[1:]:
            np.testing.assert_array_equal(data[alias], picks)


def test_synthetic_ui_fixture_metadata_contains_static_correction_ui_fields(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(tmp_path)

    metadata = module.build_fixture_metadata(config)

    assert metadata['files']['sgy'] == 'synthetic_static_2d_one_layer.sgy'
    assert metadata['files']['pick_artifact'] == 'predicted_picks_time_s.npz'
    assert metadata['pick_source']['kind'] == 'uploaded_npz'
    assert metadata['pick_source']['file_field'] == 'pick_npz'
    assert metadata['pick_source']['file_name'] == 'predicted_picks_time_s.npz'
    assert metadata['geometry_headers']['source_id_byte'] == 9
    assert metadata['geometry_headers']['receiver_id_byte'] == 13
    assert metadata['trace_store_sort_headers']['key1_byte'] == 9
    assert metadata['trace_store_sort_headers']['key2_byte'] == 13
    assert metadata['trace_store_sort_headers']['key1_label'] == 'source_id'
    assert metadata['trace_store_sort_headers']['key2_label'] == 'receiver_id'
    assert (
        metadata['trace_store_sort_headers']['ordering']
        == 'source_major_receiver_minor'
    )
    assert metadata['recommended_static_correction']['model_preset'] == 'one_layer_global'
    assert metadata['recommended_static_correction']['linkage'] == {'mode': 'none'}
    assert metadata['recommended_static_correction']['v1_m_s'] == 800.0
    assert metadata['recommended_static_correction']['v2_initial_m_s'] == 2400.0
    assert metadata['recommended_static_correction']['key1_byte'] == 9
    assert metadata['recommended_static_correction']['key2_byte'] == 13
    assert metadata['recommended_static_correction']['register_corrected_file'] is True

    readme = module.build_readme(config)
    assert 'Trace sorting for Static Correction UI' in readme
    assert 'key1_byte = 9  # source_id' in readme
    assert 'key2_byte = 13  # receiver_id' in readme


def test_synthetic_ui_fixture_expected_summary_contains_truth_ranges(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(tmp_path)
    fixture = module.build_synthetic_fixture(config)

    summary = module.build_expected_static_summary(config, fixture)
    truth = summary['truth']

    assert truth['v1_m_s'] == 800.0
    assert truth['v2_m_s'] == 2400.0
    assert truth['n_traces'] == 24 * 96
    assert truth['pick_time_min_s'] == pytest.approx(float(np.min(fixture.pick_time_s)))
    assert truth['pick_time_max_s'] == pytest.approx(float(np.max(fixture.pick_time_s)))
    assert truth['source_t1_min_ms'] == pytest.approx(
        float(np.min(fixture.source_t1_s) * 1000.0)
    )
    assert truth['source_t1_max_ms'] == pytest.approx(
        float(np.max(fixture.source_t1_s) * 1000.0)
    )
    assert truth['receiver_t1_min_ms'] == pytest.approx(
        float(np.min(fixture.receiver_t1_s) * 1000.0)
    )
    assert truth['receiver_t1_max_ms'] == pytest.approx(
        float(np.max(fixture.receiver_t1_s) * 1000.0)
    )
    assert truth['weathering_thickness_min_m'] > 0.0
    assert truth['weathering_thickness_max_m'] > truth['weathering_thickness_min_m']
    assert truth['weathering_correction_min_ms'] < truth['weathering_correction_max_ms']
    assert truth['weathering_correction_max_ms'] < 0.0


def test_ui_fixture_expected_summary_is_physically_consistent(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(tmp_path)
    fixture = module.build_synthetic_fixture(config)

    summary = module.build_expected_static_summary(config, fixture)
    truth = summary['truth']
    denominator = math.sqrt(config.v2_m_s**2 - config.v1_m_s**2)
    t1_factor = denominator / (config.v2_m_s * config.v1_m_s)

    expected_source_t1_s = fixture.source_thickness_m * t1_factor
    expected_receiver_t1_s = fixture.receiver_thickness_m * t1_factor
    source_sh1_m = fixture.source_t1_s * config.v1_m_s * config.v2_m_s / denominator
    receiver_sh1_m = (
        fixture.receiver_t1_s * config.v1_m_s * config.v2_m_s / denominator
    )
    source_wcor_s = source_sh1_m * (1.0 / config.v2_m_s - 1.0 / config.v1_m_s)
    receiver_wcor_s = receiver_sh1_m * (1.0 / config.v2_m_s - 1.0 / config.v1_m_s)
    all_sh1_m = np.concatenate([source_sh1_m, receiver_sh1_m])
    all_wcor_ms = np.concatenate([source_wcor_s, receiver_wcor_s]) * 1000.0

    np.testing.assert_allclose(fixture.source_t1_s, expected_source_t1_s)
    np.testing.assert_allclose(fixture.receiver_t1_s, expected_receiver_t1_s)
    np.testing.assert_allclose(source_sh1_m, fixture.source_thickness_m)
    np.testing.assert_allclose(receiver_sh1_m, fixture.receiver_thickness_m)
    assert truth['weathering_thickness_min_m'] == pytest.approx(float(np.min(all_sh1_m)))
    assert truth['weathering_thickness_max_m'] == pytest.approx(float(np.max(all_sh1_m)))
    assert truth['weathering_correction_min_ms'] == pytest.approx(
        float(np.min(all_wcor_ms)),
    )
    assert truth['weathering_correction_max_ms'] == pytest.approx(
        float(np.max(all_wcor_ms)),
    )
    assert truth['weathering_thickness_min_m'] > 0.0
    assert truth['weathering_correction_max_ms'] < 0.0


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
