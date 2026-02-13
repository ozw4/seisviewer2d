from app.core.paths import (
    get_app_data_dir,
    get_picks_npy_dir,
    get_pipeline_jobs_dir,
)


def test_sv_app_data_dir_base_for_default_dirs(monkeypatch, tmp_path):
    base = tmp_path / 'sv_app_data'
    monkeypatch.setenv('SV_APP_DATA_DIR', str(base))
    monkeypatch.delenv('PICKS_NPY_DIR', raising=False)
    monkeypatch.delenv('PIPELINE_JOBS_DIR', raising=False)
    monkeypatch.delenv('RUNNER_TEMP', raising=False)
    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)

    assert get_app_data_dir() == base
    assert get_picks_npy_dir() == base / 'picks_npy'
    assert get_pipeline_jobs_dir() == base / 'pipeline_jobs'
    assert not base.exists()


def test_picks_npy_dir_env_override_has_priority(monkeypatch, tmp_path):
    base = tmp_path / 'sv_base'
    override = tmp_path / 'override' / 'picks'
    monkeypatch.setenv('SV_APP_DATA_DIR', str(base))
    monkeypatch.setenv('PICKS_NPY_DIR', str(override))

    assert get_picks_npy_dir() == override


def test_pipeline_jobs_dir_env_override_has_priority(monkeypatch, tmp_path):
    base = tmp_path / 'sv_base'
    override = tmp_path / 'override' / 'pipeline_jobs'
    monkeypatch.setenv('SV_APP_DATA_DIR', str(base))
    monkeypatch.setenv('PIPELINE_JOBS_DIR', str(override))

    assert get_pipeline_jobs_dir() == override


def test_runner_temp_default_for_app_data_dir(monkeypatch, tmp_path):
    runner_temp = tmp_path / 'runner_temp'
    monkeypatch.delenv('SV_APP_DATA_DIR', raising=False)
    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)
    monkeypatch.setenv('RUNNER_TEMP', str(runner_temp))

    assert get_app_data_dir() == runner_temp / 'seisviewer2d_app_data'
