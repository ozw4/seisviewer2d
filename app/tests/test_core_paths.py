from app.core.paths import (
    get_app_data_dir,
    get_picks_npy_dir,
    get_processed_upload_dir,
    get_pipeline_jobs_dir,
    get_trace_store_dir,
    get_upload_dir,
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


def test_upload_paths_default_under_sv_app_data_dir(monkeypatch, tmp_path):
    base = tmp_path / 'sv_uploads'
    monkeypatch.setenv('SV_APP_DATA_DIR', str(base))
    monkeypatch.delenv('SV_UPLOAD_DIR', raising=False)
    monkeypatch.delenv('SV_PROCESSED_DIR', raising=False)
    monkeypatch.delenv('SV_TRACE_DIR', raising=False)
    monkeypatch.delenv('RUNNER_TEMP', raising=False)
    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)

    assert get_upload_dir() == base / 'uploads'
    assert get_processed_upload_dir() == base / 'uploads' / 'processed'
    assert get_trace_store_dir() == base / 'uploads' / 'processed' / 'traces'
    assert not base.exists()


def test_upload_dir_env_override_has_priority(monkeypatch, tmp_path):
    base = tmp_path / 'sv_base'
    override = tmp_path / 'override' / 'uploads'
    monkeypatch.setenv('SV_APP_DATA_DIR', str(base))
    monkeypatch.setenv('SV_UPLOAD_DIR', str(override))
    monkeypatch.delenv('SV_PROCESSED_DIR', raising=False)
    monkeypatch.delenv('SV_TRACE_DIR', raising=False)

    assert get_upload_dir() == override
    assert get_processed_upload_dir() == override / 'processed'
    assert get_trace_store_dir() == override / 'processed' / 'traces'
    assert not override.exists()


def test_processed_upload_dir_env_override_has_priority(monkeypatch, tmp_path):
    base = tmp_path / 'sv_base'
    upload_override = tmp_path / 'override' / 'uploads'
    processed_override = tmp_path / 'override' / 'processed'
    monkeypatch.setenv('SV_APP_DATA_DIR', str(base))
    monkeypatch.setenv('SV_UPLOAD_DIR', str(upload_override))
    monkeypatch.setenv('SV_PROCESSED_DIR', str(processed_override))
    monkeypatch.delenv('SV_TRACE_DIR', raising=False)

    assert get_processed_upload_dir() == processed_override
    assert get_trace_store_dir() == processed_override / 'traces'
    assert not processed_override.exists()


def test_trace_store_dir_env_override_has_priority(monkeypatch, tmp_path):
    base = tmp_path / 'sv_base'
    upload_override = tmp_path / 'override' / 'uploads'
    processed_override = tmp_path / 'override' / 'processed'
    trace_override = tmp_path / 'override' / 'traces'
    monkeypatch.setenv('SV_APP_DATA_DIR', str(base))
    monkeypatch.setenv('SV_UPLOAD_DIR', str(upload_override))
    monkeypatch.setenv('SV_PROCESSED_DIR', str(processed_override))
    monkeypatch.setenv('SV_TRACE_DIR', str(trace_override))

    assert get_trace_store_dir() == trace_override
    assert not trace_override.exists()
