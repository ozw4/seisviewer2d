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
        assert data['artifact_kind'].item() == 'placeholder_first_break_picks'
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

    summary = json.loads(
        (output_dir / 'expected_static_summary.json').read_text(encoding='utf-8')
    )
    assert summary['schema_version'] == 1
    assert summary['scenario'] == 'one_layer_2d_clean'
    assert summary['status'] == 'placeholder'


def test_refraction_static_ui_fixture_readme_is_created(tmp_path: Path) -> None:
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
