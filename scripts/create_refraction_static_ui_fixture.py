from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np


SCENARIO_ONE_LAYER_2D_CLEAN = 'one_layer_2d_clean'
SUPPORTED_SCENARIOS = (SCENARIO_ONE_LAYER_2D_CLEAN,)

SGY_NAME = 'synthetic_static_2d_one_layer.sgy'
PICK_ARTIFACT_NAME = 'predicted_picks_time_s.npz'
METADATA_NAME = 'fixture_metadata.json'
EXPECTED_SUMMARY_NAME = 'expected_static_summary.json'
README_NAME = 'README.md'


@dataclass(frozen=True)
class FixtureConfig:
    output_dir: Path
    scenario: str
    seed: int
    n_shots: int
    n_receivers: int
    shot_interval_m: float
    receiver_interval_m: float
    dt_s: float
    n_samples: int
    v1_m_s: float
    v2_m_s: float
    noise_std: float
    overwrite: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Create a small deterministic refraction-static fixture directory '
            'for manual Static Correction UI workflows.'
        )
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        type=Path,
        help='Directory where fixture files will be written.',
    )
    parser.add_argument(
        '--scenario',
        choices=SUPPORTED_SCENARIOS,
        default=SCENARIO_ONE_LAYER_2D_CLEAN,
        help='Synthetic fixture scenario to generate.',
    )
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n-shots', type=int, default=24)
    parser.add_argument('--n-receivers', type=int, default=96)
    parser.add_argument('--shot-interval-m', type=float, default=100.0)
    parser.add_argument('--receiver-interval-m', type=float, default=25.0)
    parser.add_argument('--dt-s', type=float, default=0.002)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--v1-m-s', type=float, default=800.0)
    parser.add_argument('--v2-m-s', type=float, default=2400.0)
    parser.add_argument('--noise-std', type=float, default=0.02)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace files in an existing non-empty output directory.',
    )
    return parser


def config_from_args(args: argparse.Namespace) -> FixtureConfig:
    return FixtureConfig(
        output_dir=args.output_dir,
        scenario=args.scenario,
        seed=args.seed,
        n_shots=args.n_shots,
        n_receivers=args.n_receivers,
        shot_interval_m=args.shot_interval_m,
        receiver_interval_m=args.receiver_interval_m,
        dt_s=args.dt_s,
        n_samples=args.n_samples,
        v1_m_s=args.v1_m_s,
        v2_m_s=args.v2_m_s,
        noise_std=args.noise_std,
        overwrite=args.overwrite,
    )


def validate_config(config: FixtureConfig) -> None:
    if config.n_shots <= 0:
        raise ValueError('--n-shots must be greater than 0')
    if config.n_receivers <= 0:
        raise ValueError('--n-receivers must be greater than 0')
    if config.shot_interval_m <= 0.0:
        raise ValueError('--shot-interval-m must be greater than 0')
    if config.receiver_interval_m <= 0.0:
        raise ValueError('--receiver-interval-m must be greater than 0')
    if config.dt_s <= 0.0:
        raise ValueError('--dt-s must be greater than 0')
    if config.n_samples <= 0:
        raise ValueError('--n-samples must be greater than 0')
    if config.v1_m_s <= 0.0:
        raise ValueError('--v1-m-s must be greater than 0')
    if config.v2_m_s <= config.v1_m_s:
        raise ValueError('--v2-m-s must be greater than --v1-m-s')
    if config.noise_std < 0.0:
        raise ValueError('--noise-std must be greater than or equal to 0')


def ensure_output_dir(config: FixtureConfig) -> None:
    output_dir = config.output_dir
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f'--output-dir exists and is not a directory: {output_dir}')
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise ValueError(
            f'output directory is not empty: {output_dir}; pass --overwrite to replace '
            'fixture files'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        for child in output_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


def build_placeholder_pick_arrays(config: FixtureConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    shot_x_m = (
        np.arange(config.n_shots, dtype=np.float64) * config.shot_interval_m
    )
    receiver_x_m = (
        np.arange(config.n_receivers, dtype=np.float64) * config.receiver_interval_m
    )
    shot_index, receiver_index = np.meshgrid(
        np.arange(config.n_shots, dtype=np.int32),
        np.arange(config.n_receivers, dtype=np.int32),
        indexing='ij',
    )
    offset_m = np.abs(shot_x_m[shot_index] - receiver_x_m[receiver_index])
    base_time_s = offset_m / config.v2_m_s
    noise_s = rng.normal(
        loc=0.0,
        scale=config.noise_std,
        size=(config.n_shots, config.n_receivers),
    )

    return {
        'artifact_kind': np.asarray('placeholder_first_break_picks'),
        'schema_version': np.asarray(1, dtype=np.int32),
        'scenario': np.asarray(config.scenario),
        'seed': np.asarray(config.seed, dtype=np.int64),
        'shot_index': shot_index,
        'receiver_index': receiver_index,
        'shot_x_m': shot_x_m,
        'receiver_x_m': receiver_x_m,
        'offset_m': offset_m,
        'pick_time_s': np.maximum(base_time_s + noise_s, 0.0).astype(
            np.float64,
            copy=False,
        ),
        'placeholder': np.asarray(True),
    }


def build_fixture_metadata(config: FixtureConfig) -> dict[str, object]:
    return {
        'schema_version': 1,
        'scenario': config.scenario,
        'files': {
            'sgy': SGY_NAME,
            'pick_artifact': PICK_ARTIFACT_NAME,
        },
        'ui_workflow': {
            'upload_sgy_via_normal_ui': True,
            'static_correction_tab': 'Static Correction',
            'linkage_default': 'none',
        },
        'recommended_static_correction': {
            'model_preset': 'one_layer_global',
            'linkage': {'mode': 'none'},
            'field_corrections': {'mode': 'none'},
            'exports': ['canonical_static_table', 'lsst_plus'],
        },
        'scenario_definition': {
            'geometry': '2d_straight_line',
            'weathering_velocity': 'constant_v1',
            'bedrock_velocity': 'global_v2',
            'model': 'one_layer_t1lsst_compatible',
        },
        'generation': {
            'seed': config.seed,
            'n_shots': config.n_shots,
            'n_receivers': config.n_receivers,
            'shot_interval_m': config.shot_interval_m,
            'receiver_interval_m': config.receiver_interval_m,
            'dt_s': config.dt_s,
            'n_samples': config.n_samples,
            'v1_m_s': config.v1_m_s,
            'v2_m_s': config.v2_m_s,
            'noise_std': config.noise_std,
        },
    }


def build_expected_static_summary(config: FixtureConfig) -> dict[str, object]:
    return {
        'schema_version': 1,
        'scenario': config.scenario,
        'status': 'placeholder',
        'truth_available': False,
        'notes': [
            'Known-truth static summaries will be populated by a later issue.',
            'The CLI contract and fixture file names are fixed by this skeleton.',
        ],
        'recommended_exports': ['canonical_static_table', 'lsst_plus'],
    }


def build_readme(config: FixtureConfig) -> str:
    return (
        '# Refraction Static UI Fixture\n\n'
        'This directory is a development fixture for manually exercising the '
        'Static Correction UI refraction workflow from an uploaded SGY file and '
        'a first-break pick artifact.\n\n'
        f'- Scenario: `{config.scenario}`\n'
        f'- SGY file: `{SGY_NAME}`\n'
        f'- Pick artifact: `{PICK_ARTIFACT_NAME}`\n'
        '- Recommended Static Correction preset: `one_layer_global`\n'
        '- Linkage default: `none`\n'
        '- Recommended exports: `canonical_static_table`, `lsst_plus`\n\n'
        'The SGY and pick artifact are placeholders in this initial skeleton. '
        'Later issues will replace them with full SEG-Y samples and final '
        'first-break artifact contents while preserving this directory layout.\n'
    )


def write_fixture(config: FixtureConfig) -> None:
    output_dir = config.output_dir

    (output_dir / SGY_NAME).write_bytes(
        b'SEISVIEWER2D_PLACEHOLDER_REFRACTION_STATIC_UI_FIXTURE\n'
    )
    np.savez(
        output_dir / PICK_ARTIFACT_NAME,
        **build_placeholder_pick_arrays(config),
    )
    (output_dir / METADATA_NAME).write_text(
        json.dumps(build_fixture_metadata(config), indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / EXPECTED_SUMMARY_NAME).write_text(
        json.dumps(build_expected_static_summary(config), indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / README_NAME).write_text(build_readme(config), encoding='utf-8')


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    try:
        validate_config(config)
        ensure_output_dir(config)
        write_fixture(config)
    except ValueError as exc:
        parser.error(str(exc))

    print(f'Wrote refraction static UI fixture to {config.output_dir}')
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == '__main__':
    main()
