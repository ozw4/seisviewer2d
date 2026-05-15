from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
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
SEGYIO_REQUIRED_MESSAGE = (
    'This script requires segyio to write synthetic_static_2d_one_layer.sgy. '
    "Install segyio or run in the repository's SEG-Y-enabled environment."
)

GEOMETRY_HEADERS = {
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

TRACE_STORE_SORT_HEADERS = {
    'key1_byte': GEOMETRY_HEADERS['source_id_byte'],
    'key1_label': 'source_id',
    'key2_byte': GEOMETRY_HEADERS['receiver_id_byte'],
    'key2_label': 'receiver_id',
    'ordering': 'source_major_receiver_minor',
}


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


@dataclass(frozen=True)
class SyntheticFixture:
    traces: np.ndarray
    shot_index: np.ndarray
    receiver_index: np.ndarray
    source_id: np.ndarray
    receiver_id: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    source_elevation_m: np.ndarray
    receiver_elevation_m: np.ndarray
    offset_m: np.ndarray
    pick_time_s: np.ndarray
    source_t1_s: np.ndarray
    receiver_t1_s: np.ndarray
    source_thickness_m: np.ndarray
    receiver_thickness_m: np.ndarray


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


def _line_length_m(config: FixtureConfig) -> float:
    shot_extent = (config.n_shots - 1) * config.shot_interval_m
    receiver_extent = (config.n_receivers - 1) * config.receiver_interval_m
    return max(float(shot_extent), float(receiver_extent), 1.0)


def _weathering_thickness_m(x_m: np.ndarray, line_length_m: float) -> np.ndarray:
    phase = 2.0 * np.pi * np.asarray(x_m, dtype=np.float64) / line_length_m
    thickness = 12.0 + 4.0 * np.sin(phase) + 1.5 * np.cos(2.0 * phase)
    return np.maximum(thickness, 0.5).astype(np.float64, copy=False)


def _half_intercept_time_s(
    thickness_m: np.ndarray,
    *,
    v1_m_s: float,
    v2_m_s: float,
) -> np.ndarray:
    factor = math.sqrt(v2_m_s * v2_m_s - v1_m_s * v1_m_s) / (v2_m_s * v1_m_s)
    return np.asarray(thickness_m, dtype=np.float64) * factor


def _add_pulse(
    trace: np.ndarray,
    *,
    center_time_s: float,
    dt_s: float,
    amplitude: float,
    frequency_hz: float,
) -> None:
    center_sample = int(round(center_time_s / dt_s))
    radius = max(2, int(round(0.016 / dt_s)))
    start = max(0, center_sample - radius)
    stop = min(trace.shape[0], center_sample + radius + 1)
    if start >= stop:
        return

    sample_indices = np.arange(start, stop, dtype=np.float64)
    time_shift_s = (sample_indices - center_sample) * dt_s
    arg = np.pi * frequency_hz * time_shift_s
    wavelet = (1.0 - 2.0 * arg * arg) * np.exp(-(arg * arg))
    trace[start:stop] += amplitude * wavelet.astype(np.float32, copy=False)


def build_synthetic_fixture(config: FixtureConfig) -> SyntheticFixture:
    rng = np.random.default_rng(config.seed)
    shot_x_m = (
        np.arange(config.n_shots, dtype=np.float64) * config.shot_interval_m
    )
    receiver_x_m = (
        np.arange(config.n_receivers, dtype=np.float64) * config.receiver_interval_m
    )
    line_length_m = _line_length_m(config)
    shot_thickness_m = _weathering_thickness_m(shot_x_m, line_length_m)
    receiver_thickness_m = _weathering_thickness_m(receiver_x_m, line_length_m)
    shot_t1_s = _half_intercept_time_s(
        shot_thickness_m,
        v1_m_s=config.v1_m_s,
        v2_m_s=config.v2_m_s,
    )
    receiver_t1_s = _half_intercept_time_s(
        receiver_thickness_m,
        v1_m_s=config.v1_m_s,
        v2_m_s=config.v2_m_s,
    )

    shot_index, receiver_index = np.meshgrid(
        np.arange(config.n_shots, dtype=np.int32),
        np.arange(config.n_receivers, dtype=np.int32),
        indexing='ij',
    )
    offset_m = np.abs(shot_x_m[shot_index] - receiver_x_m[receiver_index])
    pick_time_s = (
        shot_t1_s[shot_index]
        + receiver_t1_s[receiver_index]
        + offset_m / config.v2_m_s
    )

    n_traces = config.n_shots * config.n_receivers
    traces = rng.normal(
        loc=0.0,
        scale=config.noise_std,
        size=(n_traces, config.n_samples),
    ).astype(np.float32)

    flat_pick_time_s = pick_time_s.reshape(-1)
    flat_source_x_m = shot_x_m[shot_index].reshape(-1)
    flat_receiver_x_m = receiver_x_m[receiver_index].reshape(-1)
    midpoint_x_m = 0.5 * (flat_source_x_m + flat_receiver_x_m)
    later_variation_s = 0.04 * np.sin(2.0 * np.pi * midpoint_x_m / line_length_m)
    later_time_s = flat_pick_time_s + 0.35 + later_variation_s

    for trace, pick_time, reflection_time in zip(
        traces,
        flat_pick_time_s,
        later_time_s,
        strict=True,
    ):
        _add_pulse(
            trace,
            center_time_s=float(pick_time),
            dt_s=config.dt_s,
            amplitude=1.0,
            frequency_hz=55.0,
        )
        _add_pulse(
            trace,
            center_time_s=float(reflection_time),
            dt_s=config.dt_s,
            amplitude=0.35,
            frequency_hz=35.0,
        )

    source_id = (shot_index + 1).astype(np.int32, copy=False)
    receiver_id = (receiver_index + 1).astype(np.int32, copy=False)
    source_elevation_m = 100.0 + 0.01 * shot_x_m[shot_index]
    receiver_elevation_m = 100.0 + 0.01 * receiver_x_m[receiver_index]

    return SyntheticFixture(
        traces=np.ascontiguousarray(traces, dtype=np.float32),
        shot_index=shot_index,
        receiver_index=receiver_index,
        source_id=source_id.reshape(-1),
        receiver_id=receiver_id.reshape(-1),
        source_x_m=flat_source_x_m,
        source_y_m=np.zeros(n_traces, dtype=np.float64),
        receiver_x_m=flat_receiver_x_m,
        receiver_y_m=np.zeros(n_traces, dtype=np.float64),
        source_elevation_m=source_elevation_m.reshape(-1),
        receiver_elevation_m=receiver_elevation_m.reshape(-1),
        offset_m=offset_m.reshape(-1),
        pick_time_s=flat_pick_time_s,
        source_t1_s=shot_t1_s[shot_index].reshape(-1),
        receiver_t1_s=receiver_t1_s[receiver_index].reshape(-1),
        source_thickness_m=shot_thickness_m[shot_index].reshape(-1),
        receiver_thickness_m=receiver_thickness_m[receiver_index].reshape(-1),
    )


def build_pick_arrays(config: FixtureConfig, fixture: SyntheticFixture) -> dict[str, np.ndarray]:
    n_traces = fixture.pick_time_s.size
    pick_time_s = np.ascontiguousarray(fixture.pick_time_s, dtype=np.float32)
    return {
        'artifact_kind': np.asarray('synthetic_first_break_picks'),
        'schema_version': np.asarray(1, dtype=np.int32),
        'scenario': np.asarray(config.scenario),
        'seed': np.asarray(config.seed, dtype=np.int64),
        'n_traces': np.asarray(n_traces, dtype=np.int64),
        'n_samples': np.asarray(config.n_samples, dtype=np.int64),
        'dt': np.asarray(config.dt_s, dtype=np.float64),
        'order': np.asarray('original_trace_order'),
        'sorted_to_original': np.arange(n_traces, dtype=np.int64),
        'shot_index': fixture.shot_index,
        'receiver_index': fixture.receiver_index,
        'source_id': fixture.source_id.reshape(config.n_shots, config.n_receivers),
        'receiver_id': fixture.receiver_id.reshape(config.n_shots, config.n_receivers),
        'shot_x_m': (
            np.arange(config.n_shots, dtype=np.float64) * config.shot_interval_m
        ),
        'receiver_x_m': (
            np.arange(config.n_receivers, dtype=np.float64) * config.receiver_interval_m
        ),
        'offset_m': fixture.offset_m.reshape(config.n_shots, config.n_receivers),
        'pick_time_s': pick_time_s,
        'picks_time_s': pick_time_s,
        'predicted_picks_time_s': pick_time_s,
        'first_break_time_s': pick_time_s,
        'source_t1_s': fixture.source_t1_s.reshape(config.n_shots, config.n_receivers),
        'receiver_t1_s': fixture.receiver_t1_s.reshape(config.n_shots, config.n_receivers),
        'source_weathering_thickness_m': fixture.source_thickness_m.reshape(
            config.n_shots,
            config.n_receivers,
        ),
        'receiver_weathering_thickness_m': fixture.receiver_thickness_m.reshape(
            config.n_shots,
            config.n_receivers,
        ),
    }


def write_synthetic_segy(
    path: Path,
    *,
    config: FixtureConfig,
    fixture: SyntheticFixture,
) -> None:
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError(SEGYIO_REQUIRED_MESSAGE) from exc

    spec = segyio.spec()
    spec.format = 5
    spec.samples = list(range(config.n_samples))
    spec.tracecount = int(fixture.traces.shape[0])

    sample_interval_us = int(round(config.dt_s * 1_000_000.0))
    with segyio.create(str(path), spec) as segy_file:
        segy_file.bin[segyio.BinField.Interval] = sample_interval_us
        segy_file.bin[segyio.BinField.Samples] = int(config.n_samples)

        for trace_index in range(fixture.traces.shape[0]):
            segy_file.header[trace_index] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: trace_index + 1,
                segyio.TraceField.TRACE_SEQUENCE_FILE: trace_index + 1,
                segyio.TraceField.TRACE_SAMPLE_COUNT: int(config.n_samples),
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: sample_interval_us,
                GEOMETRY_HEADERS['source_id_byte']: int(fixture.source_id[trace_index]),
                GEOMETRY_HEADERS['receiver_id_byte']: int(fixture.receiver_id[trace_index]),
                GEOMETRY_HEADERS['offset_byte']: int(round(fixture.offset_m[trace_index])),
                GEOMETRY_HEADERS['source_elevation_byte']: int(round(fixture.source_elevation_m[trace_index])),
                GEOMETRY_HEADERS['receiver_elevation_byte']: int(round(fixture.receiver_elevation_m[trace_index])),
                GEOMETRY_HEADERS['elevation_scalar_byte']: 1,
                GEOMETRY_HEADERS['coordinate_scalar_byte']: 1,
                GEOMETRY_HEADERS['source_x_byte']: int(round(fixture.source_x_m[trace_index])),
                GEOMETRY_HEADERS['source_y_byte']: int(round(fixture.source_y_m[trace_index])),
                GEOMETRY_HEADERS['receiver_x_byte']: int(round(fixture.receiver_x_m[trace_index])),
                GEOMETRY_HEADERS['receiver_y_byte']: int(round(fixture.receiver_y_m[trace_index])),
            }
            segy_file.trace[trace_index] = fixture.traces[trace_index]


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
        'pick_source': {
            'kind': 'uploaded_npz',
            'file_field': 'pick_npz',
            'file_name': PICK_ARTIFACT_NAME,
        },
        'recommended_static_correction': {
            'model_preset': 'one_layer_global',
            'linkage': {'mode': 'none'},
            'field_corrections': {'mode': 'none'},
            'key1_byte': TRACE_STORE_SORT_HEADERS['key1_byte'],
            'key2_byte': TRACE_STORE_SORT_HEADERS['key2_byte'],
            'v1_m_s': config.v1_m_s,
            'v2_initial_m_s': config.v2_m_s,
            'min_offset_m': 300.0,
            'max_offset_m': 1800.0,
            'exports': ['canonical_static_table', 'lsst_plus'],
            'register_corrected_file': True,
        },
        'scenario_definition': {
            'geometry': '2d_straight_line',
            'weathering_velocity': 'constant_v1',
            'bedrock_velocity': 'global_v2',
            'model': 'one_layer_t1lsst_compatible',
        },
        'geometry_headers': GEOMETRY_HEADERS,
        'trace_store_sort_headers': TRACE_STORE_SORT_HEADERS,
        'synthetic_model': {
            'v1_m_s': config.v1_m_s,
            'v2_m_s': config.v2_m_s,
            'dt_s': config.dt_s,
            'n_samples': config.n_samples,
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


def _finite_range(values: np.ndarray, *, scale: float = 1.0) -> tuple[float, float]:
    scaled = np.asarray(values, dtype=np.float64) * scale
    return float(np.min(scaled)), float(np.max(scaled))


def _weathering_correction_s(
    thickness_m: np.ndarray,
    *,
    v1_m_s: float,
    v2_m_s: float,
) -> np.ndarray:
    return np.asarray(thickness_m, dtype=np.float64) * (1.0 / v2_m_s - 1.0 / v1_m_s)


def build_expected_static_summary(
    config: FixtureConfig,
    fixture: SyntheticFixture,
) -> dict[str, object]:
    source_wcor_s = _weathering_correction_s(
        fixture.source_thickness_m,
        v1_m_s=config.v1_m_s,
        v2_m_s=config.v2_m_s,
    )
    receiver_wcor_s = _weathering_correction_s(
        fixture.receiver_thickness_m,
        v1_m_s=config.v1_m_s,
        v2_m_s=config.v2_m_s,
    )
    weathering_thickness_m = np.concatenate(
        [
            np.asarray(fixture.source_thickness_m, dtype=np.float64),
            np.asarray(fixture.receiver_thickness_m, dtype=np.float64),
        ]
    )
    weathering_correction_s = np.concatenate([source_wcor_s, receiver_wcor_s])
    pick_min_s, pick_max_s = _finite_range(fixture.pick_time_s)
    source_t1_min_ms, source_t1_max_ms = _finite_range(fixture.source_t1_s, scale=1000.0)
    receiver_t1_min_ms, receiver_t1_max_ms = _finite_range(
        fixture.receiver_t1_s,
        scale=1000.0,
    )
    thickness_min_m, thickness_max_m = _finite_range(weathering_thickness_m)
    wcor_min_ms, wcor_max_ms = _finite_range(weathering_correction_s, scale=1000.0)
    return {
        'schema_version': 1,
        'scenario': config.scenario,
        'truth': {
            'v1_m_s': config.v1_m_s,
            'v2_m_s': config.v2_m_s,
            'n_traces': int(fixture.pick_time_s.size),
            'pick_time_min_s': pick_min_s,
            'pick_time_max_s': pick_max_s,
            'source_t1_min_ms': source_t1_min_ms,
            'source_t1_max_ms': source_t1_max_ms,
            'receiver_t1_min_ms': receiver_t1_min_ms,
            'receiver_t1_max_ms': receiver_t1_max_ms,
            'weathering_thickness_min_m': thickness_min_m,
            'weathering_thickness_max_m': thickness_max_m,
            'weathering_correction_min_ms': wcor_min_ms,
            'weathering_correction_max_ms': wcor_max_ms,
        },
        'recommended_exports': ['canonical_static_table', 'lsst_plus'],
    }


def build_readme(config: FixtureConfig) -> str:
    return (
        '# Refraction Static UI Fixture\n\n'
        'This directory is a development fixture for manually exercising the '
        'Static Correction UI refraction workflow from an uploaded SGY file and '
        'a directly selected first-break pick NPZ.\n\n'
        f'- Scenario: `{config.scenario}`\n'
        f'- SGY file: `{SGY_NAME}`\n'
        f'- Pick NPZ: `{PICK_ARTIFACT_NAME}`\n'
        '- Recommended Static Correction preset: `one_layer_global`\n'
        '- Linkage default: `none`\n'
        '- Recommended exports: `canonical_static_table`, `lsst_plus`\n\n'
        'The SGY file contains deterministic one-layer synthetic traces with '
        'clear first-break pulses and geometry headers listed in '
        f'`{METADATA_NAME}`.\n\n'
        '## UI workflow\n\n'
        f'1. Open `{SGY_NAME}` in the viewer Loader with `key1=9`, `key2=13`.\n'
        f'2. Open the `Static Correction` tab and confirm the Target shows the '
        'current viewer file and sort keys.\n'
        f'3. Select `{PICK_ARTIFACT_NAME}` directly in `First-break pick NPZ`.\n'
        '4. Configure geometry, model, output, and export settings.\n'
        '5. Run Static Correction.\n'
        '6. Confirm the completed result auto-loads in `Refraction QC`.\n\n'
        'You do not need a `file_id` input, pick `job_id`, `artifact_name`, or '
        'manual batch job registration for the primary UI workflow.\n\n'
        '## Settings\n\n'
        f'- Target: current viewer file loaded with `key1=9`, `key2=13`\n'
        f'- First-break pick NPZ: `{PICK_ARTIFACT_NAME}`\n'
        f'- Geometry byte locations: see `{METADATA_NAME}`\n'
        '- Model preset: `one_layer_global`\n'
        '- Linkage: unchecked / `none`\n'
        '- Recommended exports: `canonical_static_table`, `lsst_plus`\n\n'
        'For the legacy developer-only manual registration workflow, including '
        '`get_job_dir(job_id)`, `create_batch_apply_job`, the required '
        '`file_id`/`key1_byte`/`key2_byte` metadata, and '
        '`/batch/job/<job_id>/files` verification, see '
        '`docs/statics/refraction_static_ui_fixture.md'
        '#legacy-developer-appendix-job-artifact-registration`.\n\n'
        '## Trace sorting for Static Correction UI\n\n'
        'Use:\n'
        '  key1_byte = 9  # source_id\n'
        '  key2_byte = 13  # receiver_id\n\n'
        'The generated pick artifact is written in the same sorted trace order. '
        'Changing these sort headers may cause the pick artifact to no longer '
        'align with the imported TraceStore order.\n'
    )


def write_fixture(config: FixtureConfig) -> None:
    output_dir = config.output_dir
    fixture = build_synthetic_fixture(config)

    write_synthetic_segy(
        output_dir / SGY_NAME,
        config=config,
        fixture=fixture,
    )
    np.savez(
        output_dir / PICK_ARTIFACT_NAME,
        **build_pick_arrays(config, fixture),
    )
    (output_dir / METADATA_NAME).write_text(
        json.dumps(build_fixture_metadata(config), indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / EXPECTED_SUMMARY_NAME).write_text(
        json.dumps(build_expected_static_summary(config, fixture), indent=2) + '\n',
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
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f'Wrote refraction static UI fixture to {config.output_dir}')
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == '__main__':
    main()
