from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticDatumRequest,
    RefractionStaticSolverRequest,
)
from app.services.refraction_static_artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
)
from app.services.refraction_static_multilayer_service import (
    compute_refraction_multilayer_datum_statics_from_input_model,
)
from app.tests._refraction_multilayer_3layer_helpers import (
    compute_three_layer_workflow,
)
from app.tests._refraction_multilayer_synthetic import (
    SYNTHETIC_MULTILAYER_V1_M_S,
    SYNTHETIC_MULTILAYER_V2_M_S,
    SYNTHETIC_MULTILAYER_V3_M_S,
    SYNTHETIC_MULTILAYER_VSUB_M_S,
)
from app.tests.test_refraction_static_multilayer_2layer_e2e import (
    _make_two_layer_fixture,
    _resolved_first_layer,
)


_CORE_FINAL_ARTIFACT_NAMES = {
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
}

_CELL_VELOCITY_ARTIFACT_NAMES = {
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
}


def test_three_layer_job_manifest_lists_multilayer_artifacts(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    compute_three_layer_workflow(job_dir=job_dir)

    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    manifest_names = {item['name'] for item in manifest['artifacts']}

    assert _CORE_FINAL_ARTIFACT_NAMES <= manifest_names
    assert _CELL_VELOCITY_ARTIFACT_NAMES.isdisjoint(manifest_names)
    for artifact_name in manifest_names:
        assert (job_dir / artifact_name).is_file()

    qc = json.loads(
        (job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['layer_count'] == 3
    assert qc['enabled_layer_kinds'] == ['v2_t1', 'v3_t2', 'vsub_t3']
    assert qc['velocity']['layer_velocity_modes'] == {
        'v2_t1': 'fixed_global',
        'v3_t2': 'fixed_global',
        'vsub_t3': 'fixed_global',
    }
    assert qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_three_layer_solution_npz_contains_t3_sh3_and_vsub_arrays(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    dataset, _input_model, _model, workflow = compute_three_layer_workflow(
        job_dir=job_dir,
    )
    result = workflow.datum_result

    expected_arrays = {
        'source_t1_s',
        'source_t2_s',
        'source_t3_s',
        'receiver_t1_s',
        'receiver_t2_s',
        'receiver_t3_s',
        'source_sh1_m',
        'source_sh2_m',
        'source_sh3_m',
        'receiver_sh1_m',
        'receiver_sh2_m',
        'receiver_sh3_m',
        'source_v2_m_s',
        'source_v3_m_s',
        'source_vsub_m_s',
        'receiver_v2_m_s',
        'receiver_v3_m_s',
        'receiver_vsub_m_s',
        'source_weathering_correction_s',
        'receiver_weathering_correction_s',
        'refraction_trace_shift_s_sorted',
    }
    with np.load(
        job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        allow_pickle=False,
    ) as data:
        assert expected_arrays <= set(data.files)
        np.testing.assert_allclose(data['source_t1_s'], result.source_half_intercept_time_s)
        np.testing.assert_allclose(data['source_t2_s'], result.source_t2_time_s)
        np.testing.assert_allclose(data['source_t3_s'], result.source_t3_time_s)
        np.testing.assert_allclose(data['source_sh3_m'], dataset.true_source_endpoint_sh3_m)
        np.testing.assert_allclose(data['receiver_sh3_m'], dataset.true_receiver_endpoint_sh3_m)
        np.testing.assert_allclose(data['source_vsub_m_s'], SYNTHETIC_MULTILAYER_VSUB_M_S)
        np.testing.assert_allclose(data['receiver_vsub_m_s'], SYNTHETIC_MULTILAYER_VSUB_M_S)
        np.testing.assert_allclose(
            data['source_weathering_correction_s'],
            result.source_weathering_replacement_shift_s,
        )
        np.testing.assert_allclose(
            data['receiver_weathering_correction_s'],
            result.receiver_weathering_replacement_shift_s,
        )


def test_three_layer_npz_is_pickle_free(tmp_path: Path) -> None:
    job_dir = tmp_path / 'job'
    compute_three_layer_workflow(job_dir=job_dir)

    for artifact_name in (
        REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    ):
        with np.load(job_dir / artifact_name, allow_pickle=False) as data:
            for key in data.files:
                assert data[key].dtype != object, f'{artifact_name}:{key}'


def test_two_layer_artifact_manifest_regression_still_passes(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )
    job_dir = tmp_path / 'job'

    compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
        job_dir=job_dir,
    )

    manifest = json.loads(
        (job_dir / REFRACTION_STATIC_ARTIFACTS_JSON_NAME).read_text(
            encoding='utf-8'
        )
    )
    manifest_names = {item['name'] for item in manifest['artifacts']}
    assert _CORE_FINAL_ARTIFACT_NAMES <= manifest_names
    assert _CELL_VELOCITY_ARTIFACT_NAMES.isdisjoint(manifest_names)

    qc = json.loads(
        (job_dir / REFRACTION_STATIC_QC_JSON_NAME).read_text(encoding='utf-8')
    )
    assert qc['layer_count'] == 2
    assert qc['enabled_layer_kinds'] == ['v2_t1', 'v3_t2']
    assert qc['velocity']['layer_velocity_modes'] == {
        'v2_t1': 'solve_global',
        'v3_t2': 'solve_global',
    }
    with np.load(
        job_dir / REFRACTION_STATIC_SOLUTION_NPZ_NAME,
        allow_pickle=False,
    ) as data:
        assert 'source_t3_s' not in data.files
        assert 'source_sh3_m' not in data.files
        assert 'source_vsub_m_s' not in data.files
        np.testing.assert_allclose(
            data['v1_weathering_velocity_m_s'],
            SYNTHETIC_MULTILAYER_V1_M_S,
        )
        np.testing.assert_allclose(
            data['source_v2_m_s'],
            SYNTHETIC_MULTILAYER_V2_M_S,
            rtol=1.0e-9,
        )
        np.testing.assert_allclose(
            data['source_v3_m_s'],
            SYNTHETIC_MULTILAYER_V3_M_S,
            rtol=1.0e-9,
        )


def test_multilayer_residual_csv_contains_layer_kind_and_layer_index(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    compute_three_layer_workflow(job_dir=job_dir)

    rows = _read_csv(job_dir / FIRST_BREAK_RESIDUALS_CSV_NAME)

    expected_columns = {
        'trace_index_sorted',
        'layer_kind',
        'layer_index',
        'source_endpoint_key',
        'receiver_endpoint_key',
        'offset_m',
        'observed_pick_time_s',
        'modeled_pick_time_s',
        'residual_time_s',
        'used',
        'rejected_by_robust',
        'rejection_reason',
        'midpoint_cell_id',
        'row_velocity_m_s',
    }
    assert rows
    assert expected_columns <= set(rows[0])
    assert {'v2_t1', 'v3_t2', 'vsub_t3'} <= {
        row['layer_kind'] for row in rows
    }
    vsub_rows = [row for row in rows if row['layer_kind'] == 'vsub_t3']
    assert vsub_rows
    assert {row['layer_index'] for row in vsub_rows} == {'3'}
    assert all(row['residual_time_s'] != '' for row in vsub_rows)


def test_multilayer_residual_csv_reports_cell_ids_for_solve_cell_layer(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='line_2d_projected',
        v2_velocity_mode='solve_cell',
    )
    job_dir = tmp_path / 'job'

    compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
        job_dir=job_dir,
    )

    rows = _read_csv(job_dir / FIRST_BREAK_RESIDUALS_CSV_NAME)
    v2_rows = [row for row in rows if row['layer_kind'] == 'v2_t1']

    assert v2_rows
    assert all(row['midpoint_cell_id'] != '' for row in v2_rows)
    assert all(row['row_velocity_m_s'] != '' for row in v2_rows)
    assert {
        round(float(row['row_velocity_m_s']), 6) for row in v2_rows
    } == {round(SYNTHETIC_MULTILAYER_V2_M_S, 6)}


def test_multilayer_endpoint_tables_include_layer_qc_fields(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    compute_three_layer_workflow(job_dir=job_dir)

    source_rows = _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME)
    receiver_rows = _read_csv(job_dir / RECEIVER_STATIC_TABLE_CSV_NAME)

    for row in (source_rows + receiver_rows):
        assert 'pick_count_by_layer' in row
        assert 'used_pick_count_by_layer' in row
        assert 'residual_rms_by_layer_ms' in row
        assert 'residual_mad_by_layer_ms' in row

    assert any(
        'vsub_t3' in json.loads(row['pick_count_by_layer'])
        for row in source_rows + receiver_rows
    )


def test_two_layer_endpoint_tables_populate_layer_qc_fields(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_global',
    )
    job_dir = tmp_path / 'job'

    compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=250.0),
        resolved_first_layer=_resolved_first_layer(),
        job_dir=job_dir,
    )

    source_rows = _read_csv(job_dir / SOURCE_STATIC_TABLE_CSV_NAME)
    receiver_rows = _read_csv(job_dir / RECEIVER_STATIC_TABLE_CSV_NAME)

    for row in source_rows + receiver_rows:
        pick_count_by_layer = json.loads(row['pick_count_by_layer'])
        used_pick_count_by_layer = json.loads(row['used_pick_count_by_layer'])
        residual_rms_by_layer_ms = json.loads(row['residual_rms_by_layer_ms'])
        residual_mad_by_layer_ms = json.loads(row['residual_mad_by_layer_ms'])

        assert pick_count_by_layer
        assert used_pick_count_by_layer
        assert residual_rms_by_layer_ms
        assert residual_mad_by_layer_ms
        assert set(pick_count_by_layer) <= {'v2_t1', 'v3_t2'}
        assert set(used_pick_count_by_layer) <= {'v2_t1', 'v3_t2'}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))
