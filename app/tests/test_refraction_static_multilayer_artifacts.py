from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticDatumRequest,
    RefractionStaticSolverRequest,
)
from app.statics.refraction.artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    REFRACTION_STATIC_ARTIFACTS_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    write_refraction_static_artifacts,
)
from app.statics.refraction.application.datum import build_refraction_datum_statics
from app.statics.refraction.application.design_matrix import (
    refraction_design_matrix_layer_node_diagnostics_csv_name,
    refraction_design_matrix_layer_qc_json_name,
)
from app.statics.refraction.application.multilayer_service import (
    _artifact_request_for_multilayer_workflow,
    build_refraction_multilayer_weathering_replacement_statics,
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
    _layer,
    _make_two_layer_fixture,
    _resolved_first_layer,
)


_CORE_FINAL_ARTIFACT_NAMES = {
    REFRACTION_STATIC_SOLUTION_NPZ_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
    REFRACTION_STATICS_CSV_NAME,
    NEAR_SURFACE_MODEL_CSV_NAME,
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_CSV_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_NPZ_NAME,
    REFRACTION_FIRST_BREAK_FIT_QC_JSON_NAME,
    REFRACTION_REDUCED_TIME_QC_CSV_NAME,
    REFRACTION_REDUCED_TIME_QC_NPZ_NAME,
    REFRACTION_REDUCED_TIME_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENTS_CSV_NAME,
    SOURCE_STATIC_TABLE_CSV_NAME,
    RECEIVER_STATIC_TABLE_CSV_NAME,
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME,
}

_CELL_VELOCITY_ARTIFACT_NAMES = {
    REFRACTION_CELL_SOLVER_HISTORY_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_CELLS_CSV_NAME,
    REFRACTION_REFRACTOR_VELOCITY_GRID_NPZ_NAME,
    REFRACTION_REFRACTOR_VELOCITY_QC_JSON_NAME,
}


def _design_matrix_diagnostic_artifact_names(
    layer_kinds: tuple[str, ...],
) -> set[str]:
    names: set[str] = set()
    for layer_kind in layer_kinds:
        names.add(refraction_design_matrix_layer_qc_json_name(layer_kind))
        names.add(
            refraction_design_matrix_layer_node_diagnostics_csv_name(layer_kind)
        )
    return names


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
    assert _design_matrix_diagnostic_artifact_names(
        ('v2_t1', 'v3_t2', 'vsub_t3')
    ) <= manifest_names
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


def test_time_term_spreadsheet_contains_multilayer_values(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    compute_three_layer_workflow(job_dir=job_dir)

    rows = _read_csv(job_dir / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME)

    assert rows
    assert rows[0]['endpoint_kind'] == 'source'
    assert rows[0]['t1_ms']
    assert rows[0]['t2_ms']
    assert rows[0]['t3_ms']
    assert rows[0]['v1_m_s'] == f'{SYNTHETIC_MULTILAYER_V1_M_S:.3f}'
    assert rows[0]['v2_m_s'] == f'{SYNTHETIC_MULTILAYER_V2_M_S:.3f}'
    assert rows[0]['v3_m_s'] == f'{SYNTHETIC_MULTILAYER_V3_M_S:.3f}'
    assert rows[0]['vsub_m_s'] == f'{SYNTHETIC_MULTILAYER_VSUB_M_S:.3f}'
    assert rows[0]['sh1_m']
    assert rows[0]['sh2_m']
    assert rows[0]['sh3_m']
    assert rows[0]['layer1_base_elevation_m']
    assert rows[0]['layer2_base_elevation_m']
    assert rows[0]['final_refractor_elevation_m']
    assert rows[0]['pick_count_by_layer']
    assert rows[0]['used_pick_count_by_layer']
    assert rows[0]['residual_rms_by_layer_ms']
    assert rows[0]['residual_mad_by_layer_ms']


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
    assert _design_matrix_diagnostic_artifact_names(('v2_t1', 'v3_t2')) <= (
        manifest_names
    )
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

    rows = _read_csv(job_dir / REFRACTION_TIME_TERM_SPREADSHEET_CSV_NAME)
    assert rows[0]['t2_ms']
    assert rows[0]['t3_ms'] == ''
    assert rows[0]['v3_m_s'] == f'{SYNTHETIC_MULTILAYER_V3_M_S:.3f}'
    assert rows[0]['vsub_m_s'] == ''
    assert rows[0]['sh2_m']
    assert rows[0]['sh3_m'] == ''


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


def test_first_break_time_export_contains_layer_kind_for_multilayer(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    compute_three_layer_workflow(job_dir=job_dir)

    rows = _read_csv(job_dir / REFRACTION_FIRST_BREAK_TIME_EXPORT_CSV_NAME)

    assert rows
    assert {'v2_t1', 'v3_t2', 'vsub_t3'} <= {
        row['layer_kind'] for row in rows
    }
    for row in rows:
        assert row['format_name'] == 'first_break_time'
        assert row['format_version'] == '1'
        assert row['sorted_trace_index']
        assert row['source_endpoint_key']
        assert row['receiver_endpoint_key']
        assert row['source_id']
        assert row['receiver_id']
        assert row['observed_pick_time_ms']
        assert row['modeled_pick_time_ms']
        assert row['used_in_solve']


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


def test_solve_cell_presolve_rejects_keep_layer_identity(
    tmp_path: Path,
) -> None:
    fixture = _make_two_layer_fixture(
        coordinate_mode='grid_3d',
        v2_velocity_mode='solve_cell',
    )
    job_dir = tmp_path / 'job'
    solver = RefractionStaticSolverRequest(
        damping=0.0,
        robust={'enabled': False},
    )
    datum = RefractionStaticDatumRequest(mode='none')
    apply_options = RefractionStaticApplyOptions(max_abs_shift_ms=250.0)

    workflow = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=fixture.input_model,
        model=fixture.model,
        solver=solver,
        datum=datum,
        apply_options=apply_options,
        resolved_first_layer=_resolved_first_layer(),
    )
    v2_layer = _layer(workflow.solve_result, 'v2_t1')
    assert v2_layer.candidate_observation_mask_sorted is not None
    assert v2_layer.rejection_reason_sorted is not None
    used = np.asarray(v2_layer.used_observation_mask_sorted, dtype=bool).copy()
    candidate = np.asarray(
        v2_layer.candidate_observation_mask_sorted,
        dtype=bool,
    ).copy()
    reason = np.asarray(v2_layer.rejection_reason_sorted, dtype='<U32').copy()
    rejected_trace = int(np.flatnonzero(used)[-1])
    used[rejected_trace] = False
    candidate[rejected_trace] = True
    reason[rejected_trace] = 'below_min_observations_per_cell'
    patched_v2 = replace(
        v2_layer,
        used_observation_mask_sorted=used,
        candidate_observation_mask_sorted=candidate,
        rejection_reason_sorted=reason,
    )
    patched_solve = replace(
        workflow.solve_result,
        layer_results=(patched_v2, _layer(workflow.solve_result, 'v3_t2')),
    )
    replacement = build_refraction_multilayer_weathering_replacement_statics(
        input_model=fixture.input_model,
        model=fixture.model,
        solve_result=patched_solve,
        apply_options=apply_options,
        resolved_first_layer=_resolved_first_layer(),
    )
    datum_result = build_refraction_datum_statics(
        weathering_replacement_result=replacement,
        datum=datum,
        apply_options=apply_options,
        resolved_first_layer=_resolved_first_layer(),
    )
    job_dir.mkdir(parents=True)
    write_refraction_static_artifacts(
        result=datum_result,
        req=_artifact_request_for_multilayer_workflow(
            input_model=fixture.input_model,
            model=fixture.model,
            solver=solver,
            datum=datum,
            apply_options=apply_options,
            file_id=None,
            key1_byte=None,
            key2_byte=None,
        ),
        job_dir=job_dir,
        resolved_first_layer=_resolved_first_layer(),
    )

    rows = _read_csv(job_dir / FIRST_BREAK_RESIDUALS_CSV_NAME)
    low_fold_rows = [
        row
        for row in rows
        if row['rejection_reason'] == 'below_min_observations_per_cell'
    ]

    assert low_fold_rows
    assert {row['layer_kind'] for row in low_fold_rows} == {'v2_t1'}
    assert {row['layer_index'] for row in low_fold_rows} == {'1'}
    assert {row['used'] for row in low_fold_rows} == {'false'}
    assert {row['rejected_by_robust'] for row in low_fold_rows} == {'false'}


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
