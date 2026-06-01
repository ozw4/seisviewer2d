from __future__ import annotations

import csv
import json
from pathlib import Path

from app.api.schemas import (
    RefractionStaticApplyOptions,
    RefractionStaticDatumRequest,
    RefractionStaticSolverRequest,
)
from app.statics.refraction.artifacts import (
    FIRST_BREAK_RESIDUALS_CSV_NAME,
    REFRACTION_STATIC_QC_JSON_NAME,
)
from app.statics.refraction.application.multilayer_service import (
    compute_refraction_multilayer_datum_statics_from_input_model,
)
from app.tests._refraction_multilayer_3layer_helpers import (
    make_three_layer_dataset,
    make_three_layer_input_model,
    make_three_layer_model,
    resolved_first_layer,
)
from app.tests.test_refraction_static_multilayer_2layer_e2e import (
    _make_two_layer_fixture,
    _resolved_first_layer,
)
from app.tests.test_refraction_static_multilayer_v3_t2_solver import (
    V2_OFFSET_M,
    _input_model as _v3_input_model,
    _model as _v3_model,
    _resolved_first_layer as _v3_resolved_first_layer,
)


def test_three_layer_qc_contains_v2_v3_vsub_sections(tmp_path: Path) -> None:
    dataset = make_three_layer_dataset()
    job_dir = tmp_path / 'job'

    compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=make_three_layer_input_model(dataset),
        model=make_three_layer_model(),
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={'enabled': False},
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=500.0),
        resolved_first_layer=resolved_first_layer(),
        job_dir=job_dir,
    )

    qc = _read_json(job_dir / REFRACTION_STATIC_QC_JSON_NAME)
    layers = qc['layers']

    assert set(layers) == {'v2_t1', 'v3_t2', 'vsub_t3'}
    for kind, index in (('v2_t1', 1), ('v3_t2', 2), ('vsub_t3', 3)):
        assert layers[kind]['layer_kind'] == kind
        assert layers[kind]['layer_index'] == index
        assert layers[kind]['n_candidate_observations'] > 0
        assert layers[kind]['n_used_observations'] > 0
        assert 'residual_rms_ms' in layers[kind]
        assert 'residual_mad_ms' in layers[kind]


def test_multilayer_qc_reports_robust_rejections_per_layer(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'job'
    input_model = _v3_input_model(outlier_trace_index=int(V2_OFFSET_M.size))

    compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=input_model,
        model=_v3_model(v3_velocity_mode='solve_global'),
        solver=RefractionStaticSolverRequest(
            damping=0.0,
            robust={
                'enabled': True,
                'method': 'mad',
                'threshold': 2.5,
                'min_used_fraction': 0.5,
            },
        ),
        datum=RefractionStaticDatumRequest(mode='none'),
        apply_options=RefractionStaticApplyOptions(max_abs_shift_ms=500.0),
        resolved_first_layer=_v3_resolved_first_layer(),
        job_dir=job_dir,
    )

    qc = _read_json(job_dir / REFRACTION_STATIC_QC_JSON_NAME)
    v3_qc = qc['layers']['v3_t2']

    assert v3_qc['n_rejected_by_robust'] >= 1
    assert v3_qc['n_used_observations'] < v3_qc['n_observation_gate_used_observations']
    assert (
        qc['layers']['v2_t1']['observation_gate_rejection_counts'][
            'outside_layer_offset_gate'
        ]
        > 0
    )

    residual_rows = _read_csv(job_dir / FIRST_BREAK_RESIDUALS_CSV_NAME)
    robust_rows = [
        row
        for row in residual_rows
        if row['layer_kind'] == 'v3_t2' and row['rejected_by_robust'] == 'true'
    ]
    assert robust_rows
    assert {row['rejection_reason'] for row in robust_rows} == {'robust_outlier'}


def test_two_layer_qc_regression_still_contains_expected_keys(
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

    qc = _read_json(job_dir / REFRACTION_STATIC_QC_JSON_NAME)

    assert qc['layer_count'] == 2
    assert qc['enabled_layer_kinds'] == ['v2_t1', 'v3_t2']
    assert set(qc['layers']) == {'v2_t1', 'v3_t2'}
    assert 'observation_gates' in qc
    assert qc['velocity']['layer_velocity_modes'] == {
        'v2_t1': 'solve_global',
        'v3_t2': 'solve_global',
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))
