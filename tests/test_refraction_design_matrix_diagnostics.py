from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from app.api.schemas import RefractionStaticModelRequest, RefractionStaticSolverRequest
from app.statics.refraction.application import design_matrix as design_matrix_module
from app.statics.refraction.application.design_matrix import (
    REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME,
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
    build_refraction_static_design_matrix_from_arrays,
    write_refraction_design_matrix_diagnostics_artifacts,
)
from app.statics.refraction.application.solver import (
    RefractionStaticSolverError,
    solve_refraction_static_bounded_ls,
)


def _fixed_model() -> RefractionStaticModelRequest:
    return RefractionStaticModelRequest(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode='fixed_global',
        bedrock_velocity_m_s=2500.0,
    )


def test_design_matrix_node_diagnostics_success_counts_match_matrix_nonzeros() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20, 0.25, 0.30]),
        valid_observation_mask_sorted=np.asarray([True, True, False]),
        source_node_id_sorted=np.asarray([10, 10, 30]),
        receiver_node_id_sorted=np.asarray([20, 20, 20]),
        source_endpoint_key_sorted=np.asarray(['source:1001', 'source:1001', 'source:3001']),
        receiver_endpoint_key_sorted=np.asarray(
            ['receiver:2001', 'receiver:2001', 'receiver:2001']
        ),
        distance_m_sorted=np.asarray([500.0, 600.0, 700.0]),
        node_id=np.asarray([10, 20, 30]),
        node_kind=np.asarray(['source', 'receiver', 'source']),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        rejection_reason_sorted=np.asarray(['ok', 'ok', 'offset_gate']),
        include_diagnostics=True,
    )

    diagnostics = {item.node_id: item for item in design.node_diagnostics}
    assert diagnostics[10].n_rows_pre_filter == 2
    assert diagnostics[10].n_rows_post_filter == 2
    assert diagnostics[10].n_nonzero_entries == int(
        design.matrix.getnnz(axis=0)[diagnostics[10].matrix_column]
    )
    assert diagnostics[30].active is False
    assert diagnostics[30].reason == 'all_observations_filtered_by_offset_gate'
    assert design.design_matrix_qc is not None
    assert design.design_matrix_qc['n_all_zero_active_node_columns'] == 0


def test_all_zero_error_message_includes_endpoint_key_and_column() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20]),
        valid_observation_mask_sorted=np.asarray([True]),
        source_node_id_sorted=np.asarray([17]),
        receiver_node_id_sorted=np.asarray([21]),
        source_endpoint_key_sorted=np.asarray(['source:1007']),
        receiver_endpoint_key_sorted=np.asarray(['receiver:2001']),
        distance_m_sorted=np.asarray([500.0]),
        node_id=np.asarray([17, 21]),
        node_kind=np.asarray(['source', 'receiver']),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        rejection_reason_sorted=np.asarray(['ok']),
        include_diagnostics=True,
    )
    source_col = design.node_id_to_col[17]
    matrix = design.matrix.tolil()
    matrix[:, source_col] = 0.0
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    diagnostics = tuple(
        replace(
            item,
            n_nonzero_entries=0,
            status='all_zero_active_column',
            reason='unknown',
        )
        if item.node_id == 17
        else item
        for item in design.node_diagnostics
    )
    design = replace(design, matrix=matrix, node_diagnostics=diagnostics)

    with pytest.raises(RefractionStaticSolverError) as exc_info:
        solve_refraction_static_bounded_ls(
            design_matrix=design,
            model=_fixed_model(),
            solver=RefractionStaticSolverRequest(),
        )

    message = str(exc_info.value)
    assert 'all-zero active-node columns' in message
    assert 'node_id=17' in message
    assert 'endpoint_key=source:1007' in message
    assert f'column={source_col}' in message


def test_design_matrix_qc_artifacts_written_on_failure(tmp_path) -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20]),
        valid_observation_mask_sorted=np.asarray([True]),
        source_node_id_sorted=np.asarray([17]),
        receiver_node_id_sorted=np.asarray([21]),
        source_endpoint_key_sorted=np.asarray(['source:1007']),
        receiver_endpoint_key_sorted=np.asarray(['receiver:2001']),
        distance_m_sorted=np.asarray([500.0]),
        node_id=np.asarray([17, 21]),
        node_kind=np.asarray(['source', 'receiver']),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        rejection_reason_sorted=np.asarray(['ok']),
    )
    assert design.node_diagnostics == ()

    write_refraction_design_matrix_diagnostics_artifacts(tmp_path, design)

    qc = json.loads((tmp_path / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).read_text())
    assert qc['n_active_nodes'] == 2
    with (tmp_path / REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME).open(
        newline='',
        encoding='utf-8',
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['endpoint_key'] == 'source:1007'
    assert rows[0]['n_rows_pre_filter'] == '1'


def test_design_matrix_node_diagnostics_are_lazy_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_if_called(**_: object) -> tuple[object, ...]:
        raise AssertionError('node diagnostics should be lazy by default')

    monkeypatch.setattr(
        design_matrix_module,
        '_build_node_diagnostics',
        _fail_if_called,
    )

    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20]),
        valid_observation_mask_sorted=np.asarray([True]),
        source_node_id_sorted=np.asarray([17]),
        receiver_node_id_sorted=np.asarray([21]),
        distance_m_sorted=np.asarray([500.0]),
        node_id=np.asarray([17, 21]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
    )

    assert design.node_diagnostics == ()
    assert design.design_matrix_qc is not None
    assert design.design_matrix_qc['node_status_counts'] == {}
