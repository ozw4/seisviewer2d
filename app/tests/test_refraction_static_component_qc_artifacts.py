from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_artifacts import (
    REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME,
    REFRACTION_STATIC_COMPONENT_QC_JSON_NAME,
    REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME,
    REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME,
    write_refraction_static_artifacts,
)
from app.tests._refraction_static_artifact_helpers import _request, _result


def test_static_component_qc_trace_values_match_solution_npz(
    tmp_path: Path,
) -> None:
    result = _field_composed_result(apply_to_trace_shift=True)
    paths = write_refraction_static_artifacts(
        result=result,
        req=_field_request(apply_to_trace_shift=True),
        job_dir=tmp_path,
    )

    assert paths.refraction_static_component_qc_trace_csv.name == (
        REFRACTION_STATIC_COMPONENT_QC_TRACE_CSV_NAME
    )
    assert paths.refraction_static_component_qc_npz.name == (
        REFRACTION_STATIC_COMPONENT_QC_NPZ_NAME
    )

    rows = _read_csv(paths.refraction_static_component_qc_trace_csv)
    assert rows[0]['source_endpoint_key'] == 's0'
    assert rows[0]['receiver_endpoint_key'] == 'r0'
    assert float(rows[0]['refraction_shift_ms']) == pytest.approx(4.8)
    assert float(rows[0]['computed_field_shift_ms']) == pytest.approx(6.5)
    assert float(rows[0]['applied_field_shift_ms']) == pytest.approx(6.5)
    assert float(rows[0]['final_trace_shift_ms']) == pytest.approx(11.3)

    with (
        np.load(paths.refraction_static_component_qc_npz, allow_pickle=False) as qc,
        np.load(paths.solution_npz, allow_pickle=False) as solution,
    ):
        np.testing.assert_array_equal(
            qc['source_endpoint_key'],
            solution['source_endpoint_key_sorted'],
        )
        np.testing.assert_array_equal(
            qc['receiver_endpoint_key'],
            solution['receiver_endpoint_key_sorted'],
        )
        np.testing.assert_allclose(
            qc['refraction_shift_s'],
            solution['base_refraction_trace_shift_s_sorted'],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            qc['final_trace_shift_s'],
            solution['final_trace_shift_s_sorted'],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            qc['applied_trace_shift_s'],
            solution['final_trace_shift_s_sorted'],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            qc['applied_field_shift_s'],
            solution['applied_field_shift_s_sorted'],
            equal_nan=True,
        )


def test_static_component_qc_endpoint_values_match_static_tables(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_field_composed_result(apply_to_trace_shift=True),
        req=_field_request(apply_to_trace_shift=True),
        job_dir=tmp_path,
    )

    assert paths.refraction_static_component_qc_endpoint_csv.name == (
        REFRACTION_STATIC_COMPONENT_QC_ENDPOINT_CSV_NAME
    )
    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    endpoint_rows = _read_csv(paths.refraction_static_component_qc_endpoint_csv)

    source_qc = endpoint_rows[0]
    assert source_qc['endpoint_kind'] == 'source'
    assert source_qc['endpoint_key'] == source_rows[0]['source_endpoint_key']
    assert float(source_qc['weathering_correction_ms']) == pytest.approx(
        float(source_rows[0]['weathering_correction_ms'])
    )
    assert float(source_qc['elevation_correction_ms']) == pytest.approx(
        float(source_rows[0]['elevation_correction_ms'])
    )
    assert float(source_qc['field_correction_ms']) == pytest.approx(
        float(source_rows[0]['source_field_shift_ms'])
    )
    assert float(source_qc['total_static_ms']) == pytest.approx(
        float(source_rows[0]['total_static_ms'])
    )
    assert float(source_qc['total_with_field_shift_ms']) == pytest.approx(
        float(source_rows[0]['source_total_with_field_shift_ms'])
    )

    receiver_qc = endpoint_rows[2]
    assert receiver_qc['endpoint_kind'] == 'receiver'
    assert receiver_qc['endpoint_key'] == receiver_rows[0]['receiver_endpoint_key']
    assert receiver_qc['source_depth_correction_ms'] == ''
    assert receiver_qc['uphole_correction_ms'] == ''
    assert float(receiver_qc['field_correction_ms']) == pytest.approx(
        float(receiver_rows[0]['receiver_field_shift_ms'])
    )
    assert float(receiver_qc['total_with_field_shift_ms']) == pytest.approx(
        float(receiver_rows[0]['receiver_total_with_field_shift_ms'])
    )

    with (
        np.load(paths.refraction_static_component_qc_npz, allow_pickle=False) as qc,
        np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as table,
    ):
        np.testing.assert_allclose(
            qc['endpoint_field_correction_s'][:2],
            table['source_field_shift_s'],
        )
        np.testing.assert_allclose(
            qc['endpoint_field_correction_s'][2:],
            table['receiver_field_shift_s'],
        )


def test_static_component_qc_apply_false_distinguishes_computed_and_applied_field_shift(
    tmp_path: Path,
) -> None:
    paths = write_refraction_static_artifacts(
        result=_field_composed_result(apply_to_trace_shift=False),
        req=_field_request(apply_to_trace_shift=False),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_static_component_qc_trace_csv)
    assert rows[0]['apply_to_trace_shift'] == 'false'
    assert float(rows[0]['computed_field_shift_ms']) == pytest.approx(6.5)
    assert float(rows[0]['field_shift_ms']) == pytest.approx(6.5)
    assert float(rows[0]['applied_field_shift_ms']) == pytest.approx(0.0)
    assert float(rows[0]['final_trace_shift_ms']) == pytest.approx(
        float(rows[0]['refraction_shift_ms'])
    )
    assert float(rows[0]['applied_trace_shift_ms']) == pytest.approx(
        float(rows[0]['refraction_shift_ms'])
    )

    endpoint_rows = _read_csv(paths.refraction_static_component_qc_endpoint_csv)
    assert float(endpoint_rows[0]['computed_field_correction_ms']) == pytest.approx(
        4.5
    )
    assert float(endpoint_rows[0]['applied_field_correction_ms']) == pytest.approx(0.0)

    with np.load(paths.refraction_static_component_qc_npz, allow_pickle=False) as qc:
        np.testing.assert_allclose(qc['computed_field_shift_s'][0], 0.0065)
        np.testing.assert_allclose(qc['applied_field_shift_s'], np.zeros(4))
        np.testing.assert_allclose(
            qc['final_trace_shift_s'],
            qc['refraction_shift_s'],
            equal_nan=True,
        )


def test_static_component_qc_json_summary_stats(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_field_composed_result(apply_to_trace_shift=True),
        req=_field_request(apply_to_trace_shift=True),
        job_dir=tmp_path,
    )

    assert paths.refraction_static_component_qc_json.name == (
        REFRACTION_STATIC_COMPONENT_QC_JSON_NAME
    )
    payload = json.loads(
        paths.refraction_static_component_qc_json.read_text(encoding='utf-8')
    )

    assert payload['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert payload['apply_to_trace_shift'] is True
    assert payload['trace']['row_count'] == 4
    assert payload['endpoint']['row_count'] == 4
    trace_stats = payload['trace']['component_summary_ms']
    assert trace_stats['refraction_shift_ms']['min'] == pytest.approx(4.4)
    assert trace_stats['refraction_shift_ms']['median'] == pytest.approx(4.6)
    assert trace_stats['refraction_shift_ms']['max'] == pytest.approx(4.8)
    assert trace_stats['computed_field_shift_ms']['min'] == pytest.approx(3.5)
    assert trace_stats['computed_field_shift_ms']['median'] == pytest.approx(7.25)
    assert trace_stats['computed_field_shift_ms']['max'] == pytest.approx(11.0)
    endpoint_stats = payload['endpoint']['component_summary_ms']
    assert endpoint_stats['field_correction_ms']['min'] == pytest.approx(-1.0)
    assert endpoint_stats['field_correction_ms']['median'] == pytest.approx(3.25)
    assert endpoint_stats['field_correction_ms']['max'] == pytest.approx(9.0)


def _field_request(*, apply_to_trace_shift: bool) -> RefractionStaticApplyRequest:
    payload = _request().model_dump(mode='json')
    payload['geometry']['source_depth_byte'] = 115
    payload['field_corrections'] = {
        'source_depth': {'mode': 'weathering_velocity_time'},
        'uphole': {
            'mode': 'header_time',
            'uphole_time_byte': 95,
            'uphole_time_unit': 's',
        },
        'manual_static': {
            'mode': 'inline_table',
            'sign_convention': 'applied_shift_s',
            'source_inline_table': [
                {'endpoint_id': 0, 'value': 0.0005},
                {'endpoint_id': 1, 'value': 0.001},
            ],
            'receiver_inline_table': [
                {'endpoint_id': 0, 'value': 0.002},
                {'endpoint_id': 1, 'value': -0.001},
            ],
        },
        'composition': {
            'enabled': True,
            'apply_to_trace_shift': apply_to_trace_shift,
        },
    }
    return RefractionStaticApplyRequest.model_validate(payload)


def _field_composed_result(
    *,
    apply_to_trace_shift: bool,
):
    base = _result()
    source_endpoint_key_sorted = np.asarray(['s0', 's1', 's0', 's1'], dtype='<U2')
    receiver_endpoint_key_sorted = np.asarray(['r0', 'r1', 'r1', 'r0'], dtype='<U2')
    source_depth_shift = np.asarray([0.005, 0.010], dtype=np.float64)
    uphole_shift = np.asarray([-0.001, -0.002], dtype=np.float64)
    source_manual = np.asarray([0.0005, 0.001], dtype=np.float64)
    receiver_manual = np.asarray([0.002, -0.001], dtype=np.float64)
    source_field = source_depth_shift + uphole_shift + source_manual
    receiver_field = receiver_manual
    source_field_sorted = np.asarray(
        [source_field[0], source_field[1], source_field[0], source_field[1]],
        dtype=np.float64,
    )
    receiver_field_sorted = np.asarray(
        [receiver_field[0], receiver_field[1], receiver_field[1], receiver_field[0]],
        dtype=np.float64,
    )
    trace_field = source_field_sorted + receiver_field_sorted
    base_refraction = np.asarray(base.refraction_trace_shift_s_sorted, dtype=np.float64)
    applied_field = (
        np.where(np.isfinite(base_refraction), trace_field, 0.0)
        if apply_to_trace_shift
        else np.zeros(base_refraction.shape, dtype=np.float64)
    )
    final_trace_shift = base_refraction + applied_field
    final_trace_shift[~np.isfinite(base_refraction)] = np.nan
    return replace(
        base,
        source_endpoint_key_sorted=source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=receiver_endpoint_key_sorted,
        source_depth_m=np.asarray([4.0, 8.0], dtype=np.float64),
        source_depth_shift_s=source_depth_shift,
        source_depth_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_depth_field_correction_qc={
            'source_depth_mode': 'weathering_velocity_time',
            'component_name': 'source_depth_shift_s',
        },
        source_uphole_time_s=np.asarray([0.010, 0.020], dtype=np.float64),
        source_uphole_shift_s=uphole_shift,
        source_uphole_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_uphole_field_correction_qc={
            'uphole_mode': 'header_time',
            'component_name': 'uphole_shift_s',
        },
        source_manual_static_shift_s=source_manual,
        source_manual_static_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        receiver_manual_static_shift_s=receiver_manual,
        receiver_manual_static_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        manual_static_field_correction_qc={
            'manual_static_mode': 'inline_table',
            'manual_static_sign_convention': 'applied_shift_s',
            'component_name': 'manual_static_shift_s',
        },
        source_field_shift_s=source_field,
        source_field_static_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        receiver_field_shift_s=receiver_field,
        receiver_field_static_status=np.asarray(['ok', 'ok'], dtype='<U48'),
        source_field_shift_s_sorted=source_field_sorted,
        receiver_field_shift_s_sorted=receiver_field_sorted,
        trace_field_shift_s_sorted=trace_field,
        trace_field_static_status_sorted=np.asarray(
            ['ok', 'ok', 'ok', 'ok'],
            dtype='<U48',
        ),
        trace_field_static_valid_mask_sorted=np.asarray(
            [True, True, True, True],
            dtype=bool,
        ),
        base_refraction_trace_shift_s_sorted=base_refraction,
        final_trace_shift_s_sorted=final_trace_shift,
        final_trace_static_status_sorted=base.trace_static_status_sorted,
        final_trace_static_valid_mask_sorted=base.trace_static_valid_mask_sorted,
        applied_field_shift_s_sorted=applied_field,
        field_composition_qc={
            'composition_enabled': True,
            'apply_to_trace_shift': apply_to_trace_shift,
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
