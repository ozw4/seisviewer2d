from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_artifacts import (
    SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME,
    write_refraction_static_artifacts,
)
from app.services.refraction_static_datum import (
    REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME,
    write_refraction_datum_statics_artifacts,
)
from app.tests._refraction_static_artifact_helpers import _request, _result


def test_field_component_table_columns_exist_when_field_corrections_disabled(
    tmp_path: Path,
) -> None:
    result = _result()
    paths = write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    trace_rows = _read_csv(paths.refraction_statics_csv)

    assert float(source_rows[0]['source_depth_m']) == pytest.approx(0.0)
    assert float(source_rows[0]['source_depth_shift_ms']) == pytest.approx(0.0)
    assert source_rows[0]['source_depth_status'] == 'not_enabled'
    assert float(source_rows[0]['uphole_time_ms']) == pytest.approx(0.0)
    assert float(source_rows[0]['uphole_shift_ms']) == pytest.approx(0.0)
    assert source_rows[0]['uphole_status'] == 'not_enabled'
    assert float(source_rows[0]['manual_static_shift_ms']) == pytest.approx(0.0)
    assert source_rows[0]['manual_static_status'] == 'not_enabled'
    assert float(source_rows[0]['source_field_shift_ms']) == pytest.approx(0.0)
    assert source_rows[0]['source_field_static_status'] == 'not_enabled'
    assert float(source_rows[0]['source_total_with_field_shift_ms']) == pytest.approx(
        float(source_rows[0]['total_applied_shift_ms'])
    )

    assert float(receiver_rows[0]['manual_static_shift_ms']) == pytest.approx(0.0)
    assert receiver_rows[0]['manual_static_status'] == 'not_enabled'
    assert float(receiver_rows[0]['receiver_field_shift_ms']) == pytest.approx(0.0)
    assert receiver_rows[0]['receiver_field_static_status'] == 'not_enabled'
    assert float(
        receiver_rows[0]['receiver_total_with_field_shift_ms']
    ) == pytest.approx(float(receiver_rows[0]['total_applied_shift_ms']))

    assert float(trace_rows[0]['source_field_shift_ms']) == pytest.approx(0.0)
    assert float(trace_rows[0]['receiver_field_shift_ms']) == pytest.approx(0.0)
    assert float(trace_rows[0]['trace_field_shift_ms']) == pytest.approx(0.0)
    assert trace_rows[0]['trace_field_static_status'] == 'not_enabled'
    assert float(trace_rows[0]['final_trace_shift_ms']) == pytest.approx(
        float(trace_rows[0]['refraction_trace_shift_ms'])
    )

    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_depth_shift_s'], [0.0, 0.0])
        np.testing.assert_array_equal(
            data['source_depth_status'],
            ['not_enabled', 'not_enabled'],
        )
        np.testing.assert_allclose(data['source_manual_static_shift_s'], [0.0, 0.0])
        np.testing.assert_array_equal(
            data['source_manual_static_status'],
            ['not_enabled', 'not_enabled'],
        )
        np.testing.assert_allclose(
            data['source_total_with_field_shift_s'],
            data['source_total_applied_shift_s'],
        )
        np.testing.assert_allclose(
            data['receiver_total_with_field_shift_s'],
            data['receiver_total_applied_shift_s'],
        )

    with np.load(paths.solution_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data['trace_field_shift_s_sorted'],
            np.zeros_like(data['refraction_trace_shift_s_sorted']),
        )
        np.testing.assert_array_equal(
            data['trace_field_static_status_sorted'],
            ['not_enabled', 'not_enabled', 'not_enabled', 'not_enabled'],
        )
        np.testing.assert_allclose(
            data['final_trace_shift_s_sorted'],
            data['refraction_trace_shift_s_sorted'],
        )

    write_refraction_datum_statics_artifacts(tmp_path, result)
    datum_rows = _read_csv(tmp_path / REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME)
    assert float(datum_rows[0]['trace_field_shift_ms']) == pytest.approx(0.0)
    assert datum_rows[0]['trace_field_static_status'] == 'not_enabled'
    assert float(datum_rows[0]['final_trace_shift_ms']) == pytest.approx(
        float(datum_rows[0]['refraction_trace_shift_ms'])
    )


def test_field_correction_endpoint_tables_include_component_totals(
    tmp_path: Path,
) -> None:
    result = _field_composed_result()
    paths = write_refraction_static_artifacts(
        result=result,
        req=_field_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    receiver_rows = _read_csv(paths.receiver_static_table_csv)

    assert source_rows[0]['source_depth_m'] == '4.0'
    assert float(source_rows[0]['source_depth_shift_ms']) == pytest.approx(5.0)
    assert float(source_rows[0]['uphole_shift_ms']) == pytest.approx(-1.0)
    assert float(source_rows[0]['manual_static_shift_ms']) == pytest.approx(0.5)
    assert float(source_rows[0]['source_field_shift_ms']) == pytest.approx(4.5)
    assert source_rows[0]['source_field_static_status'] == 'ok'
    assert float(source_rows[0]['source_total_with_field_shift_ms']) == pytest.approx(
        7.0
    )

    assert float(receiver_rows[0]['manual_static_shift_ms']) == pytest.approx(2.0)
    assert float(receiver_rows[0]['receiver_field_shift_ms']) == pytest.approx(2.0)
    assert receiver_rows[0]['receiver_field_static_status'] == 'ok'
    assert float(
        receiver_rows[0]['receiver_total_with_field_shift_ms']
    ) == pytest.approx(4.3)

    assert paths.source_receiver_static_table_npz.name == (
        SOURCE_RECEIVER_STATIC_TABLE_NPZ_NAME
    )
    with np.load(paths.source_receiver_static_table_npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data['source_field_shift_s'], [0.0045, 0.009])
        np.testing.assert_allclose(
            data['source_total_with_field_shift_s'],
            [0.007, 0.0113],
        )
        np.testing.assert_allclose(
            data['receiver_field_shift_s'],
            [0.002, -0.001],
        )
        np.testing.assert_allclose(
            data['receiver_total_with_field_shift_s'],
            [0.0043, 0.0011],
        )


def test_trace_preview_includes_refraction_field_and_final_shift(
    tmp_path: Path,
) -> None:
    result = _field_composed_result()
    paths = write_refraction_static_artifacts(
        result=result,
        req=_field_request(),
        job_dir=tmp_path,
    )

    rows = _read_csv(paths.refraction_statics_csv)
    assert float(rows[0]['source_field_shift_ms']) == pytest.approx(4.5)
    assert float(rows[0]['receiver_field_shift_ms']) == pytest.approx(2.0)
    assert float(rows[0]['trace_field_shift_ms']) == pytest.approx(6.5)
    assert float(rows[0]['refraction_trace_shift_ms']) == pytest.approx(4.8)
    assert float(rows[0]['final_trace_shift_ms']) == pytest.approx(11.3)
    assert rows[0]['trace_field_static_status'] == 'ok'
    assert float(rows[0]['final_trace_shift_ms']) == pytest.approx(
        float(rows[0]['refraction_trace_shift_ms'])
        + float(rows[0]['trace_field_shift_ms'])
    )

    write_refraction_datum_statics_artifacts(tmp_path, result)
    datum_rows = _read_csv(tmp_path / REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME)
    assert float(datum_rows[0]['trace_field_shift_ms']) == pytest.approx(6.5)
    assert float(datum_rows[0]['refraction_trace_shift_ms']) == pytest.approx(4.8)
    assert float(datum_rows[0]['final_trace_shift_ms']) == pytest.approx(11.3)


def test_not_applicable_field_status_preserves_base_shifts(
    tmp_path: Path,
) -> None:
    base = _result()
    zero_source = np.zeros(base.source_endpoint_key.shape, dtype=np.float64)
    zero_receiver = np.zeros(base.receiver_endpoint_key.shape, dtype=np.float64)
    zero_trace = np.zeros(base.sorted_trace_index.shape, dtype=np.float64)
    endpoint_status = np.asarray(['not_applicable', 'not_applicable'], dtype='<U48')
    trace_status = np.full(
        base.sorted_trace_index.shape,
        'not_applicable',
        dtype='<U48',
    )
    result = replace(
        base,
        source_field_shift_s=zero_source,
        source_field_static_status=endpoint_status.copy(),
        receiver_field_shift_s=zero_receiver,
        receiver_field_static_status=endpoint_status.copy(),
        source_field_shift_s_sorted=zero_trace.copy(),
        receiver_field_shift_s_sorted=zero_trace.copy(),
        trace_field_shift_s_sorted=zero_trace.copy(),
        trace_field_static_status_sorted=trace_status,
        trace_field_static_valid_mask_sorted=np.zeros(
            base.sorted_trace_index.shape,
            dtype=bool,
        ),
        base_refraction_trace_shift_s_sorted=base.refraction_trace_shift_s_sorted,
        field_composition_qc={
            'composition_enabled': True,
            'apply_to_trace_shift': True,
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
        },
    )

    paths = write_refraction_static_artifacts(
        result=result,
        req=_request(),
        job_dir=tmp_path,
    )

    source_rows = _read_csv(paths.source_static_table_csv)
    assert source_rows[0]['source_field_static_status'] == 'not_applicable'
    assert float(source_rows[0]['source_total_with_field_shift_ms']) == pytest.approx(
        float(source_rows[0]['total_applied_shift_ms'])
    )

    receiver_rows = _read_csv(paths.receiver_static_table_csv)
    assert receiver_rows[0]['receiver_field_static_status'] == 'not_applicable'
    assert float(
        receiver_rows[0]['receiver_total_with_field_shift_ms']
    ) == pytest.approx(float(receiver_rows[0]['total_applied_shift_ms']))

    trace_rows = _read_csv(paths.refraction_statics_csv)
    assert trace_rows[0]['trace_field_static_status'] == 'not_applicable'
    assert float(trace_rows[0]['final_trace_shift_ms']) == pytest.approx(
        float(trace_rows[0]['refraction_trace_shift_ms'])
    )

    with np.load(paths.solution_npz, allow_pickle=False) as data:
        np.testing.assert_array_equal(
            data['trace_field_static_valid_mask_sorted'],
            [True, True, True, True],
        )
        np.testing.assert_allclose(
            data['final_trace_shift_s_sorted'],
            data['refraction_trace_shift_s_sorted'],
            equal_nan=True,
        )

    write_refraction_datum_statics_artifacts(tmp_path, result)
    datum_rows = _read_csv(tmp_path / REFRACTION_DATUM_TRACE_PREVIEW_CSV_NAME)
    assert datum_rows[0]['trace_field_static_status'] == 'not_applicable'
    assert float(datum_rows[0]['final_trace_shift_ms']) == pytest.approx(
        float(datum_rows[0]['refraction_trace_shift_ms'])
    )


def _field_request() -> RefractionStaticApplyRequest:
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
    }
    return RefractionStaticApplyRequest.model_validate(payload)


def _field_composed_result():
    base = _result()
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
    final_refraction = base_refraction + trace_field
    final_refraction[~np.isfinite(base_refraction)] = np.nan
    return replace(
        base,
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
        refraction_trace_shift_s_sorted=final_refraction,
        field_composition_qc={
            'composition_enabled': True,
            'apply_to_trace_shift': True,
            'sign_convention': 'corrected(t) = raw(t - shift_s)',
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
