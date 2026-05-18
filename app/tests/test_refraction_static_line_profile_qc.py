from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.api.schemas import RefractionStaticApplyRequest
from app.services.refraction_static_artifacts import (
    REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
    REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
    REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
    write_refraction_static_artifacts,
)
from app.tests._refraction_multilayer_3layer_helpers import compute_three_layer_workflow
from app.tests._refraction_static_synthetic import (
    synthetic_cell_refracted_arrival_input_model,
    synthetic_cell_refraction_apply_request,
    run_synthetic_cell_refraction_statics,
)


def test_line_profile_qc_generated_for_line_2d_projected(tmp_path: Path) -> None:
    paths, result = _write_line_projected_profile(tmp_path)

    for path in (
        paths.refraction_line_profile_qc_source_csv,
        paths.refraction_line_profile_qc_receiver_csv,
        paths.refraction_line_profile_qc_combined_csv,
        paths.refraction_line_profile_qc_npz,
        paths.refraction_line_profile_qc_json,
    ):
        assert path.is_file()

    manifest = json.loads(paths.manifest_json.read_text(encoding='utf-8'))
    manifest_names = {item['name'] for item in manifest['artifacts']}
    assert {
        REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME,
        REFRACTION_LINE_PROFILE_QC_RECEIVER_CSV_NAME,
        REFRACTION_LINE_PROFILE_QC_COMBINED_CSV_NAME,
        REFRACTION_LINE_PROFILE_QC_NPZ_NAME,
        REFRACTION_LINE_PROFILE_QC_JSON_NAME,
    } <= manifest_names

    qc = json.loads(paths.refraction_line_profile_qc_json.read_text(encoding='utf-8'))
    assert qc['status'] == 'available'
    assert qc['coordinate_mode'] == 'line_2d_projected'
    assert qc['sort_order'] == ['endpoint_kind', 'inline_m', 'endpoint_key']
    assert qc['source_row_count'] == int(result.source_endpoint_key.shape[0])
    assert qc['receiver_row_count'] == int(result.receiver_endpoint_key.shape[0])


def test_line_profile_qc_sorted_by_inline_distance(tmp_path: Path) -> None:
    paths, _result = _write_line_projected_profile(tmp_path)

    rows = _read_csv(paths.refraction_line_profile_qc_combined_csv)
    actual = [
        (row['endpoint_kind'], float(row['inline_m']), row['endpoint_key'])
        for row in rows
    ]
    assert actual == sorted(actual)


def test_line_profile_qc_matches_source_receiver_static_tables(
    tmp_path: Path,
) -> None:
    paths, _result = _write_line_projected_profile(tmp_path)

    source_profile = {
        row['endpoint_key']: row
        for row in _read_csv(paths.refraction_line_profile_qc_source_csv)
    }
    receiver_profile = {
        row['endpoint_key']: row
        for row in _read_csv(paths.refraction_line_profile_qc_receiver_csv)
    }
    source_table = {
        row['source_endpoint_key']: row
        for row in _read_csv(paths.source_static_table_csv)
    }
    receiver_table = {
        row['receiver_endpoint_key']: row
        for row in _read_csv(paths.receiver_static_table_csv)
    }

    assert set(source_profile) == set(source_table)
    assert set(receiver_profile) == set(receiver_table)
    for endpoint_key, profile_row in source_profile.items():
        _assert_profile_matches_static_row(
            profile_row,
            source_table[endpoint_key],
            endpoint_prefix='source',
        )
    for endpoint_key, profile_row in receiver_profile.items():
        _assert_profile_matches_static_row(
            profile_row,
            receiver_table[endpoint_key],
            endpoint_prefix='receiver',
        )


def test_line_profile_qc_csv_and_npz_agree(tmp_path: Path) -> None:
    paths, _result = _write_line_projected_profile(tmp_path)

    rows = _read_csv(paths.refraction_line_profile_qc_combined_csv)
    assert rows
    with np.load(paths.refraction_line_profile_qc_npz, allow_pickle=False) as data:
        assert set(rows[0]) <= set(data.files)
        assert int(data['endpoint_kind'].shape[0]) == len(rows)
        for index, row in enumerate(rows):
            for column in row:
                _assert_csv_value_matches_npz(row[column], data[column][index])


def test_line_profile_qc_has_three_layer_columns_when_unavailable(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / 'three-layer'
    compute_three_layer_workflow(job_dir=job_dir)

    columns = _csv_header(job_dir / REFRACTION_LINE_PROFILE_QC_SOURCE_CSV_NAME)
    assert {
        't3_ms',
        'vsub_m_s',
        'sh3_m',
        'layer2_base_elevation_m',
    } <= set(columns)
    with np.load(job_dir / REFRACTION_LINE_PROFILE_QC_NPZ_NAME, allow_pickle=False) as data:
        assert {'t3_ms', 'vsub_m_s', 'sh3_m', 'layer2_base_elevation_m'} <= set(
            data.files
        )
    qc = json.loads((job_dir / REFRACTION_LINE_PROFILE_QC_JSON_NAME).read_text())
    assert qc['status'] == 'unavailable'


def test_line_profile_qc_grid_3d_unavailable_status(tmp_path: Path) -> None:
    req = synthetic_cell_refraction_apply_request()
    result = run_synthetic_cell_refraction_statics(req=req)

    paths = write_refraction_static_artifacts(result=result, req=req, job_dir=tmp_path)

    qc = json.loads(paths.refraction_line_profile_qc_json.read_text(encoding='utf-8'))
    assert qc['status'] == 'unavailable'
    assert qc['coordinate_mode'] == 'grid_3d'
    assert qc['availability_reason'] == (
        'projected_inline_coordinates_unavailable_for_grid_3d'
    )
    assert _read_csv(paths.refraction_line_profile_qc_combined_csv) == []
    with np.load(paths.refraction_line_profile_qc_npz, allow_pickle=False) as data:
        assert int(data['endpoint_kind'].shape[0]) == 0


def _write_line_projected_profile(tmp_path: Path):
    line_origin_x_m = 1000.0
    line_origin_y_m = 2000.0
    line_azimuth_deg = 45.0
    req = synthetic_cell_refraction_apply_request()
    payload = req.model_dump(mode='json')
    payload['model']['refractor_cell'].update(
        {
            'coordinate_mode': 'line_2d_projected',
            'line_origin_x_m': line_origin_x_m,
            'line_origin_y_m': line_origin_y_m,
            'line_azimuth_deg': line_azimuth_deg,
        }
    )
    line_req = RefractionStaticApplyRequest.model_validate(payload)
    input_model = _map_inline_input_model_to_line_coordinates(
        synthetic_cell_refracted_arrival_input_model(),
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )
    result = run_synthetic_cell_refraction_statics(
        req=line_req,
        input_model=input_model,
    )
    paths = write_refraction_static_artifacts(
        result=result,
        req=line_req,
        job_dir=tmp_path,
    )
    return paths, result


def _map_inline_input_model_to_line_coordinates(
    input_model,
    *,
    line_origin_x_m: float,
    line_origin_y_m: float,
    line_azimuth_deg: float,
):
    azimuth_rad = np.deg2rad(line_azimuth_deg)

    def to_map(inline_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inline = np.asarray(inline_m, dtype=np.float64)
        return (
            line_origin_x_m + inline * np.sin(azimuth_rad),
            line_origin_y_m + inline * np.cos(azimuth_rad),
        )

    source_x, source_y = to_map(input_model.source_x_m_sorted)
    receiver_x, receiver_y = to_map(input_model.receiver_x_m_sorted)
    node_x, node_y = to_map(input_model.node_x_m)
    endpoint_x, endpoint_y = to_map(input_model.endpoint_table.x_m)
    endpoint_table = replace(
        input_model.endpoint_table,
        x_m=np.ascontiguousarray(endpoint_x, dtype=np.float64),
        y_m=np.ascontiguousarray(endpoint_y, dtype=np.float64),
    )
    return replace(
        input_model,
        source_x_m_sorted=np.ascontiguousarray(source_x, dtype=np.float64),
        source_y_m_sorted=np.ascontiguousarray(source_y, dtype=np.float64),
        receiver_x_m_sorted=np.ascontiguousarray(receiver_x, dtype=np.float64),
        receiver_y_m_sorted=np.ascontiguousarray(receiver_y, dtype=np.float64),
        node_x_m=np.ascontiguousarray(node_x, dtype=np.float64),
        node_y_m=np.ascontiguousarray(node_y, dtype=np.float64),
        endpoint_table=endpoint_table,
    )


def _assert_profile_matches_static_row(
    profile_row: dict[str, str],
    static_row: dict[str, str],
    *,
    endpoint_prefix: str,
) -> None:
    assert profile_row['endpoint_kind'] == endpoint_prefix
    assert profile_row['endpoint_key'] == static_row[f'{endpoint_prefix}_endpoint_key']
    assert profile_row['node_id'] == static_row[f'{endpoint_prefix}_node_id']
    assert profile_row['static_status'] == static_row['static_status']
    assert profile_row['solution_status'] == static_row['solution_status']

    field_column = f'{endpoint_prefix}_field_shift_ms'
    for profile_column, static_column in (
        ('x_m', 'x_m'),
        ('y_m', 'y_m'),
        ('surface_elevation_m', 'surface_elevation_m'),
        ('pick_count', 'pick_count'),
        ('used_pick_count', 'used_pick_count'),
        ('residual_rms_ms', 'residual_rms_ms'),
        ('residual_mad_ms', 'residual_mad_ms'),
        ('v1_m_s', 'v1_m_s'),
        ('v2_m_s', 'v2_m_s'),
        ('t1_ms', 't1_ms'),
        ('sh1_m', 'sh1_weathering_thickness_m'),
        ('weathering_correction_ms', 'weathering_correction_ms'),
        ('elevation_correction_ms', 'elevation_correction_ms'),
        ('field_correction_ms', field_column),
        ('manual_static_ms', 'manual_static_shift_ms'),
        ('total_static_ms', 'total_static_ms'),
        ('total_applied_shift_ms', 'total_applied_shift_ms'),
    ):
        _assert_float_text_equal(profile_row[profile_column], static_row[static_column])


def _assert_float_text_equal(left: str, right: str) -> None:
    if left == '' or right == '':
        assert left == right
    else:
        assert float(left) == pytest.approx(float(right))


def _assert_csv_value_matches_npz(text: str, value: object) -> None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        assert text == value
        return
    if isinstance(value, bytes):
        assert text == value.decode('utf-8')
        return
    numeric = float(value)
    if text == '':
        assert not np.isfinite(numeric) or numeric < 0
    else:
        assert float(text) == pytest.approx(numeric)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _csv_header(path: Path) -> list[str]:
    with path.open(encoding='utf-8', newline='') as handle:
        return next(csv.reader(handle))
