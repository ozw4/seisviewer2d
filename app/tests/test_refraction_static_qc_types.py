from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from app.services.refraction_static_qc_types import (
    REFRACTION_STATIC_QC_SIGN_CONVENTION,
    RefractionCellQcSeries,
    RefractionFirstBreakQcSeries,
    RefractionProfileQcSeries,
    RefractionStaticComponentQcSeries,
    refraction_qc_series_to_csv_rows,
    refraction_qc_series_to_csv_text,
    refraction_qc_series_to_json_safe_dict,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORTS = {
    'app.api.routers',
    'app.api.schemas',
    'app.main',
    'app.services.reader',
    'app.services.refraction_static_inputs',
    'app.services.refraction_static_service',
    'app.trace_store.reader',
    'segyio',
}
_FORBIDDEN_SOURCE_STRINGS = _FORBIDDEN_IMPORTS | {
    'BaseModel',
    'TraceStoreSectionReader',
}


def _float_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _int_array(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _bool_array(values: list[bool]) -> np.ndarray:
    return np.asarray(values, dtype=np.bool_)


def _text_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=np.str_)


def _empty_float_array() -> np.ndarray:
    return np.asarray([], dtype=np.float64)


def test_refraction_static_qc_types_are_dependency_light() -> None:
    code = f"""
from __future__ import annotations

import importlib
import json
import sys

for name in {sorted(_FORBIDDEN_IMPORTS)!r}:
    sys.modules.pop(name, None)

module = importlib.import_module('app.services.refraction_static_qc_types')
assert module.RefractionFirstBreakQcSeries is not None
assert module.RefractionProfileQcSeries is not None
assert module.RefractionCellQcSeries is not None
assert module.RefractionStaticComponentQcSeries is not None

forbidden = set({sorted(_FORBIDDEN_IMPORTS)!r})
print(json.dumps(sorted(name for name in sys.modules if name in forbidden)))
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert json.loads(result.stdout) == []

    module = importlib.import_module('app.services.refraction_static_qc_types')
    source = Path(module.__file__ or '').read_text(encoding='utf-8')
    for forbidden in _FORBIDDEN_SOURCE_STRINGS:
        assert forbidden not in source


def test_first_break_qc_series_serializes_to_json_safe_dict() -> None:
    series = RefractionFirstBreakQcSeries(
        trace_index_sorted=_int_array([4, 9]),
        source_endpoint_key=_text_array(['source:1001', 'source:1002']),
        receiver_endpoint_key=_text_array(['receiver:2001', 'receiver:2002']),
        offset_m=_float_array([120.0, 240.0]),
        observed_time_s=_float_array([0.120, np.nan]),
        modeled_time_s=_float_array([0.118, 0.245]),
        residual_time_s=_float_array([0.002, np.nan]),
        layer_kind=_text_array(['v2_t1', 'v3_t2']),
        status=_text_array(['ok', 'invalid_pick']),
    )

    payload = refraction_qc_series_to_json_safe_dict(series)

    assert payload == {
        'trace_index_sorted': [4, 9],
        'source_endpoint_key': ['source:1001', 'source:1002'],
        'receiver_endpoint_key': ['receiver:2001', 'receiver:2002'],
        'offset_m': [120.0, 240.0],
        'observed_time_s': [0.120, None],
        'modeled_time_s': [0.118, 0.245],
        'residual_time_s': [0.002, None],
        'layer_kind': ['v2_t1', 'v3_t2'],
        'status': ['ok', 'invalid_pick'],
    }
    json.dumps(payload, allow_nan=False)

    rows = refraction_qc_series_to_csv_rows(series)
    assert rows[0]['trace_index_sorted'] == 4
    assert rows[0]['source_endpoint_key'] == 'source:1001'
    assert rows[0]['observed_time_s'] == 0.120
    assert rows[1]['observed_time_s'] == ''
    assert rows[1]['status'] == 'invalid_pick'


def test_profile_qc_series_serializes_nan_as_none_or_configured_value() -> None:
    series = RefractionProfileQcSeries(
        endpoint_kind='source',
        endpoint_key=_text_array(['source:1001', 'source:1002']),
        inline_m=_float_array([1000.0, 1100.0]),
        t1_s=_float_array([0.010, np.nan]),
        t2_s=None,
        t3_s=None,
        velocity_m_s={'v2': _float_array([2400.0, np.inf])},
        static_components_s={'total_applied_shift': _float_array([-0.002, np.nan])},
        status=_text_array(['ok', 'invalid_weathering']),
    )

    payload = refraction_qc_series_to_json_safe_dict(series)
    assert payload['endpoint_kind'] == 'source'
    assert payload['t1_s'] == [0.010, None]
    assert payload['t2_s'] is None
    assert payload['velocity_m_s'] == {'v2': [2400.0, None]}
    assert payload['static_components_s'] == {
        'total_applied_shift': [-0.002, None],
    }
    json.dumps(payload, allow_nan=False)

    configured = refraction_qc_series_to_json_safe_dict(
        series,
        nonfinite_value='NaN',
    )
    assert configured['t1_s'] == [0.010, 'NaN']
    assert configured['velocity_m_s'] == {'v2': [2400.0, 'NaN']}

    rows = refraction_qc_series_to_csv_rows(series)
    assert rows == [
        {
            'endpoint_kind': 'source',
            'endpoint_key': 'source:1001',
            'inline_m': 1000.0,
            't1_s': 0.010,
            't2_s': None,
            't3_s': None,
            'velocity_m_s_v2': 2400.0,
            'static_components_s_total_applied_shift': -0.002,
            'status': 'ok',
        },
        {
            'endpoint_kind': 'source',
            'endpoint_key': 'source:1002',
            'inline_m': 1100.0,
            't1_s': '',
            't2_s': None,
            't3_s': None,
            'velocity_m_s_v2': '',
            'static_components_s_total_applied_shift': '',
            'status': 'invalid_weathering',
        },
    ]


def test_cell_qc_series_handles_empty_arrays() -> None:
    series = RefractionCellQcSeries(
        layer_kind='v2_t1',
        cell_id=_int_array([]),
        ix=_int_array([]),
        iy=_int_array([]),
        x_min_m=_empty_float_array(),
        x_max_m=_empty_float_array(),
        y_min_m=_empty_float_array(),
        y_max_m=_empty_float_array(),
        x_center_m=_empty_float_array(),
        y_center_m=_empty_float_array(),
        active=_bool_array([]),
        n_observations=_int_array([]),
        n_used_observations=_int_array([]),
        n_rejected_observations=_int_array([]),
        velocity_m_s=_empty_float_array(),
        slowness_s_per_m=_empty_float_array(),
        residual_rms_s=_empty_float_array(),
        residual_mad_s=_empty_float_array(),
        residual_mean_s=_empty_float_array(),
        residual_p95_abs_s=_empty_float_array(),
        smoothing_neighbor_count=_int_array([]),
        status=_text_array([]),
    )

    payload = refraction_qc_series_to_json_safe_dict(series)
    assert payload['layer_kind'] == 'v2_t1'
    assert payload['cell_id'] == []
    assert payload['velocity_m_s'] == []
    json.dumps(payload, allow_nan=False)

    assert refraction_qc_series_to_csv_rows(series) == []
    csv_text = refraction_qc_series_to_csv_text(series)
    assert csv_text.startswith(
        'layer_kind,cell_id,ix,iy,x_min_m,x_max_m,y_min_m,y_max_m,'
    )
    assert csv_text.count('\n') == 1


def test_static_component_qc_series_serializes_strings_and_sign_convention() -> None:
    series = RefractionStaticComponentQcSeries(
        endpoint_kind=_text_array(['source', 'receiver']),
        endpoint_key=_text_array(['source:1001', 'receiver:2001']),
        component_shift_s={
            'weathering_correction': _float_array([-0.004, -0.003]),
            'manual_static': _float_array([0.001, np.nan]),
        },
        component_status={
            'weathering_correction': _text_array(['ok', 'ok']),
            'manual_static': _text_array(['ok', 'not_enabled']),
        },
        total_static_s=_float_array([-0.003, -0.003]),
        total_applied_shift_s=_float_array([-0.003, -0.003]),
        status=_text_array(['ok', 'ok']),
    )

    payload = refraction_qc_series_to_json_safe_dict(series)
    assert payload['sign_convention'] == REFRACTION_STATIC_QC_SIGN_CONVENTION
    assert payload['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert payload['component_status']['manual_static'] == ['ok', 'not_enabled']
    assert payload['component_shift_s']['manual_static'] == [0.001, None]
    json.dumps(payload, allow_nan=False)

    rows = refraction_qc_series_to_csv_rows(series)
    assert rows[0]['endpoint_kind'] == 'source'
    assert rows[0]['component_shift_s_weathering_correction'] == -0.004
    assert rows[0]['component_status_manual_static'] == 'ok'
    assert rows[0]['sign_convention'] == REFRACTION_STATIC_QC_SIGN_CONVENTION
    assert rows[1]['component_shift_s_manual_static'] == ''
