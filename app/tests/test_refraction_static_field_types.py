from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, get_type_hints

import numpy as np
import pytest

from app.statics.refraction.contracts.result_types import (
    REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES,
    RefractionEndpointFieldCorrectionResult,
    RefractionFieldCorrectionComponentName,
    RefractionTraceFieldCorrectionResult,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORTS = {
    'app.api.routers',
    'app.api.schemas',
    'app.main',
    'app.statics.refraction.application.input_model',
    'app.statics.refraction.application.workflow',
    'app.trace_store.reader',
    'segyio',
}
_FORBIDDEN_SOURCE_STRINGS = _FORBIDDEN_IMPORTS | {'TraceStoreSectionReader'}


def _float_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _int_array(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _status_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=np.str_)


def _component_shift_s() -> dict[RefractionFieldCorrectionComponentName, np.ndarray]:
    return {
        'source_depth_shift_s': _float_array([0.001, 0.002, 0.003]),
        'uphole_shift_s': _float_array([-0.004, -0.005, -0.006]),
        'manual_static_shift_s': _float_array([0.000, 0.001, -0.001]),
    }


def _component_status() -> dict[RefractionFieldCorrectionComponentName, np.ndarray]:
    return {
        'source_depth_shift_s': _status_array(['ok', 'ok', 'missing']),
        'uphole_shift_s': _status_array(['ok', 'ok', 'ok']),
        'manual_static_shift_s': _status_array(['none', 'ok', 'ok']),
    }


def test_refraction_static_field_types_import_dependency_light() -> None:
    code = f"""
from __future__ import annotations

import importlib
import json
import sys

for name in {sorted(_FORBIDDEN_IMPORTS)!r}:
    sys.modules.pop(name, None)

module = importlib.import_module('app.statics.refraction.contracts.result_types')
assert module.RefractionEndpointFieldCorrectionResult is not None
assert module.RefractionTraceFieldCorrectionResult is not None

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

    module = importlib.import_module('app.statics.refraction.contracts.result_types')
    source = Path(module.__file__ or '').read_text(encoding='utf-8')
    for forbidden in _FORBIDDEN_SOURCE_STRINGS:
        assert forbidden not in source


def test_refraction_endpoint_field_correction_result_shape_contract() -> None:
    result = RefractionEndpointFieldCorrectionResult(
        endpoint_kind=_status_array(['source', 'source', 'receiver']),
        endpoint_key=_int_array([1001, 1002, 2001]),
        endpoint_id=_int_array([11, 12, 21]),
        node_id=_int_array([0, 1, 2]),
        component_shift_s=_component_shift_s(),
        component_status=_component_status(),
        total_field_shift_s=_float_array([-0.003, -0.002, -0.004]),
        field_static_status=_status_array(['ok', 'ok', 'missing']),
        qc={'component_names': REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES},
    )

    assert REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES == (
        'source_depth_shift_s',
        'uphole_shift_s',
        'manual_static_shift_s',
    )
    assert set(result.component_shift_s) == set(
        REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES
    )
    assert set(result.component_status) == set(
        REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES
    )
    assert result.endpoint_key.shape == result.total_field_shift_s.shape
    assert result.endpoint_key.shape == result.field_static_status.shape
    for component in REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES:
        assert result.component_shift_s[component].shape == result.endpoint_key.shape
        assert result.component_status[component].shape == result.endpoint_key.shape

    hints = get_type_hints(RefractionEndpointFieldCorrectionResult)
    assert hints['component_shift_s'] == dict[
        RefractionFieldCorrectionComponentName, np.ndarray
    ]
    assert hints['component_status'] == dict[
        RefractionFieldCorrectionComponentName, np.ndarray
    ]
    assert hints['qc'] == dict[str, Any]

    with pytest.raises(FrozenInstanceError):
        result.qc = {}


def test_refraction_trace_field_correction_result_shape_contract() -> None:
    result = RefractionTraceFieldCorrectionResult(
        source_endpoint_key_sorted=_int_array([1001, 1001, 1002, 1002]),
        receiver_endpoint_key_sorted=_int_array([2001, 2002, 2001, 2002]),
        source_field_shift_s_sorted=_float_array([-0.001, -0.001, -0.002, -0.002]),
        receiver_field_shift_s_sorted=_float_array(
            [-0.003, -0.004, -0.003, -0.004]
        ),
        trace_field_shift_s_sorted=_float_array([-0.004, -0.005, -0.005, -0.006]),
        trace_field_static_status_sorted=_status_array(['ok', 'ok', 'ok', 'ok']),
        qc={'source_endpoint_count': 2, 'receiver_endpoint_count': 2},
    )

    assert result.source_endpoint_key_sorted.shape == result.trace_field_shift_s_sorted.shape
    assert (
        result.receiver_endpoint_key_sorted.shape
        == result.trace_field_static_status_sorted.shape
    )
    assert result.source_field_shift_s_sorted.shape == result.trace_field_shift_s_sorted.shape
    assert (
        result.receiver_field_shift_s_sorted.shape
        == result.trace_field_shift_s_sorted.shape
    )
    np.testing.assert_allclose(
        result.trace_field_shift_s_sorted,
        result.source_field_shift_s_sorted + result.receiver_field_shift_s_sorted,
    )
    assert result.trace_field_static_status_sorted.dtype.kind in {'U', 'S'}

    with pytest.raises(FrozenInstanceError):
        result.trace_field_shift_s_sorted = np.asarray([], dtype=np.float64)
