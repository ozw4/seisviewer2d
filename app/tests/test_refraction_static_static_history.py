from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.api.schemas import RefractionStaticApplyRequest
from app.core.state import AppState
from app.statics.refraction.artifacts import (
    REFRACTION_STATIC_HISTORY_JSON_NAME,
    build_refraction_static_history_payload,
    write_refraction_static_artifacts,
)
from app.statics.refraction.application.workflow import (
    _with_static_history_double_application_qc,
)
from app.tests._refraction_static_artifact_helpers import _request, _result


def _field_request(
    *,
    policy: str = 'warn',
    apply_to_trace_shift: bool = True,
) -> RefractionStaticApplyRequest:
    payload = _request().model_dump(mode='json')
    payload['field_corrections'] = {
        'source_depth': {
            'mode': 'weathering_velocity_time',
            'source_depth_byte': 115,
        },
        'uphole': {
            'mode': 'header_time',
            'uphole_time_byte': 95,
        },
        'manual_static': {
            'mode': 'inline_table',
            'sign_convention': 'applied_shift_s',
            'source_inline_table': [{'endpoint_id': 0, 'value': 0.001}],
            'receiver_inline_table': [{'endpoint_id': 0, 'value': -0.002}],
        },
        'composition': {
            'apply_to_trace_shift': apply_to_trace_shift,
            'double_application_policy': policy,
        },
    }
    return RefractionStaticApplyRequest.model_validate(payload)


def _state_with_lineage(tmp_path: Path, components: list[str]) -> AppState:
    state = AppState()
    store = tmp_path / 'trace-store'
    store.mkdir()
    (store / 'meta.json').write_text(
        json.dumps(
            {
                'derived': {
                    'components': [
                        {
                            'name': 'refraction_static_correction',
                            'static_components_applied': components,
                            'field_corrections_applied_to_trace_shift': (
                                any(item != 'refraction' for item in components)
                            ),
                            'field_correction_components_requested': [
                                item for item in components if item != 'refraction'
                            ],
                        }
                    ]
                }
            }
        ),
        encoding='utf-8',
    )
    state.file_registry.update('raw-file-id', store_path=str(store), dt=0.004)
    return state


def test_static_history_written_for_refraction_job(tmp_path: Path) -> None:
    paths = write_refraction_static_artifacts(
        result=_result(),
        req=_request(),
        job_dir=tmp_path,
    )

    history = json.loads(paths.static_history_json.read_text(encoding='utf-8'))
    assert paths.static_history_json.name == REFRACTION_STATIC_HISTORY_JSON_NAME
    assert history['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert history['input_file_id'] == 'raw-file-id'
    assert history['output_file_id'] is None
    assert history['components'][0] == {
        'name': 'refraction',
        'applied_to_trace_shift': True,
        'artifact': 'refraction_static_solution.npz',
    }


def test_static_history_lists_field_components_when_enabled() -> None:
    history = build_refraction_static_history_payload(
        result=_result(),
        req=_field_request(),
        output_file_id='corrected-file-id',
    )

    components = {item['name']: item for item in history['components']}
    assert history['output_file_id'] == 'corrected-file-id'
    assert history['cumulative_shift_field'] == 'final_trace_shift_s_sorted'
    assert components['source_depth']['applied_to_trace_shift'] is True
    assert components['uphole']['applied_to_trace_shift'] is True
    assert components['manual_static']['applied_to_trace_shift'] is True


def test_double_application_policy_warn_records_warning(tmp_path: Path) -> None:
    req = _field_request(policy='warn')
    result = _with_static_history_double_application_qc(
        result=_result(),
        req=req,
        state=_state_with_lineage(tmp_path, ['refraction', 'source_depth']),
    )

    qc = result.qc['static_history']
    assert qc['status'] == 'duplicate_warned'
    assert qc['duplicate_components'] == ['refraction', 'source_depth']
    assert 'double_application_policy=warn' in qc['warnings'][0]


def test_double_application_policy_fail_raises_clear_error(tmp_path: Path) -> None:
    req = _field_request(policy='fail')

    with pytest.raises(ValueError, match='double_application_policy=fail'):
        _with_static_history_double_application_qc(
            result=_result(),
            req=req,
            state=_state_with_lineage(tmp_path, ['refraction', 'source_depth']),
        )


def test_double_application_policy_allow_records_duplicate_component(
    tmp_path: Path,
) -> None:
    req = _field_request(policy='allow')
    result = _with_static_history_double_application_qc(
        result=_result(),
        req=req,
        state=_state_with_lineage(tmp_path, ['manual_static']),
    )

    qc = result.qc['static_history']
    assert qc['status'] == 'duplicate_allowed'
    assert qc['duplicate_components'] == ['manual_static', 'refraction']
    assert 'double_application_policy=allow' in qc['warnings'][0]


def test_static_history_marks_field_components_artifact_only() -> None:
    req = _field_request(apply_to_trace_shift=False)
    history = build_refraction_static_history_payload(
        result=_result(),
        req=req,
    )

    components = {item['name']: item for item in history['components']}
    assert history['cumulative_shift_field'] == 'refraction_trace_shift_s_sorted'
    assert components['source_depth']['applied_to_trace_shift'] is False
    assert components['manual_static']['applied_to_trace_shift'] is False
