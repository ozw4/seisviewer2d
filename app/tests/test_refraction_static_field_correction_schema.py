from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    RefractionStaticApplyRequest,
    RefractionStaticFieldCorrectionArtifactRequest,
    RefractionStaticFieldCorrectionCompositionRequest,
    RefractionStaticFieldCorrectionsRequest,
    RefractionStaticManualStaticInlineEntry,
    RefractionStaticManualStaticRequest,
    RefractionStaticSourceDepthCorrectionRequest,
    RefractionStaticUpholeCorrectionRequest,
)
from app.services.refraction_static_service import (
    RefractionFieldCorrectionNotImplemented,
    _with_source_depth_field_correction,
    reject_unsupported_refraction_field_corrections,
)
from app.services.refraction_static_source_depth import resolve_refraction_source_depth
from app.services.refraction_static_types import ResolvedRefractionFirstLayer
from app.tests._refraction_static_synthetic import (
    SYNTHETIC_V1_M_S,
    run_synthetic_refraction_statics,
    synthetic_refracted_arrival_input_model,
    synthetic_refraction_apply_request,
)


def _base_payload() -> dict[str, Any]:
    return synthetic_refraction_apply_request().model_dump(mode='json')


def _payload_with_field_corrections(
    field_corrections: dict[str, Any],
) -> dict[str, Any]:
    payload = _base_payload()
    payload['field_corrections'] = field_corrections
    return payload


def _artifact_ref(name: str = 'manual_static_table.csv') -> dict[str, str]:
    return {
        'job_id': 'manual-static-job',
        'artifact_name': name,
    }


def _all_none_field_corrections() -> dict[str, Any]:
    return {
        'source_depth': {'mode': 'none'},
        'uphole': {'mode': 'none'},
        'manual_static': {'mode': 'none'},
        'composition': {
            'enabled': True,
            'apply_to_trace_shift': True,
            'invalid_component_policy': 'fail',
            'double_application_policy': 'warn',
        },
    }


def test_refraction_static_field_correction_models_forbid_extra_fields() -> None:
    for model_cls in (
        RefractionStaticFieldCorrectionArtifactRequest,
        RefractionStaticSourceDepthCorrectionRequest,
        RefractionStaticUpholeCorrectionRequest,
        RefractionStaticManualStaticInlineEntry,
        RefractionStaticManualStaticRequest,
        RefractionStaticFieldCorrectionCompositionRequest,
        RefractionStaticFieldCorrectionsRequest,
    ):
        assert model_cls.model_config.get('extra') == 'forbid'


def test_refraction_static_field_corrections_omitted_preserves_legacy_behavior() -> None:
    req = synthetic_refraction_apply_request()

    assert 'field_corrections' not in req.model_fields_set
    assert req.field_corrections.source_depth.mode == 'none'
    assert req.field_corrections.uphole.mode == 'none'
    assert req.field_corrections.manual_static.mode == 'none'

    result = run_synthetic_refraction_statics(req=req)

    assert np.any(result.refraction_trace_shift_s_sorted != 0.0)
    assert np.all(result.trace_static_valid_mask_sorted)


def test_refraction_static_field_corrections_none_modes_are_valid() -> None:
    legacy_req = synthetic_refraction_apply_request()
    none_req = RefractionStaticApplyRequest.model_validate(
        _payload_with_field_corrections(_all_none_field_corrections())
    )
    input_model = synthetic_refracted_arrival_input_model()

    legacy_result = run_synthetic_refraction_statics(
        req=legacy_req,
        input_model=input_model,
    )
    none_result = run_synthetic_refraction_statics(
        req=none_req,
        input_model=input_model,
    )

    np.testing.assert_allclose(
        none_result.refraction_trace_shift_s_sorted,
        legacy_result.refraction_trace_shift_s_sorted,
    )
    np.testing.assert_allclose(
        none_result.source_refraction_shift_s_sorted,
        legacy_result.source_refraction_shift_s_sorted,
    )
    np.testing.assert_allclose(
        none_result.receiver_refraction_shift_s_sorted,
        legacy_result.receiver_refraction_shift_s_sorted,
    )
    np.testing.assert_array_equal(
        none_result.trace_static_status_sorted,
        legacy_result.trace_static_status_sorted,
    )


def test_refraction_static_source_depth_mode_requires_source_depth_input() -> None:
    with pytest.raises(
        ValidationError,
        match='field_corrections.source_depth.source_depth_byte or '
        'geometry.source_depth_byte is required',
    ):
        RefractionStaticApplyRequest.model_validate(
            _payload_with_field_corrections(
                {'source_depth': {'mode': 'weathering_velocity_time'}}
            )
        )


def test_refraction_static_source_depth_mode_accepts_existing_geometry_depth() -> None:
    payload = _payload_with_field_corrections(
        {'source_depth': {'mode': 'weathering_velocity_time'}}
    )
    payload['geometry']['source_depth_byte'] = 115

    req = RefractionStaticApplyRequest.model_validate(payload)

    assert req.geometry.source_depth_byte == 115
    assert req.field_corrections.source_depth.source_depth_byte is None


def test_refraction_static_source_depth_mode_is_no_longer_rejected_by_service() -> None:
    req = RefractionStaticApplyRequest.model_validate(
        _payload_with_field_corrections(
            {
                'source_depth': {
                    'mode': 'weathering_velocity_time',
                    'source_depth_byte': 115,
                }
            }
        )
    )

    reject_unsupported_refraction_field_corrections(req)


def test_source_depth_weathering_time_adds_source_only_component() -> None:
    req = RefractionStaticApplyRequest.model_validate(
        _payload_with_field_corrections(
            {
                'source_depth': {
                    'mode': 'weathering_velocity_time',
                    'source_depth_byte': 115,
                }
            }
        )
    )
    input_model = synthetic_refracted_arrival_input_model()
    source_depth_m_sorted = (
        2.0 + np.asarray(input_model.source_node_id_sorted, dtype=np.float64)
    )
    source_depth_result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        source_endpoint_id_sorted=input_model.source_node_id_sorted,
        source_node_id_sorted=input_model.source_node_id_sorted,
        source_depth_m_sorted=source_depth_m_sorted,
        mode='weathering_velocity_time',
        source_depth_byte=115,
    )
    input_with_depth = replace(
        input_model,
        source_depth_m_sorted=source_depth_m_sorted,
        source_depth_result=source_depth_result,
    )
    base_result = run_synthetic_refraction_statics(
        req=req,
        input_model=input_with_depth,
    )

    result_with_component = _with_source_depth_field_correction(
        result=base_result,
        input_model=input_with_depth,
        req=req,
        resolved_first_layer=ResolvedRefractionFirstLayer(
            mode='constant',
            weathering_velocity_m_s=SYNTHETIC_V1_M_S,
            status='constant',
            qc={},
        ),
    )

    shift_by_source_key = {
        str(key): float(depth) / SYNTHETIC_V1_M_S
        for key, depth in zip(
            source_depth_result.source_endpoint_key,
            source_depth_result.source_depth_m,
            strict=True,
        )
    }
    expected_source_shift = np.asarray(
        [shift_by_source_key[str(key)] for key in base_result.source_endpoint_key],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        result_with_component.source_depth_shift_s,
        expected_source_shift,
    )
    np.testing.assert_allclose(
        result_with_component.source_refraction_shift_s,
        base_result.source_refraction_shift_s,
    )
    np.testing.assert_allclose(
        result_with_component.source_refraction_shift_s_sorted,
        base_result.source_refraction_shift_s_sorted,
    )
    np.testing.assert_allclose(
        result_with_component.refraction_trace_shift_s_sorted,
        base_result.refraction_trace_shift_s_sorted,
    )
    np.testing.assert_array_equal(
        result_with_component.trace_static_status_sorted,
        base_result.trace_static_status_sorted,
    )
    np.testing.assert_array_equal(
        result_with_component.trace_static_valid_mask_sorted,
        base_result.trace_static_valid_mask_sorted,
    )
    np.testing.assert_allclose(
        result_with_component.receiver_refraction_shift_s,
        base_result.receiver_refraction_shift_s,
    )
    assert result_with_component.source_depth_field_correction_qc is not None
    assert (
        result_with_component.source_depth_field_correction_qc['component_name']
        == 'source_depth_shift_s'
    )


def test_refraction_static_uphole_header_mode_requires_header_byte() -> None:
    with pytest.raises(
        ValidationError,
        match='field_corrections.uphole.uphole_time_byte is required',
    ):
        RefractionStaticApplyRequest.model_validate(
            _payload_with_field_corrections({'uphole': {'mode': 'header_time'}})
        )


def test_refraction_static_uphole_manual_mode_requires_table_reference() -> None:
    with pytest.raises(
        ValidationError,
        match='field_corrections.uphole.manual_table is required',
    ):
        RefractionStaticApplyRequest.model_validate(
            _payload_with_field_corrections({'uphole': {'mode': 'manual_table'}})
        )


def test_refraction_static_manual_static_artifact_mode_requires_table_reference() -> None:
    with pytest.raises(
        ValidationError,
        match='source_table_artifact or receiver_table_artifact is required',
    ):
        RefractionStaticApplyRequest.model_validate(
            _payload_with_field_corrections(
                {
                    'manual_static': {
                        'mode': 'artifact_table',
                        'sign_convention': 'applied_shift_s',
                    }
                }
            )
        )


def test_refraction_static_manual_static_requires_explicit_sign_convention() -> None:
    with pytest.raises(
        ValidationError,
        match='field_corrections.manual_static.sign_convention is required',
    ):
        RefractionStaticApplyRequest.model_validate(
            _payload_with_field_corrections(
                {
                    'manual_static': {
                        'mode': 'artifact_table',
                        'source_table_artifact': _artifact_ref('source_statics.csv'),
                    }
                }
            )
        )


@pytest.mark.parametrize(
    ('field_corrections', 'expected_mode'),
    [
        (
            {
                'uphole': {
                    'mode': 'header_time',
                    'uphole_time_byte': 119,
                    'uphole_time_unit': 'ms',
                }
            },
            'field_corrections.uphole.mode=header_time',
        ),
        (
            {
                'manual_static': {
                    'mode': 'artifact_table',
                    'sign_convention': 'applied_shift_s',
                    'source_table_artifact': _artifact_ref('source_statics.csv'),
                }
            },
            'field_corrections.manual_static.mode=artifact_table',
        ),
        (
            {
                'manual_static': {
                    'mode': 'inline_table',
                    'sign_convention': 'delay_positive_ms',
                    'receiver_inline_table': [
                        {'endpoint_id': 101, 'value': 4.5},
                    ],
                }
            },
            'field_corrections.manual_static.mode=inline_table',
        ),
    ],
)
def test_refraction_static_unsupported_field_correction_modes_fail_clearly(
    field_corrections: dict[str, Any],
    expected_mode: str,
) -> None:
    req = RefractionStaticApplyRequest.model_validate(
        _payload_with_field_corrections(field_corrections)
    )

    with pytest.raises(
        RefractionFieldCorrectionNotImplemented,
        match='M4 field-correction follow-up implementation',
    ) as exc_info:
        reject_unsupported_refraction_field_corrections(req)

    assert re.search(re.escape(expected_mode), str(exc_info.value))
