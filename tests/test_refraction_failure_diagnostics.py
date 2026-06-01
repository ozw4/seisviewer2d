from app.statics.refraction.application.design_matrix import (
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
)
from app.statics.refraction.application.preflight_diagnostics import (
    RefractionStaticPreflightError,
)
from app.statics.refraction.application.workflow import _failed_refraction_static_stage
from app.statics.refraction.domain.solver import RefractionStaticSolverError


def test_failed_refraction_static_stage_classifies_preflight_error(tmp_path) -> None:
    stage = _failed_refraction_static_stage(
        RefractionStaticPreflightError(
            'No valid refraction observations remain after preflight filtering.'
        ),
        tmp_path,
    )

    assert stage == 'preflight'


def test_failed_refraction_static_stage_prefers_solver_message_over_design_artifacts(
    tmp_path,
) -> None:
    (tmp_path / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).write_text(
        '{}', encoding='utf-8'
    )

    stage = _failed_refraction_static_stage(
        RefractionStaticSolverError(
            'refraction static bounded-LS failed: maximum iterations exceeded'
        ),
        tmp_path,
    )

    assert stage == 'solver'


def test_failed_refraction_static_stage_keeps_design_matrix_errors(
    tmp_path,
) -> None:
    (tmp_path / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).write_text(
        '{}', encoding='utf-8'
    )

    stage = _failed_refraction_static_stage(
        ValueError(
            'refraction design matrix contains 1 all-zero active-node columns'
        ),
        tmp_path,
    )

    assert stage == 'design_matrix'


def test_failed_refraction_static_stage_prefers_artifact_writer_message(
    tmp_path,
) -> None:
    (tmp_path / REFRACTION_DESIGN_MATRIX_QC_JSON_NAME).write_text(
        '{}', encoding='utf-8'
    )

    stage = _failed_refraction_static_stage(
        RuntimeError('writing_refraction_static_artifacts failed'),
        tmp_path,
    )

    assert stage == 'artifact_writer'
