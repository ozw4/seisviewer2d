from app.services.refraction_static_design_matrix import (
    REFRACTION_DESIGN_MATRIX_QC_JSON_NAME,
)
from app.services.refraction_static_service import _failed_refraction_static_stage
from app.services.refraction_static_solver import RefractionStaticSolverError


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
