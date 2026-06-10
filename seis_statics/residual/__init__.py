"""Residual statics model representation and design-matrix helpers."""

from __future__ import annotations

from seis_statics.residual.design_matrix import (
    ResidualStaticColumnLayout,
    ResidualStaticModelEvaluation,
    ResidualStaticObservationMatrixTriplets,
    ResidualStaticParameterParts,
    build_residual_static_column_layout,
    build_residual_static_observation_matrix_triplets,
    compute_linear_abs_offset_moveout_s,
    evaluate_residual_static_model,
    pack_residual_static_parameters,
    unpack_residual_static_parameters,
)
from seis_statics.residual.types import MoveoutModel, ResidualStaticSolverInputs


__all__ = [
    'MoveoutModel',
    'ResidualStaticColumnLayout',
    'ResidualStaticModelEvaluation',
    'ResidualStaticObservationMatrixTriplets',
    'ResidualStaticParameterParts',
    'ResidualStaticSolverInputs',
    'build_residual_static_column_layout',
    'build_residual_static_observation_matrix_triplets',
    'compute_linear_abs_offset_moveout_s',
    'evaluate_residual_static_model',
    'pack_residual_static_parameters',
    'unpack_residual_static_parameters',
]
