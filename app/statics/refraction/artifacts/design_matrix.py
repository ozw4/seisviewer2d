"""Design-matrix diagnostic artifact names for refraction statics."""

from __future__ import annotations

from app.statics.refraction.domain.types import RefractionLayerKind

REFRACTION_DESIGN_MATRIX_QC_JSON_NAME = 'refraction_design_matrix_qc.json'
REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME = (
    'refraction_design_matrix_node_diagnostics.csv'
)
_REFRACTION_DESIGN_MATRIX_LAYER_KINDS: tuple[RefractionLayerKind, ...] = (
    'v2_t1',
    'v3_t2',
    'vsub_t3',
)


def refraction_design_matrix_layer_qc_json_name(
    layer_kind: RefractionLayerKind,
) -> str:
    """Return the root artifact name for layer-specific design-matrix QC."""
    _validate_design_matrix_layer_kind(layer_kind)
    return f'refraction_design_matrix_{layer_kind}_qc.json'


def refraction_design_matrix_layer_node_diagnostics_csv_name(
    layer_kind: RefractionLayerKind,
) -> str:
    """Return the root artifact name for layer-specific node diagnostics."""
    _validate_design_matrix_layer_kind(layer_kind)
    return f'refraction_design_matrix_{layer_kind}_node_diagnostics.csv'


def refraction_design_matrix_layer_artifact_names(
    layer_kind: RefractionLayerKind,
) -> tuple[str, str]:
    """Return layer-specific root artifact names for design-matrix diagnostics."""
    return (
        refraction_design_matrix_layer_qc_json_name(layer_kind),
        refraction_design_matrix_layer_node_diagnostics_csv_name(layer_kind),
    )


def all_refraction_design_matrix_layer_artifact_names() -> tuple[str, ...]:
    """Return all registered layer-specific design-matrix diagnostic names."""
    names: list[str] = []
    for layer_kind in _REFRACTION_DESIGN_MATRIX_LAYER_KINDS:
        names.extend(refraction_design_matrix_layer_artifact_names(layer_kind))
    return tuple(names)


def _validate_design_matrix_layer_kind(layer_kind: RefractionLayerKind) -> None:
    if layer_kind not in _REFRACTION_DESIGN_MATRIX_LAYER_KINDS:
        raise ValueError(
            f'unsupported design-matrix diagnostics layer kind: {layer_kind}'
        )


__all__ = [
    'REFRACTION_DESIGN_MATRIX_NODE_DIAGNOSTICS_CSV_NAME',
    'REFRACTION_DESIGN_MATRIX_QC_JSON_NAME',
    'all_refraction_design_matrix_layer_artifact_names',
    'refraction_design_matrix_layer_artifact_names',
    'refraction_design_matrix_layer_node_diagnostics_csv_name',
    'refraction_design_matrix_layer_qc_json_name',
]
