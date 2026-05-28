"""Shared aliases and constants for refraction static contracts."""

from typing import Literal


RefractionStaticExportFormat = Literal[
    'canonical_static_table',
    'lsst',
    'lsst_plus',
    'time_term_spreadsheet',
    'first_break_time',
]
RefractionStaticQcBundleInclude = Literal[
    'summary',
    'first_break',
    'reduced_time',
    'profiles',
    'cells',
    'static_components',
    'gather_preview',
]
RefractionStaticQcBundleCoordinateMode = Literal[
    'auto',
    'line_2d_projected',
    'grid_3d',
]
RefractionStaticGatherPreviewAxis = Literal['section', 'source', 'receiver']
RefractionStaticGatherPreviewOverlayLayer = Literal[
    'observed_first_break',
    'modeled_first_break',
    'reduced_time',
    'static_shift_trace_curve',
]
RefractionStaticGatherPreviewScaling = Literal['amax', 'tracewise']
RefractionStaticGatherPreviewSampleSource = Literal[
    'corrected_tracestore',
    'raw_tracestore_shifted_on_the_fly',
]

REFRACTION_STATIC_DEFAULT_EXPORT_FORMATS: tuple[
    RefractionStaticExportFormat, ...
] = (
    'canonical_static_table',
    'time_term_spreadsheet',
)

RefractionStaticLayerKind = Literal['v2_t1', 'v3_t2', 'vsub_t3']
RefractionStaticLayerVelocityMode = Literal[
    'fixed_global',
    'solve_global',
    'solve_cell',
]
_REFRACTION_STATIC_LAYER_ORDER: dict[RefractionStaticLayerKind, int] = {
    'v2_t1': 0,
    'v3_t2': 1,
    'vsub_t3': 2,
}
