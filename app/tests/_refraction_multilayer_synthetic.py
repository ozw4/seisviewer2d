"""Dependency-light synthetic fixtures for multi-layer refraction statics.

The builders in this module are intentionally solver-free. They generate known
endpoint layer thicknesses, derive T1/T2/T3 from forward layered T1LSST-style
relationships, and then synthesize branch-specific first-break picks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

CoordinateMode = Literal['grid_3d', 'line_2d_projected']
LayerKind = Literal['v2_t1', 'v3_t2', 'vsub_t3']

LAYER_KINDS: tuple[LayerKind, ...] = ('v2_t1', 'v3_t2', 'vsub_t3')
V2_T1: LayerKind = 'v2_t1'
V3_T2: LayerKind = 'v3_t2'
VSUB_T3: LayerKind = 'vsub_t3'

SYNTHETIC_MULTILAYER_V1_M_S = 800.0
SYNTHETIC_MULTILAYER_V2_M_S = 2400.0
SYNTHETIC_MULTILAYER_V3_M_S = 3600.0
SYNTHETIC_MULTILAYER_VSUB_M_S = 4800.0

_V2_OFFSETS_M = (360.0, 520.0, 700.0)
_V3_OFFSETS_M = (1200.0, 1450.0, 1700.0)
_VSUB_OFFSETS_M = (2400.0, 2700.0, 3000.0)

_LAYER_INDEX_BY_KIND: dict[LayerKind, int] = {
    V2_T1: 1,
    V3_T2: 2,
    VSUB_T3: 3,
}


@dataclass(frozen=True)
class SyntheticMultiLayerRefractionDataset:
    """Small known-truth multi-layer refraction fixture.

    Row-level arrays are in synthetic sorted trace order. Endpoint-level arrays
    carry the full source/receiver static-table truth. Fields prefixed with
    ``true_`` are known values used to build the observed picks.
    """

    name: str
    layer_count: int
    enabled_layer_kinds: tuple[LayerKind, ...]
    coordinate_mode: CoordinateMode
    line_origin_x_m: float | None
    line_origin_y_m: float | None
    line_azimuth_deg: float | None
    layer_offset_range_m_by_kind: dict[str, tuple[float, float | None]]

    sorted_trace_index: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    source_inline_m: np.ndarray | None
    receiver_inline_m: np.ndarray | None
    source_elevation_m: np.ndarray
    receiver_elevation_m: np.ndarray
    source_id: np.ndarray
    receiver_id: np.ndarray
    source_node_id: np.ndarray
    receiver_node_id: np.ndarray
    source_endpoint_key: np.ndarray
    receiver_endpoint_key: np.ndarray
    source_endpoint_index: np.ndarray
    receiver_endpoint_index: np.ndarray
    offset_m: np.ndarray
    first_break_time_s: np.ndarray
    noiseless_first_break_time_s: np.ndarray
    valid_mask: np.ndarray
    rejection_reason: np.ndarray
    layer_kind: np.ndarray
    layer_index: np.ndarray
    expected_layer_mask_by_kind: dict[str, np.ndarray]
    true_noise_s: np.ndarray
    outlier_mask: np.ndarray

    true_v1_m_s: float
    true_v2_m_s: float
    true_v3_m_s: float
    true_vsub_m_s: float
    true_replacement_velocity_m_s: float

    true_source_t1_s: np.ndarray
    true_source_t2_s: np.ndarray
    true_source_t3_s: np.ndarray | None
    true_receiver_t1_s: np.ndarray
    true_receiver_t2_s: np.ndarray
    true_receiver_t3_s: np.ndarray | None
    true_source_sh1_m: np.ndarray
    true_source_sh2_m: np.ndarray
    true_source_sh3_m: np.ndarray | None
    true_receiver_sh1_m: np.ndarray
    true_receiver_sh2_m: np.ndarray
    true_receiver_sh3_m: np.ndarray | None
    true_source_wcor_s: np.ndarray
    true_receiver_wcor_s: np.ndarray
    true_source_total_static_s: np.ndarray
    true_receiver_total_static_s: np.ndarray

    source_endpoint_id: np.ndarray
    receiver_endpoint_id: np.ndarray
    source_endpoint_node_id: np.ndarray
    receiver_endpoint_node_id: np.ndarray
    source_endpoint_x_m: np.ndarray
    source_endpoint_y_m: np.ndarray
    receiver_endpoint_x_m: np.ndarray
    receiver_endpoint_y_m: np.ndarray
    source_endpoint_inline_m: np.ndarray | None
    receiver_endpoint_inline_m: np.ndarray | None
    source_endpoint_elevation_m: np.ndarray
    receiver_endpoint_elevation_m: np.ndarray
    true_source_endpoint_t1_s: np.ndarray
    true_source_endpoint_t2_s: np.ndarray
    true_source_endpoint_t3_s: np.ndarray | None
    true_receiver_endpoint_t1_s: np.ndarray
    true_receiver_endpoint_t2_s: np.ndarray
    true_receiver_endpoint_t3_s: np.ndarray | None
    true_source_endpoint_sh1_m: np.ndarray
    true_source_endpoint_sh2_m: np.ndarray
    true_source_endpoint_sh3_m: np.ndarray | None
    true_receiver_endpoint_sh1_m: np.ndarray
    true_receiver_endpoint_sh2_m: np.ndarray
    true_receiver_endpoint_sh3_m: np.ndarray | None
    true_source_endpoint_wcor_s: np.ndarray
    true_receiver_endpoint_wcor_s: np.ndarray
    true_source_endpoint_total_static_s: np.ndarray
    true_receiver_endpoint_total_static_s: np.ndarray

    @property
    def pick_time_s(self) -> np.ndarray:
        """Alias used by existing one-layer synthetic fixtures."""
        return self.first_break_time_s

    def as_input_model_arrays(self) -> dict[str, np.ndarray | None]:
        """Return arrays named like ``RefractionStaticInputModel`` fields."""
        return {
            'sorted_trace_index': self.sorted_trace_index,
            'pick_time_s_sorted': self.first_break_time_s,
            'valid_pick_mask_sorted': self.valid_mask,
            'valid_observation_mask_sorted': self.valid_mask,
            'source_id_sorted': self.source_id,
            'receiver_id_sorted': self.receiver_id,
            'source_x_m_sorted': self.source_x_m,
            'source_y_m_sorted': self.source_y_m,
            'receiver_x_m_sorted': self.receiver_x_m,
            'receiver_y_m_sorted': self.receiver_y_m,
            'source_elevation_m_sorted': self.source_elevation_m,
            'receiver_elevation_m_sorted': self.receiver_elevation_m,
            'source_depth_m_sorted': None,
            'geometry_distance_m_sorted': self.offset_m,
            'offset_m_sorted': self.offset_m,
            'distance_m_sorted': self.offset_m,
            'source_endpoint_key_sorted': self.source_endpoint_key,
            'receiver_endpoint_key_sorted': self.receiver_endpoint_key,
            'source_node_id_sorted': self.source_node_id,
            'receiver_node_id_sorted': self.receiver_node_id,
            'rejection_reason_sorted': self.rejection_reason,
            'layer_kind_sorted': self.layer_kind,
        }


def make_2d_straight_two_layer_refraction_dataset(
    *,
    seed: int = 0,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
) -> SyntheticMultiLayerRefractionDataset:
    """Build a straight 2D V2/T1 + V3/T2 branch fixture."""
    return _make_dataset(
        name='2d_straight_two_layer',
        layer_count=2,
        coordinate_mode='grid_3d',
        seed=seed,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
    )


def make_2d_rotated_two_layer_refraction_dataset(
    *,
    seed: int = 0,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
    line_origin_x_m: float = 1000.0,
    line_origin_y_m: float = 2000.0,
    line_azimuth_deg: float = 37.0,
) -> SyntheticMultiLayerRefractionDataset:
    """Build a rotated 2D fixture with line_2d_projected coordinates."""
    return _make_dataset(
        name='2d_rotated_two_layer',
        layer_count=2,
        coordinate_mode='line_2d_projected',
        seed=seed,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )


def make_2d_straight_three_layer_refraction_dataset(
    *,
    seed: int = 0,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
) -> SyntheticMultiLayerRefractionDataset:
    """Build a straight 2D V2/T1 + V3/T2 + Vsub/T3 branch fixture."""
    return _make_dataset(
        name='2d_straight_three_layer',
        layer_count=3,
        coordinate_mode='grid_3d',
        seed=seed,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
    )


def make_2d_rotated_three_layer_refraction_dataset(
    *,
    seed: int = 0,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
    line_origin_x_m: float = 1000.0,
    line_origin_y_m: float = 2000.0,
    line_azimuth_deg: float = 37.0,
) -> SyntheticMultiLayerRefractionDataset:
    """Build a rotated 2D V2/T1 + V3/T2 + Vsub/T3 branch fixture."""
    return _make_dataset(
        name='2d_rotated_three_layer',
        layer_count=3,
        coordinate_mode='line_2d_projected',
        seed=seed,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )


def make_3d_small_grid_two_layer_refraction_dataset(
    *,
    seed: int = 0,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
) -> SyntheticMultiLayerRefractionDataset:
    """Build a small 3D grid V2/T1 + V3/T2 branch fixture."""
    return _make_dataset(
        name='3d_small_grid_two_layer',
        layer_count=2,
        coordinate_mode='grid_3d',
        seed=seed,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
        small_grid_3d=True,
    )


def make_3d_small_grid_three_layer_refraction_dataset(
    *,
    seed: int = 0,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
) -> SyntheticMultiLayerRefractionDataset:
    """Build a small 3D grid V2/T1 + V3/T2 + Vsub/T3 branch fixture."""
    return _make_dataset(
        name='3d_small_grid_three_layer',
        layer_count=3,
        coordinate_mode='grid_3d',
        seed=seed,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
        small_grid_3d=True,
    )


def _make_dataset(
    *,
    name: str,
    layer_count: int,
    coordinate_mode: CoordinateMode,
    seed: int,
    noise_std_s: float,
    outlier_fraction: float,
    small_grid_3d: bool = False,
    line_origin_x_m: float | None = None,
    line_origin_y_m: float | None = None,
    line_azimuth_deg: float | None = None,
) -> SyntheticMultiLayerRefractionDataset:
    if layer_count not in (2, 3):
        raise ValueError('layer_count must be 2 or 3')
    enabled_layer_kinds = LAYER_KINDS[:layer_count]
    source_inline, source_crossline = _source_inline_crossline(small_grid_3d)
    if coordinate_mode == 'line_2d_projected':
        if small_grid_3d:
            raise ValueError('line_2d_projected fixtures must be 2D')
        source_x_endpoint, source_y_endpoint = _line_to_map(
            inline_m=source_inline,
            crossline_m=source_crossline,
            line_origin_x_m=_required_float(line_origin_x_m, name='line_origin_x_m'),
            line_origin_y_m=_required_float(line_origin_y_m, name='line_origin_y_m'),
            line_azimuth_deg=_required_float(line_azimuth_deg, name='line_azimuth_deg'),
        )
        source_inline_endpoint: np.ndarray | None = source_inline
        receiver_inline_endpoint: np.ndarray | None
    else:
        source_x_endpoint = source_inline
        source_y_endpoint = source_crossline
        source_inline_endpoint = None
        receiver_inline_endpoint = None
        line_origin_x_m = None
        line_origin_y_m = None
        line_azimuth_deg = None

    (
        source_endpoint_index,
        layer_kind,
        requested_offsets_m,
    ) = _observation_plan(n_sources=source_inline.shape[0], layers=enabled_layer_kinds)
    layer_index = np.asarray(
        [_LAYER_INDEX_BY_KIND[str(kind)] for kind in layer_kind.tolist()],
        dtype=np.int64,
    )

    receiver_inline, receiver_crossline = _receiver_inline_crossline(
        source_inline=source_inline[source_endpoint_index],
        source_crossline=source_crossline[source_endpoint_index],
        requested_offsets_m=requested_offsets_m,
        layer_index=layer_index,
        small_grid_3d=small_grid_3d,
    )
    if coordinate_mode == 'line_2d_projected':
        receiver_x_endpoint, receiver_y_endpoint = _line_to_map(
            inline_m=receiver_inline,
            crossline_m=receiver_crossline,
            line_origin_x_m=float(line_origin_x_m),
            line_origin_y_m=float(line_origin_y_m),
            line_azimuth_deg=float(line_azimuth_deg),
        )
        receiver_inline_endpoint = receiver_inline
    else:
        receiver_x_endpoint = receiver_inline
        receiver_y_endpoint = receiver_crossline

    receiver_endpoint_index = np.arange(receiver_inline.shape[0], dtype=np.int64)
    source_x = source_x_endpoint[source_endpoint_index]
    source_y = source_y_endpoint[source_endpoint_index]
    receiver_x = receiver_x_endpoint[receiver_endpoint_index]
    receiver_y = receiver_y_endpoint[receiver_endpoint_index]
    offset_m = np.hypot(receiver_x - source_x, receiver_y - source_y)

    source_layers = _endpoint_layers(
        n_endpoints=source_inline.shape[0],
        endpoint_kind='source',
        layer_count=layer_count,
    )
    receiver_layers = _endpoint_layers(
        n_endpoints=receiver_inline.shape[0],
        endpoint_kind='receiver',
        layer_count=layer_count,
    )
    source_terms = _endpoint_time_terms(source_layers, layer_count=layer_count)
    receiver_terms = _endpoint_time_terms(receiver_layers, layer_count=layer_count)
    replacement_velocity = (
        SYNTHETIC_MULTILAYER_VSUB_M_S
        if layer_count == 3
        else SYNTHETIC_MULTILAYER_V3_M_S
    )
    source_wcor = _layered_wcor_s(
        source_layers,
        layer_count=layer_count,
        replacement_velocity_m_s=replacement_velocity,
    )
    receiver_wcor = _layered_wcor_s(
        receiver_layers,
        layer_count=layer_count,
        replacement_velocity_m_s=replacement_velocity,
    )

    source_t1 = source_terms.t1_s[source_endpoint_index]
    source_t2 = source_terms.t2_s[source_endpoint_index]
    receiver_t1 = receiver_terms.t1_s[receiver_endpoint_index]
    receiver_t2 = receiver_terms.t2_s[receiver_endpoint_index]
    source_t3 = _optional_take(source_terms.t3_s, source_endpoint_index)
    receiver_t3 = _optional_take(receiver_terms.t3_s, receiver_endpoint_index)
    source_sh1 = source_layers.sh1_m[source_endpoint_index]
    source_sh2 = source_layers.sh2_m[source_endpoint_index]
    receiver_sh1 = receiver_layers.sh1_m[receiver_endpoint_index]
    receiver_sh2 = receiver_layers.sh2_m[receiver_endpoint_index]
    source_sh3 = _optional_take(source_layers.sh3_m, source_endpoint_index)
    receiver_sh3 = _optional_take(receiver_layers.sh3_m, receiver_endpoint_index)

    noiseless_pick = _branch_pick_times(
        layer_kind=layer_kind,
        offset_m=offset_m,
        source_t1_s=source_t1,
        source_t2_s=source_t2,
        source_t3_s=source_t3,
        receiver_t1_s=receiver_t1,
        receiver_t2_s=receiver_t2,
        receiver_t3_s=receiver_t3,
    )
    noise_s, outlier_mask = _noise_and_outliers(
        seed=seed,
        n_observations=int(noiseless_pick.shape[0]),
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
    )
    first_break_time = noiseless_pick + noise_s

    source_id_endpoint = 1000 + np.arange(source_inline.shape[0], dtype=np.int64)
    receiver_id_endpoint = 2000 + np.arange(receiver_inline.shape[0], dtype=np.int64)
    source_node_endpoint = np.arange(source_inline.shape[0], dtype=np.int64)
    receiver_node_endpoint = (
        source_inline.shape[0] + np.arange(receiver_inline.shape[0], dtype=np.int64)
    )
    source_elevation_endpoint = _endpoint_elevation_m(
        x_m=source_x_endpoint,
        y_m=source_y_endpoint,
        endpoint_kind='source',
    )
    receiver_elevation_endpoint = _endpoint_elevation_m(
        x_m=receiver_x_endpoint,
        y_m=receiver_y_endpoint,
        endpoint_kind='receiver',
    )
    layer_masks = {
        kind: np.ascontiguousarray(layer_kind == kind, dtype=bool)
        for kind in LAYER_KINDS
    }

    return SyntheticMultiLayerRefractionDataset(
        name=name,
        layer_count=layer_count,
        enabled_layer_kinds=enabled_layer_kinds,
        coordinate_mode=coordinate_mode,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
        layer_offset_range_m_by_kind=_offset_ranges(enabled_layer_kinds),
        sorted_trace_index=_i64(np.arange(first_break_time.shape[0], dtype=np.int64)),
        source_x_m=_f64(source_x),
        source_y_m=_f64(source_y),
        receiver_x_m=_f64(receiver_x),
        receiver_y_m=_f64(receiver_y),
        source_inline_m=_optional_take(source_inline_endpoint, source_endpoint_index),
        receiver_inline_m=_optional_take(
            receiver_inline_endpoint,
            receiver_endpoint_index,
        ),
        source_elevation_m=_f64(source_elevation_endpoint[source_endpoint_index]),
        receiver_elevation_m=_f64(receiver_elevation_endpoint[receiver_endpoint_index]),
        source_id=_i64(source_id_endpoint[source_endpoint_index]),
        receiver_id=_i64(receiver_id_endpoint[receiver_endpoint_index]),
        source_node_id=_i64(source_node_endpoint[source_endpoint_index]),
        receiver_node_id=_i64(receiver_node_endpoint[receiver_endpoint_index]),
        source_endpoint_key=_i64(source_id_endpoint[source_endpoint_index]),
        receiver_endpoint_key=_i64(receiver_id_endpoint[receiver_endpoint_index]),
        source_endpoint_index=_i64(source_endpoint_index),
        receiver_endpoint_index=_i64(receiver_endpoint_index),
        offset_m=_f64(offset_m),
        first_break_time_s=_f64(first_break_time),
        noiseless_first_break_time_s=_f64(noiseless_pick),
        valid_mask=np.ones(first_break_time.shape, dtype=bool),
        rejection_reason=np.full(first_break_time.shape, 'ok', dtype='<U16'),
        layer_kind=np.ascontiguousarray(layer_kind, dtype='<U16'),
        layer_index=_i64(layer_index),
        expected_layer_mask_by_kind=layer_masks,
        true_noise_s=_f64(noise_s),
        outlier_mask=np.ascontiguousarray(outlier_mask, dtype=bool),
        true_v1_m_s=SYNTHETIC_MULTILAYER_V1_M_S,
        true_v2_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
        true_v3_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
        true_vsub_m_s=SYNTHETIC_MULTILAYER_VSUB_M_S,
        true_replacement_velocity_m_s=replacement_velocity,
        true_source_t1_s=_f64(source_t1),
        true_source_t2_s=_f64(source_t2),
        true_source_t3_s=_optional_f64(source_t3),
        true_receiver_t1_s=_f64(receiver_t1),
        true_receiver_t2_s=_f64(receiver_t2),
        true_receiver_t3_s=_optional_f64(receiver_t3),
        true_source_sh1_m=_f64(source_sh1),
        true_source_sh2_m=_f64(source_sh2),
        true_source_sh3_m=_optional_f64(source_sh3),
        true_receiver_sh1_m=_f64(receiver_sh1),
        true_receiver_sh2_m=_f64(receiver_sh2),
        true_receiver_sh3_m=_optional_f64(receiver_sh3),
        true_source_wcor_s=_f64(source_wcor[source_endpoint_index]),
        true_receiver_wcor_s=_f64(receiver_wcor[receiver_endpoint_index]),
        true_source_total_static_s=_f64(source_wcor[source_endpoint_index]),
        true_receiver_total_static_s=_f64(receiver_wcor[receiver_endpoint_index]),
        source_endpoint_id=_i64(source_id_endpoint),
        receiver_endpoint_id=_i64(receiver_id_endpoint),
        source_endpoint_node_id=_i64(source_node_endpoint),
        receiver_endpoint_node_id=_i64(receiver_node_endpoint),
        source_endpoint_x_m=_f64(source_x_endpoint),
        source_endpoint_y_m=_f64(source_y_endpoint),
        receiver_endpoint_x_m=_f64(receiver_x_endpoint),
        receiver_endpoint_y_m=_f64(receiver_y_endpoint),
        source_endpoint_inline_m=_optional_f64(source_inline_endpoint),
        receiver_endpoint_inline_m=_optional_f64(receiver_inline_endpoint),
        source_endpoint_elevation_m=_f64(source_elevation_endpoint),
        receiver_endpoint_elevation_m=_f64(receiver_elevation_endpoint),
        true_source_endpoint_t1_s=_f64(source_terms.t1_s),
        true_source_endpoint_t2_s=_f64(source_terms.t2_s),
        true_source_endpoint_t3_s=_optional_f64(source_terms.t3_s),
        true_receiver_endpoint_t1_s=_f64(receiver_terms.t1_s),
        true_receiver_endpoint_t2_s=_f64(receiver_terms.t2_s),
        true_receiver_endpoint_t3_s=_optional_f64(receiver_terms.t3_s),
        true_source_endpoint_sh1_m=_f64(source_layers.sh1_m),
        true_source_endpoint_sh2_m=_f64(source_layers.sh2_m),
        true_source_endpoint_sh3_m=_optional_f64(source_layers.sh3_m),
        true_receiver_endpoint_sh1_m=_f64(receiver_layers.sh1_m),
        true_receiver_endpoint_sh2_m=_f64(receiver_layers.sh2_m),
        true_receiver_endpoint_sh3_m=_optional_f64(receiver_layers.sh3_m),
        true_source_endpoint_wcor_s=_f64(source_wcor),
        true_receiver_endpoint_wcor_s=_f64(receiver_wcor),
        true_source_endpoint_total_static_s=_f64(source_wcor),
        true_receiver_endpoint_total_static_s=_f64(receiver_wcor),
    )


@dataclass(frozen=True)
class _EndpointLayers:
    sh1_m: np.ndarray
    sh2_m: np.ndarray
    sh3_m: np.ndarray | None


@dataclass(frozen=True)
class _EndpointTimeTerms:
    t1_s: np.ndarray
    t2_s: np.ndarray
    t3_s: np.ndarray | None


def _source_inline_crossline(small_grid_3d: bool) -> tuple[np.ndarray, np.ndarray]:
    if small_grid_3d:
        ix = np.tile(np.arange(3, dtype=np.float64), 2)
        iy = np.repeat(np.arange(2, dtype=np.float64), 3)
        return _f64(40.0 + ix * 180.0), _f64(30.0 + iy * 220.0)
    return _f64(40.0 + np.arange(6, dtype=np.float64) * 140.0), _f64(
        np.zeros(6, dtype=np.float64)
    )


def _observation_plan(
    *,
    n_sources: int,
    layers: tuple[LayerKind, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_index: list[int] = []
    layer_kind: list[LayerKind] = []
    offset_m: list[float] = []
    for source in range(n_sources):
        for layer in layers:
            for offset in _offsets_for_layer(layer):
                source_index.append(source)
                layer_kind.append(layer)
                offset_m.append(float(offset))
    return (
        np.asarray(source_index, dtype=np.int64),
        np.asarray(layer_kind, dtype='<U16'),
        np.asarray(offset_m, dtype=np.float64),
    )


def _receiver_inline_crossline(
    *,
    source_inline: np.ndarray,
    source_crossline: np.ndarray,
    requested_offsets_m: np.ndarray,
    layer_index: np.ndarray,
    small_grid_3d: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not small_grid_3d:
        return source_inline + requested_offsets_m, source_crossline.copy()
    angle_deg = 8.0 + 5.0 * layer_index.astype(np.float64)
    angle_rad = np.deg2rad(angle_deg)
    return (
        source_inline + requested_offsets_m * np.cos(angle_rad),
        source_crossline + requested_offsets_m * np.sin(angle_rad),
    )


def _endpoint_layers(
    *,
    n_endpoints: int,
    endpoint_kind: Literal['source', 'receiver'],
    layer_count: int,
) -> _EndpointLayers:
    index = np.arange(n_endpoints, dtype=np.float64)
    if endpoint_kind == 'source':
        sh1 = 8.0 + 0.9 * (index % 5.0) + 0.04 * index
        sh2 = 14.0 + 1.1 * (index % 4.0) + 0.05 * index
        sh3 = 19.0 + 1.3 * (index % 3.0) + 0.03 * index
    else:
        sh1 = 9.0 + 0.8 * (index % 6.0) + 0.03 * index
        sh2 = 15.0 + 1.0 * (index % 5.0) + 0.04 * index
        sh3 = 21.0 + 1.2 * (index % 4.0) + 0.02 * index
    return _EndpointLayers(
        sh1_m=_f64(sh1),
        sh2_m=_f64(sh2),
        sh3_m=_f64(sh3) if layer_count == 3 else None,
    )


def _endpoint_time_terms(
    layers: _EndpointLayers,
    *,
    layer_count: int,
) -> _EndpointTimeTerms:
    t1 = _forward_time_term_s(
        thicknesses_m=(layers.sh1_m,),
        velocities_m_s=(SYNTHETIC_MULTILAYER_V1_M_S,),
        refractor_velocity_m_s=SYNTHETIC_MULTILAYER_V2_M_S,
    )
    t2 = _forward_time_term_s(
        thicknesses_m=(layers.sh1_m, layers.sh2_m),
        velocities_m_s=(SYNTHETIC_MULTILAYER_V1_M_S, SYNTHETIC_MULTILAYER_V2_M_S),
        refractor_velocity_m_s=SYNTHETIC_MULTILAYER_V3_M_S,
    )
    if layer_count == 2:
        return _EndpointTimeTerms(t1_s=t1, t2_s=t2, t3_s=None)
    if layers.sh3_m is None:
        raise ValueError('SH3 is required for three-layer fixtures')
    t3 = _forward_time_term_s(
        thicknesses_m=(layers.sh1_m, layers.sh2_m, layers.sh3_m),
        velocities_m_s=(
            SYNTHETIC_MULTILAYER_V1_M_S,
            SYNTHETIC_MULTILAYER_V2_M_S,
            SYNTHETIC_MULTILAYER_V3_M_S,
        ),
        refractor_velocity_m_s=SYNTHETIC_MULTILAYER_VSUB_M_S,
    )
    return _EndpointTimeTerms(t1_s=t1, t2_s=t2, t3_s=t3)


def _forward_time_term_s(
    *,
    thicknesses_m: tuple[np.ndarray, ...],
    velocities_m_s: tuple[float, ...],
    refractor_velocity_m_s: float,
) -> np.ndarray:
    if len(thicknesses_m) != len(velocities_m_s):
        raise ValueError('thickness and velocity layer counts must match')
    result = np.zeros(thicknesses_m[0].shape, dtype=np.float64)
    for thickness, velocity in zip(thicknesses_m, velocities_m_s, strict=True):
        result += (
            thickness
            * np.sqrt(refractor_velocity_m_s**2 - float(velocity) ** 2)
            / (float(velocity) * refractor_velocity_m_s)
        )
    return _f64(result)


def _layered_wcor_s(
    layers: _EndpointLayers,
    *,
    layer_count: int,
    replacement_velocity_m_s: float,
) -> np.ndarray:
    result = layers.sh1_m * (
        (1.0 / replacement_velocity_m_s) - (1.0 / SYNTHETIC_MULTILAYER_V1_M_S)
    )
    result += layers.sh2_m * (
        (1.0 / replacement_velocity_m_s) - (1.0 / SYNTHETIC_MULTILAYER_V2_M_S)
    )
    if layer_count == 3:
        if layers.sh3_m is None:
            raise ValueError('SH3 is required for three-layer WCOR')
        result += layers.sh3_m * (
            (1.0 / replacement_velocity_m_s) - (1.0 / SYNTHETIC_MULTILAYER_V3_M_S)
        )
    return _f64(result)


def _branch_pick_times(
    *,
    layer_kind: np.ndarray,
    offset_m: np.ndarray,
    source_t1_s: np.ndarray,
    source_t2_s: np.ndarray,
    source_t3_s: np.ndarray | None,
    receiver_t1_s: np.ndarray,
    receiver_t2_s: np.ndarray,
    receiver_t3_s: np.ndarray | None,
) -> np.ndarray:
    pick = np.empty(offset_m.shape, dtype=np.float64)
    v2_mask = layer_kind == V2_T1
    pick[v2_mask] = (
        source_t1_s[v2_mask]
        + receiver_t1_s[v2_mask]
        + offset_m[v2_mask] / SYNTHETIC_MULTILAYER_V2_M_S
    )
    v3_mask = layer_kind == V3_T2
    pick[v3_mask] = (
        source_t2_s[v3_mask]
        + receiver_t2_s[v3_mask]
        + offset_m[v3_mask] / SYNTHETIC_MULTILAYER_V3_M_S
    )
    vsub_mask = layer_kind == VSUB_T3
    if bool(np.any(vsub_mask)):
        if source_t3_s is None or receiver_t3_s is None:
            raise ValueError('T3 is required for Vsub branch picks')
        pick[vsub_mask] = (
            source_t3_s[vsub_mask]
            + receiver_t3_s[vsub_mask]
            + offset_m[vsub_mask] / SYNTHETIC_MULTILAYER_VSUB_M_S
        )
    return _f64(pick)


def _noise_and_outliers(
    *,
    seed: int,
    n_observations: int,
    noise_std_s: float,
    outlier_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if noise_std_s < 0.0:
        raise ValueError('noise_std_s must be non-negative')
    if outlier_fraction < 0.0 or outlier_fraction > 1.0:
        raise ValueError('outlier_fraction must be in [0, 1]')
    rng = np.random.default_rng(seed)
    noise = np.zeros(n_observations, dtype=np.float64)
    if noise_std_s > 0.0:
        noise = rng.normal(0.0, noise_std_s, size=n_observations)

    outlier_mask = np.zeros(n_observations, dtype=bool)
    if outlier_fraction > 0.0 and n_observations > 0:
        n_outliers = max(1, int(round(float(n_observations) * outlier_fraction)))
        outlier_index = rng.choice(n_observations, size=n_outliers, replace=False)
        outlier_mask[outlier_index] = True
        sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=n_outliers)
        outlier_scale_s = max(0.050, noise_std_s * 20.0)
        noise[outlier_index] += sign * outlier_scale_s
    return _f64(noise), outlier_mask


def _offsets_for_layer(layer: LayerKind) -> tuple[float, ...]:
    if layer == V2_T1:
        return _V2_OFFSETS_M
    if layer == V3_T2:
        return _V3_OFFSETS_M
    if layer == VSUB_T3:
        return _VSUB_OFFSETS_M
    raise ValueError(f'unknown layer kind: {layer}')


def _offset_ranges(
    layers: tuple[LayerKind, ...],
) -> dict[str, tuple[float, float | None]]:
    ranges: dict[str, tuple[float, float | None]] = {}
    for layer in layers:
        offsets = _offsets_for_layer(layer)
        ranges[layer] = (float(min(offsets)), float(max(offsets)))
    if VSUB_T3 in layers:
        ranges[VSUB_T3] = (float(min(_VSUB_OFFSETS_M)), None)
    return ranges


def _endpoint_elevation_m(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    endpoint_kind: Literal['source', 'receiver'],
) -> np.ndarray:
    base = 90.0 if endpoint_kind == 'source' else 92.0
    return _f64(base + 0.002 * np.asarray(x_m) + 0.001 * np.asarray(y_m))


def _line_to_map(
    *,
    inline_m: np.ndarray,
    crossline_m: np.ndarray,
    line_origin_x_m: float,
    line_origin_y_m: float,
    line_azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth_rad = np.deg2rad(float(line_azimuth_deg))
    inline_unit_x = float(np.sin(azimuth_rad))
    inline_unit_y = float(np.cos(azimuth_rad))
    dx = inline_m * inline_unit_x + crossline_m * inline_unit_y
    dy = inline_m * inline_unit_y - crossline_m * inline_unit_x
    return _f64(float(line_origin_x_m) + dx), _f64(float(line_origin_y_m) + dy)


def _required_float(value: float | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f'{name} is required')
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _optional_take(values: np.ndarray | None, index: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    return _f64(values[index])


def _optional_f64(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return _f64(values)


def _f64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _i64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


__all__ = [
    'CoordinateMode',
    'LAYER_KINDS',
    'LayerKind',
    'SYNTHETIC_MULTILAYER_V1_M_S',
    'SYNTHETIC_MULTILAYER_V2_M_S',
    'SYNTHETIC_MULTILAYER_V3_M_S',
    'SYNTHETIC_MULTILAYER_VSUB_M_S',
    'SyntheticMultiLayerRefractionDataset',
    'V2_T1',
    'V3_T2',
    'VSUB_T3',
    'make_2d_rotated_two_layer_refraction_dataset',
    'make_2d_rotated_three_layer_refraction_dataset',
    'make_2d_straight_three_layer_refraction_dataset',
    'make_2d_straight_two_layer_refraction_dataset',
    'make_3d_small_grid_three_layer_refraction_dataset',
    'make_3d_small_grid_two_layer_refraction_dataset',
]
