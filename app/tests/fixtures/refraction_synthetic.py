"""Dependency-light synthetic refraction-static fixture builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

CoordinateMode = Literal['grid_3d', 'line_2d_projected']


@dataclass(frozen=True)
class SyntheticRefractionCellDataset:
    coordinate_mode: CoordinateMode
    cell_size_x_m: float
    cell_size_y_m: float | None
    x_coordinate_origin_m: float
    y_coordinate_origin_m: float
    line_origin_x_m: float | None
    line_origin_y_m: float | None
    line_azimuth_deg: float | None
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    source_inline_m: np.ndarray | None
    receiver_inline_m: np.ndarray | None
    source_id: np.ndarray
    receiver_id: np.ndarray
    source_node_id: np.ndarray
    receiver_node_id: np.ndarray
    source_endpoint_index: np.ndarray
    receiver_endpoint_index: np.ndarray
    offset_m: np.ndarray
    pick_time_s: np.ndarray
    valid_mask: np.ndarray
    true_v1_m_s: float
    true_cell_v2_m_s: np.ndarray
    true_source_t1_s: np.ndarray
    true_receiver_t1_s: np.ndarray
    true_source_sh1_m: np.ndarray
    true_receiver_sh1_m: np.ndarray
    true_source_wcor_s: np.ndarray
    true_receiver_wcor_s: np.ndarray
    true_source_static_s: np.ndarray
    true_receiver_static_s: np.ndarray
    true_source_v2_m_s: np.ndarray
    true_receiver_v2_m_s: np.ndarray
    true_midpoint_v2_m_s: np.ndarray
    true_cell_ix_for_pick: np.ndarray
    true_cell_iy_for_pick: np.ndarray
    true_cell_id_for_pick: np.ndarray
    true_noise_s: np.ndarray
    outlier_mask: np.ndarray
    cell_observation_count: np.ndarray
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
    true_source_endpoint_t1_s: np.ndarray
    true_receiver_endpoint_t1_s: np.ndarray
    true_source_endpoint_sh1_m: np.ndarray
    true_receiver_endpoint_sh1_m: np.ndarray
    true_source_endpoint_wcor_s: np.ndarray
    true_receiver_endpoint_wcor_s: np.ndarray
    true_source_endpoint_static_s: np.ndarray
    true_receiver_endpoint_static_s: np.ndarray
    true_source_endpoint_v2_m_s: np.ndarray
    true_receiver_endpoint_v2_m_s: np.ndarray


def make_clean_2d_cell_refraction_dataset(
    *,
    seed: int = 0,
    v1_m_s: float = 800.0,
    cell_v2_m_s: Sequence[float] | np.ndarray | None = None,
    n_sources: int = 9,
    n_receivers: int = 9,
    cell_size_x_m: float = 100.0,
    cell_size_y_m: float | None = None,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
) -> SyntheticRefractionCellDataset:
    """Build a small 2D midpoint-cell fixture using pick=Tsrc+Trec+offset/V2."""
    v2 = _cell_v2_2d(cell_v2_m_s, default=(2200.0, 2600.0, 3000.0))
    return _make_dataset(
        coordinate_mode='grid_3d',
        seed=seed,
        v1_m_s=v1_m_s,
        cell_v2_m_s=v2,
        n_sources=n_sources,
        n_receivers=n_receivers,
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=cell_size_y_m,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
    )


def make_rotated_2d_line_refraction_dataset(
    *,
    seed: int = 0,
    v1_m_s: float = 800.0,
    cell_v2_m_s: Sequence[float] | np.ndarray | None = None,
    n_sources: int = 9,
    n_receivers: int = 9,
    cell_size_x_m: float = 100.0,
    cell_size_y_m: float | None = None,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
    line_origin_x_m: float = 1000.0,
    line_origin_y_m: float = 2000.0,
    line_azimuth_deg: float = 37.0,
) -> SyntheticRefractionCellDataset:
    """Build a rotated line fixture with explicit line_2d_projected truth."""
    if cell_size_y_m is not None:
        raise ValueError('line_2d_projected synthetic fixtures require cell_size_y_m=None')
    v2 = _cell_v2_2d(cell_v2_m_s, default=(2200.0, 2600.0, 3000.0))
    return _make_dataset(
        coordinate_mode='line_2d_projected',
        seed=seed,
        v1_m_s=v1_m_s,
        cell_v2_m_s=v2,
        n_sources=n_sources,
        n_receivers=n_receivers,
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=None,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )


def make_clean_3d_cell_refraction_dataset(
    *,
    seed: int = 0,
    v1_m_s: float = 800.0,
    cell_v2_m_s: Sequence[Sequence[float]] | np.ndarray | None = None,
    n_sources: int = 12,
    n_receivers: int = 12,
    cell_size_x_m: float = 100.0,
    cell_size_y_m: float | None = 120.0,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
) -> SyntheticRefractionCellDataset:
    """Build a 3D grid fixture with known per-cell V2 and endpoint T1 truth."""
    v2 = _cell_v2_3d(
        cell_v2_m_s,
        default=((2200.0, 2600.0, 3000.0), (2400.0, 2800.0, 3200.0)),
    )
    return _make_dataset(
        coordinate_mode='grid_3d',
        seed=seed,
        v1_m_s=v1_m_s,
        cell_v2_m_s=v2,
        n_sources=n_sources,
        n_receivers=n_receivers,
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=cell_size_y_m,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
    )


def make_low_fold_empty_cell_refraction_dataset(
    *,
    seed: int = 0,
    v1_m_s: float = 800.0,
    cell_v2_m_s: Sequence[float] | np.ndarray | None = None,
    n_sources: int = 12,
    n_receivers: int = 12,
    cell_size_x_m: float = 100.0,
    cell_size_y_m: float | None = None,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
    min_observations_per_cell: int = 3,
) -> SyntheticRefractionCellDataset:
    """Build a 2D fixture with one empty cell and one below-fold cell."""
    v2 = _cell_v2_2d(cell_v2_m_s, default=(2200.0, 2600.0, 3000.0, 3400.0))
    desired_counts = {
        0: min_observations_per_cell + 2,
        1: max(1, min_observations_per_cell - 1),
        2: 0,
        3: min_observations_per_cell + 2,
    }
    return _make_dataset(
        coordinate_mode='grid_3d',
        seed=seed,
        v1_m_s=v1_m_s,
        cell_v2_m_s=v2,
        n_sources=n_sources,
        n_receivers=n_receivers,
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=cell_size_y_m,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
        desired_observations_by_cell=desired_counts,
    )


def make_outlier_refraction_dataset(
    *,
    seed: int = 0,
    v1_m_s: float = 800.0,
    cell_v2_m_s: Sequence[float] | np.ndarray | None = None,
    n_sources: int = 9,
    n_receivers: int = 9,
    cell_size_x_m: float = 100.0,
    cell_size_y_m: float | None = None,
    noise_std_s: float = 0.0005,
    outlier_fraction: float = 0.1,
) -> SyntheticRefractionCellDataset:
    """Build a 2D fixture with deterministic large pick-time outliers."""
    v2 = _cell_v2_2d(cell_v2_m_s, default=(2200.0, 2600.0, 3000.0))
    return _make_dataset(
        coordinate_mode='grid_3d',
        seed=seed,
        v1_m_s=v1_m_s,
        cell_v2_m_s=v2,
        n_sources=n_sources,
        n_receivers=n_receivers,
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=cell_size_y_m,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
    )


def make_v2_spike_smoothing_dataset(
    *,
    seed: int = 0,
    v1_m_s: float = 800.0,
    cell_v2_m_s: Sequence[float] | np.ndarray | None = None,
    n_sources: int = 13,
    n_receivers: int = 13,
    cell_size_x_m: float = 100.0,
    cell_size_y_m: float | None = None,
    noise_std_s: float = 0.0,
    outlier_fraction: float = 0.0,
) -> SyntheticRefractionCellDataset:
    """Build a 2D fixture with a central V2 spike for smoothing tests."""
    v2 = _cell_v2_2d(
        cell_v2_m_s,
        default=(2600.0, 2600.0, 3600.0, 2600.0, 2600.0),
    )
    return _make_dataset(
        coordinate_mode='grid_3d',
        seed=seed,
        v1_m_s=v1_m_s,
        cell_v2_m_s=v2,
        n_sources=n_sources,
        n_receivers=n_receivers,
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=cell_size_y_m,
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
    )


def _make_dataset(
    *,
    coordinate_mode: CoordinateMode,
    seed: int,
    v1_m_s: float,
    cell_v2_m_s: np.ndarray,
    n_sources: int,
    n_receivers: int,
    cell_size_x_m: float,
    cell_size_y_m: float | None,
    noise_std_s: float,
    outlier_fraction: float,
    line_origin_x_m: float | None = None,
    line_origin_y_m: float | None = None,
    line_azimuth_deg: float | None = None,
    desired_observations_by_cell: dict[int, int] | None = None,
) -> SyntheticRefractionCellDataset:
    rng = np.random.default_rng(seed)
    v2 = _validated_v2(cell_v2_m_s, v1_m_s=v1_m_s)
    n_cell_y, n_cell_x = v2.shape
    _validate_positive_int(n_sources, name='n_sources')
    _validate_positive_int(n_receivers, name='n_receivers')
    cell_size_x = _positive_float(cell_size_x_m, name='cell_size_x_m')
    cell_size_y = _effective_cell_size_y(
        cell_size_y_m=cell_size_y_m,
        number_of_cell_y=n_cell_y,
    )
    if coordinate_mode == 'line_2d_projected' and n_cell_y != 1:
        raise ValueError('line_2d_projected synthetic fixtures require one cell row')

    source_cell_x, source_cell_y = _endpoint_cell_coordinates(
        n_endpoints=n_sources,
        number_of_cell_x=n_cell_x,
        number_of_cell_y=n_cell_y,
        cell_size_x_m=cell_size_x,
        cell_size_y_m=cell_size_y,
        rng=rng,
        x_fraction=0.25,
        y_fraction=0.35,
    )
    receiver_cell_x, receiver_cell_y = _endpoint_cell_coordinates(
        n_endpoints=n_receivers,
        number_of_cell_x=n_cell_x,
        number_of_cell_y=n_cell_y,
        cell_size_x_m=cell_size_x,
        cell_size_y_m=cell_size_y,
        rng=rng,
        x_fraction=0.75,
        y_fraction=0.65,
    )

    if coordinate_mode == 'line_2d_projected':
        source_x, source_y = _line_to_map(
            inline_m=source_cell_x,
            crossline_m=np.zeros(source_cell_x.shape, dtype=np.float64),
            line_origin_x_m=_required_float(
                line_origin_x_m,
                name='line_origin_x_m',
            ),
            line_origin_y_m=_required_float(
                line_origin_y_m,
                name='line_origin_y_m',
            ),
            line_azimuth_deg=_required_float(
                line_azimuth_deg,
                name='line_azimuth_deg',
            ),
        )
        receiver_x, receiver_y = _line_to_map(
            inline_m=receiver_cell_x,
            crossline_m=np.zeros(receiver_cell_x.shape, dtype=np.float64),
            line_origin_x_m=float(line_origin_x_m),
            line_origin_y_m=float(line_origin_y_m),
            line_azimuth_deg=float(line_azimuth_deg),
        )
        source_inline_endpoint = source_cell_x
        receiver_inline_endpoint = receiver_cell_x
    else:
        source_x = source_cell_x
        source_y = source_cell_y
        receiver_x = receiver_cell_x
        receiver_y = receiver_cell_y
        source_inline_endpoint = None
        receiver_inline_endpoint = None
        line_origin_x_m = None
        line_origin_y_m = None
        line_azimuth_deg = None

    source_ix, source_iy = _cell_indices_for_points(
        x_m=source_cell_x,
        y_m=source_cell_y,
        number_of_cell_x=n_cell_x,
        number_of_cell_y=n_cell_y,
        cell_size_x_m=cell_size_x,
        cell_size_y_m=cell_size_y,
    )
    receiver_ix, receiver_iy = _cell_indices_for_points(
        x_m=receiver_cell_x,
        y_m=receiver_cell_y,
        number_of_cell_x=n_cell_x,
        number_of_cell_y=n_cell_y,
        cell_size_x_m=cell_size_x,
        cell_size_y_m=cell_size_y,
    )

    source_v2 = v2[source_iy, source_ix]
    receiver_v2 = v2[receiver_iy, receiver_ix]
    source_sh1 = _endpoint_sh1_m(rng, n_sources)
    receiver_sh1 = _endpoint_sh1_m(rng, n_receivers)
    source_t1 = _t1_from_sh1(source_sh1, v1_m_s=v1_m_s, v2_m_s=source_v2)
    receiver_t1 = _t1_from_sh1(receiver_sh1, v1_m_s=v1_m_s, v2_m_s=receiver_v2)
    source_wcor = _wcor_from_sh1(source_sh1, v1_m_s=v1_m_s, v2_m_s=source_v2)
    receiver_wcor = _wcor_from_sh1(receiver_sh1, v1_m_s=v1_m_s, v2_m_s=receiver_v2)

    source_index, receiver_index = _observation_endpoint_indices(
        n_sources=n_sources,
        n_receivers=n_receivers,
    )
    row_source_cell_x = source_cell_x[source_index]
    row_receiver_cell_x = receiver_cell_x[receiver_index]
    row_source_cell_y = source_cell_y[source_index]
    row_receiver_cell_y = receiver_cell_y[receiver_index]
    midpoint_x = 0.5 * (row_source_cell_x + row_receiver_cell_x)
    midpoint_y = 0.5 * (row_source_cell_y + row_receiver_cell_y)
    midpoint_ix, midpoint_iy = _cell_indices_for_points(
        x_m=midpoint_x,
        y_m=midpoint_y,
        number_of_cell_x=n_cell_x,
        number_of_cell_y=n_cell_y,
        cell_size_x_m=cell_size_x,
        cell_size_y_m=cell_size_y,
    )
    midpoint_cell_id = midpoint_iy * n_cell_x + midpoint_ix
    selected = _select_observation_rows(
        midpoint_cell_id=midpoint_cell_id,
        desired_observations_by_cell=desired_observations_by_cell,
        n_cells=int(v2.size),
    )
    source_index = source_index[selected]
    receiver_index = receiver_index[selected]
    midpoint_ix = midpoint_ix[selected]
    midpoint_iy = midpoint_iy[selected]
    midpoint_cell_id = midpoint_cell_id[selected]

    obs_source_x = source_x[source_index]
    obs_source_y = source_y[source_index]
    obs_receiver_x = receiver_x[receiver_index]
    obs_receiver_y = receiver_y[receiver_index]
    offset_m = np.hypot(obs_receiver_x - obs_source_x, obs_receiver_y - obs_source_y)
    midpoint_v2 = v2[midpoint_iy, midpoint_ix]
    noiseless_pick = (
        source_t1[source_index]
        + receiver_t1[receiver_index]
        + offset_m / midpoint_v2
    )
    noise_s, outlier_mask = _noise_and_outliers(
        rng=rng,
        n_observations=int(noiseless_pick.shape[0]),
        noise_std_s=noise_std_s,
        outlier_fraction=outlier_fraction,
    )
    pick_time_s = noiseless_pick + noise_s

    source_id_endpoint = 1000 + np.arange(n_sources, dtype=np.int64)
    receiver_id_endpoint = 2000 + np.arange(n_receivers, dtype=np.int64)
    source_node_endpoint = np.arange(n_sources, dtype=np.int64)
    receiver_node_endpoint = n_sources + np.arange(n_receivers, dtype=np.int64)
    cell_observation_count = np.bincount(midpoint_cell_id, minlength=int(v2.size))

    return SyntheticRefractionCellDataset(
        coordinate_mode=coordinate_mode,
        cell_size_x_m=cell_size_x,
        cell_size_y_m=cell_size_y_m,
        x_coordinate_origin_m=0.0,
        y_coordinate_origin_m=0.0,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
        source_x_m=_f64(obs_source_x),
        source_y_m=_f64(obs_source_y),
        receiver_x_m=_f64(obs_receiver_x),
        receiver_y_m=_f64(obs_receiver_y),
        source_inline_m=_optional_take(source_inline_endpoint, source_index),
        receiver_inline_m=_optional_take(receiver_inline_endpoint, receiver_index),
        source_id=_i64(source_id_endpoint[source_index]),
        receiver_id=_i64(receiver_id_endpoint[receiver_index]),
        source_node_id=_i64(source_node_endpoint[source_index]),
        receiver_node_id=_i64(receiver_node_endpoint[receiver_index]),
        source_endpoint_index=_i64(source_index),
        receiver_endpoint_index=_i64(receiver_index),
        offset_m=_f64(offset_m),
        pick_time_s=_f64(pick_time_s),
        valid_mask=np.ones(pick_time_s.shape, dtype=bool),
        true_v1_m_s=float(v1_m_s),
        true_cell_v2_m_s=_f64(v2),
        true_source_t1_s=_f64(source_t1[source_index]),
        true_receiver_t1_s=_f64(receiver_t1[receiver_index]),
        true_source_sh1_m=_f64(source_sh1[source_index]),
        true_receiver_sh1_m=_f64(receiver_sh1[receiver_index]),
        true_source_wcor_s=_f64(source_wcor[source_index]),
        true_receiver_wcor_s=_f64(receiver_wcor[receiver_index]),
        true_source_static_s=_f64(source_wcor[source_index]),
        true_receiver_static_s=_f64(receiver_wcor[receiver_index]),
        true_source_v2_m_s=_f64(source_v2[source_index]),
        true_receiver_v2_m_s=_f64(receiver_v2[receiver_index]),
        true_midpoint_v2_m_s=_f64(midpoint_v2),
        true_cell_ix_for_pick=_i64(midpoint_ix),
        true_cell_iy_for_pick=_i64(midpoint_iy),
        true_cell_id_for_pick=_i64(midpoint_cell_id),
        true_noise_s=_f64(noise_s),
        outlier_mask=np.ascontiguousarray(outlier_mask, dtype=bool),
        cell_observation_count=_i64(cell_observation_count),
        source_endpoint_id=_i64(source_id_endpoint),
        receiver_endpoint_id=_i64(receiver_id_endpoint),
        source_endpoint_node_id=_i64(source_node_endpoint),
        receiver_endpoint_node_id=_i64(receiver_node_endpoint),
        source_endpoint_x_m=_f64(source_x),
        source_endpoint_y_m=_f64(source_y),
        receiver_endpoint_x_m=_f64(receiver_x),
        receiver_endpoint_y_m=_f64(receiver_y),
        source_endpoint_inline_m=_optional_f64(source_inline_endpoint),
        receiver_endpoint_inline_m=_optional_f64(receiver_inline_endpoint),
        true_source_endpoint_t1_s=_f64(source_t1),
        true_receiver_endpoint_t1_s=_f64(receiver_t1),
        true_source_endpoint_sh1_m=_f64(source_sh1),
        true_receiver_endpoint_sh1_m=_f64(receiver_sh1),
        true_source_endpoint_wcor_s=_f64(source_wcor),
        true_receiver_endpoint_wcor_s=_f64(receiver_wcor),
        true_source_endpoint_static_s=_f64(source_wcor),
        true_receiver_endpoint_static_s=_f64(receiver_wcor),
        true_source_endpoint_v2_m_s=_f64(source_v2),
        true_receiver_endpoint_v2_m_s=_f64(receiver_v2),
    )


def _cell_v2_2d(
    values: Sequence[float] | np.ndarray | None,
    *,
    default: Sequence[float],
) -> np.ndarray:
    raw = np.asarray(default if values is None else values, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError('2D synthetic cell_v2_m_s must be one-dimensional')
    return raw.reshape(1, raw.shape[0])


def _cell_v2_3d(
    values: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    default: Sequence[Sequence[float]],
) -> np.ndarray:
    raw = np.asarray(default if values is None else values, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError('3D synthetic cell_v2_m_s must be two-dimensional')
    return raw


def _validated_v2(cell_v2_m_s: np.ndarray, *, v1_m_s: float) -> np.ndarray:
    v1 = _positive_float(v1_m_s, name='v1_m_s')
    v2 = np.ascontiguousarray(cell_v2_m_s, dtype=np.float64)
    if v2.ndim != 2 or v2.size == 0:
        raise ValueError('cell_v2_m_s must be a non-empty 2D array')
    if not np.all(np.isfinite(v2)) or np.any(v2 <= v1):
        raise ValueError('all synthetic V2 cells must be finite and greater than V1')
    return v2


def _effective_cell_size_y(
    *,
    cell_size_y_m: float | None,
    number_of_cell_y: int,
) -> float | None:
    if number_of_cell_y == 1 and cell_size_y_m is None:
        return None
    if cell_size_y_m is None:
        raise ValueError('cell_size_y_m is required when more than one cell row exists')
    return _positive_float(cell_size_y_m, name='cell_size_y_m')


def _endpoint_cell_coordinates(
    *,
    n_endpoints: int,
    number_of_cell_x: int,
    number_of_cell_y: int,
    cell_size_x_m: float,
    cell_size_y_m: float | None,
    rng: np.random.Generator,
    x_fraction: float,
    y_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    total_cells = number_of_cell_x * number_of_cell_y
    cell_id = np.arange(n_endpoints, dtype=np.int64) % total_cells
    ix = cell_id % number_of_cell_x
    iy = cell_id // number_of_cell_x
    x_jitter = rng.uniform(-0.06, 0.06, size=n_endpoints) * cell_size_x_m
    x_m = (ix.astype(np.float64) + x_fraction) * cell_size_x_m + x_jitter
    if cell_size_y_m is None:
        y_m = np.zeros(n_endpoints, dtype=np.float64)
    else:
        y_jitter = rng.uniform(-0.06, 0.06, size=n_endpoints) * cell_size_y_m
        y_m = (iy.astype(np.float64) + y_fraction) * cell_size_y_m + y_jitter
    return _f64(x_m), _f64(y_m)


def _observation_endpoint_indices(
    *,
    n_sources: int,
    n_receivers: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_index = np.repeat(np.arange(n_sources, dtype=np.int64), n_receivers)
    receiver_index = np.tile(np.arange(n_receivers, dtype=np.int64), n_sources)
    return source_index, receiver_index


def _cell_indices_for_points(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    number_of_cell_x: int,
    number_of_cell_y: int,
    cell_size_x_m: float,
    cell_size_y_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.floor(np.asarray(x_m, dtype=np.float64) / cell_size_x_m).astype(np.int64)
    if cell_size_y_m is None:
        iy = np.zeros(ix.shape, dtype=np.int64)
        inside_y = np.ones(ix.shape, dtype=bool)
    else:
        iy = np.floor(np.asarray(y_m, dtype=np.float64) / cell_size_y_m).astype(np.int64)
        inside_y = (iy >= 0) & (iy < number_of_cell_y)
    inside_x = (ix >= 0) & (ix < number_of_cell_x)
    if not bool(np.all(inside_x & inside_y)):
        raise ValueError('synthetic coordinates produced points outside the cell grid')
    return ix, iy


def _select_observation_rows(
    *,
    midpoint_cell_id: np.ndarray,
    desired_observations_by_cell: dict[int, int] | None,
    n_cells: int,
) -> np.ndarray:
    if desired_observations_by_cell is None:
        return np.arange(midpoint_cell_id.shape[0], dtype=np.int64)

    desired = {int(key): int(value) for key, value in desired_observations_by_cell.items()}
    for cell_id, count in desired.items():
        if cell_id < 0 or cell_id >= n_cells or count < 0:
            raise ValueError('desired observation counts must reference valid cells')

    selected: list[int] = []
    used = np.zeros(n_cells, dtype=np.int64)
    for row_index, raw_cell_id in enumerate(midpoint_cell_id.tolist()):
        cell_id = int(raw_cell_id)
        wanted = desired.get(cell_id)
        if wanted is None:
            selected.append(row_index)
            continue
        if used[cell_id] < wanted:
            selected.append(row_index)
            used[cell_id] += 1

    missing = {
        cell_id: count - int(used[cell_id])
        for cell_id, count in desired.items()
        if used[cell_id] < count
    }
    if missing:
        raise ValueError(f'not enough synthetic observations for cells: {missing}')
    return np.asarray(selected, dtype=np.int64)


def _endpoint_sh1_m(
    rng: np.random.Generator,
    n_endpoints: int,
) -> np.ndarray:
    return _f64(rng.uniform(6.0, 24.0, size=n_endpoints))


def _t1_from_sh1(
    sh1_m: np.ndarray,
    *,
    v1_m_s: float,
    v2_m_s: np.ndarray,
) -> np.ndarray:
    # One-layer T1LSST relation inverted from SH1 truth.
    return _f64(sh1_m * np.sqrt(v2_m_s**2 - v1_m_s**2) / (v1_m_s * v2_m_s))


def _wcor_from_sh1(
    sh1_m: np.ndarray,
    *,
    v1_m_s: float,
    v2_m_s: np.ndarray,
) -> np.ndarray:
    # Repo convention uses this negative weathering correction as the shift.
    return _f64(sh1_m * ((1.0 / v2_m_s) - (1.0 / v1_m_s)))


def _noise_and_outliers(
    *,
    rng: np.random.Generator,
    n_observations: int,
    noise_std_s: float,
    outlier_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if noise_std_s < 0.0:
        raise ValueError('noise_std_s must be non-negative')
    if outlier_fraction < 0.0 or outlier_fraction > 1.0:
        raise ValueError('outlier_fraction must be in [0, 1]')
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
    return (
        _f64(float(line_origin_x_m) + dx),
        _f64(float(line_origin_y_m) + dy),
    )


def _positive_float(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be positive and finite')
    return result


def _required_float(value: float | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f'{name} is required')
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _validate_positive_int(value: int, *, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f'{name} must be positive')


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
    'SyntheticRefractionCellDataset',
    'make_clean_2d_cell_refraction_dataset',
    'make_clean_3d_cell_refraction_dataset',
    'make_low_fold_empty_cell_refraction_dataset',
    'make_outlier_refraction_dataset',
    'make_rotated_2d_line_refraction_dataset',
    'make_v2_spike_smoothing_dataset',
]
