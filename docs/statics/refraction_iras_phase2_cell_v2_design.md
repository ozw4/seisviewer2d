# IRAS-style refraction statics Phase 2 cell-based V2 design

## 1. Scope and non-goals

This document defines the Phase 2 implementation reference for cell-based
refractor velocity (`V2`) inversion in the refraction statics workflow. It is
self-contained and should be read with
`docs/statics/refraction_iras_phase1_design.md`.

Phase 2 is in scope for:

- `model.bedrock_velocity_mode="solve_cell"`.
- `model.refractor_cell` request validation and grid construction.
- Midpoint-cell first-break observation assignment.
- Sparse GLI design-matrix columns for per-cell refractor slowness.
- Bounded least-squares inversion of one active `V2` per cell.
- Optional 4-neighbor smoothing regularization on active cell slowness.
- Endpoint-local `V2` projection for T1LSST one-layer conversion.
- Cell-aware source, receiver, solution, QC, and refractor velocity artifacts.

Non-goals:

- Original IRAS documents stored in this repository.
- Full IRAS compatibility.
- Path-weighted or raypath-integrated cell slowness.
- Refraction tomography.
- Spatially varying V1.
- Multi-layer weathering conversion.
- GRM or plus-minus methods.
- SEG-Y static header write-back.
- Browser controls for cell model editing.

## 2. Relationship to Phase 1

Phase 1 defines the canonical one-layer refraction statics workflow:

```text
first-break picks
  -> resolve V1
  -> solve GLI T1 and V2
  -> convert T1 to SH1 and WCOR
  -> write source/receiver static tables
  -> optionally apply trace shifts with the repo sign convention
```

Phase 2 keeps the Phase 1 endpoint time-term model and T1LSST one-layer
conversion. The only conceptual change is the refractor moveout term: Phase 1
uses one global `V2`, while Phase 2 uses one `V2` per refractor cell. The solved
cell velocity is still a bedrock/refractor velocity used in the same T1LSST
one-layer formulas.

For backward compatibility, `solve_global` and `fixed_global` requests remain
valid and do not accept `model.refractor_cell`.

## 3. Request examples for solve_global vs solve_cell

Global V2 solve, using Phase 1 behavior:

```json
{
  "model": {
    "method": "gli_variable_thickness",
    "first_layer": {
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
    },
    "bedrock_velocity_mode": "solve_global",
    "initial_bedrock_velocity_m_s": 2400.0,
    "min_bedrock_velocity_m_s": 1200.0,
    "max_bedrock_velocity_m_s": 6000.0
  },
  "conversion": {
    "mode": "t1lsst_1layer"
  }
}
```

Cell V2 solve:

```json
{
  "model": {
    "method": "gli_variable_thickness",
    "first_layer": {
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
    },
    "bedrock_velocity_mode": "solve_cell",
    "initial_bedrock_velocity_m_s": 2600.0,
    "min_bedrock_velocity_m_s": 1200.0,
    "max_bedrock_velocity_m_s": 6000.0,
    "refractor_cell": {
      "number_of_cell_x": 20,
      "size_of_cell_x_m": 500.0,
      "x_coordinate_origin_m": 0.0,
      "number_of_cell_y": 1,
      "size_of_cell_y_m": null,
      "y_coordinate_origin_m": 0.0,
      "assignment_mode": "midpoint",
      "outside_grid_policy": "reject",
      "min_observations_per_cell": 5,
      "velocity_smoothing_weight": 0.0,
      "smoothing_reference_distance_m": null
    }
  },
  "conversion": {
    "mode": "t1lsst_1layer"
  }
}
```

`model.bedrock_velocity_m_s` is only allowed with
`model.bedrock_velocity_mode="fixed_global"` and must be omitted for
`solve_cell`.

## 4. Refractor cell grid definition

The schema is:

```python
class RefractionStaticRefractorCellRequest(BaseModel):
    number_of_cell_x: int
    size_of_cell_x_m: float
    x_coordinate_origin_m: float

    number_of_cell_y: int = 1
    size_of_cell_y_m: float | None = None
    y_coordinate_origin_m: float = 0.0

    assignment_mode: Literal["midpoint"] = "midpoint"
    outside_grid_policy: Literal["reject"] = "reject"

    min_observations_per_cell: int = 5
    velocity_smoothing_weight: float = 0.0
    smoothing_reference_distance_m: float | None = None
```

Validation rules:

- `model.refractor_cell` is required for `solve_cell`.
- `model.refractor_cell` is forbidden for `solve_global` and `fixed_global`.
- `number_of_cell_x`, `number_of_cell_y`, and
  `min_observations_per_cell` must be positive integers.
- `size_of_cell_x_m` must be positive and finite.
- `size_of_cell_y_m`, when provided, must be positive and finite.
- `number_of_cell_y > 1` requires `size_of_cell_y_m`.
- `x_coordinate_origin_m` and `y_coordinate_origin_m` must be finite.
- `assignment_mode="midpoint"` is the only Phase 2 assignment mode.
- `outside_grid_policy="reject"` is the only Phase 2 outside-grid policy.
- `velocity_smoothing_weight` must be finite and non-negative.
- `smoothing_reference_distance_m`, when provided, must be positive and finite.

For `number_of_cell_y=1` with `size_of_cell_y_m=null`, the Y axis is treated as
unbounded for assignment. Otherwise, each axis uses half-open cell intervals:

```text
x_min <= x < x_max
y_min <= y < y_max
```

The final maximum X/Y edge is included with a small tolerance so a coordinate
exactly on the final edge remains in the last cell.

## 5. Cell ID convention

Cells are row-major with X varying fastest:

```text
cell_id = iy * number_of_cell_x + ix
```

`ix` and `iy` are zero-based. For a 1-D line grid, use
`number_of_cell_y=1`; then `cell_id == ix`.

## 6. Midpoint assignment convention

Phase 2 assigns each first-break observation to a cell by source/receiver
midpoint:

```text
midpoint_x_m = 0.5 * (source_x_m + receiver_x_m)
midpoint_y_m = 0.5 * (source_y_m + receiver_y_m)
```

Rows with midpoint coordinates outside the refractor cell grid are rejected from
the inversion with:

```text
outside_refractor_cell_grid
```

They are not clipped to the nearest cell. This is a midpoint-cell MVP. It is
not a raypath-integrated, path-weighted, or tomographic velocity model.

## 7. Per-cell solve equation

For global V2, Phase 1 solves:

```text
pick_time_s = T_source_s + T_receiver_s + offset_m / V2_global + error_s
```

For cell V2, Phase 2 solves:

```text
pick_time_s = T_source_s + T_receiver_s + offset_m * s_cell(midpoint) + error_s
s_cell = 1 / V2_cell
```

The sparse design matrix keeps the Phase 1 node T1 columns and adds one
slowness column for each active cell:

```text
0 ... n_active_nodes-1                       node T1 columns
n_active_nodes ... n_active_nodes+n_cells-1  active cell slowness columns
```

Each used observation row has:

```text
source_T_col        1
receiver_T_col      1
cell_slowness_col   offset_m
```

Cells with no used observations are inactive and do not add solver columns.
The current implementation validates `min_observations_per_cell` but active
cell selection is based on observation presence after filtering.

## 8. Slowness bounds from velocity bounds

The request supplies velocity bounds:

```text
min_bedrock_velocity_m_s
max_bedrock_velocity_m_s
```

The bounded solver applies them as slowness bounds:

```text
lower_slowness_s_per_m = 1 / max_bedrock_velocity_m_s
upper_slowness_s_per_m = 1 / min_bedrock_velocity_m_s
```

For `solve_cell`, every active cell slowness parameter uses these bounds.
`initial_bedrock_velocity_m_s`, when provided, initializes all active cells;
otherwise the solver chooses a value within the configured bounds. The scalar
`bedrock_velocity_m_s` and `bedrock_slowness_s_per_m` fields in result objects
are summary medians across active cells.

## 9. Smoothing regularization equation

When `velocity_smoothing_weight > 0`, the solver adds one sparse regularization
row for each active 4-neighbor cell edge:

```text
row_scale * (s_cell_i - s_cell_j) = 0
row_scale = velocity_smoothing_weight * smoothing_reference_distance_m
```

If `smoothing_reference_distance_m` is omitted, the implementation uses the
median used-observation `row_distance_m`. Smoothing rows are appended to the
least-squares system and are excluded from robust data-row rejection.

When `velocity_smoothing_weight=0`, no smoothing rows are added.

## 10. Endpoint-local V2 projection

The per-cell inversion assigns observations by midpoint, but T1LSST conversion
needs a local V2 at each endpoint. Phase 2 projects local V2 by assigning node,
source endpoint, and receiver endpoint coordinates directly to the same
refractor cell grid.

Projection outputs include:

```text
node_v2_cell_id
node_v2_m_s
node_v2_status
source_v2_cell_id
source_v2_m_s
source_v2_status
receiver_v2_cell_id
receiver_v2_m_s
receiver_v2_status
```

Trace-order aliases are also written for source and receiver endpoints:

```text
source_v2_cell_id_sorted
source_v2_m_s_sorted
source_v2_status_sorted
receiver_v2_cell_id_sorted
receiver_v2_m_s_sorted
receiver_v2_status_sorted
```

Endpoint status values include:

```text
ok
outside_refractor_cell_grid
inactive_v2_cell
invalid_local_v2
v2_not_greater_than_v1
```

Only endpoints with `ok` local V2 participate in valid local T1LSST conversion.

## 11. T1LSST 1-layer formula with local V2

For each node/source/receiver endpoint, use the endpoint-local V2:

```text
SH1 = T1 * V1 * V2_local / sqrt(V2_local^2 - V1^2)
WCOR = SH1 * (1 / V2_local - 1 / V1)
```

Where:

- `T1` is the source or receiver half-intercept time in seconds.
- `V1` is the resolved weathering velocity in m/s.
- `V2_local` is the endpoint-local refractor velocity in m/s.
- `SH1` is weathering thickness in m.
- `WCOR` is weathering correction in seconds.

The conversion requires `V2_local > V1`. Invalid or inactive endpoint-local V2
is status-coded and produces NaN conversion values rather than pretending that a
global V2 applies.

## 12. Sign convention

The repo static-shift convention is:

```text
corrected(t) = raw(t - shift_s)
```

Therefore:

```text
shift_s > 0  -> event appears later in corrected data
shift_s < 0  -> event appears earlier in corrected data
```

For `V2_local > V1`, the one-layer weathering correction is normally negative:

```text
WCOR = SH1 * (1 / V2_local - 1 / V1)
```

because replacing a low-velocity weathering layer with faster bedrock removes
delay. Phase 2 keeps:

```text
total_static_s == total_applied_shift_s
```

under this repo convention.

## 13. Artifact list and column definitions

Final refraction jobs still write the Phase 1 artifact package:

```text
refraction_static_solution.npz
refraction_static_qc.json
refraction_statics.csv
near_surface_model.csv
first_break_residuals.csv
refraction_static_components.csv
source_static_table.csv
receiver_static_table.csv
source_receiver_static_table.npz
refraction_static_artifacts.json
refraction_static_request.json
```

When `conversion.mode="t1lsst_1layer"`, jobs also write:

```text
refraction_t1lsst_1layer_components.csv
```

Important columns:

```text
endpoint_kind
endpoint_key
node_id
x_m
y_m
surface_elevation_m
floating_datum_elevation_m
flat_datum_elevation_m
t1_ms
v1_m_s
v2_m_s
sh1_weathering_thickness_m
refractor_elevation_m
weathering_correction_ms
floating_datum_correction_ms
flat_datum_correction_ms
elevation_correction_ms
total_static_ms
total_applied_shift_ms
solution_status
weathering_status
datum_status
static_status
sign_convention
```

When `model.bedrock_velocity_mode="solve_cell"`, jobs also write:

```text
refraction_refractor_velocity_cells.csv
refraction_refractor_velocity_grid.npz
refraction_refractor_velocity_qc.json
```

`refraction_refractor_velocity_cells.csv` columns:

```text
cell_id
ix
iy
x_min_m
x_max_m
y_min_m
y_max_m
x_center_m
y_center_m
active
n_observations
n_used_observations
n_rejected_observations
v2_m_s
slowness_s_per_m
velocity_status
residual_rms_ms
residual_mad_ms
residual_mean_ms
residual_p95_abs_ms
smoothing_neighbor_count
```

`refraction_refractor_velocity_grid.npz` contains the same per-cell arrays with
pickle disabled. `refraction_refractor_velocity_qc.json` records grid shape,
assignment mode, outside-grid counts, used observations, velocity statistics,
and smoothing-row counts.

`source_static_table.csv` adds cell-aware V2 fields:

```text
source_v2_cell_id
v2_m_s
v2_status
```

`receiver_static_table.csv` adds:

```text
receiver_v2_cell_id
v2_m_s
v2_status
```

Both source and receiver tables also include:

```text
t1_ms
v1_m_s
sh1_weathering_thickness_m
weathering_correction_ms
elevation_correction_ms
total_static_ms
total_applied_shift_ms
solution_status
weathering_status
datum_status
static_status
pick_count
used_pick_count
residual_rms_ms
residual_mad_ms
```

`source_receiver_static_table.npz` includes source and receiver endpoint arrays:

```text
source_v2_cell_id
source_v2_status
source_v2_m_s
source_t1_s
source_v1_m_s
source_sh1_m
source_weathering_correction_s
source_elevation_correction_s
source_total_static_s
source_total_applied_shift_s
source_static_status

receiver_v2_cell_id
receiver_v2_status
receiver_v2_m_s
receiver_t1_s
receiver_v1_m_s
receiver_sh1_m
receiver_weathering_correction_s
receiver_elevation_correction_s
receiver_total_static_s
receiver_total_applied_shift_s
receiver_static_status
```

`refraction_static_solution.npz` includes the machine-readable cell arrays:

```text
active_cell_id
inactive_cell_id
cell_bedrock_slowness_s_per_m
cell_bedrock_velocity_m_s
cell_velocity_status
row_midpoint_cell_id
node_v2_cell_id
node_v2_m_s
node_v2_status
source_v2_cell_id
source_v2_m_s
source_v2_status
receiver_v2_cell_id
receiver_v2_m_s
receiver_v2_status
source_v2_cell_id_sorted
source_v2_m_s_sorted
source_v2_status_sorted
receiver_v2_cell_id_sorted
receiver_v2_m_s_sorted
receiver_v2_status_sorted
```

`refraction_static_qc.json` sets `velocity.bedrock_velocity_status` to
`per_cell` and points to the cell velocity QC artifact.

## 14. Synthetic test model

The synthetic Phase 2 test model uses:

```text
V1 = 800 m/s
cell_size_x = 100 m
number_of_cell_x = 3
number_of_cell_y = 1
V2_cell = [2200, 2600, 3000] m/s
node_x_m = [25, 50, 75, 125, 150, 175, 225, 250, 275]
```

Each node has a known SH1. The test computes local T1 and WCOR with the same
one-layer formulas:

```text
T1 = SH1 * sqrt(V2_local^2 - V1^2) / (V1 * V2_local)
WCOR = SH1 * (1 / V2_local - 1 / V1)
```

First-break picks are generated from midpoint-cell V2:

```text
pick_time_s = T1_source_s + T1_receiver_s + offset_m * s_cell(midpoint)
```

The E2E tests assert that noiseless solves recover the active cell V2 values,
T1, SH1, WCOR, source/receiver static tables, inactive cell endpoint status,
outside-grid rejection, smoothing behavior, and pickle-free NPZ artifacts.

## 15. Known limitations

- Uses midpoint-cell assignment, not raypath-integrated cell slowness.
- Uses 1-layer T1LSST conversion only.
- Does not implement V3/T2 or Vsub/T3.
- Does not implement GRM or plus-minus.
- Does not write SEG-Y static headers.
- Does not store original IRAS documents in repo.
- Does not estimate spatially varying V1.
- Does not perform tomography or path-weighted inversion.
- `min_observations_per_cell` is schema-validated but is not currently a
  per-cell rejection threshold in the solver.
