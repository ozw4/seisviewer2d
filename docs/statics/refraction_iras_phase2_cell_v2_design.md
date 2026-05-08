# IRAS-style refraction statics Phase 2 cell V2 design contract

## Purpose

This document defines the Phase 2 request contract for a spatially varying
refractor velocity model in the refraction statics workflow. It extends the
Phase 1 GLI request schema with a cell-based V2 mode while keeping the existing
global V2 modes valid and unchanged.

Use `docs/statics/refraction_iras_phase1_design.md` as the canonical Phase 1
reference. The original IRAS materials are intentionally not stored in this
repository and are not required for this contract.

## Scope

In scope for this phase:

- A `solve_cell` value for `model.bedrock_velocity_mode`.
- A `model.refractor_cell` request block describing the V2 cell grid contract.
- Schema validation for required, forbidden, and bounded cell-grid fields.
- Pure refractor cell-grid utilities with midpoint observation assignment.
- Sparse design-matrix support for per-cell refractor slowness columns.
- Bounded least-squares solver support for active per-cell V2 values.
- Documentation of the request fields, validation rules, and service-layer
  execution contract.

Out of scope for this phase:

- Smoothing regularization for neighboring cell velocities.
- Cell V2 job artifacts beyond the service-layer result and QC objects.
- End-to-end API/browser tests for full refraction-static apply jobs using
  `solve_cell`.
- UI controls.
- Original IRAS reference files or manuals.

## Model Contract

Phase 1 uses a global V2 model:

```text
pick_time_s = T_source_s + T_receiver_s + offset_m / V2_global + error_s
```

Phase 2 adds a request contract for solving a V2 per spatial cell:

```text
pick_time_s = T_source_s + T_receiver_s + offset_m / V2_cell + error_s
```

The assignment of an observation to a cell is requested with
`assignment_mode="midpoint"`. The midpoint is the midpoint between the source
and receiver endpoint coordinates in the same coordinate system used by the
linked geometry workflow. Rows outside the refractor cell grid are rejected from
the inversion with the reason `outside_refractor_cell_grid`; they are not clipped
to the nearest cell.

## API Schema

`RefractionStaticModelRequest.bedrock_velocity_mode` accepts:

```python
Literal["solve_global", "fixed_global", "solve_cell"]
```

`solve_global` and `fixed_global` preserve the Phase 1 behavior. `solve_cell`
requires a `refractor_cell` block:

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

The model request includes:

```python
refractor_cell: RefractionStaticRefractorCellRequest | None = None
```

## Request Example

```json
{
  "model": {
    "method": "gli_variable_thickness",
    "first_layer": {
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
    },
    "bedrock_velocity_mode": "solve_cell",
    "initial_bedrock_velocity_m_s": 2400.0,
    "min_bedrock_velocity_m_s": 1200.0,
    "max_bedrock_velocity_m_s": 6000.0,
    "refractor_cell": {
      "number_of_cell_x": 20,
      "size_of_cell_x_m": 500.0,
      "x_coordinate_origin_m": 0.0,
      "number_of_cell_y": 1,
      "size_of_cell_y_m": 1000.0,
      "y_coordinate_origin_m": 0.0,
      "assignment_mode": "midpoint",
      "outside_grid_policy": "reject",
      "min_observations_per_cell": 5,
      "velocity_smoothing_weight": 0.1
    }
  }
}
```

## Validation Rules

- `model.refractor_cell` is required when
  `model.bedrock_velocity_mode="solve_cell"`.
- `model.refractor_cell` is forbidden when `model.bedrock_velocity_mode` is
  `solve_global` or `fixed_global`.
- `number_of_cell_x` and `number_of_cell_y` must be positive integers.
- `min_observations_per_cell` must be a positive integer.
- `size_of_cell_x_m` must be a positive finite float.
- `size_of_cell_y_m`, if provided, must be a positive finite float.
- `number_of_cell_y > 1` requires `size_of_cell_y_m`.
- `x_coordinate_origin_m` and `y_coordinate_origin_m` must be finite floats.
- `assignment_mode="midpoint"` is the only Phase 2 assignment mode.
- `outside_grid_policy="reject"` is the only Phase 2 outside-grid policy.
- `velocity_smoothing_weight` must be finite and greater than or equal to 0.
- `smoothing_reference_distance_m`, if provided, must be a positive finite
  float.
- Existing Phase 1 `solve_global` and `fixed_global` requests remain valid.

## Grid Assignment Contract

`build_refraction_cell_grid()` builds a deterministic row-major grid where:

```text
cell_id = iy * number_of_cell_x + ix
```

`ix` varies fastest. One-dimensional line-style grids use
`number_of_cell_y=1`; if `size_of_cell_y_m` is omitted for this case, the Y
axis is treated as unbounded for assignment.

Point assignment uses half-open intervals:

```text
x_min <= x < x_max
y_min <= y < y_max
```

The final maximum X/Y boundary is included with a small numerical tolerance so
coordinates exactly on the final grid edge remain inside the last cell.

The shared assignment functions are:

```python
build_refraction_cell_grid(config)
assign_points_to_refraction_cells(grid, x_m, y_m)
compute_source_receiver_midpoints(source_x_m, source_y_m, receiver_x_m, receiver_y_m)
assign_observation_midpoint_cells(grid, source_x_m, source_y_m, receiver_x_m, receiver_y_m)
```

Assignment QC reports point counts, inside/outside counts, active/inactive cell
counts, assigned coordinate extents, and per-active-cell point-count statistics.

## Design Matrix Contract

`build_refraction_static_design_matrix()` preserves the existing `solve_global`
and `fixed_global` paths. For `solve_cell`, it builds the refractor grid,
assigns each valid observation by source-receiver midpoint, rejects valid rows
outside the grid, and creates one slowness column per active cell.

Parameter ordering is:

```text
0 ... n_active_nodes-1                       node T1 columns
n_active_nodes ... n_active_nodes+n_cells-1  active cell slowness columns
```

Each used observation row has three non-zero coefficients:

```text
source_T_col        1
receiver_T_col      1
cell_slowness_col   offset_m
```

Cells with no used observations are inactive and do not add solver columns.
Design-matrix QC includes `bedrock_velocity_mode`, `cell_assignment_mode`,
active/inactive cell counts, outside-grid rejection counts, observations used,
per-active-cell observation-count statistics, `matrix_nnz`, and `matrix_shape`.

## Solver Contract

`solve_refraction_static_bounded_ls()` supports `solve_cell` design matrices
with active cell slowness columns. It solves one bounded slowness value per
active cell:

```text
lower_slowness = 1 / max_bedrock_velocity_m_s
upper_slowness = 1 / min_bedrock_velocity_m_s
```

All active cells use `initial_bedrock_velocity_m_s` as the initial V2 when it is
provided. Otherwise the solver uses a midpoint value within the configured V2
bounds. Robust rejection is applied only to data rows.

For `solve_cell`, the scalar `bedrock_velocity_m_s` and
`bedrock_slowness_s_per_m` fields are summary medians across solved active
cells. The primary per-cell outputs are:

```text
active_cell_id
inactive_cell_id
cell_bedrock_slowness_s_per_m
cell_bedrock_velocity_m_s
cell_velocity_status
row_midpoint_cell_id
row_midpoint_bedrock_velocity_m_s
```

Solver QC marks `bedrock_velocity_solution_kind="per_cell"` and reports active
cell velocity/slowness min, median, and max values, plus solved/clipped cell
status counts.

## Execution Boundary

The core service-layer `solve_cell` path is executable through the cell-grid,
design-matrix, and bounded solver functions described above. Full refraction
static apply-job artifacts, datum/static conversion tables, browser controls,
and smoothing regularization remain outside this phase unless a later issue
explicitly adds them.
