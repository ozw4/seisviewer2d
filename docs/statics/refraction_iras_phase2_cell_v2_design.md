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
- Documentation of the request fields and validation rules.

Out of scope for this phase:

- Cell-grid construction from geometry.
- Matrix columns for per-cell refractor slowness.
- Solver implementation.
- Cell V2 artifacts, QC outputs, and E2E tests.
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
linked geometry workflow. Later implementation issues will define the exact
matrix construction and QC artifacts.

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

## Execution Contract

This phase validates and carries the request shape only. Until later issues add
cell-grid construction, design-matrix support, solver support, and artifacts,
callers must not assume that a `solve_cell` request has an executable backend
path.
