# Multi-layer refraction time-term workflow

## 1. Current Supported Scope

This document is the self-contained implementation reference for the completed
M3 multi-layer refraction statics workflow. It extends the Phase 1 and Phase 2
references:

- `docs/statics/refraction_iras_phase1_design.md`
- `docs/statics/refraction_iras_phase2_cell_v2_design.md`

The public apply endpoint is:

```text
POST /statics/refraction/apply
```

The implemented public workflows are:

- 1-layer T1LSST with `model.method="gli_variable_thickness"` and
  `conversion.mode="t1lsst_1layer"`.
- 2-layer T1LSST with `model.method="multilayer_time_term"`,
  enabled layers `v2_t1` and `v3_t2`, and
  `conversion.mode="t1lsst_multilayer"` with `conversion.layer_count=2`.
- 3-layer T1LSST with `model.method="multilayer_time_term"`,
  enabled layers `v2_t1`, `v3_t2`, and `vsub_t3`, and
  `conversion.mode="t1lsst_multilayer"` with `conversion.layer_count=3`.
- Constant global V1.
- Global V1 estimation from direct-arrival picks.
- V2/T1 velocity mode `fixed_global`, `solve_global`, or `solve_cell`.
- V3/T2 and Vsub/T3 velocity modes `fixed_global` or `solve_global` in the
  public T1LSST apply workflow.
- Midpoint-cell V2 assignment for `solve_cell`.
- Refractor-cell coordinate modes `grid_3d` and `line_2d_projected`.
- Source and receiver static table exports for 1-, 2-, and 3-layer results.
- Optional TraceStore registration using the repo static-shift sign convention.
- M4 source-depth field correction with
  `field_corrections.source_depth.mode="weathering_velocity_time"`.
- M4 uphole-time field correction with
  `field_corrections.uphole.mode="header_time"`.
- M4 manual source/receiver static import with
  `field_corrections.manual_static.mode="artifact_table"` or
  `field_corrections.manual_static.mode="inline_table"`.

V3/T2 and Vsub/T3 `solve_cell` are schema/internal solver capabilities, but
they are not public T1LSST apply capabilities. Public T1LSST apply rejects cell
V3/T2 and cell Vsub/T3.

## 2. Non-goals

The implemented workflow is not full IRAS compatibility. The repo docs are the
canonical implementation reference; do not add original IRAS manuals or copied
IRAS material to the repository.

Current non-goals are:

- GRM.
- Plus-minus.
- Manual uphole correction tables.
- SEG-Y static header write-back.
- Refraction tomography.
- Path-integrated or raypath-weighted cell slowness.
- Spatially varying V1 maps.
- Public T1LSST apply with cell V3/T2.
- Public T1LSST apply with cell Vsub/T3.
- Browser controls for editing refraction cell models.
- Automatic 2-D line origin or azimuth estimation.

## 3. API Examples

All examples are valid against `RefractionStaticApplyRequest`. Omitted request
blocks use schema defaults in `app/api/schemas.py`.

### One-layer T1LSST

This request estimates one global V1 from direct arrivals, solves one global
V2, writes the one-layer T1LSST component artifact, and leaves the corrected
TraceStore unregistered.

```json
{
  "file_id": "example-file-id",
  "pick_source": {
    "kind": "batch_predicted_npz",
    "job_id": "first-break-job-id"
  },
  "linkage": {
    "mode": "required",
    "job_id": "geometry-linkage-job-id"
  },
  "model": {
    "method": "gli_variable_thickness",
    "first_layer": {
      "mode": "estimate_direct_arrival",
      "min_weathering_velocity_m_s": 250.0,
      "max_weathering_velocity_m_s": 1100.0,
      "min_direct_offset_m": 0.0,
      "max_direct_offset_m": 300.0,
      "min_picks_per_fit": 5,
      "min_groups": 3,
      "robust_enabled": true,
      "robust_threshold": 3.5
    },
    "bedrock_velocity_mode": "solve_global",
    "initial_bedrock_velocity_m_s": 2400.0,
    "min_bedrock_velocity_m_s": 1200.0,
    "max_bedrock_velocity_m_s": 6000.0
  },
  "moveout": {
    "model": "head_wave_linear_offset",
    "distance_source": "geometry",
    "min_offset_m": 300.0
  },
  "conversion": {
    "mode": "t1lsst_1layer"
  },
  "apply": {
    "register_corrected_file": false
  }
}
```

### Two-layer T1LSST with cell V2

This request uses a public two-layer model. The V2/T1 layer is solved per
midpoint cell; the V3/T2 layer is solved globally. The layer gates must not
overlap unless `model.allow_overlapping_layer_gates=true`.

```json
{
  "file_id": "example-file-id",
  "pick_source": {
    "kind": "batch_predicted_npz",
    "job_id": "first-break-job-id"
  },
  "linkage": {
    "mode": "required",
    "job_id": "geometry-linkage-job-id"
  },
  "model": {
    "method": "multilayer_time_term",
    "first_layer": {
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
    },
    "layers": [
      {
        "kind": "v2_t1",
        "enabled": true,
        "min_offset_m": 300.0,
        "max_offset_m": 1800.0,
        "velocity_mode": "solve_cell",
        "initial_velocity_m_s": 2400.0,
        "min_velocity_m_s": 1200.0,
        "max_velocity_m_s": 5000.0,
        "min_observations_per_cell": 5,
        "smoothing_weight": 0.0
      },
      {
        "kind": "v3_t2",
        "enabled": true,
        "min_offset_m": 1800.0,
        "max_offset_m": null,
        "velocity_mode": "solve_global",
        "initial_velocity_m_s": 3600.0,
        "min_velocity_m_s": 2600.0,
        "max_velocity_m_s": 7000.0
      }
    ],
    "refractor_cell": {
      "number_of_cell_x": 20,
      "size_of_cell_x_m": 500.0,
      "x_coordinate_origin_m": 0.0,
      "number_of_cell_y": 1,
      "size_of_cell_y_m": null,
      "y_coordinate_origin_m": 0.0,
      "assignment_mode": "midpoint",
      "outside_grid_policy": "reject",
      "coordinate_mode": "line_2d_projected",
      "line_origin_x_m": 1000.0,
      "line_origin_y_m": 2000.0,
      "line_azimuth_deg": 45.0,
      "min_observations_per_cell": 5,
      "velocity_smoothing_weight": 0.0,
      "smoothing_reference_distance_m": null
    }
  },
  "conversion": {
    "mode": "t1lsst_multilayer",
    "layer_count": 2
  },
  "apply": {
    "register_corrected_file": false
  }
}
```

### Three-layer T1LSST

This request solves V2/T1 and V3/T2 globally and uses a fixed global Vsub/T3
velocity. Public three-layer apply requires V3/T2 and Vsub/T3 to be global
velocities.

```json
{
  "file_id": "example-file-id",
  "pick_source": {
    "kind": "batch_predicted_npz",
    "job_id": "first-break-job-id"
  },
  "linkage": {
    "mode": "required",
    "job_id": "geometry-linkage-job-id"
  },
  "model": {
    "method": "multilayer_time_term",
    "first_layer": {
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
    },
    "layers": [
      {
        "kind": "v2_t1",
        "enabled": true,
        "min_offset_m": 300.0,
        "max_offset_m": 1600.0,
        "velocity_mode": "solve_global",
        "initial_velocity_m_s": 2400.0,
        "min_velocity_m_s": 1200.0,
        "max_velocity_m_s": 5000.0
      },
      {
        "kind": "v3_t2",
        "enabled": true,
        "min_offset_m": 1600.0,
        "max_offset_m": 3200.0,
        "velocity_mode": "solve_global",
        "initial_velocity_m_s": 3600.0,
        "min_velocity_m_s": 2600.0,
        "max_velocity_m_s": 7000.0
      },
      {
        "kind": "vsub_t3",
        "enabled": true,
        "min_offset_m": 3200.0,
        "max_offset_m": null,
        "velocity_mode": "fixed_global",
        "fixed_velocity_m_s": 5000.0,
        "min_velocity_m_s": 3800.0,
        "max_velocity_m_s": 9000.0
      }
    ]
  },
  "conversion": {
    "mode": "t1lsst_multilayer",
    "layer_count": 3
  },
  "apply": {
    "register_corrected_file": false
  }
}
```

## 4. Velocity and Layer Terminology

| Term | Request representation | Implementation meaning | Unit |
|---|---|---|---:|
| V1 | `model.first_layer.weathering_velocity_m_s`, or resolved by `estimate_direct_arrival` | Weathering velocity / first-layer velocity | m/s |
| V2 | `layers[].kind="v2_t1"` or legacy `bedrock_velocity_*` fields | First refractor / bedrock velocity for the T1 branch | m/s |
| T1 | `v2_t1` time term | Source or receiver half-intercept time for the V2 branch | s in NPZ, ms in CSV |
| V3 | `layers[].kind="v3_t2"` | Second refractor and 2-layer replacement velocity | m/s |
| T2 | `v3_t2` time term | Source or receiver half-intercept time for the V3 branch | s in NPZ, ms in CSV |
| Vsub | `layers[].kind="vsub_t3"` | Substratum velocity and 3-layer replacement velocity | m/s |
| T3 | `vsub_t3` time term | Source or receiver half-intercept time for the Vsub branch | s in NPZ, ms in CSV |
| SH1 | computed | First-layer thickness | m |
| SH2 | computed | Second-layer thickness | m |
| SH3 | computed | Third-layer thickness | m |
| WCOR | computed | Weathering replacement correction under the repo shift sign convention | s in NPZ, ms in CSV |
| source static | artifact output | Endpoint static for source rows | s in NPZ, ms in CSV |
| receiver static | artifact output | Endpoint static for receiver rows | s in NPZ, ms in CSV |

Layer order is always:

```text
v2_t1 -> v3_t2 -> vsub_t3
```

Enabled layer gates assign observations by offset. Each enabled layer must have
`min_offset_m`, `max_offset_m`, or both. Only the deepest enabled layer may use
`max_offset_m=null`.

## 5. T1LSST Formulas

The repo static-shift sign convention applies to all formulas in this section:

```text
corrected(t) = raw(t - shift_s)
```

Positive `shift_s` moves events later in corrected data. Negative `shift_s`
moves events earlier. In the usual velocity order, weathering corrections are
negative because slower near-surface time is replaced by faster deeper
velocity.

### Time-term solve

Each enabled layer is solved from the picks in its offset gate:

```text
pick_time_s = Tn_source_s + Tn_receiver_s + offset_m / Vn_m_s + error_s
```

where:

```text
v2_t1   -> Vn = V2,   Tn = T1
v3_t2   -> Vn = V3,   Tn = T2
vsub_t3 -> Vn = Vsub, Tn = T3
```

### One-layer T1LSST

Velocity order:

```text
V2 > V1
```

Definitions:

```text
C12 = sqrt(1 - (V1 / V2)^2)
SH1 = T1 * V1 / C12
SH1 = T1 * V1 * V2 / sqrt(V2^2 - V1^2)
WCOR = SH1 * (1 / V2 - 1 / V1)
```

### Two-layer T1LSST

Velocity order:

```text
V3 > V2 > V1
```

Definitions:

```text
C12 = sqrt(1 - (V1 / V2)^2)
C13 = sqrt(1 - (V1 / V3)^2)
C23 = sqrt(1 - (V2 / V3)^2)

SH1 = T1 * V1 / C12
SH2 = (T2 - SH1 * C13 / V1) * V2 / C23
WCOR = SH1 * (1 / V3 - 1 / V1)
     + SH2 * (1 / V3 - 1 / V2)
```

`WCOR` replaces the modeled V1/V2 interval with V3.

### Three-layer T1LSST

Velocity order:

```text
Vsub > V3 > V2 > V1
```

Definitions:

```text
C12 = sqrt(1 - (V1 / V2)^2)
C13 = sqrt(1 - (V1 / V3)^2)
C23 = sqrt(1 - (V2 / V3)^2)
C1sub = sqrt(1 - (V1 / Vsub)^2)
C2sub = sqrt(1 - (V2 / Vsub)^2)
C3sub = sqrt(1 - (V3 / Vsub)^2)

SH1 = T1 * V1 / C12
SH2 = (T2 - SH1 * C13 / V1) * V2 / C23
SH3 = (T3 - SH1 * C1sub / V1 - SH2 * C2sub / V2) * V3 / C3sub
WCOR = SH1 * (1 / Vsub - 1 / V1)
     + SH2 * (1 / Vsub - 1 / V2)
     + SH3 * (1 / Vsub - 1 / V3)
```

`WCOR` replaces the modeled V1/V2/V3 interval with Vsub.

Invalid non-finite inputs, invalid velocity order, and negative computed
thicknesses are status-coded and written as NaN rather than clipped.

## 6. Sign Convention

The repo convention is:

```text
corrected(t) = raw(t - shift_s)
```

Therefore:

```text
shift_s > 0  -> event appears later in corrected data
shift_s < 0  -> event appears earlier in corrected data
```

For source and receiver table rows:

```text
total_static_ms == total_applied_shift_ms
```

for artifact-only and apply jobs. Trace-level final shifts in
`refraction_statics.csv` use the same sign convention. When
`apply.register_corrected_file=true`, `refraction_trace_shift_s_sorted` is the
shift array applied to the source TraceStore.

When source-depth field correction is enabled with
`field_corrections.source_depth.mode="weathering_velocity_time"`, source depth
is positive downward and the source-only component is:

```text
source_depth_shift_s = +source_depth_m / V1_m_s
```

The value is stored in the same applied-shift convention. It is written to
source endpoint field-correction columns and QC. It is not folded into legacy
`source_refraction_shift_s` or `source_refraction_shift_s_sorted`; componentized
field totals are reported separately in `source_field_shift_s` and
`source_total_with_field_shift_s`. Trace-level field composition reports
`base_refraction_trace_shift_s_sorted`, `trace_field_shift_s_sorted`, and
`final_trace_shift_s_sorted` separately from the base refraction shift.
Receiver endpoint legacy totals are unchanged by this component.

When uphole-time field correction is enabled with
`field_corrections.uphole.mode="header_time"`, the configured
`field_corrections.uphole.uphole_time_byte` is loaded from sorted trace
headers, converted to seconds using `field_corrections.uphole.uphole_time_unit`,
and median-aggregated per source endpoint. With the default
`field_corrections.uphole.positive_time_means_delay=true`, the source-only
component is:

```text
uphole_shift_s = -uphole_time_s
```

If `positive_time_means_delay=false`, the component formula is:

```text
uphole_shift_s = +uphole_time_s
```

The value is stored in the same applied-shift convention and the selected
formula is written to QC. Missing, invalid, inconsistent repeated endpoint
values, values beyond `max_abs_uphole_time_s`, and inactive source endpoints
are status-coded. The component is written to source endpoint
field-correction columns and QC. It is not folded into legacy
`source_refraction_shift_s` or `source_refraction_shift_s_sorted`; componentized
field totals are reported separately in `source_field_shift_s` and
`source_total_with_field_shift_s`. Trace-level field composition reports the
base refraction, field, and final trace shifts separately. Receiver endpoint
legacy totals are unchanged by this component.

When manual source/receiver static import is enabled with
`field_corrections.manual_static.mode="artifact_table"` or
`field_corrections.manual_static.mode="inline_table"`, the input must declare
`field_corrections.manual_static.sign_convention`.
`applied_shift_s` means the supplied values already use the repo convention:

```text
manual_static_shift_s = manual_static_s
```

`delay_positive_ms` means positive input values are delays that should advance
the data:

```text
manual_static_shift_s = -manual_static_delay_s
```

Artifact tables are CSV files with `endpoint_kind` (`source` or `receiver`),
`endpoint_key` and/or `endpoint_id`, and either `manual_static_ms` or
`manual_static_s`. Separate source and receiver artifact references may omit
`endpoint_kind`; combined tables must include it. Inline tables supply
source/receiver endpoint-id values directly in the request. Matching is
deterministic: `endpoint_kind + endpoint_key` is preferred, then
`endpoint_kind + endpoint_id`. Node-id and coordinate matching are not used.
Duplicate matched endpoint rows raise an error. Missing endpoint values are
status-coded as `missing_manual_static` and may be rejected with
`allow_missing_endpoints=false`; unmatched rows are counted as
`unmatched_manual_static_row`.

The source and receiver components are written to endpoint field-correction
columns, `source_receiver_static_table.npz`, and QC. They are not folded into
legacy `source_refraction_shift_s` or `receiver_refraction_shift_s`; componentized
field totals are reported separately in source/receiver field columns and trace
preview final-shift columns.

## 7. Artifact List

Every public refraction statics job writes:

```text
refraction_static_request.json
refraction_static_solution.npz
refraction_static_qc.json
refraction_statics.csv
near_surface_model.csv
first_break_residuals.csv
refraction_static_components.csv
source_static_table.csv
receiver_static_table.csv
source_receiver_static_table.npz
refraction_static_history.json
refraction_static_artifacts.json
```

One-layer `conversion.mode="t1lsst_1layer"` also writes:

```text
refraction_t1lsst_1layer_components.csv
```

Direct-arrival V1 estimation also writes:

```text
refraction_v1_qc.json
refraction_v1_estimates.csv
```

Source-depth field correction with
`field_corrections.source_depth.mode="weathering_velocity_time"` also writes:

```text
refraction_source_depth_qc.json
refraction_source_depth_sources.csv
```

Uphole-time field correction with
`field_corrections.uphole.mode="header_time"` also writes:

```text
refraction_uphole_qc.json
refraction_uphole_sources.csv
```

Manual source/receiver static import writes endpoint component columns to the
standard source/receiver static table artifacts and
`source_receiver_static_table.npz`; it does not add a separate public artifact.

V2/T1 `solve_cell` also writes:

```text
refraction_refractor_velocity_cells.csv
refraction_refractor_velocity_grid.npz
refraction_refractor_velocity_qc.json
refraction_cell_solver_history.csv
```

The artifact writer has names for V3 and Vsub cell velocity artifacts:

```text
refraction_v3_refractor_velocity_cells.csv
refraction_v3_refractor_velocity_grid.npz
refraction_v3_refractor_velocity_qc.json
refraction_v3_cell_solver_history.csv
refraction_vsub_refractor_velocity_cells.csv
refraction_vsub_refractor_velocity_grid.npz
refraction_vsub_refractor_velocity_qc.json
refraction_vsub_cell_solver_history.csv
```

Those V3/Vsub cell artifacts are for internal cell-layer solver output. Public
`t1lsst_multilayer` apply does not currently accept cell V3/T2 or cell
Vsub/T3.

When `apply.register_corrected_file=true`, apply output can also include:

```text
corrected_file.json
refraction_static_apply_qc.json
```

## 8. Static History and Double-Application Guard

Every public refraction statics job writes `refraction_static_history.json`.
The history artifact records the repo sign convention, input file ID, output
file ID when a corrected TraceStore is registered, the cumulative shift artifact
and field, and each component with whether it was included in the trace-shift
field. Field components use the request path
`field_corrections.composition.apply_to_trace_shift`; when false, source-depth,
uphole, and manual-static components may be written as artifacts without being
included in the applied trace shift.

`field_corrections.composition.double_application_policy` controls best-effort
guards against applying components already present in input TraceStore lineage:

- `warn`: proceed and write a QC/history warning.
- `fail`: reject the job with a clear duplicate-component error.
- `allow`: proceed and record that duplicate components were allowed.

The guard reads TraceStore `meta.json` derived-component metadata when present.
It does not inspect SEG-Y static headers and does not implement a project-wide
static registry.

## 9. Source and Receiver Table Columns

The source and receiver static tables repeat the sign convention on every row:

```text
corrected(t) = raw(t - shift_s)
```

Source-specific identifier columns:

| Column | Definition |
|---|---|
| `source_endpoint_key` | Source endpoint key used by geometry linkage and node assignment. |
| `source_id` | Source station ID read from the configured source header byte. |
| `source_node_id` | Near-surface node ID assigned to the source endpoint. |
| `source_v2_cell_id` | Endpoint-local V2 cell ID for cell V2 workflows, or blank/NaN when not cell-based. |
| `source_depth_m` | Source depth in meters, positive downward; zero with `source_depth_status="not_enabled"` when disabled. |
| `source_depth_shift_ms` | Source-depth weathering-time shift in milliseconds; zero when disabled. |
| `source_depth_status` | Source-depth component status, including `not_enabled`, missing/invalid/max-shift states. |
| `uphole_time_ms` | Source uphole time in milliseconds after source-endpoint aggregation; zero when disabled. |
| `uphole_shift_ms` | Uphole-time source correction shift in milliseconds; zero when disabled. |
| `uphole_status` | Uphole component status, including `not_enabled`, missing/invalid/inconsistent/max-time/inactive-source states. |
| `manual_static_shift_ms` | Manual source static component in milliseconds after sign-convention conversion; zero when disabled. |
| `manual_static_status` | Manual source static status, including `not_enabled`, ok/missing/invalid states. |
| `source_field_shift_ms` | Sum of enabled source field-correction components in milliseconds; zero when disabled. |
| `source_field_status` | Legacy alias for `source_field_static_status`, retained for CSV compatibility. |
| `source_field_static_status` | Source field-composition status, including `not_enabled` and ok/error states. |
| `source_total_with_field_shift_ms` | `total_applied_shift_ms + source_field_shift_ms` when the base and field shifts are valid; equals `total_applied_shift_ms` when field corrections are disabled. |

Receiver-specific identifier columns:

| Column | Definition |
|---|---|
| `receiver_endpoint_key` | Receiver endpoint key used by geometry linkage and node assignment. |
| `receiver_id` | Receiver station ID read from the configured receiver header byte. |
| `receiver_node_id` | Near-surface node ID assigned to the receiver endpoint. |
| `receiver_v2_cell_id` | Endpoint-local V2 cell ID for cell V2 workflows, or blank/NaN when not cell-based. |
| `manual_static_shift_ms` | Manual receiver static component in milliseconds after sign-convention conversion; zero when disabled. |
| `manual_static_status` | Manual receiver static status, including `not_enabled`, ok/missing/invalid states. |
| `receiver_field_shift_ms` | Sum of enabled receiver field-correction components in milliseconds; zero when disabled. |
| `receiver_field_status` | Legacy alias for `receiver_field_static_status`, retained for CSV compatibility. |
| `receiver_field_static_status` | Receiver field-composition status, including `not_enabled` and ok/error states. |
| `receiver_total_with_field_shift_ms` | `total_applied_shift_ms + receiver_field_shift_ms` when the base and field shifts are valid; equals `total_applied_shift_ms` when field corrections are disabled. |

Shared columns:

| Column | Definition |
|---|---|
| `endpoint_kind` | `source` or `receiver`. |
| `x_m`, `y_m` | Endpoint coordinates in meters after geometry scaling and linkage. |
| `surface_elevation_m` | Endpoint surface elevation in meters. |
| `floating_datum_elevation_m` | Floating datum elevation used for the endpoint. |
| `flat_datum_elevation_m` | Flat datum elevation used for the job, if configured. |
| `t1_ms` | T1 half-intercept time in milliseconds. |
| `t2_ms` | T2 half-intercept time in milliseconds for 2- and 3-layer results. |
| `t3_ms` | T3 half-intercept time in milliseconds for 3-layer results. |
| `v1_m_s` | Resolved global V1. |
| `v2_m_s` | Endpoint-local or global V2. |
| `v2_status` | V2 endpoint status, including cell assignment and low-fold status when applicable. |
| `v3_m_s` | Global V3 for 2- and 3-layer public T1LSST results. |
| `vsub_m_s` | Global Vsub for 3-layer public T1LSST results. |
| `sh1_weathering_thickness_m` | Computed SH1 thickness. |
| `sh2_weathering_thickness_m` | Computed SH2 thickness for 2- and 3-layer results. |
| `sh3_weathering_thickness_m` | Computed SH3 thickness for 3-layer results. |
| `total_weathering_thickness_m` | Sum of valid SH thicknesses for the enabled conversion. |
| `layer1_base_elevation_m` | V1/V2 interface elevation, `surface_elevation_m - SH1`. |
| `layer2_base_elevation_m` | V2/V3 interface elevation for 3-layer results. |
| `final_refractor_elevation_m` | Final replacement interface elevation. In 2-layer output this is the V3 interface; in 3-layer output this is the Vsub interface. |
| `refractor_elevation_m` | Legacy alias for the final replacement interface elevation. |
| `weathering_correction_ms` | WCOR in milliseconds using the repo shift sign convention. |
| `floating_datum_correction_ms` | Floating datum correction component. |
| `flat_datum_correction_ms` | Flat datum correction component. |
| `elevation_correction_ms` | Sum of floating and flat datum correction components. |
| `total_static_ms` | Final endpoint static in the repo sign convention. |
| `total_applied_shift_ms` | Applied endpoint shift; same convention and value as `total_static_ms`. |
| `solution_status` | Time-term solution status. |
| `weathering_status` | T1LSST/weathering conversion status. |
| `datum_status` | Endpoint datum status. |
| `static_status` | Final endpoint static status. |
| `sign_convention` | Literal `corrected(t) = raw(t - shift_s)`. |
| `pick_count` | Number of observations associated with the endpoint/node. |
| `used_pick_count` | Number of observations used after filtering or robust rejection. |
| `residual_rms_ms` | Endpoint/node residual RMS in milliseconds. |
| `residual_mad_ms` | Endpoint/node residual MAD in milliseconds. |
| `pick_count_by_layer` | Per-layer observation counts for multi-layer results. |
| `used_pick_count_by_layer` | Per-layer used-observation counts for multi-layer results. |
| `residual_rms_by_layer_ms` | Per-layer residual RMS values for multi-layer results. |
| `residual_mad_by_layer_ms` | Per-layer residual MAD values for multi-layer results. |

Trace preview CSVs include stable field-composition columns:
`source_field_shift_ms`, `receiver_field_shift_ms`, `trace_field_shift_ms`,
`refraction_trace_shift_ms`, `final_trace_shift_ms`, and
`trace_field_static_status`. When field corrections are disabled, field shifts
are zero, field status is `not_enabled`, and `final_trace_shift_ms` equals
`refraction_trace_shift_ms` wherever the base refraction shift is finite.

`source_receiver_static_table.npz` stores the stable source and receiver
endpoint arrays used by downstream apply flows as pickle-free arrays. Time
arrays are in seconds. Source-depth arrays `source_depth_m`,
`source_depth_shift_s`, and `source_depth_status`, uphole arrays
`source_uphole_time_s`, `source_uphole_shift_s`, and `source_uphole_status`,
manual-static arrays, source/receiver field-shift arrays, and
source/receiver total-with-field arrays are always present. Disabled
components use zero shifts and `not_enabled` statuses. CSV time columns are in
milliseconds. The per-layer QC summary columns `pick_count_by_layer`,
`used_pick_count_by_layer`, `residual_rms_by_layer_ms`, and
`residual_mad_by_layer_ms` are CSV-only fields and are not part of the NPZ
schema.

## 10. Cell and Coordinate Modes

Velocity modes:

- `fixed_global`: use `fixed_velocity_m_s` for the layer.
- `solve_global`: solve one velocity for the layer from observations in that
  layer's offset gate.
- `solve_cell`: solve one velocity per active refractor cell.

Public T1LSST apply supports `solve_cell` only for V2/T1. V3/T2 and Vsub/T3
must use `fixed_global` or `solve_global`.

Cell assignment uses observation midpoint, not raypath integration:

```text
midpoint_x_m = 0.5 * (source_x_m + receiver_x_m)
midpoint_y_m = 0.5 * (source_y_m + receiver_y_m)
```

For `coordinate_mode="grid_3d"`, input X/Y coordinates are used directly for
cell assignment. This supports a 3-D grid when `number_of_cell_y > 1` and
`size_of_cell_y_m` is set.

For `coordinate_mode="line_2d_projected"`, map X/Y coordinates are projected
onto a manually supplied line before cell assignment:

```text
inline_unit_x = sin(line_azimuth_deg)
inline_unit_y = cos(line_azimuth_deg)
dx = input_x_m - line_origin_x_m
dy = input_y_m - line_origin_y_m

inline_m = dx * inline_unit_x + dy * inline_unit_y
crossline_m = dx * inline_unit_y - dy * inline_unit_x

cell_x_m = inline_m
cell_y_m = 0
```

Line mode requires `number_of_cell_y=1`; it does not estimate line origin or
azimuth automatically.

## 11. Recommended Synthetic Validation Tests

Recommended coverage for future changes:

- One-layer global V2: recover known V1, V2, T1, SH1, WCOR, source statics,
  receiver statics, and trace shifts.
- One-layer cell V2: recover known endpoint-local V2, T1, SH1, WCOR, and
  source/receiver table rows on a 1-D cell grid.
- Two-layer global V2/V3: recover known T1/T2, SH1/SH2, WCOR, final interface
  elevation, and source/receiver table fields.
- Two-layer cell V2 plus global V3: verify endpoint-local V2 propagates into
  SH1/SH2/WCOR and that low-fold or outside-grid V2 statuses produce NaN
  conversion values.
- Three-layer global V2/V3/Vsub: recover known T1/T2/T3, SH1/SH2/SH3, WCOR,
  layer base elevations, and final Vsub interface.
- Direct-arrival V1 estimation: recover known V1 and verify V1 artifacts.
- `grid_3d` cell coordinates: verify row-major cell IDs, Y-cell intervals, and
  endpoint-local cell assignment.
- `line_2d_projected` coordinates: verify projected inline cell assignment on
  a diagonal map line.
- Offset layer gates: verify non-overlap validation and correct observation
  routing to `v2_t1`, `v3_t2`, and `vsub_t3`.
- Artifact schema: verify `refraction_static_solution.npz`,
  `source_receiver_static_table.npz`, CSV columns, QC metadata, and manifest
  entries.
- Sign convention: assert `corrected(t) = raw(t - shift_s)`, table
  `total_static_ms == total_applied_shift_ms`, and trace shift application
  uses `refraction_trace_shift_s_sorted`.

## 12. Known Limitations

- GRM is not implemented.
- Plus-minus is not implemented.
- Manual uphole correction tables are not implemented.
- SEG-Y header write-back is not implemented.
- Path-integrated cell slowness is not implemented; cell velocity assignment is
  midpoint-based.
- Public T1LSST apply does not support cell V3/T2.
- Public T1LSST apply does not support cell Vsub/T3.
- Spatially varying V1 maps are not implemented.
- Automatic 2-D line origin or azimuth estimation is not implemented.
- Negative SH1, SH2, or SH3 is status-coded as
  `invalid_negative_thickness` and written as NaN; the implementation does not
  clip negative thicknesses.
