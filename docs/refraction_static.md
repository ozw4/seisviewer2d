# Refraction statics

This page documents the IRAS-compatible refraction statics workflows. The
canonical implementation references are
[statics/refraction_iras_phase1_design.md](statics/refraction_iras_phase1_design.md)
and
[statics/refraction_iras_phase2_cell_v2_design.md](statics/refraction_iras_phase2_cell_v2_design.md).

The public apply workflow supports a practical 1-layer model:

```text
first-break picks
  -> resolve global V1
  -> solve 1-layer GLI variable-thickness T1 and global V2
  -> convert T1 to SH1 and WCOR
  -> export source and receiver static tables
  -> optionally apply trace shifts with the repo sign convention
```

It also supports the M3 multi-layer public contract for
`model.method="multilayer_time_term"` with `conversion.mode="t1lsst_multilayer"`
and either:

- `conversion.layer_count=2`, when the enabled layers are exactly `v2_t1` and
  `v3_t2`
- `conversion.layer_count=3`, when the enabled layers are exactly `v2_t1`,
  `v3_t2`, and `vsub_t3`

The `v2_t1` layer may use fixed, global solved, or cell solved V2; the `v3_t2`
and `vsub_t3` layers must use fixed or global solved velocities. Cell V3 and
cell Vsub remain out of scope.

`POST /statics/refraction/apply` is the public job entry point for the
1-layer workflow and the narrow two- and three-layer multi-layer workflows. It
does not claim full IRAS compatibility.

## Terms

| Term | Meaning | Unit |
|---|---|---:|
| V1 | Weathering velocity / first-layer velocity | m/s |
| V2 | Bedrock velocity / refractor velocity | m/s |
| V3 | Second refractor / replacement velocity for the 2-layer correction | m/s |
| T1 | Source or receiver node half-intercept time from the GLI model | s in NPZ, ms in CSV |
| T2 | Source or receiver half-intercept time for the V3 refractor branch | s in NPZ, ms in CSV |
| SH1 | Weathering thickness derived from T1, V1, and V2 | m |
| SH2 | Second-layer thickness derived from T1, T2, V1, V2, and V3 | m |
| WCOR | Weathering correction from replacing modeled near-surface layers with the replacement velocity | s in NPZ, ms in CSV |
| total static | Final source, receiver, or trace static under the repo sign convention | s in NPZ, ms in CSV |
| total applied shift | Shift value used when applying statics to traces | s in NPZ, ms in CSV |

The repo static-shift convention is:

```text
corrected(t) = raw(t - shift_s)
```

So:

```text
shift_s > 0  -> events appear later in corrected data
shift_s < 0  -> events appear earlier in corrected data
```

For the Phase 1 refraction workflow, `total_static_s` and
`total_applied_shift_s` are the same convention.

## V1 Modes

### Constant V1

Constant mode uses one global weathering velocity for the job:

```json
{
  "model": {
    "method": "gli_variable_thickness",
    "first_layer": {
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
    },
    "bedrock_velocity_mode": "solve_global"
  }
}
```

Legacy requests remain valid. `model.weathering_velocity_m_s` is treated as the
same global V1 value:

```json
{
  "model": {
    "method": "gli_variable_thickness",
    "weathering_velocity_m_s": 800.0,
    "bedrock_velocity_mode": "solve_global"
  }
}
```

If both `model.weathering_velocity_m_s` and
`model.first_layer.weathering_velocity_m_s` are supplied for constant mode, they
must match.

### Direct-Arrival V1 Estimation

Direct-arrival estimation resolves one global V1 from near-offset picks before
the GLI weathering solve:

```json
{
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
    "bedrock_velocity_mode": "solve_global"
  }
}
```

The estimator fits source-endpoint groups with:

```text
pick_time_s = intercept_s + offset_m * slope_s_per_m
V1_group = 1 / slope_s_per_m
```

The intercept is not forced to zero. Valid group estimates are filtered by pick
count, velocity bounds, finite residual statistics, and the robust settings. The
resolved global V1 is the median of valid group V1 estimates. In this mode the
job writes `refraction_v1_qc.json` and `refraction_v1_estimates.csv`.

## GLI Variable-Thickness Model

The Phase 1 GLI model is:

```text
pick_time_s = T_source_s + T_receiver_s + offset_m / V2_m_s + error_s
```

`T_source_s` and `T_receiver_s` are T1-style half-intercept time terms assigned
to source and receiver nodes. `V2` is either solved globally with:

```json
{
  "bedrock_velocity_mode": "solve_global",
  "min_bedrock_velocity_m_s": 1200.0,
  "max_bedrock_velocity_m_s": 6000.0
}
```

or provided as a fixed global value:

```json
{
  "bedrock_velocity_mode": "fixed_global",
  "bedrock_velocity_m_s": 2400.0
}
```

Valid solved or fixed models require `V2 > V1`.

## T1LSST 1-Layer Components

When `conversion.mode` is `t1lsst_1layer`, the job writes
`refraction_t1lsst_1layer_components.csv` with IRAS-style 1-layer component
names. The formulas are:

```text
SH1 = T1 * V2 * V1 / sqrt(V2^2 - V1^2)
WCOR = SH1 * (1 / V2 - 1 / V1)
```

Equivalent SH1 form:

```text
SCOS1 = sqrt(1 - (V1 / V2)^2)
SH1 = T1 * V1 / SCOS1
```

For the usual case `V2 > V1`, `WCOR` is negative because replacing a low-velocity
weathering layer with a faster bedrock velocity removes delay.

The T1LSST component CSV includes one source or receiver endpoint row per output
component and includes `sign_convention` with:

```text
corrected(t) = raw(t - shift_s)
```

## V3/T2 and T1LSST 2-Layer Components

The implemented public two-layer path uses
`model.method="multilayer_time_term"` with enabled layers ordered as `v2_t1`
then `v3_t2`. Each layer solves:

```text
pick_time_s = Tn_source_s + Tn_receiver_s + offset_m / Vn_m_s
```

For `v3_t2`, `velocity_mode` may be `solve_global` or `fixed_global`. For
three-layer requests, `vsub_t3` is enabled after `v3_t2` and may also use
`solve_global` or `fixed_global`. Cell V3 and cell Vsub are not implemented.

For `conversion.mode="t1lsst_multilayer"` with `conversion.layer_count=2`, the
two-layer conversion computes:

```text
SH1 = T1 * V1 / sqrt(1 - (V1 / V2)^2)
SH2 = (T2 - SH1 * sqrt(1 - (V1 / V3)^2) / V1)
      * V2 / sqrt(1 - (V2 / V3)^2)
WCOR = SH1 * (1 / V3 - 1 / V1)
     + SH2 * (1 / V3 - 1 / V2)
```

The conversion requires `V2 > V1` and `V3 > V2`. Negative SH2 is status-coded
as `invalid_negative_thickness` and written as NaN rather than clipped. When V2
is cell-based, endpoint-local V2 is projected by endpoint cell ID; the
cell-velocity arrays are indexed by `cell_id`.

Two-layer results use the same source/receiver tables and NPZ artifacts as the
1-layer workflow, with added T2/V3/SH2 and interface-elevation fields. In
1-layer output, `refractor_elevation_m` is the V1/V2 interface:

```text
refractor_elevation_m = surface_elevation_m - SH1
```

In 2-layer `t1lsst_multilayer` output, legacy `weathering_thickness_m` values
represent the total replaced interval, and legacy `refractor_elevation_m` points
to the final V3 replacement interface:

```text
weathering_thickness_m = SH1 + SH2
layer1_base_elevation_m = surface_elevation_m - SH1
final_refractor_elevation_m = surface_elevation_m - SH1 - SH2
refractor_elevation_m = final_refractor_elevation_m
```

If SH2 is invalid and written as NaN, the total replaced interval and final
refractor elevation are also NaN. The explicit SH1 fields remain the computed
first-layer thickness.

`refraction_static_qc.json` and datum-result QC identify these outputs with:

```text
conversion_mode = t1lsst_multilayer
layer_count = 2
```

## Vsub/T3 Public Apply Contract

The public three-layer apply contract uses
`model.method="multilayer_time_term"` with enabled layers ordered as `v2_t1`,
`v3_t2`, then `vsub_t3`.

`conversion.mode="t1lsst_multilayer"` with `conversion.layer_count=3` requires
global V3 and global Vsub velocities. Cell V3 and cell Vsub remain unsupported.
The three-layer T1LSST formulas are supplied by the dependent multi-layer
conversion implementation and are not defined by this public apply contract.

`refraction_static_qc.json` and datum-result QC identify accepted three-layer
jobs with:

```text
conversion_mode = t1lsst_multilayer
layer_count = 3
enabled_layer_kinds = v2_t1, v3_t2, vsub_t3
```

## Source and Receiver Static Tables

`source_static_table.csv` has one row per source endpoint. Its source-specific
identifier columns are `source_endpoint_key`, `source_id`, and
`source_node_id`.

`receiver_static_table.csv` has one row per receiver endpoint. Its
receiver-specific identifier columns are `receiver_endpoint_key`, `receiver_id`,
and `receiver_node_id`.

Identifier columns:

| Column | Table | Definition |
|---|---|---|
| `source_endpoint_key` | `source_static_table.csv` | Source endpoint key used by geometry linkage and source-node assignment. |
| `source_id` | `source_static_table.csv` | Source station ID read from the configured source header byte. |
| `source_node_id` | `source_static_table.csv` | Near-surface node ID assigned to the source endpoint. |
| `receiver_endpoint_key` | `receiver_static_table.csv` | Receiver endpoint key used by geometry linkage and receiver-node assignment. |
| `receiver_id` | `receiver_static_table.csv` | Receiver station ID read from the configured receiver header byte. |
| `receiver_node_id` | `receiver_static_table.csv` | Near-surface node ID assigned to the receiver endpoint. |

Both CSV tables then share these component columns:

| Column | Definition |
|---|---|
| `endpoint_kind` | `source` or `receiver`. |
| `x_m`, `y_m` | Endpoint coordinates in meters after geometry scaling. |
| `surface_elevation_m` | Endpoint surface elevation in meters. |
| `floating_datum_elevation_m` | Floating datum elevation used for the endpoint. |
| `flat_datum_elevation_m` | Flat datum elevation used for the job, if configured. |
| `t1_ms` | Node T1 half-intercept time in milliseconds. |
| `v1_m_s` | Resolved global V1 weathering velocity. |
| `v2_m_s` | Solved or fixed global V2 refractor velocity. |
| `sh1_weathering_thickness_m` | SH1 first-layer thickness from the T1LSST conversion. |
| `refractor_elevation_m` | 1-layer: surface elevation minus SH1. Two-layer: final V3 replacement interface, equivalent to `final_refractor_elevation_m`. |
| `weathering_correction_ms` | WCOR in milliseconds. Usually negative when `V2 > V1`. |
| `floating_datum_correction_ms` | Floating datum correction component. |
| `flat_datum_correction_ms` | Flat datum correction component. |
| `elevation_correction_ms` | Sum of floating and flat datum correction components. |
| `total_static_ms` | Final endpoint static in the repo sign convention. |
| `total_applied_shift_ms` | Applied endpoint shift. In Phase 1 this matches `total_static_ms`. |
| `solution_status` | Node-level GLI solution status. |
| `weathering_status` | Node-level weathering conversion status. |
| `datum_status` | Endpoint datum component status. |
| `static_status` | Final endpoint static status. |
| `pick_count` | Number of observations associated with the node. |
| `used_pick_count` | Number of observations used after filtering or robust rejection. |
| `residual_rms_ms` | Node residual RMS in milliseconds. |
| `residual_mad_ms` | Node residual MAD in milliseconds. |

For two-layer `t1lsst_multilayer` results, the source and receiver CSV tables
also include:

| Column | Definition |
|---|---|
| `t2_ms` | T2 half-intercept time for the V3 branch in milliseconds. |
| `v3_m_s` | Global V3 replacement velocity used by the two-layer correction. |
| `sh2_weathering_thickness_m` | SH2 second-layer thickness from the two-layer T1LSST formula. |
| `layer1_base_elevation_m` | V1/V2 interface elevation, computed as surface elevation minus SH1. |
| `final_refractor_elevation_m` | Final V3 replacement interface elevation, computed as surface elevation minus SH1 minus SH2. |

`source_receiver_static_table.npz` stores the machine-readable source and
receiver tables. Time values in this NPZ are in seconds, and arrays are
pickle-free. Two-layer results add `source_t2_s`, `source_v3_m_s`,
`source_sh2_m`, `source_layer1_base_elevation_m`,
`source_final_refractor_elevation_m`, `receiver_t2_s`, `receiver_v3_m_s`,
`receiver_sh2_m`, `receiver_layer1_base_elevation_m`, and
`receiver_final_refractor_elevation_m`.

## Example Request

This request uses direct-arrival V1 estimation, solves global V2, writes the
T1LSST 1-layer component artifact, and exports source/receiver static tables
without registering a corrected TraceStore:

```json
{
  "file_id": "example",
  "pick_source": {
    "kind": "batch_predicted_npz",
    "job_id": "fbpick-job-id"
  },
  "linkage": {
    "mode": "required",
    "job_id": "linkage-job-id"
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
      "min_groups": 3
    },
    "bedrock_velocity_mode": "solve_global",
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
  "datum": {
    "mode": "floating_and_flat",
    "floating_datum_mode": "smoothed_topography",
    "flat_datum_elevation_m": 200.0
  },
  "apply": {
    "register_corrected_file": false
  }
}
```

Omitted request blocks and fields use the schema defaults in `app/api/schemas.py`.
For `pick_source.kind="batch_predicted_npz"`, the default artifact name is
`predicted_picks_time_s.npz`. For `linkage.mode="required"`, the default
artifact name is `geometry_linkage.npz`.

## Artifacts

Every refraction static job writes:

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
```

When `conversion.mode="t1lsst_1layer"`, the job also writes:

```text
refraction_t1lsst_1layer_components.csv
```

For two-layer `t1lsst_multilayer` results, the same core artifacts are used.
`near_surface_model.csv` adds explicit `sh1_weathering_thickness_m`,
`sh2_weathering_thickness_m`, `layer1_base_elevation_m`, and
`final_refractor_elevation_m` columns, while legacy
`weathering_thickness_m` stores SH1+SH2 and legacy `refractor_elevation_m`
stores the final V3 interface. `source_static_table.csv` and
`receiver_static_table.csv` add `t2_ms`, `v3_m_s`,
`sh2_weathering_thickness_m`, `layer1_base_elevation_m`, and
`final_refractor_elevation_m`; `source_receiver_static_table.npz` adds the
corresponding seconds/meter arrays. `refraction_static_solution.npz` adds
`node_sh2_weathering_thickness_m`, `node_layer1_base_elevation_m`,
`node_final_refractor_elevation_m`, `source_t2_time_s`, `source_v3_m_s`,
`source_sh1_weathering_thickness_m`, `source_sh2_weathering_thickness_m`,
`source_layer1_base_elevation_m`, `source_final_refractor_elevation_m`,
`receiver_t2_time_s`, `receiver_v3_m_s`,
`receiver_sh1_weathering_thickness_m`,
`receiver_sh2_weathering_thickness_m`,
`receiver_layer1_base_elevation_m`, and
`receiver_final_refractor_elevation_m`.

When `model.first_layer.mode="estimate_direct_arrival"`, the job also writes:

```text
refraction_v1_qc.json
refraction_v1_estimates.csv
```

When `apply.register_corrected_file=true`, the job can also write apply
artifacts such as:

```text
corrected_file.json
refraction_static_apply_qc.json
```

Trace-level final shifts are in `refraction_statics.csv`. Source and receiver
endpoint shifts are in `source_static_table.csv`, `receiver_static_table.csv`,
and `source_receiver_static_table.npz`.

## Current Scope and Non-Goals

Current supported scope:

```text
1-layer model
global V1
global V2
cell V2
T1 / SH1 / WCOR
two-layer V3/T2 solve
Vsub/T3 solve
public 2-layer V3/T2 apply
public 3-layer Vsub/T3 apply
public 2-layer T1LSST conversion
source static table
receiver static table
station table export
```

Out of scope:

```text
full IRAS compatibility
3-D refractor velocity grid
spatially varying V1 map
cell V3
cell Vsub
GRM / plus-minus method
tomostatics
SEG-Y header write-back
UI documentation
```
