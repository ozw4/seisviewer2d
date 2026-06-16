# Refraction QC Viewer Workflow

This page describes how to review refraction statics results with the M6 QC
artifacts and the viewer. It is a user-facing workflow guide for artifacts and
views defined by
[refraction_m6_qc_viewer_design.md](refraction_m6_qc_viewer_design.md).

The numerical model and limits remain the ones documented in:

- [refraction_iras_phase1_design.md](refraction_iras_phase1_design.md)
- [refraction_iras_phase2_cell_v2_design.md](refraction_iras_phase2_cell_v2_design.md)
- [refraction_multilayer_time_term.md](refraction_multilayer_time_term.md)
- [refraction_field_corrections.md](refraction_field_corrections.md)
- [refraction_m5_exports_table_workflow.md](refraction_m5_exports_table_workflow.md)

For a synthetic manual smoke-test workflow that ends in this QC view, see
[refraction_static_ui_fixture.md](refraction_static_ui_fixture.md).

## Sign Convention And Units

All refraction statics QC views use the repo static-shift convention:

```text
corrected(t) = raw(t - shift_s)
```

Therefore:

```text
shift_s > 0  -> event appears later in corrected data
shift_s < 0  -> event appears earlier in corrected data
```

Residuals use a separate first-break fit sign:

```text
residual_s = observed_first_break_time_s - modeled_first_break_time_s
```

Machine-readable API and NPZ time fields are seconds when the field name ends
in `_s`. CSV display fields are milliseconds when the field name ends in `_ms`.
Coordinates, elevations, depths, offsets, and thicknesses are meters (`m`).
Velocities are meters per second (`m/s`). Slowness is seconds per meter
(`s/m`).

## Generate QC Artifacts

Run a normal refraction static job with `POST /statics/refraction/apply`.
Completed jobs write the standard refraction artifact package, including:

```text
refraction_static_request.json
refraction_static_solution.npz
refraction_static_qc.json
refraction_statics.csv
near_surface_model.csv
first_break_residuals.csv
refraction_first_break_time_export.csv
refraction_static_components.csv
source_static_table.csv
receiver_static_table.csv
source_receiver_static_table.npz
refraction_time_term_spreadsheet.csv
refraction_static_history.json
refraction_static_artifacts.json
```

Successful apply jobs also write QC artifact families for the viewer:

```text
refraction_first_break_fit_qc.csv
refraction_first_break_fit_qc.npz
refraction_first_break_fit_qc.json
refraction_reduced_time_qc.csv
refraction_reduced_time_qc.npz
refraction_reduced_time_qc.json
refraction_line_profile_qc_source.csv
refraction_line_profile_qc_receiver.csv
refraction_line_profile_qc_combined.csv
refraction_line_profile_qc.npz
refraction_line_profile_qc.json
refraction_static_component_qc_trace.csv
refraction_static_component_qc_endpoint.csv
refraction_static_component_qc.npz
refraction_static_component_qc.json
```

Cell velocity jobs also write the cell artifacts and grid-map QC artifacts used
by map QC:

```text
refraction_refractor_velocity_cells.csv
refraction_refractor_velocity_grid.npz
refraction_refractor_velocity_qc.json
refraction_cell_solver_history.csv
refraction_grid_map_qc.csv
refraction_grid_map_qc.npz
refraction_grid_map_qc.json
```

If `model.first_layer.mode="estimate_direct_arrival"`, review
`refraction_v1_qc.json` and `refraction_v1_estimates.csv` before interpreting
the refraction fit. If field corrections are enabled, also review
`refraction_source_depth_qc.json`, `refraction_uphole_qc.json`, and the manual
static status columns in the source and receiver tables.

## Open The Refraction QC View

In the browser viewer, open `/`, select the `Refraction QC` tab, enter the
completed refraction `job_id`, and load the bundle. The page requests:

```http
POST /statics/refraction/qc
```

Example body:

```json
{
  "job_id": "refraction-job-123",
  "include": ["summary", "first_break", "reduced_time", "profiles", "cells", "static_components"],
  "max_points": 20000,
  "coordinate_mode": "auto"
}
```

The bundle returns the job summary, sign convention, artifact manifest,
available view names, unavailable view names, and sampled tabular records. The
logical M6 artifact families are `first_break_fit`, `reduced_time_lmo`,
`profile_2d`, `map_3d`, `static_composition`, and `gather_preview`. The
viewer surfaces are:

| View | Main artifacts |
|---|---|
| Summary | `refraction_static_qc.json`, manifest entries |
| First-break fit | `refraction_first_break_fit_qc.csv`, `first_break_residuals.csv` |
| Reduced-time / LMO | `refraction_reduced_time_qc.csv` |
| 2D profiles | `refraction_line_profile_qc_combined.csv` |
| 3D cell maps | `refraction_grid_map_qc.csv`, `refraction_refractor_velocity_cells.csv` |
| Static components | `refraction_static_component_qc_endpoint.csv`, `refraction_static_component_qc_trace.csv` |
| Gather preview | Bounded raw/corrected trace windows and overlays from `POST /statics/refraction/qc/gather-preview` |

The compact bundle route provides sampled tabular QC records and manifest
metadata. The gather-preview route is the dedicated M6 endpoint for bounded
seismic samples because preview windows can be larger than the compact bundle.
When a plotted endpoint or cell needs non-seismic details, use the sampled
bundle rows and the referenced artifacts to inspect time terms, velocities,
thicknesses, static components, pick counts, statuses, fold, residuals, and
contributing observations.

## Complete Synthetic Workflow

This small example uses synthetic IDs only. It does not require real SEG-Y data
in the documentation.

1. Start with an uploaded/opened TraceStore, a first-break job, and a geometry
   linkage job:

```text
file_id = synthetic-line-file
pick job_id = synthetic-first-breaks
linkage job_id = synthetic-linkage
```

2. Run a one-layer refraction job and ask it to register a corrected file:

```json
{
  "file_id": "synthetic-line-file",
  "pick_source": {
    "kind": "batch_predicted_npz",
    "job_id": "synthetic-first-breaks"
  },
  "linkage": {
    "mode": "required",
    "job_id": "synthetic-linkage"
  },
  "model": {
    "method": "gli_variable_thickness",
    "first_layer": {
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
    },
    "bedrock_velocity_mode": "solve_cell",
    "initial_bedrock_velocity_m_s": 2400.0,
    "min_bedrock_velocity_m_s": 1200.0,
    "max_bedrock_velocity_m_s": 5000.0,
    "refractor_cell": {
      "number_of_cell_x": 4,
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
  "moveout": {
    "model": "head_wave_linear_offset",
    "distance_source": "geometry",
    "min_offset_m": 300.0
  },
  "conversion": {
    "mode": "t1lsst_1layer"
  },
  "apply": {
    "register_corrected_file": true
  }
}
```

3. Poll the job until `/statics/job/<job_id>/status` reports completion, then
   list artifacts with `/statics/job/<job_id>/files`.

4. Open `/`, select `Refraction QC`, enter the refraction job ID, and load the
   bundle.

5. Review in this order:
   first-break fit residuals, reduced-time / LMO, 2D profiles, cell map, static
   component waterfall, then before/after gather preview if the viewer reports
   that preview as available.

6. If a few traces or a cell look suspicious, filter the endpoint or cell rows
   in the bundle and referenced artifacts. If a corrected TraceStore was
   registered, compare the same gather or section in the normal viewer.

## First-Break Fit Residuals

First-break fit QC compares the picked arrival with the fitted model time. Use
the residual scatter and histogram to decide whether errors are global,
localized, offset-dependent, or tied to rejected picks.

Synthetic observation example:

| offset_m | observed_first_break_time_s | modeled_first_break_time_s | residual_time_ms | status |
|---:|---:|---:|---:|---|
| 500.0 | 0.312 | 0.307 | 5.0 | `ok` |
| 1000.0 | 0.523 | 0.530 | -7.0 | `ok` |
| 1500.0 | 0.730 | 0.690 | 40.0 | `rejected` |

Read this as: the first pick is 5 ms later than the model, the second is 7 ms
earlier than the model, and the third is not part of the solve. A pattern where
near offsets are positive and far offsets are negative usually points to a
velocity or gate problem. A tight cluster of large residuals around one source,
receiver, or cell usually points to local geometry, pick, or cell issues.

## Reduced-Time / LMO

Reduced-time QC subtracts linear moveout from first-break times:

```text
reduced_time_s = observed_first_break_time_s - offset_m / reduction_velocity_m_s
modeled_reduced_time_s = modeled_first_break_time_s - offset_m / reduction_velocity_m_s
```

Use this view to check whether each layer gate is close to a flat trend after
subtracting the selected reduction velocity. The gate overlays come from the
request and solution metadata for direct-arrival V1, V2/T1, V3/T2, and
Vsub/T3.

Synthetic LMO example:

| layer_kind | offset_m | observed_first_break_time_s | reduction_velocity_m_s | reduced_time_ms |
|---|---:|---:|---:|---:|
| `v2_t1` | 500.0 | 0.310 | 2500.0 | 110.0 |
| `v2_t1` | 1000.0 | 0.510 | 2500.0 | 110.0 |
| `v2_t1` | 1500.0 | 0.712 | 2500.0 | 112.0 |

These rows are internally consistent because the reduced times are nearly flat.
If reduced time tilts upward with offset, the reduction velocity is too fast for
that branch or the picks include another layer. If it tilts downward, the
reduction velocity is too slow.

## 2D Line Profiles

The 2D profile view plots endpoint and cell quantities against `inline_m`. Use
it for line-oriented workflows, especially `coordinate_mode="line_2d_projected"`.
For `grid_3d`, a profile is valid only when the viewer or API states how the
line, swath, or sorted axis was chosen.

Inspect these tracks together:

- T1, T2, T3 time terms in ms or s.
- V1, V2, V3, and Vsub in m/s.
- SH1, SH2, SH3, elevations, and refractor interfaces in m.
- Weathering, datum/elevation, field, manual, and total applied shifts.
- Residual RMS/MAD and pick fold.

Small synthetic endpoint profile:

| inline_m | endpoint_kind | t1_ms | v2_m_s | sh1_m | weathering_correction_ms | total_applied_shift_ms |
|---:|---|---:|---:|---:|---:|---:|
| 0.0 | `source` | 12.0 | 2400.0 | 10.5 | -8.2 | -10.0 |
| 500.0 | `source` | 13.5 | 2500.0 | 11.2 | -8.6 | -10.4 |
| 1000.0 | `source` | 25.0 | 1800.0 | 23.8 | -25.0 | -29.0 |

The third row is a local anomaly: time term, thickness, and total shift all
depart together. Check its residuals, fold, V2 status, and neighboring cells
before deciding whether it is geology, geometry, or a bad pick cluster.

## 3D Cell Velocity, Fold, And Residual Maps

Use cell maps for `solve_cell` V2 workflows. The map is plotted by cell `ix`,
`iy`, and the recorded coordinate mode:

- `grid_3d` uses map X/Y coordinates directly and can produce a true 2D map
  when `number_of_cell_y > 1`.
- `line_2d_projected` stores projected inline/crossline coordinates and usually
  appears as a one-cell-high strip or line profile.

Review maps in this order:

1. `status`: find `low_fold`, `inactive`, or unsolved areas first.
2. `fold` and `used_fold`: confirm solved cells have enough observations.
3. `cell_velocity_m_s`: look for isolated spikes or velocity reversals.
4. `residual_rms_s`, `residual_mad_s`, and `residual_p95_abs_s`: find cells
   where the model fits poorly.

Synthetic cell map example:

| cell_id | ix | iy | n_observations | n_used_observations | velocity_m_s | status |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 24 | 23 | 2400.0 | `solved` |
| 1 | 1 | 0 | 3 | 0 |  | `low_fold` |
| 2 | 2 | 0 | 0 | 0 |  | `inactive` |
| 3 | 3 | 0 | 28 | 27 | 2475.0 | `solved` |

Do not interpolate a solved velocity into `low_fold` or `inactive` cells during
QC. Empty and low-fold cells are intentionally blank so they cannot masquerade
as valid velocity estimates.

## Static Component Waterfalls

The static component view explains how endpoint and trace shifts were assembled.
Positive bars are positive `shift_s` values, so they delay the displayed event
after correction.

Trace-level composition is:

```text
trace_refraction_shift_s =
  source_total_applied_shift_s + receiver_total_applied_shift_s

trace_field_shift_s =
  source_field_shift_s + receiver_field_shift_s

final_trace_shift_s when field corrections are applied =
  trace_refraction_shift_s + trace_field_shift_s

final_trace_shift_s when field corrections are artifact-only =
  trace_refraction_shift_s
```

Example waterfall in milliseconds:

| component | shift_ms |
|---|---:|
| source weathering/elevation | -8.0 |
| receiver weathering/elevation | -6.5 |
| source depth | 4.0 |
| uphole | -2.0 |
| manual source static | 1.5 |
| manual receiver static | -0.5 |
| final trace shift | -11.5 |

This final shift is negative, so the corrected event should appear earlier. If
the corrected gather moves later instead, check for a sign inversion in an
external table before importing or applying it.

## Before/After Gather Preview

Before/after gather preview is the M6 concept for comparing raw and corrected
samples for the same section, source gather, or receiver gather. The preview
should align overlays for observed picks, modeled picks, residuals,
reduced-time picks, and the final trace-shift curve with the trace window.

The compact QC bundle does not embed gather-preview windows. The viewer uses
the dedicated bounded route:

```text
POST /statics/refraction/qc/gather-preview
```

A minimal source-gather request carries:

```json
{
  "job_id": "refraction-job-123",
  "file_id": "synthetic-line-file",
  "gather_axis": "source",
  "endpoint_key": "source:1001",
  "time_start_s": 0.0,
  "time_end_s": 1.5,
  "key1_byte": 189,
  "key2_byte": 193,
  "scaling": "amax",
  "max_traces": 200
}
```

Use raw/corrected side-by-side inspection when the refraction job registered a
corrected TraceStore. If no corrected store was registered, the preview still
returns bounded raw samples and overlay metadata with a corrected-data status
that explains why corrected samples are unavailable.

Check that:

- observed and modeled picks move with the corrected traces;
- high residual traces are visible and explainable;
- the final trace-shift curve has the expected sign and smoothness;
- correction improves event alignment in the target gate without damaging other
  offsets.

## Status Values

Statuses are machine-readable strings. Use them as the source of truth rather
than inferring validity from color alone.

| Status | Where it appears | Meaning |
|---|---|---|
| `ok` | Observation, endpoint, component, or apply rows | Valid for the relevant product. |
| `rejected` | First-break fit rows | The pick was excluded from the solve; inspect `rejection_reason`, for example `robust_outlier` or `below_min_observations_per_cell`. |
| `solved` | Cell velocity rows | The cell is active and has a solved velocity. |
| `low_fold` | Cell velocity rows | The cell had observations but fewer than `min_observations_per_cell`; no velocity is solved. |
| `inactive` | Cell velocity or node rows | No active observations or the endpoint/cell is not active in the solve. |
| `empty` | Summary language or troubleshooting shorthand | No rows or no observations are available for the requested product; cell artifacts usually record empty cells as `inactive` with zero observations. |
| `outside_refractor_cell_grid` | Observation or endpoint V2 status | Coordinates fell outside the configured cell grid and were rejected rather than clipped. |
| `inactive_v2_cell` | Endpoint V2 status | The endpoint projects into a cell without a solved V2. |
| `low_fold_v2_cell` | Endpoint V2 status | The endpoint projects into a low-fold cell. |
| `v2_not_greater_than_v1` | Endpoint V2/weathering status | Local V2 is not greater than V1, so T1LSST conversion is invalid. |
| `invalid_velocity_order` | Weathering or multilayer status family | Layer velocities do not satisfy the required order, for example V2 > V1 or V3 > V2. |
| `invalid_negative_thickness` | Weathering status | T1LSST produced a negative SH value; the numeric thickness is written as NaN. |
| `missing_manual_static` | Manual static status | A required manual source or receiver static was not supplied. |
| `not_enabled` | Field-correction component status | The component was disabled and contributes zero shift. |
| `not_registered` | Gather-preview concept or corrected-window reference | No corrected TraceStore was registered for the job. |

## Troubleshooting

Residuals are high everywhere:
Check first-break pick units, `dt`, geometry linkage, offset units, and layer
gates. In reduced-time view, a consistent tilt usually means the selected layer
velocity or gate is wrong. If V1 was estimated, review `refraction_v1_qc.json`
for insufficient or biased direct-arrival groups.

Residuals are localized in one cell:
Filter the cell rows and inspect contributing source and receiver endpoints.
Look for low fold, outside-grid rejection, a bad coordinate projection, or a
cluster of robust outliers. Compare the cell residual map with the first-break
fit scatter filtered to that cell.

V2 map has spikes:
First check fold and status. Isolated spikes in low-fold neighborhoods are
usually underconstrained. If fold is good, compare residual RMS/MAD and
neighbor velocities. Consider whether the configured cell size is too small or
whether smoothing should be part of a new solve.

Low-fold cells dominate:
The grid is too fine for the pick density, the selected offset gate is too
narrow, coordinates fall outside the grid, or `min_observations_per_cell` is
too high for the survey. Do not treat low-fold cells as solved cells.

Static profile oscillates:
Compare the static profile with T1, V2, residual RMS/MAD, and pick fold. Rapid
oscillation with low fold suggests an unstable solve. Oscillation that matches
surface elevation may be datum or source-depth related. Oscillation only after
manual statics points to the imported manual table.

Corrected gather looks worse:
Confirm the sign convention and units first. A millisecond table imported as
seconds, or a delay-positive table treated as applied shift, can overcorrect
severely. Then check whether the preview uses the intended corrected file,
whether field corrections were applied to trace shifts or only written as
artifact components, and whether the selected gather is outside the modeled
offset gate.

## Non-Goals

The M6 QC viewer workflow does not implement:

- GRM, plus-minus, refraction tomography, or path-integrated cell slowness.
- New solver math or changes to T1LSST formulas.
- SEG-Y static header write-back.
- Browser controls for editing refraction cell models.
- Pick editing from the QC view.
- Spatially varying V1 maps.
- Public T1LSST apply with cell V3/T2 or cell Vsub/T3.
- Automatic 2D line origin or azimuth estimation.
- Full IRAS compatibility or original IRAS manual content.

## Limitations

- M6 QC does not change the solver or recompute statics from viewer edits.
- The compact QC bundle samples large tables independently by view.
- Gather preview intentionally uses a dedicated bounded API instead of embedding
  heavy seismic windows in the compact QC bundle.
- Public T1LSST apply supports cell V2/T1, but public cell V3/T2 and cell
  Vsub/T3 apply remain out of scope.
- Cell velocity assignment is midpoint-cell based, not path-integrated
  tomography.
- SEG-Y static header write-back is not implemented.
- External tables must use the documented repo sign convention before apply.
