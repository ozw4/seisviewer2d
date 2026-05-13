# Refraction M6 QC and viewer integration design

## 1. Purpose

This document defines the M6 refraction QC and viewer integration contract.
It is a design target for future backend and frontend issues; it does not
change the numerical solver, existing artifacts, or viewer code by itself.

Read this document with the existing implementation references:

- `docs/statics/refraction_iras_phase1_design.md`
- `docs/statics/refraction_iras_phase2_cell_v2_design.md`
- `docs/statics/refraction_multilayer_time_term.md`
- `docs/statics/refraction_field_corrections.md`
- `docs/statics/refraction_m5_exports_table_workflow.md`

M6 is a product and data-contract layer over the Phase 1 through M5 artifacts.
The goal is to make first-break fit QC, reduced-time QC, profile plots, maps,
static composition, and before/after gather preview independently implementable
without changing the solved statics math.

## 2. Existing artifact baseline

M6 must reuse the current artifact package before adding new artifacts. Current
public refraction jobs can write:

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

The Phase 1 / M1 one-layer T1LSST workflow also writes:

```text
refraction_t1lsst_1layer_components.csv
refraction_v1_qc.json
refraction_v1_estimates.csv
```

The Phase 2 / M2-M2.5 cell V2 workflow also writes:

```text
refraction_refractor_velocity_cells.csv
refraction_refractor_velocity_grid.npz
refraction_refractor_velocity_qc.json
refraction_cell_solver_history.csv
```

The M3 multi-layer workflow extends the endpoint and solution artifacts with
`T2`, `T3`, `V3`, `Vsub`, `SH2`, and `SH3` fields. Internal cell-layer writer
names already exist for:

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

Those V3/Vsub cell artifacts are internal solver outputs. Public
`t1lsst_multilayer` apply currently supports cell maps only for V2/T1.

M4 field-correction artifacts and columns are part of the QC contract:

```text
refraction_source_depth_qc.json
refraction_source_depth_sources.csv
refraction_uphole_qc.json
refraction_uphole_sources.csv
```

Manual statics do not add a separate public artifact. They are represented in
`source_static_table.csv`, `receiver_static_table.csv`,
`source_receiver_static_table.npz`, `refraction_static_solution.npz`,
`refraction_static_qc.json`, and `refraction_static_history.json`.

M5 export/import artifacts are external review and apply views over the same
model:

```text
canonical_source_static_table.csv
canonical_receiver_static_table.csv
canonical_source_receiver_static_table.csv
refraction_lsst_plus.csv
refraction_lsst_plus_cards.txt
```

The M5 `first_break_time` export is an observation-level view derived from
`first_break_residuals.csv` and solution observation arrays.

## 3. Sign convention

Every M6 artifact, API response, axis label, hover value, and viewer overlay
must use the repo static-shift convention:

```text
corrected(t) = raw(t - shift_s)
```

Therefore:

```text
shift_s > 0  -> event appears later in corrected data
shift_s < 0  -> event appears earlier in corrected data
```

Do not silently invert this convention for QC labels or display-only fields.
If a future UI exposes a delay-positive value, the field name must include
`delay` and the response must also expose the converted `*_shift_s` value in
the repo convention.

Current endpoint tables keep:

```text
total_static_ms == total_applied_shift_ms
```

Trace-level apply artifacts use the same convention for
`refraction_trace_shift_s_sorted`, `trace_field_shift_s_sorted`, and
`final_trace_shift_s_sorted`.

## 4. Data model conventions

Future M6 APIs should be thin, typed views over existing artifacts.

- Machine-readable API time and shift values should use seconds in fields
  ending `_s`; display-facing CSV exports may continue to use milliseconds in
  fields ending `_ms`.
- Coordinates, offsets, inline distances, elevations, and thicknesses use
  meters.
- Velocities use meters per second.
- Observation rows must retain `observation_index`, `sorted_trace_index`,
  `source_endpoint_key`, `receiver_endpoint_key`, and available source/receiver
  station IDs.
- Endpoint rows must retain `endpoint_kind`, `endpoint_key`, endpoint ID,
  node ID, coordinates, static status, and sign convention.
- Status values must remain machine-readable strings. The viewer maps statuses
  to visual classes, but the API remains the source of truth.
- Large dense arrays should be returned by paged or binary endpoints where
  practical. A bundle endpoint should primarily return manifests, summaries,
  small samples, artifact references, and URLs for detailed products.

The implemented compact bundle endpoint is:

```text
POST /statics/refraction/qc
```

with a JSON body:

```json
{
  "job_id": "refraction-job-123",
  "include": ["summary", "first_break", "profiles", "cells", "static_components"],
  "max_points": 20000,
  "coordinate_mode": "auto"
}
```

The current M6 API contract is the bundle route above. Specialized detail
endpoints may be added later, but this document does not define callable route
names for them.

Logical M6 product family names are:

| Family | Product kind | Default detail shape |
|---|---|---|
| `first_break_fit` | observation table | paged JSON rows or CSV view |
| `reduced_time_lmo` | observation table with display math | paged JSON rows or CSV view |
| `profile_2d` | endpoint/cell line profile | JSON rows for plotted tracks |
| `map_3d` | cell grid/map | JSON metadata plus dense grid or cell rows |
| `static_composition` | endpoint and trace component audit | JSON rows and summaries |
| `gather_preview` | trace-window references plus overlays | JSON metadata plus existing window endpoints |

If a future backend materializes any family as a stored artifact, the stored
artifact should use the same family name and fields defined here. The API shape
remains the viewer contract; materialized files are an implementation choice.

## 5. First-break fit QC

First-break fit QC is an observation-level product. It compares picked
first-break times to the model time used by the solver.

Required fields:

```text
observation_index
sorted_trace_index
source_endpoint_key
receiver_endpoint_key
source_id
receiver_id
offset_m
inline_distance_m
observed_time_s
modeled_time_s
residual_s
used_in_solve
reject_reason
layer_kind
sign_convention
```

`inline_distance_m` is optional for true `grid_3d` map workflows and required
when the product is shown as a 2D line profile. For multi-layer workflows,
`layer_kind` must identify `v2_t1`, `v3_t2`, or `vsub_t3` where assignment is
known.

Existing artifact inputs:

- `first_break_residuals.csv`
- `refraction_first_break_time_export.csv`
- `refraction_static_solution.npz`
- `refraction_static_request.json`

Viewer use:

- residual vs offset scatter;
- observed vs modeled first-break time scatter;
- residual histogram and RMS/MAD summaries;
- map/profile brushing by source endpoint, receiver endpoint, layer, status,
  and reject reason;
- overlay of observed and modeled picks on gather preview panels.

Residual sign is:

```text
residual_s = observed_time_s - modeled_time_s
```

This residual sign is independent of the static-shift sign convention, but the
response must still carry the static sign convention because fit QC is usually
shown alongside static components.

## 6. Reduced-time / LMO QC

Reduced-time QC shows first-break picks after subtracting a selected moveout
velocity:

```text
reduced_time_s = observed_time_s - offset_m / reduction_velocity_m_s
```

For modeled picks:

```text
modeled_reduced_time_s = modeled_time_s - offset_m / reduction_velocity_m_s
```

Required fields:

```text
observation_index
sorted_trace_index
source_endpoint_key
receiver_endpoint_key
offset_m
layer_kind
reduction_velocity_m_s
observed_time_s
modeled_time_s
reduced_time_s
modeled_reduced_time_s
used_in_solve
reject_reason
```

The reduction velocity can be:

- a user-selected global display value;
- the solved or fixed global velocity for a layer;
- the endpoint-local or cell velocity for V2/T1 cell workflows when the viewer
  explicitly requests a layer-specific view.

Offset gates must be reported from the request and solution metadata:

```text
v1_direct_arrival: min_direct_offset_m, max_direct_offset_m
v2_t1: min_offset_m, max_offset_m
v3_t2: min_offset_m, max_offset_m
vsub_t3: min_offset_m, max_offset_m
```

Existing artifact inputs:

- `refraction_static_request.json`
- `first_break_residuals.csv`
- `refraction_first_break_time_export.csv`
- `refraction_static_solution.npz`
- `refraction_v1_qc.json` and `refraction_v1_estimates.csv` when V1 is
  estimated from direct arrivals

Viewer use:

- reduced-time gather or scatter panels;
- LMO line overlays by layer;
- visual gate bands for V1, V2/T1, V3/T2, and Vsub/T3;
- outlier and rejected-pick inspection without rerunning the solver.

Reduced-time QC is display math over existing observations. It must not modify
pick times, solver inputs, or stored static tables.

## 7. 2D profile QC

2D profile QC is an endpoint or cell profile plotted against inline distance.
It is distinct from a 3D map: the independent axis is one-dimensional
`inline_m`, not map X/Y.

Required profile families:

```text
T1, T2, T3
V1, V2, V3, Vsub
SH1, SH2, SH3
weathering_correction_s
elevation_correction_s
source_depth_shift_s
uphole_shift_s
manual_static_shift_s
source_field_shift_s
receiver_field_shift_s
refraction_trace_shift_s
trace_field_shift_s
final_trace_shift_s
residual_rms_s
residual_mad_s
pick_fold
used_pick_fold
```

Endpoint profile row shape:

```text
endpoint_kind
endpoint_key
endpoint_id
node_id
inline_m
x_m
y_m
surface_elevation_m
t1_s
t2_s
t3_s
v1_m_s
v2_m_s
v3_m_s
vsub_m_s
sh1_m
sh2_m
sh3_m
weathering_correction_s
elevation_correction_s
manual_static_shift_s
field_shift_s
total_applied_shift_s
static_status
residual_rms_s
residual_mad_s
pick_fold
used_pick_fold
```

For source rows, `field_shift_s` is `source_field_shift_s`. For receiver rows,
it is `receiver_field_shift_s`.

Existing artifact inputs:

- `source_static_table.csv`
- `receiver_static_table.csv`
- `source_receiver_static_table.npz`
- `refraction_time_term_spreadsheet.csv`
- `refraction_t1lsst_1layer_components.csv`
- `refraction_static_solution.npz`
- `refraction_static_components.csv`

Viewer use:

- synchronized source and receiver line plots;
- endpoint static component tracks;
- time-term and velocity tracks by layer;
- weathering thickness and refractor elevation tracks;
- residual RMS/MAD and fold tracks for QC triage.

For `coordinate_mode="line_2d_projected"`, `inline_m` comes from the manual
line projection defined by `line_origin_x_m`, `line_origin_y_m`, and
`line_azimuth_deg`. For `grid_3d`, a 2D profile requires an explicit selected
line, swath, or sorted endpoint axis; the API must state how `inline_m` was
derived and must not imply that a 3D grid is a single line.

## 8. 3D grid/map QC

3D grid/map QC is a cell product plotted in map X/Y or projected cell
coordinates. It is distinct from a 2D line profile.

Required map families:

```text
cell_velocity_m_s
cell_slowness_s_per_m
fold
used_fold
rejected_fold
residual_rms_s
residual_mad_s
residual_mean_s
residual_p95_abs_s
status
```

Cell row shape:

```text
layer_kind
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
velocity_m_s
slowness_s_per_m
velocity_status
residual_rms_s
residual_mad_s
residual_mean_s
residual_p95_abs_s
smoothing_neighbor_count
```

Existing artifact inputs:

- `refraction_refractor_velocity_cells.csv`
- `refraction_refractor_velocity_grid.npz`
- `refraction_refractor_velocity_qc.json`
- `refraction_cell_solver_history.csv`
- internal V3/Vsub cell artifact names when future public workflows expose
  those maps

Coordinate mode handling:

- `grid_3d` uses input map X/Y coordinates directly and can be rendered as a
  2D map when `number_of_cell_y > 1`.
- `line_2d_projected` converts map coordinates to manual inline/crossline
  coordinates. Public cell artifacts remain indexed by `cell_id`, `ix`, and
  `iy`; because line mode requires `number_of_cell_y=1`, the viewer should
  render it as a 2D profile or one-cell-high strip rather than a geographic
  area map.
- The API must report `coordinate_mode`, grid shape, cell size, origin, and
  line projection metadata exactly as recorded in
  `refraction_refractor_velocity_qc.json`.

Viewer use:

- velocity heatmap for solved cells;
- fold and used-fold maps;
- residual RMS/MAD maps;
- status map for `solved`, `inactive`, `low_fold`, and outside-grid rejection
  counts;
- cell inspection linking back to first-break observations and endpoint
  profile rows.

## 9. Static composition QC

Static composition QC explains how endpoint and trace shifts were assembled.
It is both an endpoint product and a trace-preview product.

Required component names:

```text
refraction_shift_s
weathering_correction_s
datum_correction_s
field_correction_s
manual_static_shift_s
final_applied_trace_shift_s
```

Current artifacts expose datum effects mainly through floating datum, flat
datum, and aggregate `elevation_correction_*` fields. M6 viewer labels may group
those fields as datum/elevation correction, but underlying artifact fields must
keep their documented names.

Concrete current fields:

```text
source_total_applied_shift_s
receiver_total_applied_shift_s
source_weathering_correction_s
receiver_weathering_correction_s
source_elevation_correction_s
receiver_elevation_correction_s
source_depth_shift_s
source_uphole_shift_s
source_manual_static_shift_s
receiver_manual_static_shift_s
source_field_shift_s
receiver_field_shift_s
refraction_trace_shift_s_sorted
trace_field_shift_s_sorted
final_trace_shift_s_sorted
```

Composition formulas:

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

Existing artifact inputs:

- `source_static_table.csv`
- `receiver_static_table.csv`
- `source_receiver_static_table.npz`
- `refraction_static_solution.npz`
- `refraction_statics.csv`
- `refraction_static_components.csv`
- `refraction_static_history.json`
- `refraction_static_qc.json`
- `refraction_source_depth_qc.json`
- `refraction_uphole_qc.json`

Viewer use:

- source and receiver component bar charts;
- trace-level final-shift profile;
- warnings when field components are present but not included in the applied
  trace shift;
- audit display of static history, cumulative shift field, and double-
  application guard result.

Every static composition view must show or expose the sign convention string.
Positive bars represent positive `shift_s`, which delays the displayed event
after correction under `corrected(t) = raw(t - shift_s)`.

## 10. Before/after gather preview

Before/after gather preview shows raw traces, corrected traces, and pick/model
overlays for the same gather or section window.

Required products:

```text
raw_gather
corrected_gather
observed_first_break_overlay
modeled_first_break_overlay
optional_lmo_or_reduced_time_overlay
static_shift_trace_curve
```

Preview request shape:

```text
job_id
file_id
key1
key1_byte
key2_byte
gather_axis
x0
x1
y0
y1
step_x
step_y
scaling
reduction_velocity_m_s
overlay_layers
```

Preview response shape:

```text
raw_window_ref
corrected_window_ref
dt_s
shape
trace_indices
offset_m
source_endpoint_key
receiver_endpoint_key
observed_pick_time_s
modeled_pick_time_s
residual_s
final_trace_shift_s
reduced_observed_time_s
reduced_modeled_time_s
```

The gather preview should reuse existing TraceStore section-window behavior for
heavy trace I/O. A QC bundle should return references and overlay arrays, not
embed full seismic windows by default.

Existing artifact inputs:

- original TraceStore window data;
- corrected TraceStore when `apply.register_corrected_file=true`;
- `refraction_static_solution.npz`;
- `refraction_statics.csv`;
- `first_break_residuals.csv`;
- `refraction_first_break_time_export.csv`;
- `corrected_file.json`;
- `refraction_static_apply_qc.json`.

Viewer use:

- side-by-side raw/corrected gather;
- optional reduced-time or LMO display;
- observed and modeled first-break curves;
- trace-shift curve aligned to the gather;
- click-through from residual outliers to trace preview.

If a job did not register a corrected TraceStore, the preview endpoint may
return only overlay and shift arrays plus a `corrected_window_ref` status of
`not_registered`. It must not synthesize a hidden fallback TraceStore.

## 11. QC bundle response example

The implemented bundle endpoint provides a compact manifest plus sampled
tabular views. Its top-level response shape is the
`RefractionStaticQcBundleResponse` contract: `job_id`, `statics_kind`,
`sign_convention`, `coordinate_mode`, `summary`, `artifacts`,
`available_views`, `unavailable_views`, `views`, and `downsampling`.

`sign_convention` is the repo convention string, not an object:
`corrected(t)=raw(t-shift_s)`. The route does not currently return
`schema_version`, `kind`, `source_artifacts`, or `products` fields.

```json
{
  "job_id": "refraction-job-123",
  "statics_kind": "refraction",
  "sign_convention": "corrected(t)=raw(t-shift_s)",
  "coordinate_mode": "line_2d_projected",
  "summary": {
    "status": "ok",
    "job_state": "ready",
    "method": "multilayer_time_term",
    "conversion_mode": "t1lsst_multilayer",
    "layer_count": 2,
    "enabled_layer_kinds": ["v2_t1", "v3_t2"],
    "first_break_fit": {
      "residual_sign": "observed - modeled",
      "used_observations": 17290,
      "residual_rms_s": 0.0124,
      "residual_mad_s": 0.0071
    },
    "observation_gates": [
      {
        "layer_kind": "v2_t1",
        "min_offset_m": 300.0,
        "max_offset_m": 1800.0
      },
      {
        "layer_kind": "v3_t2",
        "min_offset_m": 1800.0,
        "max_offset_m": null
      }
    ]
  },
  "artifacts": {
    "first_break_residuals": "first_break_residuals.csv",
    "refraction_first_break_fit_qc": "refraction_first_break_fit_qc.csv",
    "refraction_first_break_time_export": "refraction_first_break_time_export.csv",
    "refraction_refractor_velocity_cells": "refraction_refractor_velocity_cells.csv",
    "refraction_static_artifacts": "refraction_static_artifacts.json",
    "refraction_static_components": "refraction_static_components.csv",
    "refraction_static_qc": "refraction_static_qc.json",
    "refraction_static_request": "refraction_static_request.json",
    "source_static_table": "source_static_table.csv",
    "receiver_static_table": "receiver_static_table.csv"
  },
  "available_views": [
    "summary",
    "first_break_fit",
    "static_components"
  ],
  "unavailable_views": ["profiles", "cells", "gather_preview"],
  "views": {
    "first_break_fit": {
      "artifact": "refraction_first_break_fit_qc.csv",
      "columns": [
        "trace_index_sorted",
        "source_endpoint_key",
        "receiver_endpoint_key",
        "offset_m",
        "observed_first_break_time_s",
        "modeled_first_break_time_s",
        "residual_time_s",
        "residual_time_ms",
        "layer_kind"
      ],
      "total_points": 18420,
      "returned_points": 18420,
      "downsampled": false,
      "downsampling_method": "even_index_floor_first_last",
      "records": [
        {
          "trace_index_sorted": "0",
          "source_endpoint_key": "1001",
          "receiver_endpoint_key": "2101",
          "offset_m": "625.0",
          "observed_first_break_time_s": "0.284",
          "modeled_first_break_time_s": "0.279",
          "residual_time_s": "0.005",
          "residual_time_ms": "5.0",
          "layer_kind": "v2_t1"
        }
      ]
    },
    "static_components": {
      "artifact": "refraction_static_components.csv",
      "columns": [
        "trace_index_sorted",
        "refraction_trace_shift_s",
        "trace_field_shift_s",
        "final_trace_shift_s"
      ],
      "total_points": 18420,
      "returned_points": 18420,
      "downsampled": false,
      "downsampling_method": "even_index_floor_first_last",
      "records": []
    }
  },
  "downsampling": {
    "first_break_fit": {
      "total_points": 18420,
      "returned_points": 18420,
      "downsampled": false,
      "method": "even_index_floor_first_last"
    },
    "static_components": {
      "total_points": 18420,
      "returned_points": 18420,
      "downsampled": false,
      "method": "even_index_floor_first_last"
    }
  }
}
```

Each key in `views` is a sampled tabular artifact. `records` are JSON-safe CSV
rows and therefore currently contain string values from the source CSV. Large
tables are sampled independently per view using the deterministic method named
in both the view payload and the `downsampling` entry. If a requested family has
no matching existing artifact, the family name appears in `unavailable_views`.
`gather_preview` is currently reported unavailable by this compact bundle route;
a future preview endpoint must reuse TraceStore window APIs for heavy trace I/O.

## 12. Viewer mapping summary

M6 products map to viewer surfaces as follows:

| Product | Primary view | Existing source |
|---|---|---|
| First-break fit QC | observation scatter, residual histogram, gather overlays | `first_break_residuals.csv`, `refraction_first_break_time_export.csv`, solution NPZ |
| Reduced-time / LMO QC | reduced-time scatter or gather overlay with offset gates | request JSON, residual/export observations, V1 QC |
| 2D profile QC | inline tracks for time terms, velocities, thicknesses, statics, residuals, fold | source/receiver tables, endpoint NPZ, time-term spreadsheet |
| 3D grid/map QC | cell heatmaps for velocity, fold, residuals, and status | refractor velocity cells/grid/QC artifacts |
| Static composition QC | endpoint and trace component audit panels | source/receiver tables, solution NPZ, static history, field QC |
| Before/after gather preview | raw/corrected gather with pick/model/LMO overlays | TraceStores, solution NPZ, residual/export observations, corrected file metadata |

2D line plots and 3D maps must remain separate viewer concepts. A
`line_2d_projected` cell workflow is a projected inline product, even if source
data started as map X/Y coordinates. A `grid_3d` workflow with multiple Y cells
is a map product, even if a viewer later extracts a profile from it.

## 13. Non-goals

M6 does not implement the following:

- UI or service code in this issue.
- New solver math or changes to existing T1LSST formulas.
- GRM.
- Plus-minus.
- Refraction tomography.
- Path-integrated or raypath-weighted cell slowness.
- Spatially varying V1 maps.
- SEG-Y static header write-back.
- Pick-editing workflows.
- Browser controls for editing refraction cell models.
- Automatic 2D line origin or azimuth estimation.
- Full IRAS compatibility or original IRAS manual content in the repository.
