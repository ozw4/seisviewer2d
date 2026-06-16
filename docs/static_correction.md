# Static correction

`seisviewer2d` exposes API-centered static-correction workflows through the
`/statics/...` API family. The viewer can open corrected TraceStores that these
jobs register, and refraction statics also have a dedicated browser workflow.
The browser workflow for the `Static Correction` page is documented in
[statics/static_correction_ui_workflow.md](statics/static_correction_ui_workflow.md).

## Workflows

Core static-correction job launch endpoints:

- `POST /statics/datum/apply`
  - Computes datum trace shifts and can register a datum-corrected TraceStore.
- `POST /statics/first-break/qc`
  - Validates first-break picks against a datum solution and writes QC artifacts.
- `POST /statics/linkage/build`
  - Builds `geometry_linkage.npz`, which maps each sorted trace's source and
    receiver endpoints to near-surface node IDs.
- `POST /statics/residual/apply`
  - Estimates residual applied event-time shifts and can register a
    residual-corrected TraceStore.
- `POST /statics/time-term/apply`
  - Estimates source/receiver node time terms, converts estimated delays to
    applied weathering shifts, writes artifacts, and can register a
    time-term-corrected TraceStore.

Refraction static apply and validation endpoints:

- `POST /statics/refraction/apply`
  - JSON request body API for first-break based near-surface refraction statics.
  - `pick_source.kind="uploaded_npz"` is not accepted here. Directly uploaded
    first-break NPZ files must use the multipart endpoints below.
  - Builds the GLI variable-thickness near-surface model and writes the final
    refraction statics artifact package.
  - When requested, applies the final refraction shifts to a derived TraceStore
    and registers the corrected file for viewer access. SEG-Y writing is not
    part of this workflow.
- `POST /statics/refraction/apply-with-picks`
  - `multipart/form-data` apply endpoint used by the `Static Correction` UI.
  - Form fields are `request_json`, containing the JSON
    `RefractionStaticApplyRequest`, and `pick_npz`, containing the selected
    first-break pick NPZ file.
  - The request JSON must use `pick_source.kind="uploaded_npz"`.
- `POST /statics/refraction/validate-with-picks`
  - `multipart/form-data` validation endpoint used by the `Static Correction`
    UI before launching an apply job.
  - Form fields are `request_json`, containing the JSON
    `RefractionStaticApplyRequest`, and `pick_npz`, containing the selected
    first-break pick NPZ file.
  - The request JSON must use `pick_source.kind="uploaded_npz"`.

Refraction QC endpoints:

- `POST /statics/refraction/qc`
- `POST /statics/refraction/qc/endpoints`
- `POST /statics/refraction/qc/pick-map`
- `POST /statics/refraction/qc/drilldown`
- `POST /statics/refraction/qc/station-structure`
- `POST /statics/refraction/qc/gather-preview`

The refraction QC artifact and viewer workflow is documented in
[statics/refraction_qc_viewer_workflow.md](statics/refraction_qc_viewer_workflow.md).

M5 export and static-table apply endpoints:

- `POST /statics/refraction/export`
- `POST /statics/refraction/static-table/apply`

The M5 export/import and static-table apply workflow is documented in
[statics/refraction_m5_exports_table_workflow.md](statics/refraction_m5_exports_table_workflow.md).
The IRAS-compatible 1-layer V1/T1LSST/source-receiver table workflow is
documented in [refraction_static.md](refraction_static.md). The UI workflow for
running refraction statics from the current viewer file and a directly selected
first-break pick NPZ is documented in
[statics/static_correction_ui_workflow.md](statics/static_correction_ui_workflow.md).

The detailed time-term inversion API, sign convention, apply modes, and artifact
contract are documented in [time_term_static_correction.md](time_term_static_correction.md).

## Job Lifecycle

Static-correction jobs use common job endpoints:

```bash
curl http://localhost:8000/statics/job/<job_id>/status
curl -X POST http://localhost:8000/statics/job/<job_id>/cancel
curl http://localhost:8000/statics/job/<job_id>/files
curl -L "http://localhost:8000/statics/job/<job_id>/download?name=<artifact_name>" -o <artifact_name>
```

`/statics/job/<job_id>/status` returns `state`, `progress`, and `message`.
`/statics/job/<job_id>/cancel` requests cancellation and returns the same status
payload shape.
`/statics/job/<job_id>/files` returns artifact names and sizes from the job
artifact directory. `/statics/job/<job_id>/download` accepts only a plain file
basename in the `name` query parameter.

## Trace Order

Static-correction artifacts that carry per-trace arrays use TraceStore sorted
trace order. The order is controlled by the uploaded/opened TraceStore key bytes:

- `key1_byte`: groups traces into sections, usually `189`
- `key2_byte`: sorts traces within a section, usually `193`

Do not reorder artifact arrays before applying them to a TraceStore. Per-trace
shift fields named `*_sorted` are aligned with this sorted order.
Refraction statics input-model per-trace arrays also use this TraceStore sorted
trace order.

## Refraction Statics Artifacts

`POST /statics/refraction/apply` writes a standard refraction artifact package
to the common static job artifact directory. Successful apply jobs include these
core solution and table artifacts:

- `refraction_static_request.json`: validated request payload used for the job.
- `refraction_static_solution.npz`: compressed, pickle-free machine-readable
  solution with trace, node, endpoint, component, and residual arrays.
- `refraction_static_qc.json`: human-readable QC summary with request,
  velocity, datum, observation, fit, statics, status-count, and artifact
  metadata.
- `refraction_statics.csv`: one row per trace in TraceStore sorted order.
- `near_surface_model.csv`: one row per near-surface node with elevations,
  thickness, half-intercept time, replacement shift, statuses, and residual
  summaries.
- `first_break_residuals.csv`: one row per GLI design-matrix observation.
- `refraction_static_components.csv`: one combined source/receiver endpoint
  component table.
- `source_static_table.csv`, `receiver_static_table.csv`, and
  `source_receiver_static_table.npz`: endpoint static tables for spreadsheet
  and machine-readable use.
- `refraction_time_term_spreadsheet.csv`: endpoint time-term spreadsheet view.
- `refraction_static_history.json`: static lineage, sign convention,
  cumulative shift, and double-application audit history.
- `refraction_static_artifacts.json`: manifest generated from the current
  artifact registry entries for the request.

Successful apply jobs also include viewer-ready QC families:

- first-break fit: `refraction_first_break_fit_qc.csv`,
  `refraction_first_break_fit_qc.npz`,
  `refraction_first_break_fit_qc.json`
- reduced-time QC: `refraction_reduced_time_qc.csv`,
  `refraction_reduced_time_qc.npz`, `refraction_reduced_time_qc.json`
- static component QC: `refraction_static_component_qc_trace.csv`,
  `refraction_static_component_qc_endpoint.csv`,
  `refraction_static_component_qc.npz`,
  `refraction_static_component_qc.json`
- line-profile QC: `refraction_line_profile_qc_source.csv`,
  `refraction_line_profile_qc_receiver.csv`,
  `refraction_line_profile_qc_combined.csv`,
  `refraction_line_profile_qc.npz`, `refraction_line_profile_qc.json`

Conditional artifacts are documented in
[refraction_static.md](refraction_static.md) and the statics workflow guides.

NPZ time arrays are stored in seconds unless the array name ends in `_ms`.
CSV time-shift, half-intercept, and residual columns are in milliseconds.
Geometry, elevation, and thickness columns are in meters. Velocities are in
meters per second, and slowness values are in seconds per meter.

The implementation lives under `app/statics/refraction/artifacts` with
`__init__.py` kept as the public re-export facade. The package responsibilities
are split by artifact family:

- `contract.py`: artifact names, column definitions, and common constants.
- `registry.py`: manifest and artifact registry helpers.
- `writer.py`: final artifact package writer.
- `solution.py`, `qc.py`, `final_tables.py`, `first_break.py`,
  `components.py`, `static_tables.py`, `cell_velocity.py`, `grid_map.py`, and
  `line_profile.py`: artifact-family builders and writers.
- `arrays.py`, `validation.py`, `formatters.py`, `io.py`, and `stats.py`:
  shared array coercion, validation, formatting, atomic writes, and statistics.

Review new artifact code with the same split in mind: an artifact module should
normally stay below about 1,500 lines, a public writer should normally stay
below about 120 lines, and builders over about 200 lines are candidates for
extracting smaller helpers.

Per-trace arrays in `refraction_static_solution.npz` and rows in
`refraction_statics.csv` use TraceStore sorted trace order. The shift sign
convention is:

```text
corrected(t) = raw(t - shift_s)
```

## Refraction Statics TraceStore Application

Refraction jobs with `apply.register_corrected_file=false` complete as
artifact-only jobs. They write the final refraction statics artifacts and do not
create `corrected_file.json`.

Refraction jobs with `apply.register_corrected_file=true` write the same
solution artifacts, then apply `refraction_trace_shift_s_sorted` to the source
TraceStore in sorted trace order. Application uses linear interpolation and the
repo-wide statics sign convention:

```text
corrected(t) = raw(t - shift_s)
```

Positive shifts delay events in the corrected TraceStore. Negative shifts move
events earlier. Samples shifted outside the source time range use
`apply.fill_value`, and the P0 output dtype is `float32`.

Before writing the derived TraceStore, the backend verifies that
`sorted_trace_index` matches the source TraceStore `sorted_to_original` index,
that every trace static is valid, that shifts are finite, and that no shift
exceeds `apply.max_abs_shift_ms`. Failed validation does not register a
corrected file.

Corrected outputs add:

- `corrected_file.json`: corrected file registration metadata and provenance.
- `refraction_static_apply_qc.json`: apply summary, shift statistics, status
  counts, and corrected TraceStore write status.

The workflow does not write SEG-Y files or SEG-Y static headers.
