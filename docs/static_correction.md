# Static correction

`seisviewer2d` exposes backend-only static-correction workflows through the
`/statics/...` API family. The viewer can open corrected TraceStores that these
jobs register, but the static-correction APIs are primarily developer-facing.
The browser workflow for the `Static Correction` tab is documented in
[statics/static_correction_ui_workflow.md](statics/static_correction_ui_workflow.md).

## Workflows

- Datum statics: `POST /statics/datum/apply`
  - Computes datum trace shifts and can register a datum-corrected TraceStore.
- First-break QC: `POST /statics/first-break/qc`
  - Validates first-break picks against a datum solution and writes QC artifacts.
- Residual statics: `POST /statics/residual/apply`
  - Estimates residual applied event-time shifts and can register a residual-corrected TraceStore.
- Geometry linkage: `POST /statics/linkage/build`
  - Builds `geometry_linkage.npz`, which maps each sorted trace's source and receiver endpoints to near-surface node IDs.
- Time-term statics: `POST /statics/time-term/apply`
  - Estimates source/receiver node time terms, converts estimated delays to applied weathering shifts, writes artifacts, and can register a time-term-corrected TraceStore.
- Refraction statics: `POST /statics/refraction/apply`
  - API entry point for first-break based near-surface refraction statics.
  - Builds the GLI variable-thickness near-surface model and writes the final
    refraction statics artifact package.
  - When requested, applies the final refraction shifts to a derived TraceStore
    and registers the corrected file for viewer access. SEG-Y writing is not
    part of this workflow.
  - The IRAS-compatible 1-layer V1/T1LSST/source-receiver table workflow is
    documented in [refraction_static.md](refraction_static.md).
  - The refraction QC artifact and viewer workflow is documented in
    [statics/refraction_qc_viewer_workflow.md](statics/refraction_qc_viewer_workflow.md).
  - The UI workflow for running refraction statics from the current viewer file
    and a directly selected first-break pick NPZ is documented in
    [statics/static_correction_ui_workflow.md](statics/static_correction_ui_workflow.md).

The detailed time-term inversion API, sign convention, apply modes, and artifact
contract are documented in [time_term_static_correction.md](time_term_static_correction.md).

## Job Lifecycle

Static-correction jobs use common job endpoints:

```bash
curl http://localhost:8000/statics/job/<job_id>/status
curl http://localhost:8000/statics/job/<job_id>/files
curl -L "http://localhost:8000/statics/job/<job_id>/download?name=<artifact_name>" -o <artifact_name>
```

`/statics/job/<job_id>/status` returns `state`, `progress`, and `message`.
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

`POST /statics/refraction/apply` writes these final refraction artifacts to the
common static job artifact directory:

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
- `refraction_static_artifacts.json`: manifest listing the final artifacts.

NPZ time arrays are stored in seconds unless the array name ends in `_ms`.
CSV time-shift, half-intercept, and residual columns are in milliseconds.
Geometry, elevation, and thickness columns are in meters. Velocities are in
meters per second, and slowness values are in seconds per meter.

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
