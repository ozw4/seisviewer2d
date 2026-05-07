# Time-term static correction

Time-term static correction estimates near-surface delay terms for source and
receiver nodes from first-break picks. The PR6 backend implementation writes
pickle-free artifacts and can register a corrected TraceStore for viewer use.

## Overview

The inversion fits first-break picks after existing datum/residual statics to a
moveout term plus source and receiver node time terms:

```text
pick_time_after_static_s[i]
  = moveout_time_s[i]
  + node_time_term_s[source_node_id[i]]
  + node_time_term_s[receiver_node_id[i]]
  + error[i]
```

The input builder uses this sign convention for picks:

```text
pick_time_after_static_s
  = pick_time_raw_s
  + datum_trace_shift_s
  + residual_applied_shift_s
```

For the current endpoint, the standard workflow is to pass a TraceStore that has
already had datum/residual correction applied, then run time-term correction in
`weathering_only` mode. The time-term solution artifacts still include datum,
residual, weathering, and final shift component fields so downstream code can
verify the composition contract.

The current moveout implementation computes a positive propagation-time term:

```text
moveout_time_s = source_receiver_distance_m / refractor_velocity_m_s
```

Accepted moveout models are `head_wave_linear_offset`,
`reciprocal_head_wave`, `linear_offset`, and `none`; non-`none` models currently
use the selected distance source and `refractor_velocity_m_s`.

## Processing Flow

1. Resolve the input TraceStore by `file_id`, `key1_byte`, and `key2_byte`.
2. Load first-break picks from a batch predicted NPZ, manual NPZ artifact, or manual memmap.
3. Resolve geometry linkage if `linkage.mode` requires or supplies it.
4. Read sorted source/receiver IDs, coordinates, elevations, and optional offset headers.
5. Build the moveout vector and source/receiver node design matrix.
6. Solve for `node_time_term_s` with damping, gauge, and optional robust rejection.
7. Convert estimated delays into applied event-time shifts.
8. Write `time_term_static_solution.npz`, `time_term_static_qc.json`, and `time_term_statics.csv`.
9. If `apply.register_corrected_file=true`, apply the selected shift field to a new TraceStore and write `corrected_file.json`.

## Sign Convention

TraceStore static application uses:

```text
corrected(t) = raw(t - shift_s)
```

Meaning:

- `shift_s > 0`: the event is delayed in the corrected trace.
- `shift_s < 0`: the event is advanced in the corrected trace.

The solver estimates delays:

```text
node_time_term_s
estimated_trace_time_term_delay_s_sorted
```

A positive estimated delay means the observed first-break is later than the
moveout model.

Estimated delays are not applied directly. They are converted to applied
event-time shifts:

```text
applied_weathering_shift_s_sorted
  = -estimated_trace_time_term_delay_s_sorted
```

The final shift composition is:

```text
final_trace_shift_s_sorted
  = datum_trace_shift_s_sorted
  + residual_applied_shift_s_sorted
  + applied_weathering_shift_s_sorted
```

Do not pass an `estimated_*delay*` array to a TraceStore builder. TraceStore
builders take applied `*_shift*` arrays only.

## Recommended Workflow

1. Upload or open a SEG-Y file and create a TraceStore.
2. Run datum static correction.
3. Prepare first-break picks.
4. Review first-break QC.
5. Run residual static correction.
6. Run geometry linkage.
7. Run `POST /statics/time-term/apply` with `apply.mode="weathering_only"`.
8. Open the `file_id` from `corrected_file.json` in the viewer.

For the standard weathering-only flow:

- Input `file_id`: a datum/residual corrected TraceStore.
- Apply mode: `weathering_only`.
- Shift field used for TraceStore application: `applied_weathering_shift_s_sorted`.

## API

### POST /statics/time-term/apply

Request example:

```json
{
  "file_id": "residual-corrected-file-id",
  "key1_byte": 189,
  "key2_byte": 193,
  "pick_source": {
    "kind": "batch_predicted_npz",
    "job_id": "batch-job-id",
    "artifact_name": "predicted_picks_time_s.npz"
  },
  "geometry": {
    "source_id_byte": 9,
    "receiver_id_byte": 13,
    "source_x_byte": 73,
    "source_y_byte": 77,
    "receiver_x_byte": 81,
    "receiver_y_byte": 85,
    "source_elevation_byte": 45,
    "receiver_elevation_byte": 41,
    "source_depth_byte": null,
    "coordinate_scalar_byte": 71,
    "elevation_scalar_byte": 69,
    "coordinate_unit": "m",
    "elevation_unit": "m"
  },
  "linkage": {
    "mode": "required",
    "job_id": "linkage-job-id",
    "artifact_name": "geometry_linkage.npz"
  },
  "velocity": {
    "replacement_velocity_m_s": 2000.0,
    "refractor_velocity_m_s": 4500.0,
    "weathering_velocity_m_s": null
  },
  "moveout": {
    "model": "head_wave_linear_offset",
    "distance_source": "geometry",
    "offset_byte": 37,
    "allow_missing_offset": false,
    "max_geometry_offset_mismatch_m": null
  },
  "solver": {
    "damping": 0.01,
    "gauge": "reference_node",
    "reference_node_id": 0,
    "robust": {
      "enabled": true,
      "method": "mad",
      "threshold": 3.5,
      "max_iterations": 5,
      "min_used_fraction": 0.5,
      "min_used_observations": 1
    }
  },
  "apply": {
    "mode": "weathering_only",
    "interpolation": "linear",
    "fill_value": 0.0,
    "max_abs_shift_ms": 250.0,
    "output_dtype": "float32",
    "register_corrected_file": true
  }
}
```

Response example:

```json
{
  "job_id": "uuid",
  "state": "queued"
}
```

Important current schema notes:

- `apply.mode` currently accepts only `weathering_only`.
- `apply.register_corrected_file` defaults to `false`; set it to `true` to write `corrected_file.json` and register a corrected TraceStore.
- `solver.gauge` currently accepts `mean_zero` or `reference_node` through the API. `reference_node` requires `solver.reference_node_id`.
- `velocity.refractor_velocity_m_s` must be greater than `velocity.replacement_velocity_m_s`.

### Job Status

```bash
curl http://localhost:8000/statics/job/<job_id>/status
```

Response shape:

```json
{
  "state": "done",
  "progress": 1.0,
  "message": "done"
}
```

### Files and Download

```bash
curl http://localhost:8000/statics/job/<job_id>/files
curl -L "http://localhost:8000/statics/job/<job_id>/download?name=time_term_static_solution.npz" -o time_term_static_solution.npz
curl -L "http://localhost:8000/statics/job/<job_id>/download?name=time_term_static_qc.json" -o time_term_static_qc.json
curl -L "http://localhost:8000/statics/job/<job_id>/download?name=time_term_statics.csv" -o time_term_statics.csv
curl -L "http://localhost:8000/statics/job/<job_id>/download?name=corrected_file.json" -o corrected_file.json
```

`corrected_file.json` exists only when `apply.register_corrected_file=true`.

## Inputs

### First-break Picks

`pick_source.kind` may be:

- `batch_predicted_npz`: resolves `predicted_picks_time_s.npz` from a batch job.
- `manual_npz_artifact`: resolves a manual-pick NPZ artifact from a static, batch, or pipeline job.
- `manual_memmap`: reads the dataset's manual-pick memmap; omit `job_id` and `artifact_name`.

Pick arrays are interpreted in TraceStore sorted trace order and seconds.

### Datum and Residual Components

The inversion input contract is:

```text
pick_time_after_static_s
  = pick_time_raw_s
  + datum_trace_shift_s
  + residual_applied_shift_s
```

The artifact contract always includes:

- `datum_trace_shift_s_sorted`
- `residual_applied_shift_s_sorted`
- `applied_weathering_shift_s_sorted`
- `final_trace_shift_s_sorted`

Current API requests do not include explicit datum/residual solution artifact
fields. In normal operation, use a datum/residual corrected input TraceStore and
`weathering_only` mode so the time-term job applies only the weathering component
on top of that corrected input.

### Geometry Linkage

`geometry_linkage.npz` maps source and receiver endpoints to shared near-surface
node IDs. The time-term unknown is `node_time_term_s[node_id]`.

The design equation uses:

```text
pick_time_after_static_s
  = moveout_time_s
  + node_time_term_s[source_node_id]
  + node_time_term_s[receiver_node_id]
  + error
```

The time-term input loader uses these main linkage fields:

- `source_node_id_sorted`
- `receiver_node_id_sorted`
- `n_nodes`

When `linkage.mode="required"`, `linkage.job_id` must reference a statics
`geometry_linkage` job whose artifact matches the same `file_id`, key bytes, and
trace count.

### Header Bytes and Units

Default time-term request header bytes:

```text
source_id_byte: 9
receiver_id_byte: 13
offset_byte: 37

source_elevation_byte: 45
receiver_elevation_byte: 41
source_depth_byte: optional / null

coordinate_scalar_byte: 71
elevation_scalar_byte: 69

source_x_byte: 73
source_y_byte: 77
receiver_x_byte: 81
receiver_y_byte: 85

key1_byte: usually 189
key2_byte: usually 193
```

Units:

- `coordinate_unit`: `m` or `ft`.
- `elevation_unit`: `m` or `ft`.
- Velocities are meters per second.
- NPZ artifact times are seconds unless the field name says `_ms`.
- CSV and QC summary fields are milliseconds when labelled `_ms`.

SEG-Y scalar convention:

- Positive scalar: multiply by the scalar.
- Negative scalar: divide by `abs(scalar)`.
- Zero scalar: treat as `1`.

## Moveout Model

`moveout.distance_source` may be:

- `geometry`: use source/receiver coordinates.
- `offset_header`: use absolute offset from `moveout.offset_byte`; `offset_byte` is required.
- `auto`: use valid geometry distance when available and offset otherwise.

`max_geometry_offset_mismatch_m`, when set, rejects inputs where geometry distance
and offset-header distance differ by more than the configured tolerance.

## Solver

### Gauge

Gauge constraints control the absolute level or nullspace of node time terms.
The lower-level solver supports `none`, `mean_zero`, `component_mean_zero`, and
`reference_node`; the current API schema exposes `mean_zero` and `reference_node`.

- `mean_zero`: constrains all node time terms toward a zero mean.
- `reference_node`: fixes the absolute level by referencing one node ID; use
  `reference_node_id`.

Recommended usage:

- Small synthetic or known-node tests: `reference_node`.
- General field data through the current API: `mean_zero`.

### Damping

`solver.damping` maps to `damping_lambda`. It regularizes node time terms toward
the damping prior, currently zero through the API. Use small nonzero damping for
field data unless a test intentionally needs an undamped exact solution.

### Robust Rejection

Robust options:

- `enabled`: run iterative outlier rejection when `true`.
- `method`: `mad` or `sigma`.
- `threshold`: score threshold for rejection.
- `max_iterations`: maximum rejection iterations.
- `min_used_fraction`: minimum fraction of observations that must remain.
- `min_used_observations`: minimum count of observations that must remain.

Rejected traces are excluded from solver fitting after rejection. They still get
`applied_weathering_shift_s_sorted` and `final_trace_shift_s_sorted` from the
final node model because TraceStore application needs a shift value for every
trace. The policy is recorded as `rejected_trace_policy = "use_final_model"`.

QC fields for reviewing rejection include:

- `rejected_trace_mask_sorted`
- `rejected_iteration_sorted`
- `final_used_trace_mask_sorted`
- `robust.iterations` in `time_term_static_qc.json`

## Apply Modes

### weathering_only

Current status: implemented and accepted by the API.

Use this mode when the input TraceStore is already datum/residual corrected. The
TraceStore apply step uses:

```text
applied_weathering_shift_s_sorted
```

This avoids double-applying datum and residual components.

Lineage validation rejects:

- a raw/original TraceStore with no derived datum/residual metadata,
- a source TraceStore with a non-null `source_sha256`,
- a source TraceStore already containing `time_term_static_correction`, and
- a source TraceStore already containing `weathering_static_correction`.

### final_from_raw

Current status: not implemented and not accepted by the API schema. The lower
apply selector names the intended field:

```text
final_trace_shift_s_sorted
```

The intended use is applying datum + residual + weathering to a raw TraceStore in
one pass. Until implemented, use `weathering_only` with a datum/residual corrected
TraceStore.

## Artifacts

### time_term_static_solution.npz

Contract:

- Read with `np.load(path, allow_pickle=False)`.
- Contains no object dtype arrays.
- `order == "trace_store_sorted"`.
- All `*_sorted` trace arrays are in TraceStore sorted trace order.
- `applied_weathering_shift_s_sorted == -estimated_trace_time_term_delay_s_sorted`.
- `final_trace_shift_s_sorted == datum_trace_shift_s_sorted + residual_applied_shift_s_sorted + applied_weathering_shift_s_sorted`.

Main scalar fields:

```text
schema_version
artifact_kind
order
job_id
input_file_id
key1_byte
key2_byte
n_traces
n_samples
dt
n_nodes
n_observations
n_final_used_traces
n_rejected_traces
pick_source_description
datum_solution_path
residual_solution_path
linkage_artifact_path
header_source_segy_path
moveout_model
refractor_velocity_m_s
moveout_distance_source
solver_name
solver_istop
solver_iterations
solver_stop_message
gauge_mode
damping_lambda
robust_enabled
robust_method
robust_threshold
robust_stop_reason
robust_n_iterations
sign_convention
delay_to_shift_convention
final_shift_convention
rejected_trace_policy
solver_result_kind
```

Main trace-level arrays:

```text
sorted_trace_index
pick_time_raw_s_sorted
valid_pick_mask_sorted
pick_time_after_static_s_sorted
moveout_time_s_sorted
moveout_distance_m_sorted
valid_moveout_mask_sorted
source_node_id_sorted
receiver_node_id_sorted
source_node_time_term_s_sorted
receiver_node_time_term_s_sorted
estimated_trace_time_term_delay_s_sorted
applied_weathering_shift_s_sorted
datum_trace_shift_s_sorted
residual_applied_shift_s_sorted
final_trace_shift_s_sorted
final_used_trace_mask_sorted
rejected_trace_mask_sorted
rejected_iteration_sorted
source_id_sorted
receiver_id_sorted
offset_sorted
source_x_m_sorted
source_y_m_sorted
receiver_x_m_sorted
receiver_y_m_sorted
source_elevation_m_sorted
receiver_elevation_m_sorted
source_depth_m_sorted
```

Main node-level arrays:

```text
node_id
node_time_term_s
node_time_term_ms
source_observation_count_by_node
receiver_observation_count_by_node
total_observation_count_by_node
component_id_by_node
```

Main row-level arrays:

```text
row_trace_index_sorted
row_source_node_id
row_receiver_node_id
row_pick_time_after_static_s
row_moveout_time_s
row_data_s
row_estimated_time_term_delay_s
row_residual_before_s
row_residual_after_s
row_residual_after_ms
initial_row_used_mask
final_row_used_mask
final_row_rejected_mask
row_rejected_iteration
```

Robust iteration arrays:

```text
robust_iteration_index
robust_iteration_n_used
robust_iteration_n_rejected_total
robust_iteration_n_rejected_this_iteration
robust_iteration_center_s
robust_iteration_scale_s
robust_iteration_threshold_s
robust_iteration_rms_residual_after_s
```

### time_term_static_qc.json

Top-level sections:

```text
schema_version
artifact_kind
order
job
inputs
counts
moveout
solver
robust
time_terms
components
sign_convention
request
```

`time_terms` summarizes node and trace delay/shift fields in milliseconds.
`components` summarizes datum, residual, and final trace shift fields in
milliseconds. `sign_convention` records the estimated-delay and applied-shift
rules used to write the artifacts.

### time_term_statics.csv

Columns:

```text
sorted_trace_index
source_id
receiver_id
source_node_id
receiver_node_id
offset_m
source_x_m
source_y_m
receiver_x_m
receiver_y_m
source_elevation_m
receiver_elevation_m
source_depth_m
pick_time_raw_s
valid_pick
pick_time_after_static_s
moveout_time_s
moveout_distance_m
source_node_time_term_ms
receiver_node_time_term_ms
estimated_trace_time_term_delay_ms
applied_weathering_shift_ms
datum_trace_shift_ms
residual_applied_shift_ms
final_trace_shift_ms
final_used
rejected
rejected_iteration
row_index
row_residual_before_ms
row_residual_after_ms
```

There is one row per TraceStore sorted trace. Missing optional numeric values are
blank. Boolean values are written as `true` or `false`.

### corrected_file.json

When `apply.register_corrected_file=true`, the job writes `corrected_file.json`
and registers the corrected TraceStore under `file_id`.

Example shape:

```json
{
  "schema_version": 1,
  "artifact_kind": "corrected_file",
  "file_id": "corrected-file-id",
  "store_path": "/path/to/corrected/store",
  "store_name": "line001.sgy.statics.time_term.12345678",
  "derived_from_file_id": "source-file-id",
  "derived_from_store_path": "/path/to/source/store",
  "derived_by": "time_term_static_correction",
  "job_id": "time-term-job-id",
  "solution_artifact": "time_term_static_solution.npz",
  "apply_mode": "weathering_only",
  "applied_shift_field": "applied_weathering_shift_s_sorted",
  "key1_byte": 189,
  "key2_byte": 193,
  "dt": 0.004,
  "n_traces": 1200,
  "n_samples": 1500,
  "shift_ms": {
    "min": -12.0,
    "max": 8.0,
    "mean": -1.5,
    "max_abs": 12.0
  },
  "sign_convention": "corrected(t)=raw(t-shift_s); positive_shift_delays_events",
  "delay_to_shift_convention": "applied_weathering_shift_s_sorted = -estimated_trace_time_term_delay_s_sorted",
  "final_shift_convention": "final_trace_shift_s_sorted = datum_trace_shift_s_sorted + residual_applied_shift_s_sorted + applied_weathering_shift_s_sorted"
}
```

The corrected TraceStore `meta.json` keeps the source `original_segy_path`, sets
`source_sha256` to `null`, and appends a `time_term_static_correction` component
to derived metadata.

## Common Failure Cases

- `file_id` is not registered or cannot be opened with the requested key bytes.
- Pick source artifact is missing, has the wrong job kind, or does not match the input file/key bytes.
- Geometry linkage artifact is missing or its `source_node_id_sorted` / `receiver_node_id_sorted` shape does not match `n_traces`.
- A resolved datum/residual solution artifact is missing or has incompatible `n_traces`, `dt`, or key bytes.
- Required SEG-Y headers cannot be read.
- `velocity.refractor_velocity_m_s <= velocity.replacement_velocity_m_s`.
- `moveout.distance_source="offset_header"` but `moveout.offset_byte` is null.
- Too few valid observations remain to build or solve the system.
- Robust rejection leaves fewer than `min_used_fraction` or `min_used_observations`.
- `max_abs_shift_ms` is exceeded by weathering or final shifts.
- `weathering_only` is applied to a raw/original TraceStore.
- The source TraceStore is incomplete, including missing split raw-baseline artifacts.
- A time-term corrected TraceStore is used as the source for another time-term application.

## Synthetic E2E Regression

`app/tests/test_time_term_static_end_to_end.py` builds a synthetic TraceStore with
known source/receiver node time terms and verifies:

- the inversion recovers the known estimated delays,
- `applied_weathering_shift_s_sorted = -estimated_trace_time_term_delay_s_sorted`,
- robust outlier rejection removes the intended outlier from fitting,
- the corrected TraceStore impulse shift has the documented sign, and
- the job artifacts download round-trips through the static job API.

Run:

```bash
python -m pytest -q app/tests/test_time_term_static_end_to_end.py
```

## Developer Commands

Focused PR6 checks:

```bash
python -m pytest -q \
  app/tests/test_time_term_import_layering.py \
  app/tests/test_time_term_static_request.py \
  app/tests/test_time_term_static_inputs.py \
  app/tests/test_time_term_moveout.py \
  app/tests/test_time_term_design_matrix.py \
  app/tests/test_time_term_sparse_solver.py \
  app/tests/test_time_term_robust_solver.py \
  app/tests/test_time_term_apply_shift.py \
  app/tests/test_time_term_static_artifacts.py \
  app/tests/test_time_term_static_apply_trace_store.py \
  app/tests/test_time_term_static_job_api.py \
  app/tests/test_time_term_static_end_to_end.py
```

Repository-level checks after code changes remain:

```bash
python -m compileall -q app
ruff check app
pytest
```
