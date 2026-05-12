# Refraction M5 exports and static-table workflow

This page documents the implemented M5 workflow for repo-owned IRAS-style
exports, canonical static-table import, and static-table TraceStore application.
Read it with:

- `docs/statics/refraction_iras_phase1_design.md`
- `docs/statics/refraction_iras_phase2_cell_v2_design.md`
- `docs/statics/refraction_multilayer_time_term.md`
- `docs/statics/refraction_field_corrections.md`

Original IRAS references are not required in this repository. The repo docs are
the implementation reference, and M5 intentionally defines a repo-owned subset
for export, spreadsheet review, validated import, and application.

## Overview

The practical workflow is:

```text
run refraction statics
  -> export tables/cards
  -> inspect or edit
  -> import
  -> validate
  -> apply to TraceStore
```

There are two ways to produce M5 export artifacts:

- Request exports while creating a refraction job with
  `POST /statics/refraction/apply`.
- Run a standalone export job from a completed refraction job with
  `POST /statics/refraction/export`.

Only the canonical static table is intended for import. `lsst`, `lsst_plus`,
`time_term_spreadsheet`, and `first_break_time` are review and handoff formats.
If a table is edited in a spreadsheet, normalize the apply values into
`canonical_static_table` before applying it to a TraceStore.

All job artifacts are accessed through the common static job endpoints:

```bash
curl http://localhost:8000/statics/job/<job_id>/status
curl http://localhost:8000/statics/job/<job_id>/files
curl -L "http://localhost:8000/statics/job/<job_id>/download?name=<artifact_name>" -o <artifact_name>
```

## Sign Convention

All exported and imported applied shifts use the repo convention:

```text
corrected(t) = raw(t - shift_s)
```

`shift_s > 0` delays events in the corrected data. `shift_s < 0` moves events
earlier. CSV files express shifts, time terms, corrections, first-break times,
and residuals in milliseconds unless a column name ends in `_s`.

For current source and receiver static tables:

```text
total_static_ms == total_applied_shift_ms
```

The canonical import column `applied_shift_ms` or `applied_shift_s` is
authoritative for static-table apply. Other total or component columns are audit
metadata.

## Export Formats

Set `export.enabled=true` to request M5 exports. If `formats` is empty, the
default export formats are:

```text
canonical_static_table
time_term_spreadsheet
```

Supported `formats` values and generated artifacts are:

| Format | Artifacts | Purpose |
|---|---|---|
| `lsst` | `refraction_lsst.csv`, `refraction_lsst_cards.txt` | Compact IRAS-style endpoint static export derived from source/receiver static tables. |
| `lsst_plus` | `refraction_lsst_plus.csv`, `refraction_lsst_plus_cards.txt` | Component-rich endpoint export including M4 field-correction columns when present. |
| `time_term_spreadsheet` | `refraction_time_term_spreadsheet.csv` | Endpoint spreadsheet for time terms, layer parameters, component shifts, statuses, and residual summaries. |
| `first_break_time` | `refraction_first_break_time_export.csv` | Observation-level first-break audit export with observed, modeled, and residual pick times. |
| `canonical_static_table` | `canonical_source_static_table.csv`, `canonical_receiver_static_table.csv`, `canonical_source_receiver_static_table.csv` | Importable source/receiver static tables for TraceStore application. |

Standalone export jobs also write `job_meta.json` and
`refraction_static_export_request.json` in the export job artifact directory.

### `lsst`

`lsst` writes one row per source or receiver endpoint. Required columns are:

```text
format_name
format_version
source_job_id
endpoint_kind
endpoint_key
endpoint_id
node_id
x_m
y_m
surface_elevation_m
t1_ms
v1_m_s
v2_m_s
sh1_weathering_thickness_m
weathering_correction_ms
elevation_correction_ms
total_static_ms
total_applied_shift_ms
static_status
sign_convention
```

Two- and three-layer jobs may add `t2_ms`, `t3_ms`, `v3_m_s`, `vsub_m_s`,
`sh2_weathering_thickness_m`, `sh3_weathering_thickness_m`, and
`total_weathering_thickness_m`.

### `lsst_plus`

`lsst_plus` is a superset of `lsst`. It includes all stable source-depth,
uphole, manual-static, and field-static columns when present.

Additional field-correction columns include:

```text
source_depth_m
source_depth_shift_ms
source_depth_status
uphole_time_ms
uphole_shift_ms
uphole_status
manual_static_shift_ms
manual_static_status
source_field_shift_ms
source_field_static_status
source_total_with_field_shift_ms
receiver_field_shift_ms
receiver_field_static_status
receiver_total_with_field_shift_ms
```

### `time_term_spreadsheet`

`time_term_spreadsheet` writes `refraction_time_term_spreadsheet.csv`.
Required columns are:

```text
schema_version
format_name
format_version
source_job_id
endpoint_kind
endpoint_key
endpoint_id
station_id
node_id
x_m
y_m
elevation_m
surface_elevation_m
t1_ms
t2_ms
t3_ms
v1_m_s
v2_m_s
v3_m_s
vsub_m_s
sh1_m
sh2_m
sh3_m
layer1_base_elevation_m
layer2_base_elevation_m
final_refractor_elevation_m
weathering_correction_ms
elevation_correction_ms
source_depth_correction_ms
uphole_correction_ms
manual_static_ms
field_correction_ms
total_applied_shift_ms
pick_count
used_pick_count
pick_count_by_layer
used_pick_count_by_layer
residual_rms_ms
residual_mad_ms
residual_rms_by_layer_ms
residual_mad_by_layer_ms
solution_status
weathering_status
datum_status
source_depth_status
uphole_status
manual_static_status
field_static_status
static_status
sign_convention
```

The export writes source endpoint rows followed by receiver endpoint rows.
Columns that do not apply to a job are left blank rather than synthesized.

### `first_break_time`

`first_break_time` writes `refraction_first_break_time_export.csv`. Required
columns are:

```text
schema_version
trace_index_sorted
source_endpoint_key
receiver_endpoint_key
source_node_id
receiver_node_id
offset_m
midpoint_x_m
midpoint_y_m
cell_ix
cell_iy
layer_kind
used_for_layer
observed_pick_time_ms
modeled_pick_time_ms
residual_ms
moveout_time_ms
source_time_term_ms
receiver_time_term_ms
velocity_m_s
rejection_reason
observation_status
sign_convention
```

`residual_ms` is observed minus modeled in the time basis used by the
refraction solve.

## Canonical Static Table

`canonical_static_table` is the import/apply schema. It is a single logical
endpoint table with one row per source or receiver endpoint. It can be supplied
as one combined CSV or as separate source and receiver CSV artifacts.

Required columns:

```text
format_name
format_version
source_job_id
endpoint_kind
endpoint_key
endpoint_id
static_status
sign_convention
```

The table must also provide exactly one applied-shift column:

- `applied_shift_ms`: applied shift in milliseconds.
- `applied_shift_s`: applied shift in seconds.
- `applied_shift`: accepted only by lower-level import callers that provide
  explicit units metadata. Public static-table apply artifacts should use
  `applied_shift_ms` or `applied_shift_s`.

`format_name` must be `canonical_static_table`. `format_version` is `1`.
`endpoint_kind` must be `source` or `receiver`. `endpoint_key` is the default
identity used for matching. `endpoint_id` must be present and is required to be
non-empty when a request uses `endpoint_id` matching.

Optional audit columns:

```text
x_m
y_m
source_id
receiver_id
node_id
total_static_ms
total_applied_shift_ms
source_field_shift_ms
receiver_field_shift_ms
source_total_with_field_shift_ms
receiver_total_with_field_shift_ms
manual_static_shift_ms
source_depth_shift_ms
uphole_shift_ms
t1_ms
t2_ms
t3_ms
v1_m_s
v2_m_s
v3_m_s
vsub_m_s
sh1_weathering_thickness_m
sh2_weathering_thickness_m
sh3_weathering_thickness_m
weathering_correction_ms
elevation_correction_ms
comment
```

Units:

| Value type | Unit convention |
|---|---|
| Columns ending in `_ms` | milliseconds |
| Columns ending in `_s` | seconds |
| Coordinates, elevations, depths, thicknesses, offsets | meters |
| Velocities | meters per second |
| Trace and endpoint identifiers | unitless text/integer identifiers |

## Import Validation

Static-table apply validates the imported table before writing or registering a
corrected TraceStore. Validation fails closed.

The importer validates that:

- the artifact ID has the form `<job_id>:<artifact_name>`;
- the table is either one combined artifact or a source/receiver artifact pair;
- `source_receiver_static_table.npz` has the required source and receiver arrays
  and the repo sign convention;
- CSV `format_name`, `format_version`, `sign_convention`, `endpoint_kind`, and
  applied-shift units are valid;
- every required column is present;
- exactly one applied-shift column is populated;
- rows with `static_status!="ok"` are rejected for apply;
- applied shifts and optional numeric metadata are finite when present;
- endpoint keys and endpoint IDs are unique within each endpoint kind;
- endpoint matching uses `endpoint_key` by default, or `endpoint_id` when
  `source_key_header` or `receiver_key_header` requests it;
- every source and receiver endpoint required by the target TraceStore geometry
  is present unless `missing_static_policy="zero"` and the corresponding
  `allow_missing_source_static` or `allow_missing_receiver_static` flag is true;
- imported endpoint shifts and final trace shifts do not exceed
  `max_abs_shift_ms`;
- sorted trace order and source/receiver endpoint arrays match the target
  TraceStore and geometry;
- duplicate-application guards are satisfied. By default the job rejects
  same-table reapplication; set `allow_reapply_same_static_table=true` only for
  an intentional reapply.

Failed validation does not register a corrected TraceStore.

Static-table apply writes these artifacts:

```text
static_table_apply_request.json
static_table_import_qc.json
static_table_apply_solution.npz
static_table_apply_qc.json
static_table_apply_trace_shifts.csv
static_table_apply_history.json
refraction_static_history.json
```

When `register_corrected_file=true`, it also writes `corrected_file.json` and
registers the corrected TraceStore.

## API Examples

### Refraction apply with exports

```bash
curl -X POST http://localhost:8000/statics/refraction/apply \
  -H "Content-Type: application/json" \
  -d @refraction_apply_with_exports.json
```

Example body:

```json
{
  "file_id": "line001",
  "key1_byte": 189,
  "key2_byte": 193,
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
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
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
  "export": {
    "enabled": true,
    "formats": [
      "canonical_static_table",
      "lsst",
      "lsst_plus",
      "time_term_spreadsheet",
      "first_break_time"
    ],
    "include_inactive_endpoints": true,
    "fail_on_invalid_static_status": true
  },
  "apply": {
    "register_corrected_file": false
  }
}
```

Response:

```json
{
  "job_id": "refraction-job-id",
  "state": "queued",
  "requested_formats": [
    "canonical_static_table",
    "lsst",
    "lsst_plus",
    "time_term_spreadsheet",
    "first_break_time"
  ]
}
```

### Standalone export from a source job

Use this when the refraction job is already complete and you want to generate
or regenerate M5 export artifacts without rerunning the inversion.

```bash
curl -X POST http://localhost:8000/statics/refraction/export \
  -H "Content-Type: application/json" \
  -d @refraction_export.json
```

Example body:

```json
{
  "source_job_id": "completed-refraction-job-id",
  "export": {
    "enabled": true,
    "formats": [
      "canonical_static_table",
      "lsst_plus"
    ],
    "include_inactive_endpoints": true,
    "fail_on_invalid_static_status": true
  }
}
```

Response:

```json
{
  "job_id": "export-job-id",
  "state": "queued",
  "source_job_id": "completed-refraction-job-id",
  "requested_formats": [
    "canonical_static_table",
    "lsst_plus"
  ]
}
```

The `source_job_id` must reference a completed `statics_kind="refraction"` job
with the source artifacts needed by the requested formats.

### Static-table apply

Use an artifact ID in the form `<job_id>:<artifact_name>`. For a standalone
export job, the combined canonical table artifact is usually:

```text
<export-job-id>:canonical_source_receiver_static_table.csv
```

Apply that table to a TraceStore:

```bash
curl -X POST http://localhost:8000/statics/refraction/static-table/apply \
  -H "Content-Type: application/json" \
  -d @static_table_apply.json
```

Example body:

```json
{
  "file_id": "line001",
  "key1_byte": 189,
  "key2_byte": 193,
  "combined_table_artifact_id": "export-job-id:canonical_source_receiver_static_table.csv",
  "source_key_header": "endpoint_key",
  "receiver_key_header": "endpoint_key",
  "register_corrected_file": true,
  "output_name": "line001.refraction.static-table",
  "missing_static_policy": "fail",
  "allow_reapply_same_static_table": false,
  "max_abs_shift_ms": 250.0
}
```

Response:

```json
{
  "job_id": "static-table-apply-job-id",
  "state": "queued"
}
```

Separate source and receiver artifacts are also accepted:

```json
{
  "file_id": "line001",
  "source_table_artifact_id": "export-job-id:canonical_source_static_table.csv",
  "receiver_table_artifact_id": "export-job-id:canonical_receiver_static_table.csv",
  "register_corrected_file": true
}
```

`source_receiver_static_table.npz`, `source_static_table.csv`, and
`receiver_static_table.csv` from a refraction job can also be imported through
the same artifact-ID mechanism. CSV source/receiver static tables are
canonicalized internally from their documented endpoint columns.

## Relationship To Base Refraction Artifacts

M5 export/import is layered over the existing refraction artifact package. It
does not rename or replace:

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
refraction_time_term_spreadsheet.csv
refraction_static_history.json
refraction_static_artifacts.json
```

`source_static_table.csv` and `receiver_static_table.csv` are endpoint CSV views
for spreadsheet inspection. `source_receiver_static_table.npz` is the
pickle-free machine-readable endpoint table. `refraction_static_solution.npz`
stores trace, node, endpoint, component, and residual arrays in TraceStore
sorted trace order.

## Non-Goals And Current Limitations

M5 does not include:

- SEG-Y header write-back;
- exact legacy IRAS byte-for-byte compatibility;
- viewer plots;
- GRM or plus-minus;
- refraction tomography;
- new inversion math;
- original IRAS manuals or copied IRAS material in the repository.

The public apply workflow remains limited to the refraction models documented in
`docs/statics/refraction_multilayer_time_term.md`: 1-layer T1LSST, 2-layer
T1LSST, and 3-layer T1LSST with the stated velocity-mode restrictions.
