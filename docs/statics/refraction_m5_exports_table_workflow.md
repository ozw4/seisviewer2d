# Refraction M5 exports and static table workflow

## 1. Current baseline

This document defines the M5 design target for repo-owned IRAS-style exports,
static-table import, and TraceStore application. It is self-contained and must
be read with:

- `docs/statics/refraction_iras_phase1_design.md`
- `docs/statics/refraction_iras_phase2_cell_v2_design.md`
- `docs/statics/refraction_multilayer_time_term.md`
- `docs/statics/refraction_field_corrections.md`

Original IRAS materials are not required in this repository and must not be
copied into it. M5 defines the repo-owned subset of IRAS-style behavior needed
for export, spreadsheet review, validated import, and static application.

The completed Phase 1-M4 refraction workflow already writes the stable public
artifact package. The core table and solution artifacts used by M5 exports are:

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

The same successful apply package also includes the first-break time export,
first-break fit QC, reduced-time QC, static-component QC, and line-profile QC
families described in
[refraction_qc_viewer_workflow.md](refraction_qc_viewer_workflow.md).

`source_static_table.csv` and `receiver_static_table.csv` are the endpoint CSV
views intended for spreadsheet inspection. `source_receiver_static_table.npz`
is the pickle-free machine-readable endpoint table. `refraction_static_solution.npz`
stores trace, node, endpoint, component, and residual arrays in TraceStore
sorted trace order. `refraction_static_history.json` records the sign
convention, input and output file IDs, cumulative shift field, component
inclusion, and double-application policy.

The current model terms are:

| Term | Meaning | Unit |
|---|---|---:|
| `V1` | Weathering / first-layer velocity | m/s |
| `V2` | First refractor / bedrock velocity for `T1` | m/s |
| `V3` | Second refractor velocity for `T2` | m/s |
| `Vsub` | Substratum velocity for `T3` | m/s |
| `T1` | Source/receiver half-intercept time for the `V2` branch | s in NPZ, ms in CSV |
| `T2` | Source/receiver half-intercept time for the `V3` branch | s in NPZ, ms in CSV |
| `T3` | Source/receiver half-intercept time for the `Vsub` branch | s in NPZ, ms in CSV |
| `SH1` | First-layer thickness | m |
| `SH2` | Second-layer thickness | m |
| `SH3` | Third-layer thickness | m |
| `WCOR` | Weathering replacement correction in the repo shift convention | s in NPZ, ms in CSV |

M4 field-correction components are source depth, uphole time, and manual
source/receiver statics. They are written as component columns in
`source_static_table.csv`, `receiver_static_table.csv`,
`source_receiver_static_table.npz`, `refraction_static_solution.npz`,
`refraction_static_qc.json`, and `refraction_static_history.json`.

## 2. M5 scope and non-goals

M5 is in scope for:

- exporting source/receiver statics in repo-defined IRAS-style formats;
- exporting time-term and static components in spreadsheet-friendly form;
- exporting observed, modeled, and residual first-break times;
- defining one canonical static-table schema for validated import;
- applying a validated imported table to a TraceStore using the existing static
  application sign convention;
- recording import and apply provenance in static history artifacts.

M5 does not implement:

- SEG-Y header write-back;
- viewer plots in M5;
- GRM;
- plus-minus;
- tomography;
- new inversion math;
- full IRAS compatibility or exact legacy IRAS byte-for-byte compatibility;
- original IRAS manuals or copied IRAS material in the repository.

## 3. Sign convention

All exported and imported applied shifts use the repo convention:

```text
corrected(t) = raw(t - shift_s)
```

Therefore:

```text
shift_s > 0  -> event appears later in corrected data
shift_s < 0  -> event appears earlier in corrected data
```

Imported and exported applied shifts must be expressed in this convention unless
a field name explicitly says otherwise, such as a field containing `delay`.
Delay-positive fields are metadata unless a request or schema explicitly
defines a conversion to applied shift.

For current source and receiver static tables:

```text
total_static_ms == total_applied_shift_ms
```

`source_total_with_field_shift_ms` and `receiver_total_with_field_shift_ms`
represent endpoint totals after enabled M4 field components. Trace-level apply
artifacts use the same convention for `refraction_trace_shift_s_sorted`,
`trace_field_shift_s_sorted`, and `final_trace_shift_s_sorted`.

## 4. Units and rounding

Machine-readable NPZ arrays remain second-based unless an array name ends in
`_ms`. CSV and spreadsheet exports use units in column names:

| Value type | CSV unit convention |
|---|---:|
| Static shifts, time terms, corrections, first-break times, residuals | ms |
| Coordinates, elevations, thicknesses, depths, offsets | m |
| Velocities | m/s |
| Slowness | s/m |
| Trace and endpoint identifiers | unitless, not rounded |

M5 CSV writers should use deterministic decimal output:

- time columns in milliseconds: 6 decimal places;
- coordinates, elevations, thicknesses, depths, and offsets in meters: 3 decimal
  places;
- velocities in meters per second: 3 decimal places;
- slowness in seconds per meter: at least 12 significant digits;
- status, kind, ID, and key columns: exact text or integer values.

Current public export requests are constrained to this millisecond CSV schema:

- `export.units` must be `milliseconds`;
- `export.rounding_ms` is reserved for future display/card outputs and does
  not control machine-readable CSV precision; only the default `0.001` or
  `null` is accepted;
- `export.include_legacy_alias_columns` must be `true`; current M5 exports
  always write the documented column sets and do not support alias suppression.

`canonical_static_table` import converts millisecond columns to seconds before
TraceStore application. Import validation must reject non-finite values in
required apply fields.

## 5. API examples

### Refraction apply with inline M5 export

Use `POST /statics/refraction/apply` to run a normal refraction statics job and
write M5 export artifacts from the same source job directory. The `export`
block is top-level in the refraction apply request; the model below is a compact
one-layer example, and the same export block also applies to valid two- and
three-layer requests.

```json
{
  "file_id": "input-file-id",
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
    "bedrock_velocity_mode": "solve_global"
  },
  "conversion": {
    "mode": "t1lsst_1layer"
  },
  "apply": {
    "register_corrected_file": false
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
    "units": "milliseconds",
    "fail_on_invalid_static_status": true,
    "include_inactive_endpoints": false
  }
}
```

### Standalone refraction export

Use `POST /statics/refraction/export` when the refraction solve already
completed and only the M5 export artifacts need to be generated.

```json
{
  "source_job_id": "refraction-job-id",
  "export": {
    "enabled": true,
    "formats": ["canonical_static_table", "lsst_plus"],
    "units": "milliseconds"
  }
}
```

For that request, the expected generated artifacts are:

```text
canonical_source_static_table.csv
canonical_receiver_static_table.csv
canonical_source_receiver_static_table.csv
refraction_lsst_plus.csv
refraction_lsst_plus_cards.txt
```

### Static-table apply

Use `POST /statics/refraction/static-table/apply` to apply a canonical
source/receiver table to a TraceStore. The combined canonical table form uses
one artifact ID:

```json
{
  "file_id": "input-file-id",
  "key1_byte": 189,
  "key2_byte": 193,
  "combined_table_artifact_id": "export-job-id:canonical_source_receiver_static_table.csv",
  "register_corrected_file": true,
  "missing_static_policy": "fail",
  "allow_reapply_same_static_table": false
}
```

The separate source/receiver form uses paired artifact IDs instead of
`combined_table_artifact_id`:

```json
{
  "file_id": "input-file-id",
  "key1_byte": 189,
  "key2_byte": 193,
  "source_table_artifact_id": "export-job-id:canonical_source_static_table.csv",
  "receiver_table_artifact_id": "export-job-id:canonical_receiver_static_table.csv",
  "register_corrected_file": true,
  "missing_static_policy": "fail",
  "allow_reapply_same_static_table": false
}
```

Both forms apply shifts with the repo convention:

```text
corrected(t) = raw(t - shift_s)
```

## 6. Export formats

M5 defines these logical export formats. The public API routes are shown above,
and the column meanings and sign convention are fixed by this document.

### `lsst`

`lsst` is the compact IRAS-style endpoint static export. It is derived from
`source_static_table.csv` and `receiver_static_table.csv` without renaming those
baseline artifacts.

Required columns:

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

For two- and three-layer jobs, `lsst` may also include `t2_ms`, `t3_ms`,
`v3_m_s`, `vsub_m_s`, `sh2_weathering_thickness_m`,
`sh3_weathering_thickness_m`, and `total_weathering_thickness_m`.

### `lsst_plus`

`lsst_plus` is the component-rich endpoint export for review and handoff. It is
a superset of `lsst` and includes all stable M4 field-correction columns when
present.

Additional source columns include:

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
```

Additional receiver columns include:

```text
manual_static_shift_ms
manual_static_status
receiver_field_shift_ms
receiver_field_static_status
receiver_total_with_field_shift_ms
```

`lsst_plus` is intended for spreadsheet inspection and audit. Importable values
must be normalized into `canonical_static_table` before applying them to a
TraceStore.

### `time_term_spreadsheet`

`time_term_spreadsheet` is the generated
`refraction_time_term_spreadsheet.csv` endpoint export for spreadsheet review
of time terms, layer parameters, static components, statuses, and residual
summaries. It is derived from `source_static_table.csv`,
`receiver_static_table.csv`, `source_receiver_static_table.npz`, and
`refraction_static_solution.npz`.

Required columns, in deterministic order:

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

The export writes one row per source endpoint followed by one row per receiver
endpoint. Layer-specific numeric columns and optional component values that do
not apply to the job should be blank in CSV rather than synthesized.
Receiver rows use `not_applicable` for source-only source-depth and uphole
status fields. All time/static/correction columns use milliseconds, coordinate
and elevation columns use meters, velocity columns use meters per second, and
`sign_convention` is the repo convention from Section 3.

### `first_break_time`

`first_break_time` is an observation-level export for first-break audit. It is
derived from `first_break_residuals.csv` and the observation arrays in
`refraction_static_solution.npz`.

Required columns:

```text
format_name
format_version
source_job_id
observation_index
sorted_trace_index
source_endpoint_key
receiver_endpoint_key
source_id
receiver_id
offset_m
layer_kind
observed_pick_time_ms
modeled_pick_time_ms
residual_ms
used_in_solve
reject_reason
sign_convention
```

`observed_pick_time_ms` is the picked first-break time in the time basis used by
the refraction solve. `modeled_pick_time_ms` is the fitted model time for the
same observation. `residual_ms` is observed minus modeled.

### `canonical_static_table`

`canonical_static_table` is the only M5 table intended for import and TraceStore
application. It is a single endpoint table with one row per source or receiver
endpoint.

Canonical export status behavior:

- The default canonical export fails when any row has a non-`ok`
  `static_status`.
- `fail_on_invalid_static_status=false` with `include_inactive_endpoints=false`
  filters invalid rows and produces an apply-ready canonical table.
- `fail_on_invalid_static_status=false` with `include_inactive_endpoints=true`
  produces an audit table that can include inactive or invalid endpoints and may
  not be import/apply-ready until those rows are removed or corrected.

Required columns:

```text
format_name
format_version
source_job_id
endpoint_kind
endpoint_key
endpoint_id
static_status
```

Shift columns:

- `applied_shift_ms` is the default canonical shift column.
- `applied_shift_s` is also accepted and is converted to seconds internally
  without millisecond scaling.
- An unqualified `applied_shift` column is accepted only when import metadata
  explicitly declares `shift_units` as `milliseconds` or `seconds`.
- A table must provide exactly one applied-shift column.

Sign convention:

- `sign_convention` is required in the table unless the import request provides
  an explicit sign-convention override.
- The table value or override must be exactly
  `corrected(t) = raw(t - shift_s)`.

Optional metadata columns:

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

The applied-shift column is authoritative for import. Existing total/component
columns are preserved as audit metadata unless a future schema version defines
another explicit apply field.

## 7. Canonical import validation

The import validator must be deterministic and fail closed. A valid
`canonical_static_table` must satisfy all of these checks:

- `format_name` is `canonical_static_table`.
- `format_version` is a supported integer version.
- `sign_convention` is exactly `corrected(t) = raw(t - shift_s)`, or the import
  request supplies that exact value as an explicit override.
- `endpoint_kind` is `source` or `receiver`.
- Every required column is present.
- Exactly one of `applied_shift_ms`, `applied_shift_s`, or metadata-qualified
  `applied_shift` is present.
- The applied shift is finite for every row with `static_status="ok"`.
- Rows with non-`ok` `static_status` are rejected for apply.
- Endpoint identity is unique for each `endpoint_kind`.
- The import request declares whether matching uses `endpoint_key` or
  `endpoint_id`; coordinate matching is not used.
- Every source and receiver endpoint required by the target TraceStore linkage
  is present exactly once.
- No imported absolute shift exceeds the request maximum.
- Optional numeric metadata columns, when present, are finite or blank.
- Optional existing-total columns, when present, must use millisecond units and
  the repo applied-shift convention.

Validation produces a machine-readable QC artifact with row counts, endpoint
coverage, duplicate counts, missing endpoint counts, maximum absolute shift,
sign-convention status, and the selected endpoint identity mode.

## 8. Static table import/apply workflow

The intended M5 workflow is:

```text
refraction job
  -> source/receiver static table
  -> export
  -> optional edit
  -> import
  -> validate
  -> apply to TraceStore
```

The export step may produce any M5 format. If a user edits a spreadsheet, the
table must be normalized to `canonical_static_table` before import. Import does
not run a new inversion and does not recompute T1LSST components. It validates
endpoint identity and applied shifts, then creates source and receiver endpoint
lookup arrays.

Apply uses the validated endpoint shifts to build a sorted trace shift:

```text
trace_shift_s_sorted =
  source_applied_shift_s[source_endpoint_for_trace]
  + receiver_applied_shift_s[receiver_endpoint_for_trace]
```

The resulting shift array is applied to the source TraceStore in sorted trace
order using the existing TraceStore static application path:

```text
corrected(t) = raw(t - shift_s)
```

Application must verify that the sorted trace index and source/receiver endpoint
arrays match the target TraceStore and geometry linkage. Failed validation must
not register a corrected TraceStore.

## 9. Static history behavior

An M5 import/apply job writes history that is compatible with the existing
`refraction_static_history.json` contract. The history record must include:

- the repo sign convention;
- the input file ID and output file ID when a corrected TraceStore is
  registered;
- source export format, source job ID, and imported table artifact name;
- import schema name and version;
- endpoint identity mode;
- cumulative shift artifact and field used for TraceStore application;
- component names included in the applied trace shift;
- validation status and warnings;
- double-application policy result.

For an imported static table, the cumulative shift field should be the generated
sorted trace shift derived from `canonical_static_table.applied_shift_ms`.
Existing Phase 1-M4 history fields are not renamed.

## 10. Relationship to Phase 1-M4 artifacts

M5 is an export/import layer over the existing Phase 1-M4 artifacts. It does
not rename or replace:

```text
source_static_table.csv
receiver_static_table.csv
source_receiver_static_table.npz
refraction_static_solution.npz
refraction_static_history.json
```

`lsst`, `lsst_plus`, and `time_term_spreadsheet` are endpoint views derived from
the source and receiver static table artifacts and the solution NPZ.
`first_break_time` is an observation view derived from
`first_break_residuals.csv` and solution observation arrays.
`canonical_static_table` is the normalized import/apply schema. It can be
exported from current source/receiver static tables or authored externally, but
it must validate against the target TraceStore before application.

M5 preserves the Phase 1 one-layer `T1`/`V2`/`SH1`/`WCOR` contract, the Phase 2
cell-based V2 contract, the M3 two- and three-layer `T1`/`T2`/`T3`,
`V1`/`V2`/`V3`/`Vsub`, `SH1`/`SH2`/`SH3` contract, and the M4
source-depth, uphole-time, and manual-static component contract.
