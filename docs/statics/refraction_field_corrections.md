# Refraction field corrections

This document is the implementation reference for the M4 field-correction
components in the public refraction statics workflow. Read it with:

- `docs/statics/refraction_iras_phase1_design.md`
- `docs/statics/refraction_iras_phase2_cell_v2_design.md`
- `docs/statics/refraction_multilayer_time_term.md`

The repo docs are the canonical reference. Original IRAS manuals are not
required in this repository and must not be copied into it.

## 1. M4 scope and non-goals

M4 adds componentized field corrections to `POST /statics/refraction/apply`:

- source-depth correction with
  `field_corrections.source_depth.mode="weathering_velocity_time"`
- uphole-time correction with `field_corrections.uphole.mode="header_time"`
- manual source/receiver statics with
  `field_corrections.manual_static.mode="artifact_table"` or
  `field_corrections.manual_static.mode="inline_table"`
- componentized source, receiver, trace, final-shift, QC, and static-history
  artifacts
- a best-effort double-application guard based on TraceStore lineage metadata

Current non-goals:

- Original IRAS manuals stored in the repo.
- Full IRAS compatibility.
- GRM, plus-minus, tomography, path-weighted cells, or SEG-Y static header
  write-back.
- Manual uphole correction tables. The schema name exists, but
  `field_corrections.uphole.mode="manual_table"` is not a public implemented
  mode.
- Project-wide static registries or inspection of SEG-Y static headers for
  double-application detection.

## 2. Repo sign convention

All field-correction components use the repo static-shift convention:

```text
corrected(t) = raw(t - shift_s)
```

Therefore:

```text
shift_s > 0  -> event appears later in corrected data
shift_s < 0  -> event appears earlier in corrected data
```

The same convention is used in CSV millisecond columns, NPZ second arrays, QC
JSON formulas, static history, and TraceStore apply paths.

## 3. Source-depth correction mode and sign

Source depth is source-only. Enable it with:

```text
field_corrections.source_depth.mode="weathering_velocity_time"
```

`source_depth_byte` may be supplied under `field_corrections.source_depth`, or
the job may reuse `geometry.source_depth_byte`. `source_depth_unit` is `m`.
Depth is normalized to positive-down meters before correction. With the default
`positive_down=true`, negative source-depth values are invalid. With
`positive_down=false`, the raw header value is negated before validation.

Resolved source depths are median-aggregated per source endpoint. The applied
shift formula is:

```text
source_depth_shift_s = +source_depth_m / V1_m_s
```

`V1_m_s` is the resolved weathering velocity from the first-layer request.
This component is written to source endpoint field-correction columns and QC.
It is not folded into legacy `source_refraction_shift_s` arrays.

## 4. Uphole correction mode and sign

Uphole correction is source-only. Enable it with:

```text
field_corrections.uphole.mode="header_time"
```

The request must provide `uphole_time_byte`. `uphole_time_unit` is `s` or `ms`.
Values are loaded from sorted trace headers, converted to seconds, and
median-aggregated per source endpoint.

With the default `positive_time_means_delay=true`, positive uphole time is a
delay, so the correction advances the data:

```text
uphole_shift_s = -uphole_time_s
```

If `positive_time_means_delay=false`, the component uses:

```text
uphole_shift_s = +uphole_time_s
```

Missing, invalid, inconsistent repeated endpoint values, values beyond
`max_abs_uphole_time_s`, and inactive source endpoints are status-coded. The
component is written to source endpoint field-correction columns and QC. It is
not folded into legacy `source_refraction_shift_s` arrays.

## 5. Manual static sign conventions

Manual statics can target source endpoints, receiver endpoints, or both. The
request must declare one sign convention whenever manual values are supplied.

For `sign_convention="applied_shift_s"`:

```text
manual_static_shift_s = manual_static_s
```

For `sign_convention="delay_positive_ms"`:

```text
manual_static_shift_s = -delay_ms / 1000.0
```

Artifact CSV tables use `manual_static_s` or `manual_static_ms`. Inline table
`value` is seconds when `sign_convention="applied_shift_s"` and milliseconds
when `sign_convention="delay_positive_ms"`.

Artifact rows may contain `endpoint_kind` (`source` or `receiver`),
`endpoint_key`, `endpoint_id`, `station_id`, `node_id`, `x_m`, `y_m`,
`manual_static_s`, `manual_static_ms`, and `comment`. Matching uses
`endpoint_kind + endpoint_key` first, then `endpoint_kind + endpoint_id`.
`station_id`, `node_id`, and coordinates are metadata only and are not matching
keys. Duplicate matched endpoint rows raise an error. Missing endpoint values
are status-coded as `missing_manual_static` and are rejected when
`allow_missing_endpoints=false`.

## 6. Component composition formula

Endpoint field components are composed in seconds:

```text
source_field_shift_s =
  source_depth_shift_s + uphole_shift_s + source_manual_static_shift_s

receiver_field_shift_s =
  receiver_manual_static_shift_s

trace_field_shift_s =
  source_field_shift_s + receiver_field_shift_s

final_trace_shift_s when apply_to_trace_shift=true =
  refraction_trace_shift_s + trace_field_shift_s

final_trace_shift_s when apply_to_trace_shift=false =
  refraction_trace_shift_s
```

The QC field formulas use the exact single-line names. The
`final_trace_shift_s` formula depends on `apply_to_trace_shift`:

```text
total_field_shift_s = source_depth_shift_s + uphole_shift_s + manual_static_shift_s
trace_field_shift_s = source_field_shift_s + receiver_field_shift_s
final_trace_shift_s = refraction_trace_shift_s + trace_field_shift_s
final_trace_shift_s = refraction_trace_shift_s
```

`field_corrections.composition.enabled` controls component composition.
`apply_to_trace_shift=true` makes the corrected TraceStore use
`final_trace_shift_s_sorted`; `apply_to_trace_shift=false` keeps the
cumulative applied shift and `final_trace_shift_s_sorted` at
`refraction_trace_shift_s_sorted` while still writing componentized field
artifacts. In artifact-only mode, `final_trace_static_status_sorted` and
`final_trace_static_valid_mask_sorted` match the base refraction trace status
arrays, and `applied_field_shift_s_sorted` is zero.
`invalid_component_policy="fail"` rejects invalid field components only when
field corrections are applied to trace shifts.
`invalid_component_policy="skip_invalid_traces"` keeps invalid traces at the
base refraction shift and applies valid field shifts to valid traces.

## 7. Artifact list

Public refraction statics jobs write the standard package described in
[refraction_static.md](../refraction_static.md#artifacts). Its core solution
and table artifacts are:

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

The successful-job package also includes first-break fit, reduced-time,
static-component, and line-profile QC artifact families used by
[refraction_qc_viewer_workflow.md](refraction_qc_viewer_workflow.md).

Source-depth correction also writes:

```text
refraction_source_depth_qc.json
refraction_source_depth_sources.csv
```

Uphole-time correction also writes:

```text
refraction_uphole_qc.json
refraction_uphole_sources.csv
```

Manual static import does not add a separate public artifact. It writes manual
component arrays and statuses into `source_static_table.csv`,
`receiver_static_table.csv`, `source_receiver_static_table.npz`,
`refraction_static_solution.npz`, `refraction_static_qc.json`, and
`refraction_static_history.json`.

When `apply.register_corrected_file=true`, apply output can also include:

```text
corrected_file.json
refraction_static_apply_qc.json
```

## 8. Source/receiver table columns

The source and receiver tables repeat the sign convention on each row. M4
stabilizes these field-correction columns in addition to the base refraction
columns documented in `refraction_multilayer_time_term.md`.

Source table M4 columns:

| Column | Definition |
|---|---|
| `source_depth_m` | Source depth in meters after direction normalization; zero when disabled. |
| `source_depth_shift_ms` | Source-depth weathering-time shift in milliseconds. |
| `source_depth_status` | Source-depth status such as `not_enabled`, `ok`, `missing_source_depth`, `invalid_source_depth`, `inconsistent_source_depth`, or max-shift/depth status. |
| `uphole_time_ms` | Source uphole time in milliseconds after endpoint aggregation. |
| `uphole_shift_ms` | Uphole correction shift in milliseconds. |
| `uphole_status` | Uphole status such as `not_enabled`, `ok`, `missing_uphole_time`, `invalid_uphole_time`, `inconsistent_uphole_time`, `exceeds_max_abs_uphole_time`, or `inactive_source_endpoint`. |
| `manual_static_shift_ms` | Manual source static shift after sign-convention conversion. |
| `manual_static_status` | Manual source static status such as `not_enabled`, `ok`, `missing_manual_static`, or invalid/unmatched row status. |
| `source_field_shift_ms` | Sum of enabled source field-correction components. |
| `source_field_status` | Legacy alias for `source_field_static_status`. |
| `source_field_static_status` | Source endpoint field-composition status. |
| `source_total_with_field_shift_ms` | `total_applied_shift_ms + source_field_shift_ms` when valid. |

Receiver table M4 columns:

| Column | Definition |
|---|---|
| `manual_static_shift_ms` | Manual receiver static shift after sign-convention conversion. |
| `manual_static_status` | Manual receiver static status such as `not_enabled`, `ok`, `missing_manual_static`, or invalid/unmatched row status. |
| `receiver_field_shift_ms` | Sum of enabled receiver field-correction components. |
| `receiver_field_status` | Legacy alias for `receiver_field_static_status`. |
| `receiver_field_static_status` | Receiver endpoint field-composition status. |
| `receiver_total_with_field_shift_ms` | `total_applied_shift_ms + receiver_field_shift_ms` when valid. |

Trace preview CSVs include:

```text
source_field_shift_ms
receiver_field_shift_ms
trace_field_shift_ms
refraction_trace_shift_ms
final_trace_shift_ms
trace_field_static_status
```

`source_receiver_static_table.npz` stores second-based source/receiver arrays
for source depth, uphole time, manual statics, source/receiver field shifts,
and total-with-field shifts. Disabled components use zero shifts and
`not_enabled` statuses.

When `apply_to_trace_shift=false`, trace preview and solution artifacts still
include `trace_field_shift_s_sorted` and component columns, but
`final_trace_shift_s_sorted` remains the base
`refraction_trace_shift_s_sorted`.

## 9. Static history and double-application guard

Successful public refraction statics jobs include
`refraction_static_history.json`. It records:

- `sign_convention`
- input and optional output file IDs
- `double_application_policy`
- component names and whether each component was included in the trace shift
- cumulative shift artifact and field
- duplicate-component QC and warnings

When field corrections are applied to trace shifts, the cumulative shift field
is `final_trace_shift_s_sorted`. Otherwise it is
`refraction_trace_shift_s_sorted`.

`field_corrections.composition.double_application_policy` controls the
best-effort guard against applying components already present in TraceStore
lineage metadata:

- `warn`: proceed and write a warning.
- `fail`: reject the job with a duplicate-component error.
- `allow`: proceed and record that duplicates were allowed.

The guard reads TraceStore `meta.json` derived-component metadata when present.
It does not inspect SEG-Y static headers and does not maintain a project-wide
static registry.

## 10. Example requests

These examples show the `field_corrections` block to include in a normal
`RefractionStaticApplyRequest`.

Source-depth correction:

```json
{
  "field_corrections": {
    "source_depth": {
      "mode": "weathering_velocity_time",
      "source_depth_byte": 115,
      "source_depth_unit": "m",
      "positive_down": true,
      "max_abs_source_depth_m": 100.0
    },
    "composition": {
      "enabled": true,
      "apply_to_trace_shift": true,
      "invalid_component_policy": "fail",
      "double_application_policy": "warn"
    }
  }
}
```

Uphole-time correction from trace headers:

```json
{
  "field_corrections": {
    "uphole": {
      "mode": "header_time",
      "uphole_time_byte": 95,
      "uphole_time_unit": "ms",
      "positive_time_means_delay": true,
      "max_abs_uphole_time_s": 1.0
    },
    "composition": {
      "enabled": true,
      "apply_to_trace_shift": true,
      "invalid_component_policy": "fail",
      "double_application_policy": "warn"
    }
  }
}
```

Manual statics from artifact tables:

```json
{
  "field_corrections": {
    "manual_static": {
      "mode": "artifact_table",
      "sign_convention": "delay_positive_ms",
      "source_table_artifact": {
        "job_id": "manual-static-job",
        "artifact_name": "source_manual_statics.csv"
      },
      "receiver_table_artifact": {
        "job_id": "manual-static-job",
        "artifact_name": "receiver_manual_statics.csv"
      },
      "allow_missing_endpoints": false
    },
    "composition": {
      "enabled": true,
      "apply_to_trace_shift": true,
      "invalid_component_policy": "skip_invalid_traces",
      "double_application_policy": "fail"
    }
  }
}
```

Manual statics inline:

```json
{
  "field_corrections": {
    "manual_static": {
      "mode": "inline_table",
      "sign_convention": "applied_shift_s",
      "source_inline_table": [
        {
          "endpoint_id": 101,
          "value": 0.004
        }
      ],
      "receiver_inline_table": [
        {
          "endpoint_id": 205,
          "value": -0.002
        }
      ],
      "allow_missing_endpoints": true
    },
    "composition": {
      "enabled": true,
      "apply_to_trace_shift": true,
      "invalid_component_policy": "fail",
      "double_application_policy": "warn"
    }
  }
}
```

## 11. Synthetic validation strategy

Synthetic tests should use known source and receiver endpoints so every
component can be checked independently before composition:

- Source depth: set known positive-down depths and V1, then assert
  `source_depth_shift_s = +source_depth_m / V1_m_s`.
- Uphole: set known header uphole times in seconds and milliseconds, then
  assert the default delay-positive sign gives `uphole_shift_s = -uphole_time_s`.
- Manual statics: exercise `applied_shift_s` and `delay_positive_ms` for
  source and receiver endpoints, duplicate rows, missing endpoints, and
  endpoint-key versus endpoint-id matching.
- Composition: assert source endpoint totals, receiver endpoint totals,
  `trace_field_shift_s`; when `apply_to_trace_shift=true`, assert
  `final_trace_shift_s = refraction_trace_shift_s + trace_field_shift_s`; when
  false, assert `final_trace_shift_s = refraction_trace_shift_s` and
  `applied_field_shift_s = 0`.
- Artifact schema: assert M4 CSV columns, NPZ arrays, QC formulas, manifest
  entries, and disabled-component zero/`not_enabled` behavior.
- Static history: assert cumulative shift field selection and
  `warn`/`fail`/`allow` double-application policy behavior.
- Apply paths: when `apply_to_trace_shift=true`, assert corrected TraceStore
  registration uses `final_trace_shift_s_sorted`; when false, assert it uses
  the base refraction shift.
