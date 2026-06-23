# Static Correction Core Extraction Manifest

## Purpose

This document defines the extraction boundary for reusable static-correction
numerical code that now lives in the external `seis-statics` package while
keeping `seisviewer2d` as an application consumer.

The proposed package and import name is:

```text
seis_statics
```

The dependency direction must be one-way:

```text
seis_statics  <-- imported by --  seisviewer2d
seis_statics  <-- imported by --  seisai
```

`seis_statics` must not import from `seisviewer2d`, and must not depend on
FastAPI, `AppState`, `TraceStore`, SEG-Y readers, job lifecycle management,
artifact registries, background workers, browser UI code, or
`seisviewer2d` API response schemas.

## Current Package Surface

The external package exposes the application-consumed numerical surface:

```text
seis_statics.validation
seis_statics.trace_shift
seis_statics.datum
seis_statics.residual
seis_statics.time_term
seis_statics.refraction
```

`seisviewer2d` must import public package modules only. In particular, app code
must not import private helpers such as `seis_statics._validation` directly.

## Phase 1 Core Ownership

The Phase 1 core files are owned by `seis-statics`; the local `seis_statics/`
shadow package has been removed from this repository. `seisviewer2d` call sites
import the package surface directly from app-owned adapters or services,
without fallback import paths.

| Current file | Target area | Notes |
|---|---|---|
| `app/services/common/array_validation.py` | `seis_statics.validation` | Public validation helper imports used by app-side compatibility code. |
| `app/utils/time_shift.py` | `seis_statics.trace_shift` | Trace-wise interpolation utility. |
| `app/statics/common/trace_shift.py` | `seis_statics.trace_shift` | Static shift validation and array application. Must keep the repo sign convention: `corrected(t) = raw(t - shift_s)`. |
| `app/services/datum_static_math.py` | `seis_statics.datum` | Datum static numerical calculation and result type. |
| `app/services/residual_static_types.py` | `seis_statics.residual` | Residual-static solver input dataclasses and lightweight type aliases. |
| `app/services/residual_static_design_matrix.py` | `seis_statics.residual` | Residual-static matrix layout, triplet construction, unpacking, and model evaluation. |
| `app/services/residual_static_sparse_solver.py` | `seis_statics.residual` | Stabilized sparse least-squares solve and diagnostics. |
| `app/services/residual_static_robust_solver.py` | `seis_statics.residual` | Robust residual rejection loop around the sparse solver. |

The extracted code should be parameterized with arrays, scalars, dataclasses,
and simple Python values. It should not open files, resolve application state,
materialize `TraceStore` objects, write job artifacts, or construct API
responses.

## Phase 1 App Adapters and Services

These files should remain in `seisviewer2d` in Phase 1. They orchestrate
application state, request parsing, artifact writing, TraceStore registration,
or job lifecycle behavior around the numerical core:

```text
app/services/datum_static_service.py
app/services/datum_static_geometry.py
app/services/datum_static_validation.py
app/services/residual_static_service.py
app/services/residual_static_inputs.py
app/services/residual_static_artifacts.py
app/services/residual_static_corrected_store.py
app/services/static_artifacts.py
app/services/corrected_trace_store.py
app/services/job_manager.py
app/services/job_runner.py
app/services/pipeline_artifacts.py
app/services/reader.py
app/services/trace_store_registration.py
app/services/trace_store_headers.py
app/statics/common/corrected_store.py
```

These modules may import `seis_statics` after Phase 1 extraction, but
`seis_statics` must never import them.

## Contracts and API Schemas

API contracts and routers stay in `seisviewer2d` for Phase 1:

```text
app/contracts/statics/**
app/contracts/statics/refraction/**
app/api/routers/statics/**
app/statics/refraction/contracts/**
app/statics/refraction/api/**
```

These modules define the public `seisviewer2d` HTTP contract and should remain
free to use FastAPI/Pydantic/application-specific response models. Phase 1 must
preserve existing `seisviewer2d` API response shapes, artifact names, and schema
field names. Any package-level dataclasses in `seis_statics` should be treated
as internal numerical inputs or outputs, not as HTTP schemas.

## Explicitly Not Moved in Phase 1

Do not move these categories during Phase 1:

- FastAPI routers, dependencies, response builders, or launch/job endpoints.
- Pydantic request and response schemas under `app/contracts`.
- `TraceStore` implementations, readers, registry code, cache keys, or
  corrected-store registration.
- Job/artifact lifecycle code, including background job runners, job metadata,
  artifact path resolution, and cleanup behavior.
- Browser UI code under `app/static`.
- Refraction public workflow modules, until the refraction-specific extraction
  boundary is separately defined.

Phase 1 is a dependency inversion and packaging step only. It must not change
runtime behavior, API responses, cache behavior, artifact contents, or static
shift sign conventions.

## Completed Time-Term and Refraction Boundary

Time-term numerical core code has been extracted to `seis_statics.time_term` in
the external package.
Refraction numerical and domain core code has been extracted to
`seis_statics.refraction` in the external package. `seisviewer2d` keeps request
handling, artifact contracts, TraceStore integration, cache behavior, and job
runtime behavior.

Before moving any refraction module, verify the boundary against the canonical
refraction docs:

```text
docs/statics/refraction_iras_phase1_design.md
docs/statics/refraction_iras_phase2_cell_v2_design.md
docs/statics/refraction_multilayer_time_term.md
```

The package boundary for refraction must preserve the documented T1LSST
workflows, supported velocity modes, source/receiver static table semantics,
and static-shift sign convention.

### Extracted Time-Term Core

The `seis_statics.time_term` package owns the numerical time-term surface:

```text
seis_statics/time_term/types.py
seis_statics/time_term/moveout.py
seis_statics/time_term/design_matrix.py
seis_statics/time_term/sparse_solver.py
seis_statics/time_term/robust_solver.py
seis_statics/time_term/apply_shift.py
```

These modules are parameterized with arrays, scalar options, and small
dataclasses. They must not import `app.*`, FastAPI, Pydantic, `AppState`,
SEG-Y readers, `TraceStore`, job lifecycle helpers, artifact path resolvers, or
API response schemas. They preserve the convention that estimated time-term
delays are converted to applied weathering shifts by negation, and that final
shifts compose datum, residual, and weathering applied shifts without changing
numerical behavior.

The app compatibility modules are thin re-export shims over the extracted core:

```text
app/services/time_term_types.py
app/services/time_term_moveout.py
app/services/time_term_design_matrix.py
app/services/time_term_sparse_solver.py
app/services/time_term_robust_solver.py
app/services/time_term_apply_shift.py
```

These shims are kept for compatibility during migration only. Their
implementation should remain limited to imports from `seis_statics.time_term`
and `__all__` re-exports.

Keep these time-term modules in `seisviewer2d` unless a future issue explicitly
splits out a pure helper from them:

```text
app/services/time_term_static_service.py
app/services/time_term_static_inputs.py
app/services/time_term_static_artifacts.py
app/services/time_term_static_apply_trace_store.py
```

They currently own request orchestration, artifact loading/writing, TraceStore
registration, and job/runtime integration.

### Extracted Refraction Core

The `seis_statics.refraction` package owns documented domain calculations,
status rules, design-matrix construction, and solver orchestration that can be
expressed without FastAPI, AppState, artifact registries, or TraceStore objects:

```text
seis_statics/refraction/bedrock.py
seis_statics/refraction/cell_coordinates.py
seis_statics/refraction/cell_grid.py
seis_statics/refraction/cell_regularization.py
seis_statics/refraction/datum.py
seis_statics/refraction/design_matrix.py
seis_statics/refraction/field_composition.py
seis_statics/refraction/first_layer.py
seis_statics/refraction/half_intercept.py
seis_statics/refraction/layer_config.py
seis_statics/refraction/layer_observations.py
seis_statics/refraction/manual_static.py
seis_statics/refraction/multilayer_conversion.py
seis_statics/refraction/multilayer_solver.py
seis_statics/refraction/solver.py
seis_statics/refraction/source_depth.py
seis_statics/refraction/status.py
seis_statics/refraction/t1lsst.py
seis_statics/refraction/uphole.py
seis_statics/refraction/v1.py
seis_statics/refraction/weathering.py
seis_statics/refraction/weathering_replacement.py
```

The external core must keep the canonical Phase 1, Phase 2 cell-V2, and
multi-layer T1LSST behavior intact: first-break equations, endpoint time terms,
midpoint cell assignment, endpoint-local V2 projection, velocity-order
validation, NaN/status handling for invalid conversions, source/receiver static
table semantics, and the repo static-shift convention
`corrected(t) = raw(t - shift_s)`.

The viewer-side modules with the same workflow names are application adapters:
they translate Pydantic requests and app dataclasses into package options,
write artifacts, compose QC, and preserve public response and artifact shapes.
They must not grow replacement numerical implementations or fallback branches.

`app/statics/refraction/application/workflow.py` remains an app orchestration
boundary: it owns artifact writing, background job status, and corrected
TraceStore registration.

### Refraction App-Only Boundary

These areas are not part of the initial numerical package. Revisit them only
after Phase 1 and Phase 2 boundaries are proven:

```text
app/statics/refraction/artifacts/**
app/statics/refraction/application/apply_trace_store.py
app/statics/refraction/application/export_service.py
app/statics/refraction/application/gather_preview.py
app/statics/refraction/application/job_status.py
app/statics/refraction/application/lsst_export.py
app/statics/refraction/application/pick_map.py
app/statics/refraction/application/pick_source_loader.py
app/statics/refraction/application/preflight_diagnostics.py
app/statics/refraction/application/qc_bundle.py
app/statics/refraction/application/qc_drilldown.py
app/statics/refraction/application/qc_endpoint_search.py
app/statics/refraction/application/table_apply_service.py
app/statics/refraction/application/validation_service.py
app/statics/refraction/application/workflow.py
app/statics/refraction/api/**
app/statics/refraction/contracts/**
app/statics/refraction/ports/**
app/statics/refraction/adapters/seisviewer2d/**
```

Artifact writers and QC bundle builders may contain reusable formatting logic,
but their current responsibility is coupled to `seisviewer2d` artifact
contracts. `ports/**` and `adapters/seisviewer2d/**` describe application
integration points and should remain on the app side unless a future
refraction-core package explicitly owns a port interface.

### Required Test Strategy Before Moving Phase 2+ Modules

Any future move of time-term or refraction modules must be behavior-preserving.
Package-owned numerical unit tests live in `seis-statics`; `seisviewer2d` keeps
application integration, contract, artifact, import-layer, and sign-convention
smoke coverage.

Moved pure-core coverage is recorded here because the external checkout is
read-only for `seisviewer2d` implementation issues:

| Former viewer coverage | External owner | Viewer retained coverage |
|---|---|---|
| Cell grid assignment and line projection unit tests | `seis-statics/tests/parity_manifest.md` and package refraction tests at `v0.4.0` | Cell design-matrix, synthetic E2E, QC, artifact, and API tests |
| Cell smoothing row/unit tests | `seis-statics/tests/parity_manifest.md` and package refraction tests at `v0.4.0` | Cell smoothing synthetic workflow and solver/artifact tests |
| Direct-arrival V1 estimator error/robustness unit tests | `seis-statics/tests/parity_manifest.md` and package refraction tests at `v0.4.0` | V1 artifact smoke and API apply tests |
| T1LSST scalar/vector formula and velocity-order unit tests | `seis-statics/tests/parity_manifest.md` and package refraction tests at `v0.4.0` | T1LSST request, artifact schema, synthetic workflow, and sign-convention tests |

Minimum time-term coverage:

```text
app/tests/test_time_term_import_layering.py
app/tests/test_time_term_moveout.py
app/tests/test_time_term_design_matrix.py
app/tests/test_time_term_sparse_solver.py
app/tests/test_time_term_robust_solver.py
app/tests/test_time_term_apply_shift.py
app/tests/test_time_term_static_artifacts.py
app/tests/test_time_term_static_apply_trace_store.py
app/tests/test_time_term_static_job_api.py
app/tests/test_time_term_static_end_to_end.py
```

Minimum refraction coverage:

```text
app/tests/test_refraction_design_matrix_diagnostics.py
app/tests/test_refraction_failure_diagnostics.py
app/tests/test_refraction_gather_preview.py
app/tests/test_refraction_qc_drilldown.py
app/tests/test_refraction_qc_endpoint_search.py
app/tests/test_refraction_qc_pick_map.py
```

For cell-V2, multi-layer T1LSST, source/receiver table, or apply-artifact
changes, include the synthetic and artifact tests that cite the canonical
refraction docs. Keep import-layer tests in the move PR so accidental FastAPI,
SEG-Y reader, app-state, or job-runtime dependencies fail quickly.

## Migration Order

1. Phase 1 trace-shift, datum, residual, and validation code is externalized.
2. Time-term numerical modules are externalized behind `seis_statics.time_term`.
3. `seisviewer2d` pins `seis-statics` from an immutable release tag and does
   not ship a local `seis_statics/` package.
4. Refraction numerical/domain pieces are externalized behind
   `seis_statics.refraction`; app-only adapters, artifact writers, API schemas,
   and runtime boundaries remain in `seisviewer2d`.
