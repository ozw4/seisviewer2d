# Static Correction Core Extraction Manifest

## Purpose

This document defines the initial extraction boundary for moving reusable
static-correction numerical code from `seisviewer2d` into a future package while
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

## Initial Package Surface

Phase 1 should expose a small numerical surface:

```text
seis_statics.trace_shift
seis_statics.datum
seis_statics.residual
```

Internal helpers may live under private modules such as
`seis_statics._array_validation`, but callers should not need to import them
directly.

## Phase 1 Core Candidates

Move or wrap these files first. During migration, update `seisviewer2d` call
sites to import the package surface directly from app-owned adapters or
services, without adding fallback import paths.

| Current file | Target area | Notes |
|---|---|---|
| `app/services/common/array_validation.py` | private package helper | Pure NumPy validation helpers used by datum, residual, and trace-shift code. |
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

## Phase 2+ Time-Term and Refraction Boundary

Time-term and refraction are Phase 2+ extraction items. Do not move them during
the Phase 1 datum/residual/trace-shift package extraction, and do not move the
full refraction stack in one step. Future work should separate pure numerical
and domain modules into `seis_statics` while keeping request handling, artifact
contracts, TraceStore integration, and job runtime behavior in `seisviewer2d`.

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

### Time-Term Candidates for `seis_statics.time_term`

These modules contain the likely future `seis_statics.time_term` numerical
surface, subject to an import audit before any move:

```text
app/services/time_term_types.py
app/services/time_term_moveout.py
app/services/time_term_design_matrix.py
app/services/time_term_sparse_solver.py
app/services/time_term_robust_solver.py
app/services/time_term_apply_shift.py
```

The future package surface should be parameterized with arrays, scalar options,
and small dataclasses. It must preserve the current convention that estimated
time-term delays are converted to applied weathering shifts by negation, and
that final shifts compose datum, residual, and weathering applied shifts without
changing numerical behavior.

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

### Refraction Candidates for `seis_statics.refraction`

These refraction modules are Phase 2+ candidates because they contain
documented domain calculations, status rules, design-matrix construction, or
solver orchestration that can be expressed without FastAPI, AppState, artifact
registries, or TraceStore objects:

```text
app/statics/refraction/domain/**
app/statics/refraction/application/bedrock.py
app/statics/refraction/application/datum.py
app/statics/refraction/application/design_matrix.py
app/statics/refraction/application/half_intercept.py
app/statics/refraction/application/input_model.py
app/statics/refraction/application/multilayer_service.py
app/statics/refraction/application/weathering.py
app/statics/refraction/application/weathering_replacement.py
```

These candidates must keep the canonical Phase 1, Phase 2 cell-V2, and
multi-layer T1LSST behavior intact: first-break equations, endpoint time terms,
midpoint cell assignment, endpoint-local V2 projection, velocity-order
validation, NaN/status handling for invalid conversions, source/receiver static
table semantics, and the repo static-shift convention
`corrected(t) = raw(t - shift_s)`.

Do not extract `app/statics/refraction/application/workflow.py` as a numerical
core module. It remains an app orchestration boundary unless a future issue
first separates its pure computation from artifact writing, background job
status, and corrected TraceStore registration.

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

Any future move of time-term or refraction modules must be behavior-preserving
and must not introduce imports from `seis_statics` back into `app`. Before the
move, add or confirm focused tests around the existing app modules; after the
move, run the same tests against the package imports and app adapters.

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

1. Create this manifest and agree on the extraction boundary.
2. Add a future `seis_statics` package skeleton with tests copied or adapted
   around the Phase 1 core modules.
3. Move private array validation plus trace-shift utilities, then update
   `seisviewer2d` services to import the new package surface directly.
4. Move datum static math and update `datum_static_service.py` to import the
   package surface without changing job artifacts or responses.
5. Move residual static types, design matrix, sparse solver, and robust solver;
   update residual services to import the package surface.
6. Run full `seisviewer2d` checks and compare existing datum/residual artifact
   and API response shapes.
7. Only after Phase 1 is stable, move narrowly scoped time-term numerical
   modules behind `seis_statics.time_term`.
8. Move refraction domain pieces incrementally behind `seis_statics.refraction`
   only after the app-only adapters, artifact writers, API schemas, and runtime
   boundaries are explicit.
