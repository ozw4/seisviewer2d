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

## Phase 2 Extraction Candidates

These modules contain numerical or domain logic that may be suitable for a
later extraction after the Phase 1 datum/residual/trace-shift boundary is
stable:

```text
app/services/time_term_*.py
app/statics/refraction/domain/**
app/statics/refraction/application/bedrock.py
app/statics/refraction/application/datum.py
app/statics/refraction/application/design_matrix.py
app/statics/refraction/application/half_intercept.py
app/statics/refraction/application/input_model.py
app/statics/refraction/application/multilayer_service.py
app/statics/refraction/application/station_structure.py
app/statics/refraction/application/weathering.py
app/statics/refraction/application/weathering_replacement.py
app/statics/refraction/application/workflow.py
```

Before moving these, verify the boundary against the canonical refraction docs:

```text
docs/statics/refraction_iras_phase1_design.md
docs/statics/refraction_iras_phase2_cell_v2_design.md
docs/statics/refraction_multilayer_time_term.md
```

The package boundary for refraction must preserve the documented T1LSST
workflows, supported velocity modes, source/receiver static table semantics,
and static-shift sign convention.

## Phase 3 or App-Only Candidates

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
app/statics/refraction/ports/**
app/statics/refraction/adapters/seisviewer2d/**
```

Artifact writers and QC bundle builders may contain reusable formatting logic,
but their current responsibility is coupled to `seisviewer2d` artifact
contracts. `ports/**` and `adapters/seisviewer2d/**` describe application
integration points and should remain on the app side unless a future
refraction-core package explicitly owns a port interface.

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
7. Only after Phase 1 is stable, define the separate time-term/refraction
   extraction boundary.
