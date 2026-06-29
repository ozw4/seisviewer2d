# Static Correction Core Extraction Manifest

## Purpose

This document defines the extraction boundary for reusable static-correction
numerical code that now lives in the external `seis-statics` package while
keeping `seisviewer2d` as an application consumer.

The external package import name is:

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

`seisviewer2d` pins the external package at immutable release tag `v0.4.1`:

```text
seis-statics @ git+https://github.com/ozw4/seis-statics.git@v0.4.1
```

The pin is recorded in `.devcontainer/requirements-dev.txt`. Do not replace it
with a `main` branch dependency, editable CI install, `PYTHONPATH` runtime
override, symlink, runtime path manipulation, or repository-root local
`seis_statics/` package. A local `/workspaces/seis-statics` checkout may be
bind-mounted read-only for development inspection, but CI and acceptance checks
must reproduce with the tag install.

## Final Ownership

`seis-statics` owns the reusable numerical core:

- Time-term solver, robust solver, and applied-shift composition.
- Refraction design matrix and solver.
- Bedrock and half-intercept calculations.
- Weathering and replacement calculations.
- Datum statics.
- Multi-layer solve and conversion.
- Trace-shift, residual, and datum primitives.

`seisviewer2d` owns the application boundary:

- FastAPI routes.
- Pydantic request and response contracts.
- `TraceStore` and SEG-Y I/O.
- Uploaded pick NPZ handling.
- Job orchestration.
- Artifact registry, writers, and downloads.
- UI-facing schema.
- Request and app model to `seis_statics` dataclass adapters.
- External result to app DTO and artifact mapping.

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

## Viewer-Owned Application Boundary

These categories remain in `seisviewer2d`:

- FastAPI routers, dependencies, response builders, or launch/job endpoints.
- Pydantic request and response schemas under `app/contracts`.
- `TraceStore` implementations, readers, registry code, cache keys, or
  corrected-store registration.
- Job/artifact lifecycle code, including background job runners, job metadata,
  artifact path resolution, and cleanup behavior.
- Browser UI code under `app/static`.
- Refraction public workflow modules, adapters, and artifact contracts.

The externalization is a dependency inversion and packaging step. It must not change
runtime behavior, API responses, cache behavior, artifact contents, or static
shift sign conventions.

## Completed Time-Term and Refraction Boundary

Time-term numerical core code has been extracted to `seis_statics.time_term` in
the external package.
Refraction numerical and domain core code has been extracted to
`seis_statics.refraction` in the external package. `seisviewer2d` keeps request
handling, artifact contracts, TraceStore integration, cache behavior, and job
runtime behavior.

The completed refraction package boundary follows the canonical refraction
docs:

```text
docs/statics/refraction_iras_phase1_design.md
docs/statics/refraction_iras_phase2_cell_v2_design.md
docs/statics/refraction_multilayer_time_term.md
```

The package boundary for refraction preserves the documented T1LSST
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

These areas are not part of the external numerical package:

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

### Retained Test Strategy

Any future move of app-side time-term or refraction modules must be
behavior-preserving.
Package-owned numerical unit tests live in `seis-statics`; `seisviewer2d` keeps
application integration, contract, artifact, import-layer, and sign-convention
smoke coverage.

Moved pure-core coverage is recorded here because the external checkout is
read-only for `seisviewer2d` implementation issues:

| Former viewer coverage | External owner | Viewer retained coverage |
|---|---|---|
| Cell grid assignment and line projection unit tests | `seis-statics/tests/parity_manifest.md` and package refraction tests at `v0.4.1` | Cell design-matrix, synthetic E2E, QC, artifact, and API tests |
| Cell smoothing row/unit tests | `seis-statics/tests/parity_manifest.md` and package refraction tests at `v0.4.1` | Cell smoothing synthetic workflow and solver/artifact tests |
| Direct-arrival V1 estimator error/robustness unit tests | `seis-statics/tests/parity_manifest.md` and package refraction tests at `v0.4.1` | V1 artifact smoke and API apply tests |
| T1LSST scalar/vector formula and velocity-order unit tests | `seis-statics/tests/parity_manifest.md` and package refraction tests at `v0.4.1` | T1LSST request, artifact schema, synthetic workflow, and sign-convention tests |

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

1. 026: external package dependency and local package removal completed.
2. 027: adapter and request conversion completed.
3. 028: geometry, cell, and layer helpers externalized.
4. 029: field corrections, V1, and T1LSST externalized.
5. 030: design matrix externalized.
6. 031: design matrix canonical external type preserved.
7. 032: solver, bedrock, and half-intercept externalized.
8. 033: weathering, replacement, and datum externalized.
9. 034: multilayer externalized.
10. 035: app-owned export, table, and QC moved.
11. 036/final: dependency, docs, and CI finalization.

## Deleted Viewer Paths

These viewer-side packages have been removed and must remain absent:

```text
seisviewer2d/seis_statics/
app/statics/refraction/domain/
```

Check the current checkout with:

```bash
test ! -d seis_statics
test ! -d app/statics/refraction/domain
```

## Import Boundary Checks

Production viewer code must not import the legacy refraction domain package,
private `seis_statics` modules, or direct SciPy optimization entry points for
viewer-side refraction solving:

```bash
grep -RInE "app\.statics\.refraction\.domain" app tests || true
grep -RInE "from seis_statics\._|import seis_statics\._" app tests || true
grep -RInE \
  "from scipy import optimize|import scipy\.optimize|scipy\.optimize|optimize\.lsq_linear|optimize\.least_squares|lsq_linear\(" \
  app/statics/refraction || true
```

Calls to public external solver APIs such as
`seis_statics.refraction.solve_refraction_static_design_least_squares` are the
correct dependency direction. Artifact and QC status strings containing
`least_squares` are not boundary violations.

Verify the installed import origin and version with:

```bash
python - <<'PY'
import importlib.metadata
import seis_statics

print(seis_statics.__file__)
print(importlib.metadata.version("seis-statics"))
assert importlib.metadata.version("seis-statics") == "0.4.1"
PY
```

## Verified Final State

The externalization finalization recorded these successful checks:

```text
seis-statics:
  pytest: 471 passed
  ruff: All checks passed
  compileall: OK
  git diff --check: OK

seisviewer2d:
  tests/test_external_seis_statics_dependency.py: 6 passed
  app/tests/test_refraction_static_ui_fixture_direct_npz.py: 1 passed
  app/tests -k 'refraction and datum': 41 passed
  app/tests -k 'refraction and multilayer': 131 passed
  full pytest: 2590 passed
```

Known warning: `StarletteDeprecationWarning` from `fastapi.testclient` is not a
blocking externalization issue.

## Actions Re-Run Checklist

After dependency or release-cache changes, re-run the relevant GitHub Actions:

```text
seis-statics:
  pytest
  ruff
  release-readiness

seisviewer2d:
  full pytest
  external dependency test
  UI fixture direct NPZ
  refraction datum
  refraction multilayer
```

If Actions fail after the tag bump, confirm that the remote `seis-statics`
`v0.4.1` tag exists, pip installs `v0.4.1`, and the job is not importing an old
system site-packages `seis-statics 0.4.0`.

## Release Distribution Policy

`seis-statics` distribution artifacts are not Git-managed source files:

- Keep `dist/` out of Git.
- Publish wheel and sdist files through GitHub Releases or CI artifacts.
- Do not commit `dist/*.whl` or `dist/*.tar.gz` to either repository.
