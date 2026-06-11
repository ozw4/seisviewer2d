# seis_statics Externalization Handoff

## Purpose

This handoff describes the next repository split for the reusable statics
numerical core currently living in `seisviewer2d`.

The recommended external repository name is:

```text
seis-statics
```

The Python import package name remains:

```text
seis_statics
```

The split should preserve the current dependency direction:

```text
seis_statics  <-- imported by --  seisviewer2d
seis_statics  <-- imported by --  seisai
```

`seis_statics` must not import from `app.*` or any other `seisviewer2d`
application module.

## Initial Export Surface

The first external package should export the Phase 1 numerical surface already
present in this repository:

```text
seis_statics.trace_shift
seis_statics.datum
seis_statics.residual
```

The initial copy-out should include:

```text
seis_statics/__init__.py
seis_statics/_validation.py
seis_statics/trace_shift.py
seis_statics/datum/**
seis_statics/residual/**
```

Keep private helpers private unless an external caller has a documented need
for them. The package-level APIs should continue accepting arrays, scalars,
small dataclasses, and simple Python values rather than application objects.

## Modules Left in seisviewer2d

The following responsibilities intentionally remain in `seisviewer2d` during
the initial split:

- FastAPI routers and request dependencies.
- Pydantic HTTP request and response schemas.
- SEG-Y readers, `TraceStore`, registry, and cache integration.
- Job lifecycle, background workers, artifact registries, and cleanup.
- Static correction artifact writing and public download packaging.
- Browser UI assets under `app/static`.
- Refraction workflow orchestration and app adapters.

The existing extraction manifest remains the detailed boundary reference:

```text
docs/statics/static_core_extraction_manifest.md
```

## Suggested External Repository Layout

Use a small source-layout package so tests exercise the installed package:

```text
seis-statics/
  pyproject.toml
  README.md
  src/
    seis_statics/
      __init__.py
      _validation.py
      trace_shift.py
      datum/
        __init__.py
        math.py
      residual/
        __init__.py
        api.py
        design_matrix.py
        result.py
        robust.py
        solver.py
        types.py
  tests/
    test_datum_math.py
    test_first_break_residual_api.py
    test_no_app_dependency.py
    test_residual_design_matrix.py
    test_residual_robust_solver.py
    test_residual_sparse_solver.py
    test_source_receiver_lag_api.py
    test_trace_shift.py
```

When copying tests, keep only package-level tests that import `seis_statics`.
Tests that import `app.*` remain in `seisviewer2d`.

## Dependency Policy

For the MVP, keep the core package limited to numerical dependencies:

```text
numpy
scipy
```

Do not add FastAPI, Pydantic, SEG-Y readers, job managers, artifact stores,
frontend tooling, or `seisviewer2d` app modules to the external package.

Development and test tooling can be dev-only dependencies in the external
repository.

## Local Development Workflow

Use sibling checkouts while the package is still shared across repositories:

```bash
# future layout
workspace/
  seis-statics/
  seisviewer2d/
  seisai/

cd workspace/seisviewer2d
pip install -e ../seis-statics
```

During the transition, run the package tests in `seis-statics` first, then run
the `seisviewer2d` tests that exercise wrapper imports and application
integration.

## Version Milestones

Suggested initial milestones:

```text
v0.1: trace shift + datum + S/R residual solver
v0.2: first-break residual compatibility API
v0.3: generic additive solver S/R/CMP/offset
v0.4: time-term/refraction domain extraction
```

Do not include Phase 2+ refraction or time-term extraction in the first release
unless a separate issue redefines the boundary and validates it against the
canonical statics design docs.

## Guardrails Before Copy-Out

Before creating the external repository, verify:

- `seis_statics` has no `app.*` imports.
- Existing `seisviewer2d` wrapper imports still resolve.
- Package tests pass from an installed editable checkout.
- Static-shift sign conventions and API response shapes are unchanged.
- No FastAPI, artifact, cache, job, or SEG-Y reader dependency has crossed into
  the package.

