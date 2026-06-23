# seis_statics Externalization Handoff

## Purpose

This handoff records the repository split for the reusable statics numerical
core now consumed by `seisviewer2d` from the external `seis-statics` package.
The split is complete for the currently supported trace-shift, datum,
residual, time-term, and refraction numerical core surface.

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

## External Export Surface

The external package exports the numerical surface consumed by this repository:

```text
seis_statics.validation
seis_statics.trace_shift
seis_statics.datum
seis_statics.residual
seis_statics.time_term
seis_statics.refraction
```

The local `seis_statics/` package has been removed from `seisviewer2d`; imports
must resolve to the installed external distribution. App code should use public
package modules and avoid private helpers such as `seis_statics._validation`.
The package-level APIs should continue accepting arrays, scalars, small
dataclasses, and simple Python values rather than application objects.

## Modules Left in seisviewer2d

The following responsibilities intentionally remain in `seisviewer2d` during
the initial split:

- FastAPI routers and request dependencies.
- Pydantic HTTP request and response schemas.
- SEG-Y readers, `TraceStore`, registry, and cache integration.
- Job lifecycle, background workers, artifact registries, and cleanup.
- Static correction artifact writing and public download packaging.
- Browser UI assets under `app/static`.
- Refraction workflow orchestration, app adapters, HTTP contracts, artifact
  writers, TraceStore registration, cache integration, and job lifecycle.

The existing extraction manifest remains the detailed boundary reference:

```text
docs/statics/static_core_extraction_manifest.md
```

## External Repository Layout

The external repository uses a source-layout package so tests exercise the
installed package:

```text
seis-statics/
  pyproject.toml
  README.md
  src/
    seis_statics/
      __init__.py
      validation.py
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
      time_term/
        __init__.py
        apply_shift.py
        design_matrix.py
        moveout.py
        robust_solver.py
        sparse_solver.py
        types.py
      refraction/
        __init__.py
        bedrock.py
        cell_coordinates.py
        cell_grid.py
        cell_regularization.py
        datum.py
        design_matrix.py
        field_composition.py
        first_layer.py
        half_intercept.py
        layer_config.py
        layer_observations.py
        manual_static.py
        multilayer_conversion.py
        multilayer_solver.py
        solver.py
        source_depth.py
        status.py
        t1lsst.py
        uphole.py
        v1.py
        weathering.py
        weathering_replacement.py
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

Package-level numerical unit tests live in `seis-statics`. Tests that import
`app.*`, verify HTTP/artifact contracts, or smoke-test app consumers remain in
`seisviewer2d`.

Pure refraction core coverage for cell grids, line projection, smoothing,
direct-arrival V1, T1LSST formulas, weathering, datum, solver, and multi-layer
time-term calculations is owned by `seis-statics` as of release `v0.4.0`.
`seisviewer2d` retains application integration, import-boundary, API,
artifact, TraceStore, and consumer smoke tests.

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

Use an immutable release tag or commit when installing the package for
`seisviewer2d`:

```bash
pip install "seis-statics @ git+https://github.com/ozw4/seis-statics.git@v0.4.0"
```

Do not rely on sibling checkout path injection, symlinks, editable local
fallbacks, or runtime `sys.path` manipulation.

## Version Milestones

Suggested initial milestones:

```text
v0.1: trace shift + datum + S/R residual solver
v0.2: first-break residual compatibility API
v0.3: generic additive solver S/R/CMP/offset
v0.4: time-term/refraction domain extraction
```

`seisviewer2d` currently pins the external package at immutable release tag
`v0.4.0`:

```text
seis-statics @ git+https://github.com/ozw4/seis-statics.git@v0.4.0
```

The pin is recorded in `.devcontainer/requirements-dev.txt` and used by CI
through the same requirements file. Do not replace it with `main`, a sibling
checkout, an editable install, `PYTHONPATH`, symlinks, or runtime path
manipulation.

## Guardrails Before Copy-Out

Before changing the external package pin or import surface, verify:

- `seis_statics` imports resolve outside the `seisviewer2d` repository.
- Existing `seisviewer2d` wrapper imports still resolve.
- Package tests pass in `seis-statics` for the release tag or immutable commit
  being pinned.
- Static-shift sign conventions and API response shapes are unchanged.
- No FastAPI, artifact, cache, job, or SEG-Y reader dependency has crossed into
  the package.
