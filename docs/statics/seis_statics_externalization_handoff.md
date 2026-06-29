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

The split preserves the current dependency direction:

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

`seisviewer2d` pins the external package at immutable release tag `v0.4.1`:

```text
seis-statics @ git+https://github.com/ozw4/seis-statics.git@v0.4.1
```

## Modules Left in seisviewer2d

The following responsibilities intentionally remain in `seisviewer2d`:

- FastAPI routers and request dependencies.
- Pydantic HTTP request and response schemas.
- SEG-Y readers, `TraceStore`, registry, and cache integration.
- Job lifecycle, background workers, artifact registries, and cleanup.
- Static correction artifact writing and public download packaging.
- Browser UI assets under `app/static`.
- Uploaded pick NPZ handling.
- UI-facing schema.
- Request and app model to `seis_statics` dataclass adapters.
- External result to app DTO and artifact mapping.
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
time-term calculations is owned by `seis-statics` as of release `v0.4.1`.
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

Use the immutable release tag when installing the package for
`seisviewer2d`:

```bash
pip install "seis-statics @ git+https://github.com/ozw4/seis-statics.git@v0.4.1"
```

Do not rely on `main` branch installs, editable installs in CI, sibling
checkout path injection, symlinks, `PYTHONPATH`, runtime `sys.path`
manipulation, or a repository-root `seis_statics/` package. A local
`/workspaces/seis-statics` checkout may be bind-mounted read-only for
inspection during development, but CI and acceptance checks must reproduce with
the tag install.

## Version Milestones

Suggested initial milestones:

```text
v0.1: trace shift + datum + S/R residual solver
v0.2: first-break residual compatibility API
v0.3: generic additive solver S/R/CMP/offset
v0.4.1: time-term/refraction core externalization finalization
```

`seisviewer2d` currently pins the external package at immutable release tag
`v0.4.1`:

```text
seis-statics @ git+https://github.com/ozw4/seis-statics.git@v0.4.1
```

The pin is recorded in `.devcontainer/requirements-dev.txt` and used by CI
through the same requirements file. Do not replace it with `main`, a sibling
checkout, an editable install, `PYTHONPATH`, symlinks, or runtime path
manipulation.

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

```text
app.statics.refraction.domain
seis_statics._
from scipy import optimize
import scipy.optimize
scipy.optimize
optimize.lsq_linear
optimize.least_squares
lsq_linear(
```

Use these checks before changing the external package pin or refraction import
surface:

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
`least_squares` are also not boundary violations.

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

## Guardrails Before Copy-Out

Before changing the external package pin or import surface, verify:

- `seis_statics` imports resolve outside the `seisviewer2d` repository.
- Existing `seisviewer2d` wrapper imports still resolve.
- Package tests pass in `seis-statics` for the release tag or immutable commit
  being pinned.
- Static-shift sign conventions and API response shapes are unchanged.
- No FastAPI, artifact, cache, job, or SEG-Y reader dependency has crossed into
  the package.
