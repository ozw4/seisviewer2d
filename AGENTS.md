# Agent Instructions

## Goal
Maintain `seisviewer2d`, a FastAPI application for browsing 2D seismic data from SEG-Y files with a small HTML/JavaScript frontend.

## Architecture
- FastAPI entrypoint: `app/main.py`
- API routers: `app/api/routers`
- Shared API helpers: `app/api/_helpers.py`
- SEG-Y readers and metadata helpers: `app/utils`
- Static frontend assets: `app/static`

## Work Rules
- Start from a GitHub issue.
- Restate acceptance criteria before changing code.
- Keep changes scoped to the current issue.
- Prefer extending existing routers and helpers over creating parallel code paths.
- Keep heavy SEG-Y I/O off the request path where practical.
- Preserve API response shapes unless the issue explicitly changes them.

## Checks
Run after code changes:
- `python -m compileall -q app`
- `ruff app`
- `pytest`

If the issue touches browser behavior, also run:
- `pytest -k playwright`

## Constraints
- Do not reformat unrelated files.
- Do not add fallback behavior unless the issue requires it.
- Keep background processing explicit and easy to reason about.
- Explain cache, API, and schema changes in the PR body.
