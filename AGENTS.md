# Agent Instructions

## Goal
Maintain the FastAPI-based viewer that visualizes 2D seismic data from SEG-Y files via a simple web frontend.

## Guidelines
- Use asynchronous tasks or background threads to keep SEG-Y processing responsive.
- Cache loaded sections to minimize repeated disk access.
- Place new API routes under `app/api` and helper utilities under `app/utils`.
- Store static assets in `app/static`.

## Formatting Policy (Do NOT change indent width)
Python: black==24.8.0 (4 spaces), ruff==0.6.9 (E,F)
JS/TS: prettier==3.x (tabWidth=2, useTabs=false), eslint

## Workflow
1) Edit files.
2) Run checks (fail fast):
   - `python -m compileall -q .`
   - `black --check .`
   - `ruff .`
3) If any check fails:
   - `black .`
   - `prettier --write .`
   - `eslint . --fix`
   Then repeat step 2.

## Constraints - Do NOT reformat unrelated files.
- Only touch files in the current diff.
- Never change indent width or tab/space policy.
- If formatting changes a file, explain why in the commit body.
