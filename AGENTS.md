
## Goal
Maintain the FastAPI-based viewer that visualizes 2D seismic data from SEG-Y files via a simple web frontend.

## What to Read vs. Skip
### Must Read (highest priority)
Entry point: app/main.py (app wiring, static routes).
API aggregation & routers:
app/api/endpoints.py
app/api/routers/{upload.py, section.py, pipeline.py, fbpick.py, picks.py}
app/api/schemas.py, app/api/_helpers.py
Domain & processing (app/utils/):
Core pipeline & ops: pipeline.py, ops.py
First-break & metadata: fbpick.py, segy_meta.py
Models & utilities: model.py, model_utils.py, utils.py, pick_cache_file1d_mem.py
Signal transforms: bandpass.py, denoise.py, predict.py
Frontend glue (API↔UI): app/static/pipeline_ui.js, app/static/api.js, plus index.html and upload.html.
Top-level docs & run: README.md, Dockerfile.

### Nice to Read (as needed)
UI core/layout state: app/static/viewer/**
Dev environment helpers: .devcontainer/**


### Skip (do not load bodies into context)
Test source bodies: app/tests/** → Use execution logs, coverage, and test names only. Exception: golden/property/E2E tests — extract high-level expectations only.
Large/minified vendor assets: app/static/plotly-*.min.js, app/static/msgpack*.min.js, app/static/pako*.min.js.
Generated artifacts, large binaries, images, node_modules/ (if added later).

## Formatting Policy (Do NOT change indent width)
Python: **ruff (formatter) with tabs**; ruff >= 0.6.9
Lint: ruff（E,F）
JS/TS: prettier==3.x (**tabWidth=2, useTabs=false**), eslint

## Workflow
1) Edit files.
2) Run checks (fail fast):
   - `python -m compileall -q .`
   - `ruff format --check .`
   - `ruff .`  # lint (E,F)
3) If any check fails:
   - `ruff format .`
   - `prettier --write .`
   - `eslint . --fix`
   Then repeat step 2.

## Constraints - Do NOT reformat unrelated files.
- Only touch files in the current diff.
- Never change indent width or tab/space policy (Python: **tabs**).
- If formatting changes a file, explain why in the commit body.
- Default to “fail fast” on errors—especially
- Allow fallbacks only via an explicit flag (e.g., allow_fallback=False by default; set to True only when intentionally permitted).
- Always emit loud logs or warnings; silent behavior is forbidden. Ensure these warnings are detectable by CI (e.g., fail the build or surface a gate when [FALLBACK] appears).
