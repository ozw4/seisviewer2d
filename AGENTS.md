
## Goal
Maintain the FastAPI-based viewer that visualizes 2D seismic data from SEG-Y files via a simple web frontend.

### Skip (do not load bodies into context)
Test source bodies: app/tests/** → Use execution logs, coverage, and test names only. Exception: golden/property/E2E tests — extract high-level expectations only.
Large/minified vendor assets: app/static/plotly-*.min.js, app/static/msgpack*.min.js, app/static/pako*.min.js.

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
