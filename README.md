# seisviewer2d

`seisviewer2d` is a small **FastAPI** app for exploring 2D (or pseudo-2D) seismic data stored in **SEG-Y**.

It ships with a browser-based viewer (`/upload` → `/`) that can:

- ingest a SEG-Y into a cached **TraceStore** (NumPy-backed, section-friendly layout)
- fetch section windows efficiently using **gzip + msgpack + uint8 quantization**
- run a small processing **pipeline** (currently: `bandpass`, `denoise`, `fbpick`)
- create/edit **manual picks** and export them as `.npy`

> Tip: once the server is running, open the interactive API docs at `/docs`.

## Quick start

### Local (Python)

1. Install dependencies:

```bash
python -m pip install -r .devcontainer/requirements-dev.txt
```

2. Start the server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Open the UI:

- Upload/open page: `http://localhost:8000/upload`
- Viewer: `http://localhost:8000/`

### Dev container (recommended for GPU + UI tests)

This repo includes a dev container based on an NVIDIA PyTorch image (see `Dockerfile`).
It installs Python deps, Node 20, and Playwright for the UI tests.

## What happens on upload

`POST /upload_segy` converts the uploaded SEG-Y into a cached “trace store” under:

```
app/api/uploads/processed/traces/<safe_filename>/
  traces.npy
  index.npz
  meta.json
  baseline_raw.json
```

Trace grouping and ordering are controlled by two SEG-Y trace-header words:

- `key1_byte` (default `189`) groups traces into a “section” (one section per key1 value)
- `key2_byte` (default `193`) orders traces within a section (stable sort)

The UI lets you choose both bytes.

### Reusing an existing TraceStore

- `POST /upload_segy` will reuse an existing trace store when it matches **(key1_byte, key2_byte)** and the source file hash.
- `POST /open_segy` opens an existing trace store by filename (used by the upload page to “re-open last data”).

## Model weights (optional)

Two pipeline operations use PyTorch weights that are **not included** in this repository:

- `denoise` transform: expects `model/recon_replace_eq_edgenext_small.pth`
- `fbpick` analyzer: expects `model/fbpick_edgenext_small.pth`

Create a `model/` directory at the repo root and place the weights there:

```
model/
  recon_replace_eq_edgenext_small.pth
  fbpick_edgenext_small.pth
```

If the weights are missing, the FB-pick endpoints will return HTTP 409.

## Frontend

The UI is served from `app/static/`.

- `/upload` lets you upload a SEG-Y or re-open the previously processed dataset.
- `/` is the viewer with a side panel for pipeline steps.

Keyboard shortcuts:

- `N`: toggle between the `raw` layer and the first computed layer in the layer dropdown
- Hold `Alt`: temporary pan mode

### Rebuilding frontend assets (optional)

Static assets are committed, but if you modify the web sources you can rebuild:

```bash
cd app
npm install
npm run build
```

The build output is written into `app/static/assets/` (see `app/vite.config.ts`).

## API overview

Most UI reads use the binary window endpoint (fast path).

### Upload / open

- `POST /upload_segy` (form-data): upload SEG-Y, build/reuse trace store → `{file_id, reused_trace_store}`
- `POST /open_segy` (form-data): open an existing trace store by `original_name` → `{file_id, reused_trace_store}`
- `GET /file_info?file_id=...`: return the stored dataset basename

### Sections

- `GET /get_key1_values?file_id=...&key1_byte=...&key2_byte=...`
- `GET /get_section?file_id=...&key1=...` (JSON, mostly for debugging)
- `GET /get_section_meta?file_id=...`: shape/dt/dtype/scale
- `GET /section/stats?file_id=...&baseline=raw`: per-file stats used for normalization

#### Fast window fetch

`GET /get_section_window_bin?file_id=...&key1=...&x0=...&x1=...&y0=...&y1=...`

- returns `application/octet-stream`
- response body is **gzip-compressed msgpack**
- payload contains:
  - `shape`: window shape (after transpose if requested)
  - `scale`: float scale for de-quantization
  - `data`: raw bytes (uint8) of the quantized window
  - `dt`: sample interval (seconds)

Parameters:

- `step_x`, `step_y`: integer downsampling for traces/samples
- `transpose` (default `true`): matches the viewer’s expected orientation
- `scaling`: `amax` (section-wise) or `tracewise` normalization using baseline stats
- `pipeline_key` + `tap_label`: fetch from a cached pipeline tap instead of raw

### Pipeline

Pipelines are defined by `PipelineSpec` (`steps[]`), where each step has:

- `kind`: `transform` or `analyzer`
- `name`: operation name
- `params`: operation parameters
- `label` (optional): label shown in the UI

Currently registered operations:

- transforms: `bandpass`, `denoise`
- analyzers: `fbpick`

Endpoints:

- `POST /pipeline/section?file_id=...&key1=...`: run a pipeline on one section
  - `list_only=true` caches taps and only returns tap labels
- `POST /pipeline/all?file_id=...`: run the pipeline for all key1 values (background)
- `GET /pipeline/job/{job_id}/status`
- `GET /pipeline/job/{job_id}/artifact?key1=...&tap=...`

### First-break picking

- `POST /fbpick_section_bin`: start an async job to compute a probability map
- `GET /fbpick_job_status?job_id=...`
- `GET /get_fbpick_section_bin?job_id=...`: fetch probability map (gzip+msgpack, quantized)

For “picks from probability” (server-side):

- `POST /fbpick_predict`: returns `{dt, picks[]}` where `picks` contains `{trace, time}`
  - supports `method=argmax|expectation` and a `sigma_ms_max` gate

### Manual picks

Manual picks are stored in a memmapped `.npy` file per dataset.

- `PICKS_NPY_DIR` (highest priority): direct override for memmap files.
- Default when unset: `<app_data_dir>/picks_npy`

Pipeline job artifacts are persisted on disk.

- `PIPELINE_JOBS_DIR` (highest priority): direct override for pipeline artifacts.
- Default when unset: `<app_data_dir>/pipeline_jobs`

`<app_data_dir>` is resolved in this order:

- `SV_APP_DATA_DIR`
- `RUNNER_TEMP/seisviewer2d_app_data` (CI-friendly default)
- `XDG_CACHE_HOME/seisviewer2d`
- `~/.cache/seisviewer2d`

Endpoints:

- `GET /picks?file_id=...&key1=...&key1_byte=...`
- `POST /picks` (JSON): `{file_id, trace, time, key1, key1_byte}`
- `DELETE /picks?file_id=...&key1=...&key1_byte=...&trace=...`
- `GET /export_manual_picks_all_npy?file_id=...`: export all key1 sections to a 2D int32 `.npy`

## Project layout

```
app/
  main.py                 # FastAPI app + static mounting
  api/
    routers/              # upload/section/pipeline/picks/fbpick
    _helpers.py           # shared state/cache helpers (to be refactored)
    baselines.py          # raw baseline stats used for scaling
    schemas.py            # pydantic models for requests/responses
  static/                 # HTML/JS viewer served at / and /upload
  utils/                  # SEG-Y ingest, TraceStore reader, pipeline ops
  tests/                  # backend + frontend smoke tests
.devcontainer/requirements-dev.txt
Dockerfile
```

## Tests

Backend:

```bash
pytest -q
```

Frontend unit tests (optional):

```bash
cd app
npm install
npx vitest run
```
