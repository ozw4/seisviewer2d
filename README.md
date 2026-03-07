# seisviewer2d

`seisviewer2d` is a small **FastAPI** app for exploring 2D (or pseudo‑2D) seismic data stored in **SEG‑Y**.

It ships with a browser-based viewer (`/upload` → `/`) that can:

- ingest a SEG‑Y into a cached **TraceStore** (NumPy‑backed, section‑friendly layout)
- fetch section windows efficiently using **gzip + msgpack + uint8 quantization**
- run a small processing **pipeline** (currently: `bandpass`, `denoise`, `fbpick`)
- create/edit **manual picks** stored in a memmapped `.npy` and export them as `.npz` / `.txt`

Once the server is running, open the interactive API docs at `/docs`.

## Quick start

### Local (Python)

1) Install Python dependencies:

```bash
python -m pip install -r .devcontainer/requirements-dev.txt
```

2) Build the frontend bundle (required if `app/static/assets/main.js` is missing):

```bash
cd app
npm install
npm run build
```

3) Start the server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4) Open the UI:

- Upload/open page: `http://localhost:8000/upload`
- Viewer: `http://localhost:8000/`

### Dev container (recommended for GPU and browser testing)

This repo includes a dev container based on an NVIDIA PyTorch image (see `Dockerfile`).
It installs Python deps, Node 20, and Playwright.

## What happens on upload

`POST /upload_segy` converts the uploaded SEG‑Y into a cached “trace store” under:

```
<app_data_dir>/uploads/processed/traces/<safe_filename>/
  traces.npy
  index.npz
  meta.json
  baseline_raw.json
```

Notes:

- `baseline_raw.json` is created lazily (first requested via `/get_section_meta` or `/section/stats`).
- Trace grouping and ordering are controlled by two SEG‑Y trace-header byte offsets:
  - `key1_byte` (default `189`) groups traces into a “section” (one section per key1 value)
  - `key2_byte` (default `193`) orders traces within a section (stable sort)

The upload UI lets you choose both bytes.

### Reusing an existing TraceStore

- `POST /upload_segy` will reuse an existing trace store when it matches `(key1_byte, key2_byte)` and the source file hash.
- `POST /open_segy` opens an existing trace store by `original_name` (used by the upload page to re-open previous data).

If an on-disk trace store exists but does not match the requested bytes or the file hash, it is moved aside with an `.old-<uuid>` suffix.

## Model weights

Two pipeline operations use PyTorch weights that are **not included** in this repository:

- `denoise` transform: expects `model/denoise_default.pt`
- `fbpick` analyzer: expects `model/fbpick_edgenext_small.pt`

Create a `model/` directory at the repo root and place the weights there:

```
model/
  denoise_default.pt
  fbpick_edgenext_small.pt
```

Behavior when weights are missing:

- `fbpick` endpoints return HTTP 409 (`FB pick model weights not found`).
- Any pipeline that includes `denoise` or `fbpick` will fail when it reaches that step.

Offset-aware `fbpick` variant (optional): the code supports injecting an “offset channel” when the fbpick weights filename contains `offset` (see `app/services/fbpick_support.py`).

## Frontend

The UI is served from `app/static/`.

- `/upload` lets you upload a SEG‑Y or re-open a previously processed dataset.
- `/` is the viewer with a side panel for pipeline steps.
- `/batch` is the batch-apply page for running pipeline jobs over all key1 values.

The main viewer page (`app/static/index.html`) expects a small Vite-built bundle at `/static/assets/main.js`.
If you change frontend sources under `app/web/`, rebuild with:

```bash
cd app
npm install
npm run build
```

Keyboard shortcuts:

- `N`: toggle between the `raw` layer and the first computed layer in the layer dropdown
- Hold `Alt`: temporary pan mode

## API overview

Most UI reads use the binary window endpoint (`/get_section_window_bin`) as the fast path.

### Upload / open

- `POST /upload_segy` (multipart/form-data): upload SEG‑Y, build/reuse trace store
  - form fields: `file` (file), `key1_byte` (int), `key2_byte` (int)
  - returns: `{ "file_id": "...", "reused_trace_store": true|false }`
- `POST /open_segy` (multipart/form-data): open an existing trace store by `original_name`
  - form fields: `original_name` (string), `key1_byte` (int), `key2_byte` (int)
  - returns: `{ "file_id": "...", "reused_trace_store": true|false }`
- `GET /file_info?file_id=<FILE_ID>`: return the stored dataset basename

### Sections

- `GET /get_key1_values?file_id=<FILE_ID>&key1_byte=<BYTE>&key2_byte=<BYTE>`
- `GET /get_section?file_id=<FILE_ID>&key1=<KEY1>` (JSON; mainly for debugging)
- `GET /get_section_meta?file_id=<FILE_ID>&key1_byte=<BYTE>&key2_byte=<BYTE>`: shape/dt/dtype/scale
- `GET /section/stats?file_id=<FILE_ID>&baseline=raw&key1_byte=<BYTE>&key2_byte=<BYTE>`: per-file stats used for normalization

#### Fast window fetch

`GET /get_section_window_bin` returns `application/octet-stream` with `Content-Encoding: gzip`.
The body is msgpack of a quantized uint8 array.

Required query params:

- `file_id`, `key1`
- `x0`, `x1` (trace indices, inclusive)
- `y0`, `y1` (sample indices, inclusive)

Optional query params:

- `key1_byte` (default `189`), `key2_byte` (default `193`)
- `step_x`, `step_y` (integer downsampling, default `1`)
- `transpose` (default `true`, matches the viewer’s expected orientation)
- `scaling`: `amax` (section-wise) or `tracewise` normalization using baseline stats
- `pipeline_key` + `tap_label`: fetch from a cached pipeline tap instead of raw
- `offset_byte`: included in the pipeline-tap cache key (use the same value you used when generating taps)

Payload fields:

- `shape`: window shape (after transpose if requested)
- `scale`: float scale for de-quantization
- `data`: raw bytes (uint8) of the quantized window
- `dt`: sample interval (seconds)

### Pipeline

Pipeline request bodies are JSON objects shaped like
`{"spec": ..., "taps": ..., "window": ...}`.
`spec` is required, while `taps` and `window` are optional.
`window` is accepted only by `POST /pipeline/section`, and it is not allowed when
`list_only=true`.

`spec` is a `PipelineSpec` (`steps[]`), where each step has:

- `kind`: `transform` or `analyzer`
- `name`: operation name
- `params`: operation parameters
- `label` (optional): label shown in the UI

Currently registered operations:

- transforms: `bandpass`, `denoise`
- analyzers: `fbpick`

Endpoints:

- `POST /pipeline/section?file_id=<FILE_ID>&key1=<KEY1>`: run a pipeline on one section
  - query params: `key1_byte` (default `189`), `key2_byte` (default `193`), `offset_byte` (optional), `list_only` (bool)
  - body: `{"spec": <PipelineSpec>, "taps": ["tapA", "tapB"], "window": {"tr_min": 0, "tr_max": 128, "t_min": 0, "t_max": 1024}}`
  - `list_only=true` caches taps and returns only tap labels (window slicing is not supported with `list_only=true`)
- `POST /pipeline/all?file_id=<FILE_ID>`: run the pipeline for all key1 values (background job)
  - query params: `key1_byte`, `key2_byte`, `offset_byte`, `downsample_quicklook`
  - body: `{"spec": <PipelineSpec>, "taps": ["tapA", "tapB"]}`
- `GET /pipeline/job/<JOB_ID>/status`
- `GET /pipeline/job/<JOB_ID>/artifact?key1=<KEY1>&tap=<TAP_LABEL>`

### Batch apply

Batch apply runs a pipeline over every key1 section and writes padded outputs under
`<pipeline_jobs_dir>/<JOB_ID>/`.

- `POST /batch/apply` (JSON): create a background job
  - body:
    - `file_id`, `key1_byte`, `key2_byte`
    - `pipeline_spec`: `PipelineSpec`
    - `pick_options`: `{method, subsample, sigma_ms_max, snap:{enabled, mode, refine, window_ms}}`
    - `save_picks`: when `true` and the pipeline includes `fbpick`, also writes `predicted_picks_time_s.npz`
  - returns: `{ "job_id": "...", "state": "queued" }`
- `GET /batch/job/<JOB_ID>/status`: returns `{state, progress, message}`
- `GET /batch/job/<JOB_ID>/files`: list generated artifacts
- `GET /batch/job/<JOB_ID>/download?name=<FILE_NAME>`: download one artifact

Typical output files:

- `job_meta.json`
- `key1_values.npy`
- `key2_values_padded.npy`
- `denoise_f32_padded.npy` (when the pipeline includes `denoise`)
- `fbpick_prob_f16_padded.npy` (when the pipeline includes `fbpick`)
- `predicted_picks_time_s.npz` (when `save_picks=true` and the pipeline includes `fbpick`)

### First-break picking

Model discovery:

- `GET /fbpick_models`: list available `model/fbpick_*.pt` weights and the default model id

Probability map (binary):

- `POST /fbpick_section_bin` (JSON): start an async job to compute a probability map
  - body: `file_id`, `key1`, `key1_byte`, `key2_byte`, plus tiling params (`tile_h`, `tile_w`, `overlap`, `amp`)
  - optional body fields: `offset_byte`, `pipeline_key` + `tap_label` (run fbpick on a cached tap), `model_id`, `channel`
  - returns: `{ "job_id": "...", "status": "queued"|"running"|"done" }`
- `GET /fbpick_job_status?job_id=<JOB_ID>`
- `GET /get_fbpick_section_bin?job_id=<JOB_ID>`: fetch probability map (gzip+msgpack, quantized)

Picks from probability (server-side):

- `POST /fbpick_predict` (JSON): returns `{dt, picks[]}` where `picks` contains `{trace, time}`
  - body: `file_id`, `key1`, `key1_byte`, `key2_byte`, `method`, `sigma_ms_max`
  - optional body fields: `pipeline_key` + `tap_label`, `model_id`, `channel`

### Manual picks

Manual picks are stored in a memmapped `.npy` file per dataset and exported on demand
as `.npz` or `.txt`.
`GET /export_manual_picks_npz` writes manual-pick `.npz` in seisai(psn)-compatible CSR
(`n_traces`, `p_indptr`, `p_data`, `s_indptr`, `s_data`) and also includes
`picks_time_s`.
`GET /export_manual_picks_grstat_txt` writes a grstat-compatible `.txt`.
`POST /import_manual_picks_npz` imports manual picks back into the memmap and accepts
both CSR and legacy `picks_time_s` formats.
`manual_pick_format` is `seisai_csr`, and format versioning is represented by
`format_version` (current value: `1`).

These exported `.npz` files are seisai学習で使える and can be passed directly as
`paths.phase_pick_files`.
Example:
`paths.phase_pick_files: ["/path/to/manual_picks_time_v1_lineA.npz"]`

- `PICKS_NPY_DIR` (highest priority): direct override for memmap files
- Default when unset: `<app_data_dir>/picks_npy`

Pipeline job artifacts are persisted on disk.

- `PIPELINE_JOBS_DIR` (highest priority): direct override for pipeline artifacts
- Default when unset: `<app_data_dir>/pipeline_jobs`

Upload and trace-store artifacts are persisted on disk.

- `SV_UPLOAD_DIR` (highest priority): direct override for upload root
- `SV_PROCESSED_DIR` (highest priority): direct override for processed upload root
- `SV_TRACE_DIR` (highest priority): direct override for trace-store root

Defaults when unset:

- upload root: `<app_data_dir>/uploads`
- processed upload root: `<app_data_dir>/uploads/processed`
- trace-store root: `<app_data_dir>/uploads/processed/traces`

`<app_data_dir>` is resolved in this order:

- `SV_APP_DATA_DIR`
- `RUNNER_TEMP/seisviewer2d_app_data` (CI-friendly default)
- `XDG_CACHE_HOME/seisviewer2d`
- `~/.cache/seisviewer2d`

Endpoints:

- `GET /picks?file_id=<FILE_ID>&key1=<KEY1>&key1_byte=<BYTE>&key2_byte=<BYTE>`
- `POST /picks` (JSON): `{file_id, trace, time, key1, key1_byte, key2_byte?}`
- `DELETE /picks?file_id=<FILE_ID>&key1=<KEY1>&key1_byte=<BYTE>&key2_byte=<BYTE>&trace=<TRACE>`
- `GET /export_manual_picks_npz?file_id=<FILE_ID>&key1_byte=<BYTE>&key2_byte=<BYTE>`
- `GET /export_manual_picks_grstat_txt?file_id=<FILE_ID>&key2_byte=<BYTE>`
- `POST /import_manual_picks_npz?file_id=<FILE_ID>&key1_byte=<BYTE>&key2_byte=<BYTE>&mode=replace|merge` (multipart/form-data)
  - form field: `file`

## Project layout

```
app/
  main.py                 # FastAPI app + static mounting
  api/
    routers/              # upload/section/pipeline/picks/fbpick/batch
    _helpers.py           # shared API helpers
    baselines.py          # raw baseline stats used for scaling
    schemas.py            # pydantic models for requests/responses
  core/                   # app state + data-dir resolution
  services/               # TraceStore readers, caching, pipeline taps, persistence
  static/                 # HTML/JS viewer served at / and /upload
  web/                    # Vite sources that build into static/assets
  utils/                  # SEG-Y ingest, ML wrappers, pipeline ops
  tests/                  # backend and JS unit tests
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
