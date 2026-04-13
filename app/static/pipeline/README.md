# Pipeline UI split memo (4A)

## Goal
- Keep current behavior and `window.pipelineUI` compatibility.
- Add split points so `window.pipelineUI` can be retired in later phases.

## Current `window.pipelineUI` public API (from `app/static/pipeline/index.js`)
- `initPipelineUI`
- `renderPipelineCards`
- `openInspector`
- `closeInspector`
- `graphToSpec`
- `specToGraph`
- `schedulePipelineRun`
- `runPipeline`
- `savePipelineToLocalStorage`
- `loadPipelineFromLocalStorage`
- `prepareForNewSection`
- `on`
- `off`
- `_emit`
- `cancel`

## Added files in 4A
- `app/static/pipeline/events.js`
  - Provides `window.createPipelineEventBus()`.
  - Used by `pipeline/index.js` for `on/off/_emit`.
  - `section:prepare` is emitted from viewer and handled by pipeline to reset layer/tap state.
- `app/static/pipeline/progressOverlay.js`
  - Provides `window.createPipelineProgressOverlay()`.
  - Handles overlay text/progress/open/close/error and event attachment.
- `app/static/pipeline/storage.js`
  - localStorage persistence and legacy `viewer.pipelineSpec`/`viewer.pipelineTaps` compatibility load.
- `app/static/pipeline/state.js`
  - UI state holder and state change notifications via events.
- `app/static/pipeline/render/cards.js`
  - Card rendering and D&D ordering handlers.
- `app/static/pipeline/render/inspector.js`
  - Inspector form rendering and submit handling.
- `app/static/pipeline/run.js`
  - `graphToSpec`, `specToGraph`, diagnostics, run execution, run event emit, and progress overlay bridge.
- `app/static/pipeline/index.js`
  - Main classic-script entry that assembles modules and defines `window.pipelineUI`.
  - Exposes `window.pipelineIndex.publicApiKeys`, `window.pipelineIndex.getPipelineUI()`, `window.pipelineIndex.createPipelineFacade()`.
- `app/static/pipeline_ui.js`
  - Thin compatibility wrapper that only requests `pipeline/index.js` bootstrap.

## Script loading order (classic split)
1. `pipeline/events.js`
2. `pipeline/progressOverlay.js`
3. `pipeline/storage.js`
4. `pipeline/state.js`
5. `pipeline/render/cards.js`
6. `pipeline/render/inspector.js`
7. `pipeline/run.js`
8. `pipeline/index.js`
9. `pipeline_ui.js`

`window.pipelineUI` is defined only by `pipeline/index.js`.
