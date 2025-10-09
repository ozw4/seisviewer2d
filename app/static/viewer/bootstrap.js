// /viewer/bootstrap.js
import { createStore } from './store.js';
import * as GridCore from './core/grid.js';
import { buildLayout, buildPickShapes } from './core/layout.js';
import { initPrefs, getPref } from './settings/prefs.js';

// Read initial values from existing DOM / globals
const slider   = document.getElementById('key1_idx_slider');

const initial = {
  fileId: document.getElementById('file_id')?.value || (window.currentFileId || ''),
  pickMode: !!window.isPickMode,
  wiggleDensity: Number(getPref('wiggle_density')),
  gain: Number(getPref('gain')),
  colormap: String(getPref('colormap')),
  cmReverse: !!getPref('cmReverse'),
  savedXRange: window.savedXRange || null,
  savedYRange: window.savedYRange || null,
  key1Index: Number.parseInt(slider?.value || '0', 10) || 0,
  key1Values: Array.isArray(window.key1Values) ? window.key1Values : [],
  sectionShape: window.sectionShape || null,
  pipelineKey: window.latestPipelineKey || null,
};

const store = createStore(initial);

// Expose for debugging
window.store = store;

// ---- Expose core modules to existing global code (override safely) ----
// Grid & coordinate helpers
window.Grid = GridCore.Grid;
window.setGrid = GridCore.setGrid;
window.getPlotEnv = GridCore.getPlotEnv;
window.dataXYFromClient = GridCore.dataXYFromClient;
window.snapTraceFromDataX = GridCore.snapTraceFromDataX;
window.snapTimeFromDataY = GridCore.snapTimeFromDataY;
window.traceAtPixel = GridCore.traceAtPixel;
window.pixelForTrace = GridCore.pixelForTrace;
window.timeAtPixel = GridCore.timeAtPixel;

// Layout helpers
window.buildLayout = buildLayout;
window.buildPickShapes = buildPickShapes;

// Initialize preferences (applies to DOM & sets listeners)
initPrefs({
  onChange(key, value) {
    // Patch only the fields we mirror in store (render will be triggered below)
    if (key === 'gain')             store.patch({ gain: Number(value) || 1 });
    if (key === 'colormap')         store.patch({ colormap: String(value) });
    if (key === 'cmReverse')        store.patch({ cmReverse: !!value });
    if (key === 'wiggle_density')   store.patch({ wiggleDensity: Number(value) });
    if (typeof window.renderLatestView === 'function') window.renderLatestView();
  }
});

/* ---- Wrap or listen to existing handlers to keep store in sync ---- */

// Pick mode toggle
if (typeof window.togglePickMode === 'function') {
  const _orig = window.togglePickMode;
  window.togglePickMode = function () {
    _orig();
    store.patch({ pickMode: !!window.isPickMode });
  };
}

// Key1 slider index
if (slider) {
  slider.addEventListener('input', () => {
    store.patch({ key1Index: Number.parseInt(slider.value || '0', 10) || 0 });
  });
}

// Relayout (viewport ranges)
if (typeof window.handleRelayout === 'function') {
  const _rel = window.handleRelayout;
  window.handleRelayout = async function (ev) {
    await _rel(ev);
    if ('xaxis.range[0]' in ev && 'xaxis.range[1]' in ev) {
      window.savedXRange = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
    }
    if ('yaxis.range[0]' in ev && 'yaxis.range[1]' in ev) {
      const y0 = ev['yaxis.range[0]'], y1 = ev['yaxis.range[1]'];
      window.savedYRange = y0 > y1 ? [y0, y1] : [y1, y0];
    }
    store.patch({ savedXRange: window.savedXRange, savedYRange: window.savedYRange });
  };
}

// Whenever store changes, ask existing renderer to refresh (cheap & safe)
store.subscribe(() => {
  if (typeof window.renderLatestView === 'function') {
    window.renderLatestView();
  }
});
