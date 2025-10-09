// /viewer/bootstrap.js
import { createStore } from './store.js';

// Read initial values from existing DOM / globals
const gainEl   = document.getElementById('gain');
const cmSel    = document.getElementById('colormap');
const cmRevEl  = document.getElementById('cmReverse');
const slider   = document.getElementById('key1_idx_slider');

function num(v, def) {
  const x = parseFloat(v);
  return Number.isFinite(x) ? x : def;
}

const initial = {
  fileId: document.getElementById('file_id')?.value || (window.currentFileId || ''),
  pickMode: !!window.isPickMode,
  wiggleDensity: num(localStorage.getItem('wiggle_density'), 0.20),
  gain: num(localStorage.getItem('gain') ?? (gainEl?.value ?? 1), 1),
  colormap: localStorage.getItem('colormap') || (cmSel?.value || 'Greys'),
  cmReverse: (localStorage.getItem('cmReverse') ?? (cmRevEl?.checked ? 'true' : 'false')) === 'true',
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

/* ---- Wrap or listen to existing handlers to keep store in sync ---- */

// Pick mode toggle
if (typeof window.togglePickMode === 'function') {
  const _orig = window.togglePickMode;
  window.togglePickMode = function () {
    _orig();
    store.patch({ pickMode: !!window.isPickMode });
  };
}

// Wiggle/heatmap density threshold
const wiggleInput = document.getElementById('wiggle_density');
if (wiggleInput) {
  wiggleInput.addEventListener('input', () => {
    const v = parseFloat(wiggleInput.value);
    if (Number.isFinite(v)) store.patch({ wiggleDensity: v });
  });
}

// Gain
if (gainEl && typeof window.onGainChange === 'function') {
  const _gain = window.onGainChange;
  window.onGainChange = function () {
    _gain();
    store.patch({ gain: num(gainEl.value, 1) });
  };
}

// Colormap + reverse
if (cmSel && typeof window.onColormapChange === 'function') {
  const _cm = window.onColormapChange;
  window.onColormapChange = function () {
    _cm();
    const rev = !!document.getElementById('cmReverse')?.checked;
    store.patch({ colormap: cmSel.value || 'Greys', cmReverse: rev });
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
