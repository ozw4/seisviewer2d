// /viewer/bootstrap.js
import { createStore } from './store.js';
import * as GridCore from './core/grid.js';
import { buildLayout, buildPickShapes } from './core/layout.js';
import { initPrefs, getPref } from './settings/prefs.js';
import { cfg } from './core/config.js';
import { debounce, throttle, rafDebounce } from './core/utils/timing.js';

// ------------------------------------------------------------
  // Mode decider with geometry-stability lock
  // ------------------------------------------------------------
  (function attachModeDecider() {
      // 「直前の幾何」と「直前のモード」を保持
        let lastGeom = null;   // { traces: number, samples: number }
      let lastMode = null;   // 'wiggle' | 'heatmap'

        // 既存のしきい値（ローカル保存や設定から）
        const getThreshold = () =>
            (typeof window.WIGGLE_DENSITY_THRESHOLD === 'number'
                ? window.WIGGLE_DENSITY_THRESHOLD
              : (typeof window.cfg?.getWiggleDensity === 'function'
                    ? window.cfg.getWiggleDensity()
                  : 0.20));

        // 既存の「ポイント上限」
        const getMaxPoints = () =>
            (window.cfg?.limits?.WINDOW_MAX_POINTS ?? window.cfg?.WINDOW_MAX_POINTS ?? 1_200_000);

        // 公開API：必要なら他所からも呼べるように
        window.decidePlotMode = function decidePlotMode({ traces, samples }) {
            const geom = { traces: Math.max(0, traces | 0), samples: Math.max(0, samples | 0) };
            const stable = lastGeom
                && lastGeom.traces === geom.traces
                && lastGeom.samples === geom.samples;
            if (stable && lastMode) return lastMode;              // ← 幾何不変ならモード維持

              // 幾何が変わったときだけ密度で再判定（元々の規約に合わせる）
              const density = geom.traces * geom.samples;
            const wantWiggle = density <= getThreshold() * getMaxPoints();
            const next = wantWiggle ? 'wiggle' : 'heatmap';
            lastGeom = geom;
            lastMode = next;
            return next;
          };

        // 明示的にリセットしたい場合用（ズーム/リセットなど）
        window.resetPlotModeDecision = function () {
            lastGeom = null; lastMode = null;
          };
    })();

// ---- helpers
const toNum = (v, d) => {
  const x = Number(v);
  return Number.isFinite(x) ? x : d;
};
const toStr = (v, d) => (v == null ? d : String(v));
const toBool = (v) => (typeof v === 'string' ? v === 'true' : !!v);

// Make available to non-module scripts
window.cfg = cfg;
window.debounce = debounce;
window.throttle = throttle;
window.rafDebounce = rafDebounce;

const readyQueue = window.__viewerBootstrapQueue;
window.viewerBootstrapReady = Promise.resolve();
if (Array.isArray(readyQueue)) {
  window.__viewerBootstrapQueue = null;
  for (const fn of readyQueue) {
    try {
      if (typeof fn === 'function') fn();
    } catch (err) {
      console.error('[viewer] bootstrap callback failed', err);
    }
  }
}
if (typeof window.whenViewerBootstrapReady === 'function') {
  window.whenViewerBootstrapReady = function whenViewerBootstrapReady(cb) {
    if (typeof cb !== 'function') return;
    try {
      cb();
    } catch (err) {
      console.error('[viewer] bootstrap callback failed', err);
    }
  };
}

if (!(typeof window.defaultDt === 'number' && Number.isFinite(window.defaultDt) && window.defaultDt > 0)) {
  const v = cfg.getDefaultDt();
  window.defaultDt = v;
}

// Read initial values from existing DOM / globals
const slider = document.getElementById('key1_idx_slider');

const initial = {
  fileId: document.getElementById('file_id')?.value || (window.currentFileId || ''),
  pickMode: !!window.isPickMode,
  wiggleDensity: toNum(getPref('wiggle_density'), 0.20),
  gain: toNum(getPref('gain'), toNum(document.getElementById('gain')?.value, 1)),
  colormap: toStr(getPref('colormap'), document.getElementById('colormap')?.value || 'Greys'),
  cmReverse: toBool(getPref('cmReverse')),
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

// ---- Sync global wiggle-threshold at boot
  window.WIGGLE_DENSITY_THRESHOLD = store.get().wiggleDensity;

  // ---- Wrap drawSelectedLayer to pass geometry into decidePlotMode
  // 既存の drawSelectedLayer がグローバルにある前提。なければ後段の render 呼び出し側で同様に包む。
  if (typeof window.drawSelectedLayer === 'function') {
      const _origDraw = window.drawSelectedLayer;
      window.drawSelectedLayer = function (start, end) {
          try {
              const traces = Math.max(0, Number(end) - Number(start));
              const samples =
                  Array.isArray(window.sectionShape) && window.sectionShape.length >= 2
                      ? Number(window.sectionShape[1]) | 0
                      : 0;
              const mode = window.decidePlotMode({ traces, samples });
              // どこで mode を参照するかは実装次第だが、広く使えるように保持しておく
                window.currentPlotMode = mode;
            } catch (_) { /* keep original path even if geometry not available */ }
          return _origDraw.apply(this, arguments);
        };
    }
// ---- Expose core modules to existing global code (override safely)
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
    // patch store (描画は必要時のみ行う)
    if (key === 'gain') store.patch({ gain: toNum(value, 1) });
    if (key === 'colormap') store.patch({ colormap: toStr(value, 'Greys') });
    if (key === 'cmReverse') store.patch({ cmReverse: toBool(value) });

    if (key === 'wiggle_density') {
      const v = toNum(value, 0.20);
      store.patch({ wiggleDensity: v });
      // モード判定はこのグローバルを見るので同期が必須
      window.WIGGLE_DENSITY_THRESHOLD = v;
      if (typeof window.scheduleWindowFetch === 'function') {
        window.scheduleWindowFetch();
      }
      return; // 閾値変更では即時再描画しない（fetchに任せる）
    }

    // それ以外（gain/colormap等）は pan/zoom 中でなければ軽量再描画
    if (!window.isRelayouting && !window.suppressRelayout) {
      if (typeof window.renderLatestView === 'function') {
        window.renderLatestView();
      }
    }
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

// loadSettings 後に store を実データで追従（fileId/key1Values 等）
if (typeof window.loadSettings === 'function') {
  const _load = window.loadSettings;
  window.loadSettings = async function () {
    await _load();
    window.store.patch({
      fileId: document.getElementById('file_id')?.value || (window.currentFileId || ''),
      key1Values: Array.isArray(window.key1Values) ? window.key1Values : [],
      sectionShape: window.sectionShape || null,
      pipelineKey: window.latestPipelineKey || null,
    });
    // dt が変わる可能性があるので、閾値更新は不要だが、必要ならここで再フェッチ可能
  };
}
store.subscribe(() => { });
