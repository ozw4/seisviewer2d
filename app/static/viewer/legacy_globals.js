    /* Adopt globals provided by /static/viewer/bootstrap.js */
    var setGrid, getPlotEnv, dataXYFromClient, traceAtPixel, timeAtPixel, pixelForTrace,
        snapTraceFromDataX, snapTimeFromDataY, buildLayout, buildPickShapes;
    var key1Values = [];
    var currentFileId = '';
    var currentFileName = ''; // NEW: basename (e.g., LineA.sgy)
    var currentKey1Byte = 189;
    var currentKey2Byte = 193;
    var savedXRange = null;
    var savedYRange = null;
    var latestSeismicData = null;
    // Kept for compatibility with pipeline_ui fallback checks; no full-section data is stored.
    var rawSeismicData = null;
    var latestTapData = {};
    var latestPipelineKey = null;
    var latestWindowRender = null;
    var windowFetchToken = 0;
    let windowFetchCtrl = null; // active window-fetch controller (if any)
    let cfg;
    let debounce;
    let defaultDt = 0;
    var onViewportSettled = function () { };
    var scheduleWindowFetch = function () { };
    var dragBase = 'zoom'; // will be overridden from localStorage in initViewerGlobals
    let lastHover = null;
    let redrawPending = false;
    let USE_HEATMAP_POOLING = true;
    {
      const params = new URLSearchParams(window.location.search);
      const poolParam = params.get('heatmap_pool');
      if (poolParam != null) {
        const norm = poolParam.trim().toLowerCase();
        if (norm === '0' || norm === 'false' || norm === 'off') {
          USE_HEATMAP_POOLING = false;
        }
        if (norm === '1' || norm === 'true' || norm === 'on') {
          USE_HEATMAP_POOLING = true;
        }
      }
    }
    // === A-7: Coordinate utilities (provided by /static/viewer/core via bootstrap) ===
    // UI-adjustable threshold for Wiggle/Heatmap decision (persisted)
    let WIGGLE_DENSITY_THRESHOLD = 0.10;
    const WIGGLE_MAX_POINTS = 1_000_000;

    const __pickOps = new Map();
    let flushPickOpsDebounced = null;

    async function flushPickOps() {
      if (__pickOps.size === 0) return;

      const ops = Array.from(__pickOps.values());
      __pickOps.clear();

      if (ops.every(op => !op.fileId)) return;
      const limit = Math.min(4, ops.length);
      let i = 0;

      async function worker() {
        while (i < ops.length) {
          const op = ops[i++];
          if (!op?.fileId) continue;
          const fileId = op.fileId;
          const key1Val = op.key1Val;
          const key1Byte = op.key1Byte | 0;
          if (key1Val === undefined) continue;
          if (op.op === 'upsert') {
            await fetch('/picks', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                file_id: fileId,
                trace: op.trace,
                time: op.time,
                key1: key1Val,
                key1_byte: key1Byte,
                key2_byte: currentKey2Byte
              }),
            });
          } else {
            const url = `/picks?file_id=${encodeURIComponent(fileId)}&trace=${op.trace}&key1=${encodeURIComponent(key1Val)}&key1_byte=${key1Byte}&key2_byte=${currentKey2Byte}`;
            await fetch(url, { method: 'DELETE' });
          }
        }
      }

      await Promise.all(Array.from({ length: limit }, () => worker()));
    }
    function syncWiggleInit() {
      const el = document.getElementById('wiggle_density');
      if (!el) return;
      const saved = cfg.getWiggleDensity();
      const min = parseFloat(el.min) || 0.02;
      const max = parseFloat(el.max) || 0.30;
      let v = Number.isFinite(saved) ? saved : parseFloat(el.value) || 0.20;
      if (!Number.isFinite(v)) v = 0.20;
      v = Math.min(max, Math.max(min, v));
      el.value = v.toFixed(2);
      WIGGLE_DENSITY_THRESHOLD = v;
    }
    function initViewerGlobals() {
      cfg = window.cfg;
      debounce = window.debounce;
      if (!cfg || !debounce) {
        console.warn('viewer bootstrap globals not ready yet');
        return;
      }
      flushPickOpsDebounced = debounce(flushPickOps, 120);
      defaultDt = cfg.getDefaultDt();
      window.defaultDt = defaultDt;
      WIGGLE_DENSITY_THRESHOLD = cfg.getWiggleDensity();
      syncWiggleInit();
      if (!Number.isFinite(WIGGLE_DENSITY_THRESHOLD)) {
        WIGGLE_DENSITY_THRESHOLD = 0.10;
      }
      try {
        const storedDrag = localStorage.getItem(cfg.LS_KEYS.DRAG_BASE);
        dragBase = storedDrag === 'pan' ? 'pan' : 'zoom';
      } catch (_) {
        dragBase = 'zoom';
      }
      onViewportSettled = debounce(() => {
        if (suppressRelayout || isRelayouting) return;
        checkModeFlipAndRefetch();
        maybeFetchIfOutOfWindow();
      }, cfg.FETCH_DEBOUNCE_MS);
      scheduleWindowFetch = debounce(() => {
        fetchWindowAndPlot().catch((err) => console.warn('Window fetch failed', err));
      }, cfg.WINDOW_FETCH_DEBOUNCE_MS);
    }

    if (typeof window.whenViewerBootstrapReady !== 'function') {
      window.whenViewerBootstrapReady = function whenViewerBootstrapReady(cb) {
        if (typeof cb !== 'function') return;
        if (window.cfg && window.debounce) {
          cb();
          return;
        }
        (window.__viewerBootstrapQueue ||= []).push(cb);
      };
    }
    window.whenViewerBootstrapReady(initViewerGlobals);
    function makeDebounced(fn, waitMs) {
      let timer = null;
      function debounced(...args) {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => { timer = null; fn(...args); }, waitMs);
      }
      debounced.flush = (...args) => {
        if (timer) { clearTimeout(timer); timer = null; }
        fn(...args);
      };
      debounced.cancel = () => { if (timer) { clearTimeout(timer); timer = null; } };
      return debounced;
    }

    // スライダ用：入力中は遅延、確定時は即時
    let fetchAndPlotDebounced = makeDebounced(() => {
      try { fetchAndPlot(); } catch (e) { console.warn(e); }
    }, 120); // 100〜200msくらいが体感◎
