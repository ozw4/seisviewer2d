    /* Adopt globals provided by /static/viewer/bootstrap.js */
    var setGrid, getPlotEnv, dataXYFromClient, traceAtPixel, timeAtPixel, pixelForTrace,
        snapTraceFromDataX, snapTimeFromDataY, buildLayout, buildPickShapes, buildPickMarkerTraces, buildPendingPickMarkerTrace;
    var key1Values = [];
    var currentFileId = '';
    var currentFileName = ''; // NEW: basename (e.g., LineA.sgy)
    var currentKey1Byte = 189;
    var currentKey2Byte = 193;
    const LMO_DEFAULTS = Object.freeze({
      enabled: false,
      velocityMps: 1500,
      offsetByte: 37,
      offsetScale: 1.0,
      offsetMode: 'absolute',
      refMode: 'min',
      refTrace: 0,
      polarity: 1,
    });
    const LMO_STORAGE_KEYS = Object.freeze({
      enabled: 'lmo_enabled',
      velocityMps: 'lmo_velocity_mps',
      offsetByte: 'lmo_offset_byte',
      offsetScale: 'lmo_offset_scale',
      offsetMode: 'lmo_offset_mode',
      refMode: 'lmo_ref_mode',
      refTrace: 'lmo_ref_trace',
      polarity: 'lmo_polarity',
    });
    const LMO_OFFSET_MODES = new Set(['absolute', 'signed']);
    const LMO_REF_MODES = new Set(['min', 'first', 'center', 'trace', 'zero']);
    var currentLinearMoveout = { ...LMO_DEFAULTS };
    var savedXRange = null;
    var savedYRange = null;
    var latestSeismicData = null;
    // Kept for compatibility with pipeline_ui fallback checks; no full-section data is stored.
    var rawSeismicData = null;
    var latestTapData = {};
    var latestPipelineKey = null;
    var latestWindowRender = null;
    var activeWindowFetchId = 0;
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
    window.SV_PERF = false;
    window.SV_PERF_ROWS = [];
    window.svPerfLog = function svPerfLog(rec) {
      if (!window.SV_PERF) return;
      const rows = Array.isArray(window.SV_PERF_ROWS)
        ? window.SV_PERF_ROWS
        : (window.SV_PERF_ROWS = []);
      rows.push(rec);
      if (rows.length > 60) {
        rows.splice(0, rows.length - 60);
      }
      console.info('[svperf]', rec);
    };
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
      const perfParam = params.get('perf');
      if (perfParam != null) {
        const norm = perfParam.trim().toLowerCase();
        if (norm === '0' || norm === 'false' || norm === 'off') {
          window.SV_PERF = false;
        }
        if (norm === '1' || norm === 'true' || norm === 'on') {
          window.SV_PERF = true;
        }
      }
    }

    function cloneLinearMoveoutState(state = currentLinearMoveout) {
      return {
        enabled: !!state.enabled,
        velocityMps: Number(state.velocityMps),
        offsetByte: Number(state.offsetByte),
        offsetScale: Number(state.offsetScale),
        offsetMode: String(state.offsetMode),
        refMode: String(state.refMode),
        refTrace: Number(state.refTrace),
        polarity: Number(state.polarity) === -1 ? -1 : 1,
      };
    }

    function parseLmoBoolean(value, def) {
      if (typeof value === 'boolean') return value;
      if (typeof value === 'string') {
        const norm = value.trim().toLowerCase();
        if (norm === 'true' || norm === '1' || norm === 'on') return true;
        if (norm === 'false' || norm === '0' || norm === 'off') return false;
      }
      return def;
    }

    function parseLmoFiniteNumber(value) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }

    function parseLmoInteger(value) {
      const parsed = Number(value);
      return Number.isInteger(parsed) ? parsed : null;
    }

    function normalizeLinearMoveoutState(value = {}, base = LMO_DEFAULTS) {
      const source = { ...base, ...(value && typeof value === 'object' ? value : {}) };
      const velocity = parseLmoFiniteNumber(source.velocityMps);
      const offsetByte = parseLmoInteger(source.offsetByte);
      const offsetScale = parseLmoFiniteNumber(source.offsetScale);
      const refTrace = parseLmoInteger(source.refTrace);
      const polarity = Number(source.polarity);
      const offsetMode = String(source.offsetMode);
      const refMode = String(source.refMode);

      return {
        enabled: parseLmoBoolean(source.enabled, LMO_DEFAULTS.enabled),
        velocityMps: velocity !== null && velocity > 0 ? velocity : LMO_DEFAULTS.velocityMps,
        offsetByte: offsetByte !== null && offsetByte >= 1 && offsetByte <= 240
          ? offsetByte
          : LMO_DEFAULTS.offsetByte,
        offsetScale: offsetScale !== null && offsetScale !== 0 ? offsetScale : LMO_DEFAULTS.offsetScale,
        offsetMode: LMO_OFFSET_MODES.has(offsetMode) ? offsetMode : LMO_DEFAULTS.offsetMode,
        refMode: LMO_REF_MODES.has(refMode) ? refMode : LMO_DEFAULTS.refMode,
        refTrace: refTrace !== null && refTrace >= 0 ? refTrace : LMO_DEFAULTS.refTrace,
        polarity: polarity === -1 || polarity === 1 ? polarity : LMO_DEFAULTS.polarity,
      };
    }

    function linearMoveoutStatesEqual(a, b) {
      return a.enabled === b.enabled
        && a.velocityMps === b.velocityMps
        && a.offsetByte === b.offsetByte
        && a.offsetScale === b.offsetScale
        && a.offsetMode === b.offsetMode
        && a.refMode === b.refMode
        && a.refTrace === b.refTrace
        && a.polarity === b.polarity;
    }

    function formatLmoNumber(value) {
      return String(Number(value));
    }

    function readStoredLinearMoveoutState() {
      const stored = {};
      try {
        for (const [field, key] of Object.entries(LMO_STORAGE_KEYS)) {
          const value = localStorage.getItem(key);
          if (value !== null) stored[field] = value;
        }
      } catch (_) {
      }
      return normalizeLinearMoveoutState(stored, LMO_DEFAULTS);
    }

    function persistLinearMoveoutState(state = currentLinearMoveout) {
      const lmo = normalizeLinearMoveoutState(state, LMO_DEFAULTS);
      try {
        localStorage.setItem(LMO_STORAGE_KEYS.enabled, lmo.enabled ? 'true' : 'false');
        localStorage.setItem(LMO_STORAGE_KEYS.velocityMps, formatLmoNumber(lmo.velocityMps));
        localStorage.setItem(LMO_STORAGE_KEYS.offsetByte, String(lmo.offsetByte));
        localStorage.setItem(LMO_STORAGE_KEYS.offsetScale, formatLmoNumber(lmo.offsetScale));
        localStorage.setItem(LMO_STORAGE_KEYS.offsetMode, lmo.offsetMode);
        localStorage.setItem(LMO_STORAGE_KEYS.refMode, lmo.refMode);
        localStorage.setItem(LMO_STORAGE_KEYS.refTrace, String(lmo.refTrace));
        localStorage.setItem(LMO_STORAGE_KEYS.polarity, String(lmo.polarity));
      } catch (_) {
      }
    }

    function getLinearMoveoutControlSnapshot() {
      const snapshot = cloneLinearMoveoutState(currentLinearMoveout);
      const enabled = document.getElementById('lmoEnabled');
      const velocityMps = document.getElementById('lmoVelocityMps');
      const offsetByte = document.getElementById('lmoOffsetByte');
      const offsetScale = document.getElementById('lmoOffsetScale');
      const offsetMode = document.getElementById('lmoOffsetMode');
      const refMode = document.getElementById('lmoRefMode');
      const refTrace = document.getElementById('lmoRefTrace');
      const polarity = document.getElementById('lmoPolarity');

      if (enabled) snapshot.enabled = !!enabled.checked;
      if (velocityMps) snapshot.velocityMps = velocityMps.value;
      if (offsetByte) snapshot.offsetByte = offsetByte.value;
      if (offsetScale) snapshot.offsetScale = offsetScale.value;
      if (offsetMode) snapshot.offsetMode = offsetMode.value;
      if (refMode) snapshot.refMode = refMode.value;
      if (refTrace) snapshot.refTrace = refTrace.value;
      if (polarity) snapshot.polarity = polarity.value;
      return snapshot;
    }

    function syncLinearMoveoutControls(options = {}) {
      const skipActiveNumber = options.skipActiveNumber === true;
      const lmo = normalizeLinearMoveoutState(currentLinearMoveout, LMO_DEFAULTS);
      const active = document.activeElement;

      function setNumberValue(id, value) {
        const el = document.getElementById(id);
        if (!el) return;
        if (skipActiveNumber && el === active) return;
        el.value = formatLmoNumber(value);
      }

      const enabled = document.getElementById('lmoEnabled');
      const offsetMode = document.getElementById('lmoOffsetMode');
      const refMode = document.getElementById('lmoRefMode');
      const refTrace = document.getElementById('lmoRefTrace');
      const polarity = document.getElementById('lmoPolarity');

      if (enabled) enabled.checked = lmo.enabled;
      setNumberValue('lmoVelocityMps', lmo.velocityMps);
      setNumberValue('lmoOffsetByte', lmo.offsetByte);
      setNumberValue('lmoOffsetScale', lmo.offsetScale);
      if (offsetMode) offsetMode.value = lmo.offsetMode;
      if (refMode) refMode.value = lmo.refMode;
      setNumberValue('lmoRefTrace', lmo.refTrace);
      if (refTrace) refTrace.disabled = lmo.refMode !== 'trace';
      if (polarity) polarity.value = String(lmo.polarity);
    }

    function dispatchLinearMoveoutChange() {
      window.dispatchEvent(new CustomEvent('lmo:change', {
        detail: {
          lmo: window.getCurrentLinearMoveout(),
          key: window.currentLmoKey(),
        },
      }));
    }

    function getCurrentLinearMoveout() {
      return cloneLinearMoveoutState(currentLinearMoveout);
    }

    function setCurrentLinearMoveout(patch, options = {}) {
      const next = normalizeLinearMoveoutState(patch, currentLinearMoveout);
      const changed = !linearMoveoutStatesEqual(next, currentLinearMoveout);
      currentLinearMoveout = next;
      window.currentLinearMoveout = currentLinearMoveout;
      if (options.persist !== false) persistLinearMoveoutState(currentLinearMoveout);
      if (options.applyDom !== false) syncLinearMoveoutControls({
        skipActiveNumber: options.skipActiveNumber === true,
      });
      if (changed) dispatchLinearMoveoutChange();
      return getCurrentLinearMoveout();
    }

    function currentLmoKey() {
      const lmo = normalizeLinearMoveoutState(currentLinearMoveout, LMO_DEFAULTS);
      if (!lmo.enabled) return 'lmo:off';
      return [
        'lmo:on',
        `v=${formatLmoNumber(lmo.velocityMps)}`,
        `ob=${lmo.offsetByte}`,
        `os=${formatLmoNumber(lmo.offsetScale)}`,
        `om=${lmo.offsetMode}`,
        `rm=${lmo.refMode}`,
        `rt=${lmo.refTrace}`,
        `p=${lmo.polarity}`,
      ].join('|');
    }

    function onLinearMoveoutControlChange(options = {}) {
      return setCurrentLinearMoveout(getLinearMoveoutControlSnapshot(), options);
    }

    function bindLinearMoveoutControls() {
      if (window.__linearMoveoutControlsBound) return;
      const numberIds = ['lmoVelocityMps', 'lmoOffsetByte', 'lmoOffsetScale', 'lmoRefTrace'];
      const changeIds = ['lmoEnabled', 'lmoOffsetMode', 'lmoRefMode', 'lmoPolarity'];
      let foundControl = false;

      for (const id of numberIds) {
        const el = document.getElementById(id);
        if (!el) continue;
        foundControl = true;
        el.addEventListener('input', () => onLinearMoveoutControlChange({
          applyDom: false,
        }));
        el.addEventListener('change', () => onLinearMoveoutControlChange());
      }
      for (const id of changeIds) {
        const el = document.getElementById(id);
        if (!el) continue;
        foundControl = true;
        el.addEventListener('change', () => onLinearMoveoutControlChange());
      }
      if (foundControl) window.__linearMoveoutControlsBound = true;
    }

    function initLinearMoveoutControls() {
      syncLinearMoveoutControls();
      bindLinearMoveoutControls();
    }

    currentLinearMoveout = readStoredLinearMoveoutState();
    window.currentLinearMoveout = currentLinearMoveout;
    window.getCurrentLinearMoveout = getCurrentLinearMoveout;
    window.setCurrentLinearMoveout = setCurrentLinearMoveout;
    window.currentLmoKey = currentLmoKey;
    window.onLinearMoveoutControlChange = onLinearMoveoutControlChange;
    persistLinearMoveoutState(currentLinearMoveout);
    initLinearMoveoutControls();
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initLinearMoveoutControls, { once: true });
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
    }, 300); // keep section auto-plot responsive without flooding requests
