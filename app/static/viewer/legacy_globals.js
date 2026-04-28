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
    const LMO_FIXED_OFFSET_MODE = 'absolute';
    const LMO_FIXED_REF_MODE = 'min';
    const LMO_FIXED_REF_TRACE = 0;
    const LMO_FIXED_POLARITY = 1;
    const LMO_SECTION_OFFSET_CACHE_MAX_ENTRIES = 64;
    const LMO_SHIFT_SECONDS_CACHE_MAX_ENTRIES = 128;
    const LMO_WARNING_KEY_MAX_ENTRIES = 128;
    const lmoSectionOffsetCache = new Map();
    const lmoSectionOffsetInflight = new Map();
    const lmoShiftSecondsCache = new Map();
    const lmoPickOverlayWarningKeys = new Set();
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
        offsetMode: LMO_FIXED_OFFSET_MODE,
        refMode: LMO_FIXED_REF_MODE,
        refTrace: LMO_FIXED_REF_TRACE,
        polarity: LMO_FIXED_POLARITY,
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

    function setBoundedMapEntry(map, key, value, maxEntries) {
      if (map.has(key)) {
        map.delete(key);
      }
      map.set(key, value);

      while (map.size > maxEntries) {
        const oldestKey = map.keys().next().value;
        if (oldestKey === undefined) break;
        map.delete(oldestKey);
      }

      return value;
    }

    function addBoundedSetEntry(set, value, maxEntries) {
      if (set.has(value)) {
        set.delete(value);
      }
      set.add(value);

      while (set.size > maxEntries) {
        const oldestValue = set.values().next().value;
        if (oldestValue === undefined) break;
        set.delete(oldestValue);
      }

      return value;
    }

    function clearLinearMoveoutRuntimeCaches() {
      lmoSectionOffsetCache.clear();
      lmoSectionOffsetInflight.clear();
      lmoShiftSecondsCache.clear();
      lmoPickOverlayWarningKeys.clear();
    }

    function normalizeLinearMoveoutState(value = {}, base = LMO_DEFAULTS) {
      const baseState = base && typeof base === 'object' ? base : LMO_DEFAULTS;
      const source = { ...baseState, ...(value && typeof value === 'object' ? value : {}) };
      const baseVelocity = parseLmoInteger(baseState.velocityMps);
      const baseOffsetByte = parseLmoInteger(baseState.offsetByte);
      const baseOffsetScale = parseLmoFiniteNumber(baseState.offsetScale);
      const velocity = parseLmoInteger(source.velocityMps);
      const offsetByte = parseLmoInteger(source.offsetByte);
      const offsetScale = parseLmoFiniteNumber(source.offsetScale);

      return {
        enabled: parseLmoBoolean(source.enabled, parseLmoBoolean(baseState.enabled, LMO_DEFAULTS.enabled)),
        velocityMps: velocity !== null && velocity > 0
          ? velocity
          : (baseVelocity !== null && baseVelocity > 0 ? baseVelocity : LMO_DEFAULTS.velocityMps),
        offsetByte: offsetByte !== null && offsetByte >= 1 && offsetByte <= 240
          ? offsetByte
          : (baseOffsetByte !== null && baseOffsetByte >= 1 && baseOffsetByte <= 240
            ? baseOffsetByte
            : LMO_DEFAULTS.offsetByte),
        offsetScale: offsetScale !== null && offsetScale !== 0
          ? offsetScale
          : (baseOffsetScale !== null && baseOffsetScale !== 0 ? baseOffsetScale : LMO_DEFAULTS.offsetScale),
        offsetMode: LMO_FIXED_OFFSET_MODE,
        refMode: LMO_FIXED_REF_MODE,
        refTrace: LMO_FIXED_REF_TRACE,
        polarity: LMO_FIXED_POLARITY,
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

      if (enabled) snapshot.enabled = !!enabled.checked;
      if (velocityMps) snapshot.velocityMps = velocityMps.value;
      if (offsetByte) snapshot.offsetByte = offsetByte.value;
      if (offsetScale) snapshot.offsetScale = offsetScale.value;
      snapshot.offsetMode = LMO_FIXED_OFFSET_MODE;
      snapshot.refMode = LMO_FIXED_REF_MODE;
      snapshot.refTrace = LMO_FIXED_REF_TRACE;
      snapshot.polarity = LMO_FIXED_POLARITY;
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
      if (enabled) enabled.checked = lmo.enabled;
      setNumberValue('lmoVelocityMps', lmo.velocityMps);
      setNumberValue('lmoOffsetByte', lmo.offsetByte);
      setNumberValue('lmoOffsetScale', lmo.offsetScale);
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
      return cloneLinearMoveoutState(normalizeLinearMoveoutState(currentLinearMoveout, LMO_DEFAULTS));
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
      ].join('|');
    }

    function currentKey1ValueForLmoPickTransform() {
      const slider = document.getElementById('key1_slider');
      const idxRaw = slider ? parseInt(slider.value, 10) : 0;
      const idx = Number.isFinite(idxRaw) ? idxRaw : 0;
      return Array.isArray(key1Values) ? key1Values[idx] : undefined;
    }

    function lmoPickOffsetCacheKey(context) {
      return [
        context.fileId,
        context.key1,
        context.key1Byte,
        context.key2Byte,
        context.offsetByte,
      ].map((value) => encodeURIComponent(String(value))).join('|');
    }

    function lmoPickShiftCacheKey(context) {
      return `${lmoPickOffsetCacheKey(context)}|${context.lmoKey}`;
    }

    function currentLmoPickContext() {
      const lmo = getCurrentLinearMoveout();
      if (!lmo.enabled) return { enabled: false, lmo, lmoKey: 'lmo:off' };
      const key1 = currentKey1ValueForLmoPickTransform();
      if (!currentFileId || key1 === undefined) {
        return null;
      }
      return {
        enabled: true,
        fileId: currentFileId,
        key1,
        key1Byte: currentKey1Byte,
        key2Byte: currentKey2Byte,
        offsetByte: lmo.offsetByte,
        lmo,
        lmoKey: currentLmoKey(),
      };
    }

    function decodeSectionOffsetsPayload(buffer) {
      if (!window.msgpack || typeof window.msgpack.decode !== 'function') {
        throw new Error('msgpack decoder is not available');
      }
      const payload = window.msgpack.decode(new Uint8Array(buffer));
      if (!payload || payload.dtype !== 'float32') {
        throw new Error('Unexpected section offsets payload dtype');
      }
      const shape = Array.isArray(payload.shape) ? payload.shape : [];
      const n = Number(shape[0]);
      if (!Number.isInteger(n) || n <= 0) {
        throw new Error('Unexpected section offsets payload shape');
      }
      const rawOffsets = payload.offsets;
      let bytes = null;
      if (rawOffsets instanceof Uint8Array) {
        bytes = rawOffsets;
      } else if (rawOffsets instanceof ArrayBuffer) {
        bytes = new Uint8Array(rawOffsets);
      } else if (ArrayBuffer.isView(rawOffsets)) {
        bytes = new Uint8Array(rawOffsets.buffer, rawOffsets.byteOffset, rawOffsets.byteLength);
      }
      if (!bytes || bytes.byteLength !== n * Float32Array.BYTES_PER_ELEMENT) {
        throw new Error('Unexpected section offsets byte length');
      }
      const aligned = bytes.byteOffset % Float32Array.BYTES_PER_ELEMENT === 0
        ? bytes
        : new Uint8Array(bytes);
      return new Float32Array(aligned.buffer, aligned.byteOffset, n).slice();
    }

    function requestSectionOffsetsForLmoPickTransform(context) {
      const offsetKey = lmoPickOffsetCacheKey(context);
      if (lmoSectionOffsetCache.has(offsetKey)) {
        return Promise.resolve(lmoSectionOffsetCache.get(offsetKey));
      }
      if (lmoSectionOffsetInflight.has(offsetKey)) {
        return lmoSectionOffsetInflight.get(offsetKey);
      }

      const params = new URLSearchParams({
        file_id: String(context.fileId),
        key1: String(context.key1),
        key1_byte: String(context.key1Byte),
        key2_byte: String(context.key2Byte),
        offset_byte: String(context.offsetByte),
      });
      let loaded = false;
      const promise = fetch(`/get_section_offsets_bin?${params.toString()}`)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`section offsets request failed with status ${response.status}`);
          }
          return response.arrayBuffer();
        })
        .then((buffer) => {
          const offsets = decodeSectionOffsetsPayload(buffer);
          setBoundedMapEntry(
            lmoSectionOffsetCache,
            offsetKey,
            offsets,
            LMO_SECTION_OFFSET_CACHE_MAX_ENTRIES,
          );
          loaded = true;
          return offsets;
        })
        .catch((err) => {
          console.warn('LMO section offsets fetch failed', err);
          throw err;
        })
        .finally(() => {
          lmoSectionOffsetInflight.delete(offsetKey);
          if (loaded && typeof schedulePickOverlayUpdate === 'function') {
            schedulePickOverlayUpdate();
          }
        });
      lmoSectionOffsetInflight.set(offsetKey, promise);
      return promise;
    }

    function lmoReferenceOffset(values, lmo) {
      if (lmo.refMode === 'min') {
        let best = Infinity;
        for (const value of values) best = Math.min(best, value);
        return best;
      }
      if (lmo.refMode === 'first') return values[0];
      if (lmo.refMode === 'center') return values[Math.floor(values.length / 2)];
      if (lmo.refMode === 'zero') return 0;
      if (lmo.refMode === 'trace') {
        const refTrace = lmo.refTrace | 0;
        if (refTrace < 0 || refTrace >= values.length) return NaN;
        return values[refTrace];
      }
      return NaN;
    }

    function computeLmoShiftSecondsForOffsets(offsets, lmo) {
      const n = offsets ? offsets.length : 0;
      const velocity = Number(lmo.velocityMps);
      const scale = Number(lmo.offsetScale);
      if (!n || !Number.isFinite(velocity) || velocity <= 0 || !Number.isFinite(scale) || scale === 0) {
        return null;
      }

      const values = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const scaled = Number(offsets[i]) * scale;
        if (!Number.isFinite(scaled)) return null;
        values[i] = lmo.offsetMode === 'absolute' ? Math.abs(scaled) : scaled;
      }

      const ref = lmoReferenceOffset(values, lmo);
      const polarity = Number(lmo.polarity) === -1 ? -1 : 1;
      if (!Number.isFinite(ref)) return null;

      const shifts = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const shift = polarity * (values[i] - ref) / velocity;
        if (!Number.isFinite(shift)) return null;
        shifts[i] = shift;
      }
      return shifts;
    }

    function warnLmoPickOverlaySkipped(context) {
      const key = context?.lmoKey || currentLmoKey();
      if (lmoPickOverlayWarningKeys.has(key)) return;
      addBoundedSetEntry(lmoPickOverlayWarningKeys, key, LMO_WARNING_KEY_MAX_ENTRIES);
      console.warn('LMO pick overlay skipped because section offsets are not ready.');
    }

    function getCurrentLmoShiftSeconds() {
      const context = currentLmoPickContext();
      if (!context) return null;
      if (!context.enabled) return { shifts: null, lmo: context.lmo, lmoKey: context.lmoKey };

      const shiftKey = lmoPickShiftCacheKey(context);
      if (lmoShiftSecondsCache.has(shiftKey)) {
        return { shifts: lmoShiftSecondsCache.get(shiftKey), lmo: context.lmo, lmoKey: context.lmoKey };
      }

      const offsetKey = lmoPickOffsetCacheKey(context);
      const offsets = lmoSectionOffsetCache.get(offsetKey);
      if (!offsets) {
        requestSectionOffsetsForLmoPickTransform(context).catch(() => {});
        return null;
      }

      const shifts = computeLmoShiftSecondsForOffsets(offsets, context.lmo);
      if (!shifts) return null;
      setBoundedMapEntry(lmoShiftSecondsCache, shiftKey, shifts, LMO_SHIFT_SECONDS_CACHE_MAX_ENTRIES);
      return { shifts, lmo: context.lmo, lmoKey: context.lmoKey };
    }

    function ensureLmoPickOffsetsReady() {
      const context = currentLmoPickContext();
      if (!context) return Promise.resolve(false);
      if (!context.enabled) return Promise.resolve(true);
      const current = getCurrentLmoShiftSeconds();
      if (current?.shifts) return Promise.resolve(true);
      return requestSectionOffsetsForLmoPickTransform(context)
        .then(() => {
          const updated = getCurrentLmoShiftSeconds();
          return !!updated?.shifts;
        })
        .catch(() => false);
    }

    function getLmoShiftSecondsForTrace(trace) {
      const context = currentLmoPickContext();
      if (!context) return NaN;
      if (!context.enabled) return 0;

      const current = getCurrentLmoShiftSeconds();
      const shifts = current?.shifts;
      const traceIndex = Math.round(Number(trace));
      if (
        !shifts ||
        !Number.isInteger(traceIndex) ||
        traceIndex < 0 ||
        traceIndex >= shifts.length
      ) {
        warnLmoPickOverlaySkipped(context);
        return NaN;
      }
      return shifts[traceIndex];
    }

    function rawTimeToDisplayTime(trace, rawTime) {
      const raw = Number(rawTime);
      if (!Number.isFinite(raw)) return NaN;
      const context = currentLmoPickContext();
      if (!context) return NaN;
      if (!context.enabled) return raw;
      const shift = getLmoShiftSecondsForTrace(trace);
      return Number.isFinite(shift) ? raw - shift : NaN;
    }

    function displayTimeToRawTime(trace, displayTime) {
      const display = Number(displayTime);
      if (!Number.isFinite(display)) return NaN;
      const context = currentLmoPickContext();
      if (!context) return NaN;
      if (!context.enabled) return display;
      const shift = getLmoShiftSecondsForTrace(trace);
      return Number.isFinite(shift) ? display + shift : NaN;
    }

    function pickRawTimeToDisplayTime(trace, rawTime) {
      const display = rawTimeToDisplayTime(trace, rawTime);
      if (!Number.isFinite(display)) {
        const context = currentLmoPickContext();
        if (context?.enabled) warnLmoPickOverlaySkipped(context);
      }
      return display;
    }

    function onLinearMoveoutControlChange(options = {}) {
      return setCurrentLinearMoveout(getLinearMoveoutControlSnapshot(), options);
    }

    function bindLinearMoveoutControls() {
      if (window.__linearMoveoutControlsBound) return;
      const numberIds = ['lmoVelocityMps', 'lmoOffsetByte', 'lmoOffsetScale'];
      const changeIds = ['lmoEnabled'];
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
    window.rawTimeToDisplayTime = rawTimeToDisplayTime;
    window.displayTimeToRawTime = displayTimeToRawTime;
    window.getLmoShiftSecondsForTrace = getLmoShiftSecondsForTrace;
    window.ensureLmoPickOffsetsReady = ensureLmoPickOffsetsReady;
    window.clearLinearMoveoutRuntimeCaches = clearLinearMoveoutRuntimeCaches;
    window.pickRawTimeToDisplayTime = pickRawTimeToDisplayTime;
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
