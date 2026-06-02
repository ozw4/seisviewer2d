const TIMING_KEYS = Object.freeze({
  fetch: 'fetchMs',
  decode: 'decodeMs',
  render: 'renderMs',
  overlay: 'overlayMs',
});

function defaultNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function coerceFiniteNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function createInitialState() {
  return {
    fetchMs: 0,
    decodeMs: 0,
    renderMs: 0,
    overlayMs: 0,
    payloadBytes: 0,
    visibleTraces: 0,
    visibleSamples: 0,
    requestStarted: 0,
    requestCompleted: 0,
    requestAborted: 0,
    staleResponseDropped: 0,
    lastRenderMode: null,
    lastLayer: null,
    lastKey1: null,
  };
}

export function createPerfMetrics({ now = defaultNow } = {}) {
  if (typeof now !== 'function') {
    throw new TypeError('now must be a function');
  }

  let state = createInitialState();

  function updateContext(meta = {}) {
    if (!meta || typeof meta !== 'object') return;
    if (meta.mode !== undefined) state.lastRenderMode = meta.mode;
    if (meta.renderMode !== undefined) state.lastRenderMode = meta.renderMode;
    if (meta.layer !== undefined) state.lastLayer = meta.layer;
    if (meta.key1 !== undefined) state.lastKey1 = meta.key1;
    if (meta.payloadBytes !== undefined) state.payloadBytes = coerceFiniteNumber(meta.payloadBytes);
    if (meta.bytes !== undefined) state.payloadBytes = coerceFiniteNumber(meta.bytes);
    if (meta.visibleTraces !== undefined) state.visibleTraces = coerceFiniteNumber(meta.visibleTraces);
    if (meta.visibleSamples !== undefined) state.visibleSamples = coerceFiniteNumber(meta.visibleSamples);
  }

  function recordDuration(name, durationMs) {
    const key = TIMING_KEYS[name];
    if (!key) return false;
    state[key] = Math.max(0, coerceFiniteNumber(durationMs));
    return true;
  }

  function startTimer(name) {
    if (!TIMING_KEYS[name]) {
      throw new TypeError(`unknown perf timer: ${name}`);
    }
    return {
      name,
      startedAt: coerceFiniteNumber(now()),
    };
  }

  function stopTimer(timer) {
    if (!timer || typeof timer !== 'object') return null;
    const key = TIMING_KEYS[timer.name];
    if (!key) return null;
    const durationMs = coerceFiniteNumber(now()) - coerceFiniteNumber(timer.startedAt);
    state[key] = Math.max(0, durationMs);
    return state[key];
  }

  function recordRequestStarted(meta = {}) {
    state.requestStarted += 1;
    updateContext(meta);
  }

  function recordRequestCompleted(meta = {}) {
    state.requestCompleted += 1;
    updateContext(meta);
  }

  function recordRequestAborted(meta = {}) {
    state.requestAborted += 1;
    updateContext(meta);
  }

  function recordStaleResponseDropped(meta = {}) {
    state.staleResponseDropped += 1;
    updateContext(meta);
  }

  function recordPayload(meta = {}) {
    updateContext(meta);
  }

  function snapshot() {
    return { ...state };
  }

  function reset() {
    state = createInitialState();
  }

  return {
    recordDuration,
    recordRequestStarted,
    recordRequestCompleted,
    recordRequestAborted,
    recordStaleResponseDropped,
    recordPayload,
    snapshot,
    reset,
    startTimer,
    stopTimer,
  };
}

export const viewerPerfMetrics = createPerfMetrics();
