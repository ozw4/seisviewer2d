import { viewerPerfMetrics } from './perf_metrics.js';

const EMPTY_SLOT = Object.freeze({
  requestId: 0,
  controller: null,
});

const METRIC_KEYS = Object.freeze([
  'started',
  'aborted',
  'staleDropped',
  'completed',
  'failed',
]);

function createEmptyMetrics() {
  return {
    started: 0,
    aborted: 0,
    staleDropped: 0,
    completed: 0,
    failed: 0,
  };
}

export function createRenderRequestController({ perfMetrics = viewerPerfMetrics } = {}) {
  let nextRequestId = 0;
  const slots = new Map();
  const metrics = new Map();

  function requireSlotName(slotName) {
    if (typeof slotName !== 'string' || slotName.trim() === '') {
      throw new TypeError('slotName must be a non-empty string');
    }
    return slotName;
  }

  function getSlot(slotName) {
    return slots.get(slotName) || EMPTY_SLOT;
  }

  function getMetrics(slotName) {
    if (!metrics.has(slotName)) {
      metrics.set(slotName, createEmptyMetrics());
    }
    return metrics.get(slotName);
  }

  function increment(slotName, key) {
    if (!METRIC_KEYS.includes(key)) return;
    getMetrics(slotName)[key] += 1;
  }

  function notify(methodName, meta) {
    const fn = perfMetrics && perfMetrics[methodName];
    if (typeof fn === 'function') fn(meta);
  }

  function abort(slotName) {
    const name = requireSlotName(slotName);
    const slot = getSlot(name);
    if (slot.controller && !slot.controller.signal.aborted) {
      slot.controller.abort();
      increment(name, 'aborted');
      notify('recordRequestAborted', { slotName: name, requestId: slot.requestId });
    }
    if (slot.requestId !== 0) {
      slots.set(name, { requestId: 0, controller: null });
    }
  }

  function begin(slotName) {
    const name = requireSlotName(slotName);
    abort(name);
    const controller = new AbortController();
    const requestId = allocateRequestId();
    slots.set(name, { requestId, controller });
    increment(name, 'started');
    notify('recordRequestStarted', { slotName: name, requestId });
    return { requestId, signal: controller.signal };
  }

  function allocateRequestId() {
    nextRequestId += 1;
    return nextRequestId;
  }

  function isCurrent(slotName, requestId) {
    const name = requireSlotName(slotName);
    return getSlot(name).requestId === requestId;
  }

  function abortAll() {
    for (const slotName of slots.keys()) {
      abort(slotName);
    }
  }

  function markCompleted(slotName, requestId) {
    const name = requireSlotName(slotName);
    if (!isCurrent(name, requestId)) return false;
    increment(name, 'completed');
    notify('recordRequestCompleted', { slotName: name, requestId });
    slots.set(name, { requestId, controller: null });
    return true;
  }

  function markFailed(slotName, requestId) {
    const name = requireSlotName(slotName);
    if (!isCurrent(name, requestId)) return false;
    increment(name, 'failed');
    slots.set(name, { requestId, controller: null });
    return true;
  }

  function markStaleDropped(slotName, requestId) {
    const name = requireSlotName(slotName);
    if (isCurrent(name, requestId)) return false;
    increment(name, 'staleDropped');
    notify('recordStaleResponseDropped', { slotName: name, requestId });
    return true;
  }

  function snapshotMetrics() {
    const snapshot = {};
    for (const [slotName, slotMetrics] of metrics.entries()) {
      snapshot[slotName] = { ...slotMetrics };
    }
    return snapshot;
  }

  return {
    allocateRequestId,
    begin,
    isCurrent,
    abort,
    abortAll,
    markCompleted,
    markFailed,
    markStaleDropped,
    snapshotMetrics,
  };
}

export const viewerRenderRequests = createRenderRequestController();
