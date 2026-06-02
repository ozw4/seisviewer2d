import { viewerPerfMetrics } from './perf_metrics.js';

const OVERLAY_CLASS = 'sv-viewer-perf-overlay';
const STYLE_ID = 'sv-viewer-perf-overlay-style';
const DEFAULT_MIN_INTERVAL_MS = 250;

function normalizePerfParam(value) {
  return String(value ?? '').trim().toLowerCase();
}

export function isViewerPerfOverlayEnabled(search = window.location.search) {
  const params = new URLSearchParams(search || '');
  return normalizePerfParam(params.get('perf')) === '1';
}

function formatMs(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  return `${num.toFixed(num >= 100 ? 0 : 1)} ms`;
}

function formatCount(value) {
  const num = Number(value);
  return Number.isFinite(num) ? String(Math.max(0, Math.floor(num))) : '0';
}

function formatBytes(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return '-';
  if (num >= 1024 * 1024) return `${(num / (1024 * 1024)).toFixed(1)} MB`;
  if (num >= 1024) return `${(num / 1024).toFixed(1)} KB`;
  return `${Math.floor(num)} B`;
}

function formatText(value) {
  if (value === null || value === undefined || value === '') return '-';
  return String(value);
}

function formatVisible(snapshot) {
  const traces = Number(snapshot.visibleTraces);
  const samples = Number(snapshot.visibleSamples);
  if (!Number.isFinite(traces) || !Number.isFinite(samples)) return '-';
  if (traces <= 0 && samples <= 0) return '-';
  return `${Math.max(0, Math.floor(traces))} x ${Math.max(0, Math.floor(samples))}`;
}

export function formatPerfOverlaySnapshot(snapshot = {}) {
  return [
    `fetch  ${formatMs(snapshot.fetchMs)}`,
    `decode ${formatMs(snapshot.decodeMs)}`,
    `render ${formatMs(snapshot.renderMs)}`,
    `overlay ${formatMs(snapshot.overlayMs)}`,
    `bytes  ${formatBytes(snapshot.payloadBytes)}`,
    `visible ${formatVisible(snapshot)}`,
    `request started/completed/aborted/stale ${[
      snapshot.requestStarted,
      snapshot.requestCompleted,
      snapshot.requestAborted,
      snapshot.staleResponseDropped,
    ].map(formatCount).join('/')}`,
    `key1   ${formatText(snapshot.lastKey1)}`,
    `layer  ${formatText(snapshot.lastLayer)}`,
    `mode   ${formatText(snapshot.lastRenderMode)}`,
  ].join('\n');
}

function ensureStyle(doc) {
  if (doc.getElementById(STYLE_ID)) return;
  const style = doc.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
.${OVERLAY_CLASS} {
  position: fixed;
  right: 12px;
  bottom: 12px;
  z-index: 2147483000;
  max-width: min(320px, calc(100vw - 24px));
  padding: 8px 10px;
  border: 1px solid rgba(15, 23, 42, 0.35);
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.88);
  color: #f8fafc;
  font: 11px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  white-space: pre;
  pointer-events: none;
  user-select: none;
}`;
  doc.head.appendChild(style);
}

function createOverlayNode(doc) {
  const node = doc.createElement('div');
  node.className = OVERLAY_CLASS;
  node.dataset.testid = 'viewer-perf-overlay';
  node.setAttribute('aria-hidden', 'true');
  node.textContent = formatPerfOverlaySnapshot();
  doc.body.appendChild(node);
  return node;
}

export function initViewerPerfOverlay({
  doc = document,
  win = window,
  metrics = viewerPerfMetrics,
  minIntervalMs = DEFAULT_MIN_INTERVAL_MS,
} = {}) {
  if (!doc?.body || !isViewerPerfOverlayEnabled(win?.location?.search || '')) {
    return null;
  }

  ensureStyle(doc);
  const node = createOverlayNode(doc);
  let disposed = false;
  let lastUpdatedAt = 0;

  function readSnapshot() {
    if (metrics && typeof metrics.snapshot === 'function') {
      return metrics.snapshot();
    }
    return {};
  }

  function update(force = false) {
    if (disposed) return;
    const now = typeof win?.performance?.now === 'function'
      ? win.performance.now()
      : Date.now();
    if (!force && now - lastUpdatedAt < minIntervalMs) return;
    lastUpdatedAt = now;
    node.textContent = formatPerfOverlaySnapshot(readSnapshot());
  }

  update(true);
  const intervalId = win.setInterval(() => update(false), Math.max(DEFAULT_MIN_INTERVAL_MS, minIntervalMs));

  return {
    node,
    update,
    dispose() {
      disposed = true;
      win.clearInterval(intervalId);
      node.remove();
    },
  };
}
