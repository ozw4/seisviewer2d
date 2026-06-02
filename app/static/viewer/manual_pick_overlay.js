const DEFAULT_OPTIONS = Object.freeze({
  manualColor: 'red',
  manualWidth: 2,
  manualLength: 14,
  pendingLineStroke: '#b45309',
  pendingLineFill: '#f59e0b',
  pendingLineWidth: 2,
  pendingLineSize: 16,
  deleteRangeColor: '#b91c1c',
  deleteRangeWidth: 2,
  deleteRangeDash: [6, 5],
});

let overlayState = {
  manualPicks: [],
  pending: null,
  timeTransform: null,
  options: DEFAULT_OPTIONS,
};
let redrawRaf = 0;
let initialized = false;

function finiteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function normalizePick(pick) {
  if (!pick || typeof pick !== 'object') return null;
  const trace = finiteNumber(pick.trace);
  const time = finiteNumber(pick.time);
  if (trace === null || time === null) return null;
  return { trace, time, raw: pick };
}

function normalizePending(pending) {
  if (!pending || typeof pending !== 'object') return null;
  const trace = finiteNumber(pending.trace);
  if (trace === null) return null;
  if (pending.kind === 'line') {
    const time = finiteNumber(pending.time);
    if (time === null) return null;
    return { kind: 'line', trace, time };
  }
  if (pending.kind === 'delete-range') {
    return { kind: 'delete-range', trace };
  }
  return null;
}

function resolveDisplayTime(pick, timeTransform) {
  if (typeof timeTransform !== 'function') return pick.time;
  const displayTime = finiteNumber(timeTransform(pick.trace, pick.time, pick.raw));
  return displayTime === null ? null : displayTime;
}

function clearCanvas(canvas, width, height) {
  const ctx = canvas?.getContext?.('2d');
  if (!ctx) return null;
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

function drawManualPick(ctx, point, options) {
  const half = options.manualLength / 2;
  ctx.beginPath();
  ctx.moveTo(point.relativeX - half, point.relativeY);
  ctx.lineTo(point.relativeX + half, point.relativeY);
  ctx.strokeStyle = options.manualColor;
  ctx.lineWidth = options.manualWidth;
  ctx.setLineDash([]);
  ctx.stroke();
}

function drawPendingLineAnchor(ctx, point, options) {
  const half = options.pendingLineSize / 2;
  ctx.beginPath();
  ctx.moveTo(point.relativeX, point.relativeY - half);
  ctx.lineTo(point.relativeX + half, point.relativeY);
  ctx.lineTo(point.relativeX, point.relativeY + half);
  ctx.lineTo(point.relativeX - half, point.relativeY);
  ctx.closePath();
  ctx.fillStyle = 'rgba(245, 158, 11, 0.12)';
  ctx.fill();
  ctx.strokeStyle = options.pendingLineStroke;
  ctx.lineWidth = options.pendingLineWidth;
  ctx.setLineDash([]);
  ctx.stroke();
}

function drawDeleteRangeAnchor(ctx, transform, trace, options) {
  const timeRange = transform.visibleTimeRange?.();
  if (!Array.isArray(timeRange) || timeRange.length !== 2) return;
  const p0 = transform.traceTimeToPixel(trace, timeRange[0]);
  const p1 = transform.traceTimeToPixel(trace, timeRange[1]);
  if (!p0 || !p1) return;
  ctx.beginPath();
  ctx.moveTo(p0.relativeX, p0.relativeY);
  ctx.lineTo(p1.relativeX, p1.relativeY);
  ctx.strokeStyle = options.deleteRangeColor;
  ctx.lineWidth = options.deleteRangeWidth;
  ctx.setLineDash(options.deleteRangeDash);
  ctx.stroke();
  ctx.setLineDash([]);
}

export function renderManualPickOverlay(payload = {}) {
  const canvas = payload.canvas || payload.manualPickCanvas;
  const transform = payload.transform;
  const width = finiteNumber(payload.width ?? payload.state?.width ?? canvas?.clientWidth) ?? 0;
  const height = finiteNumber(payload.height ?? payload.state?.height ?? canvas?.clientHeight) ?? 0;
  const ctx = clearCanvas(canvas, width, height);
  if (!ctx || !transform?.valid) return { manual: 0, pending: false };

  const options = { ...DEFAULT_OPTIONS, ...(overlayState.options || {}) };
  let manualCount = 0;
  const manualPicks = Array.isArray(overlayState.manualPicks) ? overlayState.manualPicks : [];
  for (const rawPick of manualPicks) {
    const pick = normalizePick(rawPick);
    if (!pick) continue;
    const displayTime = resolveDisplayTime(pick, overlayState.timeTransform);
    if (displayTime === null || !transform.isTraceTimeVisible(pick.trace, displayTime)) continue;
    const point = transform.traceTimeToPixel(pick.trace, displayTime);
    if (!point) continue;
    drawManualPick(ctx, point, options);
    manualCount += 1;
  }

  const pending = normalizePending(overlayState.pending);
  if (pending?.kind === 'line') {
    const point = transform.traceTimeToPixel(pending.trace, pending.time);
    if (point) drawPendingLineAnchor(ctx, point, options);
  } else if (pending?.kind === 'delete-range') {
    drawDeleteRangeAnchor(ctx, transform, pending.trace, options);
  }

  return { manual: manualCount, pending: !!pending };
}

export function scheduleManualPickOverlayRedraw(reason = 'manual-pick') {
  if (redrawRaf !== 0) return;
  redrawRaf = requestAnimationFrame(() => {
    redrawRaf = 0;
    const state = window.ViewerOverlayLayer?.getState?.();
    if (!state?.manualPickCanvas || !state.transform) {
      window.scheduleViewerOverlaySync?.(reason);
      return;
    }
    renderManualPickOverlay({
      reason,
      state,
      canvas: state.manualPickCanvas,
      transform: state.transform,
      width: state.width,
      height: state.height,
    });
    window.viewerPerfMetrics?.recordOverlayRender?.({ reason });
  });
}

export function updateManualPickOverlayState(next = {}, options = {}) {
  overlayState = {
    manualPicks: Array.isArray(next.manualPicks) ? next.manualPicks.slice() : [],
    pending: normalizePending(next.pending),
    timeTransform: typeof next.timeTransform === 'function' ? next.timeTransform : null,
    options: { ...DEFAULT_OPTIONS, ...(next.options || {}) },
  };
  if (options.redraw !== false) {
    scheduleManualPickOverlayRedraw(options.reason || 'manual-pick-state');
  }
  return overlayState;
}

export function initManualPickOverlay() {
  if (initialized) return window.ManualPickOverlay;
  initialized = true;
  window.onViewerOverlayRedraw?.(({ manualPickCanvas, transform, state }) => {
    renderManualPickOverlay({
      canvas: manualPickCanvas,
      transform,
      state,
      width: state?.width,
      height: state?.height,
    });
  });
  window.scheduleViewerOverlaySync?.('manual-pick-overlay-init');
  return window.ManualPickOverlay;
}

window.ManualPickOverlay = {
  init: initManualPickOverlay,
  updateState: updateManualPickOverlayState,
  scheduleRedraw: scheduleManualPickOverlayRedraw,
  render: renderManualPickOverlay,
};
window.initManualPickOverlay = initManualPickOverlay;
window.updateManualPickOverlayState = updateManualPickOverlayState;
window.scheduleManualPickOverlayRedraw = scheduleManualPickOverlayRedraw;
