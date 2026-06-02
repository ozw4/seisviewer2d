const DEFAULT_OPTIONS = Object.freeze({
  predictionColor: '#1f77b4',
  predictionWidth: 4,
  predictionLength: 18,
});

let overlayState = {
  predictedPicks: [],
  source: null,
  current: null,
  show: false,
  timeTransform: null,
  options: DEFAULT_OPTIONS,
};
let redrawRaf = 0;
let initialized = false;

function finiteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function normalizeKey(value) {
  return value === undefined ? null : value;
}

function metadataMatches(source, current) {
  if (!source || !current) return false;
  return (
    normalizeKey(source.fileId) === normalizeKey(current.fileId) &&
    normalizeKey(source.key1) === normalizeKey(current.key1) &&
    normalizeKey(source.layer) === normalizeKey(current.layer) &&
    normalizeKey(source.pipelineKey) === normalizeKey(current.pipelineKey) &&
    normalizeKey(source.modelId) === normalizeKey(current.modelId)
  );
}

function normalizeMetadata(metadata) {
  if (!metadata || typeof metadata !== 'object') return null;
  return {
    fileId: metadata.fileId ?? null,
    key1: metadata.key1 ?? null,
    layer: metadata.layer ?? 'raw',
    pipelineKey: metadata.pipelineKey ?? null,
    modelId: metadata.modelId ?? null,
  };
}

function normalizePick(pick) {
  if (!pick || typeof pick !== 'object') return null;
  const trace = finiteNumber(pick.trace);
  const time = finiteNumber(pick.time);
  if (trace === null || time === null) return null;
  return { trace, time, raw: pick };
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

function drawPredictionPick(ctx, point, options) {
  const half = options.predictionLength / 2;
  ctx.beginPath();
  ctx.moveTo(point.relativeX - half, point.relativeY);
  ctx.lineTo(point.relativeX + half, point.relativeY);
  ctx.strokeStyle = options.predictionColor;
  ctx.lineWidth = options.predictionWidth;
  ctx.setLineDash([]);
  ctx.stroke();
}

export function isPredictionOverlayFresh(source, current) {
  return metadataMatches(normalizeMetadata(source), normalizeMetadata(current));
}

export function renderPredictionOverlay(payload = {}) {
  const canvas = payload.canvas || payload.predictionCanvas;
  const transform = payload.transform;
  const width = finiteNumber(payload.width ?? payload.state?.width ?? canvas?.clientWidth) ?? 0;
  const height = finiteNumber(payload.height ?? payload.state?.height ?? canvas?.clientHeight) ?? 0;
  const ctx = clearCanvas(canvas, width, height);
  if (!ctx || !transform?.valid) return { predicted: 0, stale: false };

  const fresh = metadataMatches(overlayState.source, overlayState.current);
  if (!overlayState.show) return { predicted: 0, stale: false };
  if (!fresh) return { predicted: 0, stale: true };

  const options = { ...DEFAULT_OPTIONS, ...(overlayState.options || {}) };
  let predictedCount = 0;
  const predictedPicks = Array.isArray(overlayState.predictedPicks) ? overlayState.predictedPicks : [];
  for (const rawPick of predictedPicks) {
    const pick = normalizePick(rawPick);
    if (!pick) continue;
    const displayTime = resolveDisplayTime(pick, overlayState.timeTransform);
    if (displayTime === null || !transform.isTraceTimeVisible(pick.trace, displayTime)) continue;
    const point = transform.traceTimeToPixel(pick.trace, displayTime);
    if (!point) continue;
    drawPredictionPick(ctx, point, options);
    predictedCount += 1;
  }

  return { predicted: predictedCount, stale: false };
}

export function schedulePredictionOverlayRedraw(reason = 'prediction') {
  if (redrawRaf !== 0) return;
  redrawRaf = requestAnimationFrame(() => {
    redrawRaf = 0;
    const state = window.ViewerOverlayLayer?.getState?.();
    if (!state?.predictionCanvas || !state.transform) {
      window.scheduleViewerOverlaySync?.(reason);
      return;
    }
    renderPredictionOverlay({
      reason,
      state,
      canvas: state.predictionCanvas,
      transform: state.transform,
      width: state.width,
      height: state.height,
    });
  });
}

export function updatePredictionOverlayState(next = {}, options = {}) {
  overlayState = {
    predictedPicks: Array.isArray(next.predictedPicks) ? next.predictedPicks.slice() : [],
    source: normalizeMetadata(next.source),
    current: normalizeMetadata(next.current),
    show: next.show === true,
    timeTransform: typeof next.timeTransform === 'function' ? next.timeTransform : null,
    options: { ...DEFAULT_OPTIONS, ...(next.options || {}) },
  };
  if (options.redraw !== false) {
    schedulePredictionOverlayRedraw(options.reason || 'prediction-state');
  }
  return overlayState;
}

export function initPredictionOverlay() {
  if (initialized) return window.PredictionOverlay;
  initialized = true;
  window.onViewerOverlayRedraw?.(({ predictionCanvas, transform, state }) => {
    renderPredictionOverlay({
      canvas: predictionCanvas,
      transform,
      state,
      width: state?.width,
      height: state?.height,
    });
  });
  window.scheduleViewerOverlaySync?.('prediction-overlay-init');
  return window.PredictionOverlay;
}

window.PredictionOverlay = {
  init: initPredictionOverlay,
  updateState: updatePredictionOverlayState,
  scheduleRedraw: schedulePredictionOverlayRedraw,
  render: renderPredictionOverlay,
  isFresh: isPredictionOverlayFresh,
};
window.initPredictionOverlay = initPredictionOverlay;
window.updatePredictionOverlayState = updatePredictionOverlayState;
window.schedulePredictionOverlayRedraw = schedulePredictionOverlayRedraw;
