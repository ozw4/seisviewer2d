export const RENDER_INVALIDATION_KINDS = Object.freeze({
  base: 'base',
  overlay: 'overlay',
  unknown: 'unknown',
});

export const BASE_RENDER_REASONS = Object.freeze([
  'file-id',
  'key1',
  'layer',
  'pipeline',
  'scaling',
  'gain',
  'transpose',
  'viewport-full-res',
]);

export const OVERLAY_ONLY_REASONS = Object.freeze([
  'manual-pick-add',
  'manual-pick-move',
  'manual-pick-delete',
  'pending-pick-state',
  'prediction-toggle',
  'prediction-data-current-viewport',
  'selected-pick-change',
  'hover-pick-change',
]);

const BASE_RENDER_REASON_SET = new Set(BASE_RENDER_REASONS);
const OVERLAY_ONLY_REASON_SET = new Set(OVERLAY_ONLY_REASONS);

const REASON_ALIASES = Object.freeze({
  file_id: 'file-id',
  file: 'file-id',
  'file-change': 'file-id',
  'file-change-clear': 'file-id',
  key1_byte: 'key1',
  section: 'key1',
  'section-change': 'key1',
  'section-cache-sync': 'key1',
  tap: 'layer',
  'layer-change': 'layer',
  'pipeline-change': 'pipeline',
  'prediction-model-change': 'prediction-data-current-viewport',
  'prediction-cache-hit': 'prediction-data-current-viewport',
  'prediction-cache-miss': 'prediction-data-current-viewport',
  'prediction-request-start': 'prediction-data-current-viewport',
  'prediction-complete': 'prediction-data-current-viewport',
  'pick-overlay-update': 'pending-pick-state',
  'manual-pick': 'manual-pick-add',
  'manual-pick-state': 'manual-pick-add',
  'manual-pick-overlay': 'manual-pick-add',
  'toggle-pick-mode': 'pending-pick-state',
  'single-pick': 'pending-pick-state',
  'shift-keyup': 'pending-pick-state',
  'delete-range-complete': 'pending-pick-state',
  'line-pick-anchor': 'pending-pick-state',
  'delete-range-anchor': 'pending-pick-state',
  'viewport': 'viewport-full-res',
  'viewport-change': 'viewport-full-res',
});

export function normalizeRenderInvalidationReason(reason) {
  const normalized = String(reason ?? '').trim().toLowerCase().replace(/_/g, '-');
  return REASON_ALIASES[normalized] || normalized;
}

export function classifyRenderInvalidation(reason) {
  const normalized = normalizeRenderInvalidationReason(reason);
  if (BASE_RENDER_REASON_SET.has(normalized)) {
    return {
      kind: RENDER_INVALIDATION_KINDS.base,
      reason: normalized,
      requiresBaseRender: true,
      overlayOnly: false,
    };
  }
  if (OVERLAY_ONLY_REASON_SET.has(normalized)) {
    return {
      kind: RENDER_INVALIDATION_KINDS.overlay,
      reason: normalized,
      requiresBaseRender: false,
      overlayOnly: true,
    };
  }
  return {
    kind: RENDER_INVALIDATION_KINDS.unknown,
    reason: normalized,
    requiresBaseRender: false,
    overlayOnly: false,
  };
}

export function requiresBaseRender(reason) {
  return classifyRenderInvalidation(reason).requiresBaseRender;
}

export function isOverlayOnlyInvalidation(reason) {
  return classifyRenderInvalidation(reason).overlayOnly;
}

export function createRenderInvalidationScheduler({
  scheduleBaseRender,
  scheduleOverlayRedraw,
  beforeBaseRender,
  requestAnimationFrameImpl = globalThis.requestAnimationFrame,
} = {}) {
  let overlayRaf = 0;
  let pendingOverlayReason = null;

  function flushOverlay() {
    overlayRaf = 0;
    const reason = pendingOverlayReason || 'overlay';
    pendingOverlayReason = null;
    if (typeof scheduleOverlayRedraw === 'function') {
      scheduleOverlayRedraw(reason);
    }
  }

  function scheduleOverlay(reason) {
    pendingOverlayReason = reason;
    if (overlayRaf !== 0) return;
    if (typeof requestAnimationFrameImpl === 'function') {
      overlayRaf = requestAnimationFrameImpl(flushOverlay);
      return;
    }
    overlayRaf = 1;
    setTimeout(flushOverlay, 0);
  }

  function invalidate(reason, payload = {}) {
    const classification = classifyRenderInvalidation(reason);
    if (classification.requiresBaseRender) {
      if (typeof beforeBaseRender === 'function') {
        beforeBaseRender(classification.reason, payload);
      }
      const baseScheduler = typeof payload?.scheduleBaseRender === 'function'
        ? payload.scheduleBaseRender
        : scheduleBaseRender;
      if (typeof baseScheduler === 'function') {
        baseScheduler(classification.reason, payload);
      }
      return classification;
    }
    if (classification.overlayOnly) {
      scheduleOverlay(classification.reason);
    }
    return classification;
  }

  return {
    invalidate,
    classify: classifyRenderInvalidation,
    requiresBaseRender,
    isOverlayOnly: isOverlayOnlyInvalidation,
  };
}

window.RenderInvalidation = {
  RENDER_INVALIDATION_KINDS,
  BASE_RENDER_REASONS,
  OVERLAY_ONLY_REASONS,
  normalizeReason: normalizeRenderInvalidationReason,
  classify: classifyRenderInvalidation,
  requiresBaseRender,
  isOverlayOnly: isOverlayOnlyInvalidation,
  createScheduler: createRenderInvalidationScheduler,
};
