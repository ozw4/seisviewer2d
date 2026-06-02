import { describe, expect, test, vi } from 'vitest';
import {
  BASE_RENDER_REASONS,
  OVERLAY_ONLY_REASONS,
  classifyRenderInvalidation,
  createRenderInvalidationScheduler,
  isOverlayOnlyInvalidation,
  requiresBaseRender,
} from '../../static/viewer/render_invalidation.js';

describe('render invalidation reasons', () => {
  test.each(BASE_RENDER_REASONS)('%s requires a base render', (reason) => {
    expect(requiresBaseRender(reason)).toBe(true);
    expect(isOverlayOnlyInvalidation(reason)).toBe(false);
    expect(classifyRenderInvalidation(reason)).toMatchObject({
      kind: 'base',
      reason,
      requiresBaseRender: true,
      overlayOnly: false,
    });
  });

  test.each(OVERLAY_ONLY_REASONS)('%s redraws overlays only', (reason) => {
    expect(requiresBaseRender(reason)).toBe(false);
    expect(isOverlayOnlyInvalidation(reason)).toBe(true);
    expect(classifyRenderInvalidation(reason)).toMatchObject({
      kind: 'overlay',
      reason,
      requiresBaseRender: false,
      overlayOnly: true,
    });
  });

  test('legacy aliases normalize to canonical reasons', () => {
    expect(classifyRenderInvalidation('file_id')).toMatchObject({
      kind: 'base',
      reason: 'file-id',
    });
    expect(classifyRenderInvalidation('prediction-cache-hit')).toMatchObject({
      kind: 'overlay',
      reason: 'prediction-data-current-viewport',
    });
    expect(classifyRenderInvalidation('pick-overlay-update')).toMatchObject({
      kind: 'overlay',
      reason: 'pending-pick-state',
    });
  });

  test('unknown reasons do not silently become base or overlay work', () => {
    expect(classifyRenderInvalidation('not-a-real-reason')).toEqual({
      kind: 'unknown',
      reason: 'not-a-real-reason',
      requiresBaseRender: false,
      overlayOnly: false,
    });
  });
});

describe('render invalidation scheduler', () => {
  test('overlay-only invalidations are coalesced with requestAnimationFrame', () => {
    const base = vi.fn();
    const overlay = vi.fn();
    const rafCallbacks = [];
    const scheduler = createRenderInvalidationScheduler({
      scheduleBaseRender: base,
      scheduleOverlayRedraw: overlay,
      requestAnimationFrameImpl: (callback) => {
        rafCallbacks.push(callback);
        return rafCallbacks.length;
      },
    });

    scheduler.invalidate('manual-pick-add');
    scheduler.invalidate('prediction-toggle');

    expect(base).not.toHaveBeenCalled();
    expect(overlay).not.toHaveBeenCalled();
    expect(rafCallbacks).toHaveLength(1);

    rafCallbacks[0]();

    expect(overlay).toHaveBeenCalledTimes(1);
    expect(overlay).toHaveBeenCalledWith('prediction-toggle');
  });

  test('base invalidations schedule base render directly', () => {
    const base = vi.fn();
    const overlay = vi.fn();
    const scheduler = createRenderInvalidationScheduler({
      scheduleBaseRender: base,
      scheduleOverlayRedraw: overlay,
      requestAnimationFrameImpl: vi.fn(),
    });

    const classification = scheduler.invalidate('key1', { source: 'test' });

    expect(classification.kind).toBe('base');
    expect(base).toHaveBeenCalledWith('key1', { source: 'test' });
    expect(overlay).not.toHaveBeenCalled();
  });

  test('base invalidations can sync overlay state before base render work', () => {
    const calls = [];
    const scheduler = createRenderInvalidationScheduler({
      beforeBaseRender: (reason, payload) => calls.push(['sync', reason, payload.source]),
      scheduleBaseRender: (reason, payload) => calls.push(['base', reason, payload.source]),
      scheduleOverlayRedraw: vi.fn(),
      requestAnimationFrameImpl: vi.fn(),
    });

    scheduler.invalidate('layer-change', { source: 'layer-select' });

    expect(calls).toEqual([
      ['sync', 'layer', 'layer-select'],
      ['base', 'layer', 'layer-select'],
    ]);
  });

  test('base invalidations can run the caller-provided base work', () => {
    const base = vi.fn();
    const customBase = vi.fn();
    const scheduler = createRenderInvalidationScheduler({
      scheduleBaseRender: base,
      scheduleOverlayRedraw: vi.fn(),
      requestAnimationFrameImpl: vi.fn(),
    });

    scheduler.invalidate('viewport-full-res', {
      source: 'viewport',
      scheduleBaseRender: customBase,
    });

    expect(customBase).toHaveBeenCalledWith('viewport-full-res', expect.objectContaining({
      source: 'viewport',
    }));
    expect(base).not.toHaveBeenCalled();
  });
});
