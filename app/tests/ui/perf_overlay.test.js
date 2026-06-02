import { afterEach, expect, test, vi } from 'vitest';
import {
  formatPerfOverlaySnapshot,
  initViewerPerfOverlay,
  isViewerPerfOverlayEnabled,
} from '../../static/viewer/perf_overlay.js';

afterEach(() => {
  vi.useRealTimers();
  document.head.innerHTML = '';
  document.body.innerHTML = '';
});

function createMetrics(snapshot) {
  return {
    snapshot: vi.fn(() => snapshot),
  };
}

test('perf overlay is enabled only by perf=1 query param', () => {
  expect(isViewerPerfOverlayEnabled('?perf=1')).toBe(true);
  expect(isViewerPerfOverlayEnabled('?file_id=x&perf=1')).toBe(true);
  expect(isViewerPerfOverlayEnabled('')).toBe(false);
  expect(isViewerPerfOverlayEnabled('?perf=0')).toBe(false);
  expect(isViewerPerfOverlayEnabled('?perf=true')).toBe(false);
});

test('init does not create DOM when perf overlay is disabled', () => {
  const win = {
    location: { search: '' },
    performance: { now: () => 0 },
    setInterval: vi.fn(),
    clearInterval: vi.fn(),
  };

  const overlay = initViewerPerfOverlay({ doc: document, win, metrics: createMetrics({}) });

  expect(overlay).toBeNull();
  expect(document.querySelector('.sv-viewer-perf-overlay')).toBeNull();
});

test('init creates a non-interactive overlay with formatted metrics', () => {
  vi.useFakeTimers();
  let now = 0;
  const win = {
    location: { search: '?perf=1' },
    performance: { now: () => now },
    setInterval,
    clearInterval,
  };
  const metrics = createMetrics({
    fetchMs: 12.34,
    decodeMs: 5,
    renderMs: 103.8,
    overlayMs: 0.8,
    payloadBytes: 2048,
    visibleTraces: 64,
    visibleSamples: 128,
    requestStarted: 4,
    requestCompleted: 3,
    requestAborted: 1,
    staleResponseDropped: 2,
    lastKey1: 101,
    lastLayer: 'raw',
    lastRenderMode: 'heatmap',
  });

  const overlay = initViewerPerfOverlay({ doc: document, win, metrics });

  expect(overlay.node).toBe(document.querySelector('.sv-viewer-perf-overlay'));
  expect(overlay.node.dataset.testid).toBe('viewer-perf-overlay');
  expect(getComputedStyle(overlay.node).pointerEvents).toBe('none');
  expect(overlay.node.textContent).toContain('fetch  12.3 ms');
  expect(overlay.node.textContent).toContain('decode 5.0 ms');
  expect(overlay.node.textContent).toContain('render 104 ms');
  expect(overlay.node.textContent).toContain('overlay 0.8 ms');
  expect(overlay.node.textContent).toContain('bytes  2.0 KB');
  expect(overlay.node.textContent).toContain('visible 64 x 128');
  expect(overlay.node.textContent).toContain('request started/completed/aborted/stale 4/3/1/2');
  expect(overlay.node.textContent).toContain('key1   101');
  expect(overlay.node.textContent).toContain('layer  raw');
  expect(overlay.node.textContent).toContain('mode   heatmap');

  now = 249;
  overlay.update();
  expect(metrics.snapshot).toHaveBeenCalledTimes(1);

  now = 250;
  overlay.update();
  expect(metrics.snapshot).toHaveBeenCalledTimes(2);

  overlay.dispose();
  expect(document.querySelector('.sv-viewer-perf-overlay')).toBeNull();
});

test('snapshot formatter keeps empty values explicit', () => {
  expect(formatPerfOverlaySnapshot({})).toContain('request started/completed/aborted/stale 0/0/0/0');
  expect(formatPerfOverlaySnapshot({})).toContain('visible -');
  expect(formatPerfOverlaySnapshot({})).toContain('mode   -');
});
