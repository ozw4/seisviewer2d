import { expect, test } from 'vitest';
import { createPerfMetrics } from '../../static/viewer/perf_metrics.js';

test('timer start and stop records duration', () => {
  let t = 10;
  const metrics = createPerfMetrics({ now: () => t });
  const timer = metrics.startTimer('fetch');

  t = 32.5;

  expect(metrics.stopTimer(timer)).toBe(22.5);
  expect(metrics.snapshot().fetchMs).toBe(22.5);
});

test('stale drop increments and snapshot returns a plain object', () => {
  const metrics = createPerfMetrics({ now: () => 0 });

  metrics.recordStaleResponseDropped({ mode: 'heatmap', layer: 'raw', key1: 123 });

  const snapshot = metrics.snapshot();
  expect(snapshot).toEqual(expect.objectContaining({
    staleResponseDropped: 1,
    lastRenderMode: 'heatmap',
    lastLayer: 'raw',
    lastKey1: 123,
  }));
  expect(Object.getPrototypeOf(snapshot)).toBe(Object.prototype);
});

test('payload and request counts can be reset', () => {
  const metrics = createPerfMetrics({ now: () => 0 });

  metrics.recordRequestStarted();
  metrics.recordRequestCompleted();
  metrics.recordPayload({
    payloadBytes: 2048,
    visibleTraces: 50,
    visibleSamples: 600,
  });
  metrics.recordDuration('decode', 4);

  expect(metrics.snapshot()).toMatchObject({
    requestStarted: 1,
    requestCompleted: 1,
    payloadBytes: 2048,
    visibleTraces: 50,
    visibleSamples: 600,
    decodeMs: 4,
  });

  metrics.reset();

  expect(metrics.snapshot()).toMatchObject({
    requestStarted: 0,
    requestCompleted: 0,
    payloadBytes: 0,
    visibleTraces: 0,
    visibleSamples: 0,
    decodeMs: 0,
  });
});
