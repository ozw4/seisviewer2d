import { test, expect } from 'vitest';
import { createRenderRequestController } from '../../static/viewer/render_request_controller.js';

test('begin aborts the previous request in the same slot', () => {
  const controller = createRenderRequestController();
  const first = controller.begin('section-window');
  const second = controller.begin('section-window');

  expect(first.signal.aborted).toBe(true);
  expect(second.signal.aborted).toBe(false);
  expect(controller.isCurrent('section-window', first.requestId)).toBe(false);
  expect(controller.isCurrent('section-window', second.requestId)).toBe(true);
  expect(controller.snapshotMetrics()['section-window']).toMatchObject({
    started: 2,
    aborted: 1,
  });
});

test('slots are independent and abortAll cancels each active slot', () => {
  const controller = createRenderRequestController();
  const section = controller.begin('section-window');
  const compare = controller.begin('compare-window');

  expect(controller.isCurrent('section-window', section.requestId)).toBe(true);
  expect(controller.isCurrent('compare-window', compare.requestId)).toBe(true);

  controller.abortAll();

  expect(section.signal.aborted).toBe(true);
  expect(compare.signal.aborted).toBe(true);
  expect(controller.isCurrent('section-window', section.requestId)).toBe(false);
  expect(controller.isCurrent('compare-window', compare.requestId)).toBe(false);
  expect(controller.snapshotMetrics()['section-window'].aborted).toBe(1);
  expect(controller.snapshotMetrics()['compare-window'].aborted).toBe(1);
});

test('metrics hooks count current completions, failures, and stale drops', () => {
  const controller = createRenderRequestController();
  const first = controller.begin('pipeline-section');
  const second = controller.begin('pipeline-section');

  expect(controller.markStaleDropped('pipeline-section', first.requestId)).toBe(true);
  expect(controller.markCompleted('pipeline-section', first.requestId)).toBe(false);
  expect(controller.markFailed('pipeline-section', second.requestId)).toBe(true);

  expect(controller.snapshotMetrics()['pipeline-section']).toMatchObject({
    started: 2,
    aborted: 1,
    staleDropped: 1,
    completed: 0,
    failed: 1,
  });
});

test('completed requests are invalidated without counting as aborted', () => {
  const controller = createRenderRequestController();
  const request = controller.begin('section-window');

  expect(controller.markCompleted('section-window', request.requestId)).toBe(true);
  expect(controller.isCurrent('section-window', request.requestId)).toBe(true);

  controller.abort('section-window');

  expect(request.signal.aborted).toBe(false);
  expect(controller.isCurrent('section-window', request.requestId)).toBe(false);
  expect(controller.snapshotMetrics()['section-window']).toMatchObject({
    started: 1,
    aborted: 0,
    completed: 1,
  });
});

test('request controller notifies perf metrics for lifecycle events', () => {
  const calls = [];
  const perfMetrics = {
    recordRequestStarted: (meta) => calls.push(['started', meta]),
    recordRequestAborted: (meta) => calls.push(['aborted', meta]),
    recordRequestCompleted: (meta) => calls.push(['completed', meta]),
    recordStaleResponseDropped: (meta) => calls.push(['stale', meta]),
  };
  const controller = createRenderRequestController({ perfMetrics });
  const first = controller.begin('section-window');
  const second = controller.begin('section-window');

  controller.markStaleDropped('section-window', first.requestId);
  controller.markCompleted('section-window', second.requestId);

  expect(calls).toEqual([
    ['started', { slotName: 'section-window', requestId: first.requestId }],
    ['aborted', { slotName: 'section-window', requestId: first.requestId }],
    ['started', { slotName: 'section-window', requestId: second.requestId }],
    ['stale', { slotName: 'section-window', requestId: first.requestId }],
    ['completed', { slotName: 'section-window', requestId: second.requestId }],
  ]);
});
