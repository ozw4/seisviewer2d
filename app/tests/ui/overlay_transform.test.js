import { describe, expect, test } from 'vitest';
import { createOverlayTransform } from '../../static/viewer/overlay_transform.js';

const BASE_INPUT = {
  containerRect: { left: 10, top: 20, width: 400, height: 300 },
  plotArea: { left: 40, top: 30, width: 200, height: 100 },
  xRange: [0, 100],
  yRange: [0, 2],
  renderedStart: 0,
  renderedEnd: 100,
  renderedTimeStart: 0,
  renderedTimeEnd: 2,
  dt: 0.002,
};

function expectRoundTrip(transform, trace, timeS) {
  const px = transform.traceTimeToPixel(trace, timeS);
  expect(px).not.toBeNull();
  const data = transform.pixelToTraceTime(px.x, px.y);
  expect(data).not.toBeNull();
  expect(data.trace).toBeCloseTo(trace, 8);
  expect(data.timeS).toBeCloseTo(timeS, 8);
}

describe('createOverlayTransform', () => {
  test('round-trips normal trace/time and container pixels', () => {
    const transform = createOverlayTransform(BASE_INPUT);
    expect(transform.valid).toBe(true);

    const px = transform.traceTimeToPixel(50, 1);
    expect(px).toEqual({ x: 150, y: 100, relativeX: 140, relativeY: 80 });
    expect(transform.traceTimeToPixel(50, 0)).toMatchObject({ x: 150, y: 150 });
    expect(transform.traceTimeToPixel(50, 2)).toMatchObject({ x: 150, y: 50 });
    expectRoundTrip(transform, 50, 1);
  });

  test('round-trips reversed x and y ranges', () => {
    const transform = createOverlayTransform({
      ...BASE_INPUT,
      xRange: [100, 0],
      yRange: [2, 0],
    });

    expectRoundTrip(transform, 25, 0.5);
    expect(transform.traceTimeToPixel(100, 2)).toMatchObject({ x: 50, y: 150 });
    expect(transform.traceTimeToPixel(0, 0)).toMatchObject({ x: 250, y: 50 });
  });

  test('clips visible ranges to the rendered partial viewport', () => {
    const transform = createOverlayTransform({
      ...BASE_INPUT,
      xRange: [10, 70],
      yRange: [0.25, 1.5],
      renderedStart: 20,
      renderedEnd: 80,
      renderedTimeStart: 0.5,
      renderedTimeEnd: 1.75,
    });

    expect(transform.visibleTraceRange()).toEqual([20, 70]);
    expect(transform.visibleTimeRange()).toEqual([0.5, 1.5]);
    expect(transform.isTraceTimeVisible(20, 0.5)).toBe(true);
    expect(transform.isTraceTimeVisible(19.999, 0.5)).toBe(false);
  });

  test('uses current axis range so zoomed and panned inputs move pixels', () => {
    const full = createOverlayTransform(BASE_INPUT);
    const zoomed = createOverlayTransform({
      ...BASE_INPUT,
      xRange: [25, 75],
      yRange: [1.5, 0.5],
    });

    expect(full.traceTimeToPixel(50, 1)).toMatchObject({ x: 150, y: 100 });
    expect(zoomed.traceTimeToPixel(50, 1)).toMatchObject({ x: 150, y: 100 });
    expect(zoomed.traceTimeToPixel(25, 1.5)).toMatchObject({ x: 50, y: 150 });
    expect(zoomed.traceTimeToPixel(75, 0.5)).toMatchObject({ x: 250, y: 50 });
  });

  test('reports out-of-view trace/time points', () => {
    const transform = createOverlayTransform({
      ...BASE_INPUT,
      xRange: [10, 20],
      yRange: [1, 0],
      renderedStart: 0,
      renderedEnd: 30,
    });

    expect(transform.isTraceTimeVisible(15, 0.5)).toBe(true);
    expect(transform.isTraceTimeVisible(9.9, 0.5)).toBe(false);
    expect(transform.isTraceTimeVisible(15, 1.1)).toBe(false);
  });

  test('handles transposed trace/time axes', () => {
    const transform = createOverlayTransform({
      ...BASE_INPUT,
      transpose: true,
      xRange: [0, 2],
      yRange: [0, 100],
    });

    const px = transform.traceTimeToPixel(50, 1);
    expect(px).toEqual({ x: 150, y: 100, relativeX: 140, relativeY: 80 });
    expect(transform.traceTimeToPixel(0, 1)).toMatchObject({ x: 150, y: 150 });
    expect(transform.traceTimeToPixel(100, 1)).toMatchObject({ x: 150, y: 50 });
    expectRoundTrip(transform, 50, 1);
  });

  test('treats zero or invalid rects as an invalid safe transform', () => {
    const transform = createOverlayTransform({
      ...BASE_INPUT,
      containerRect: { left: 0, top: 0, width: 0, height: 100 },
    });

    expect(transform.valid).toBe(false);
    expect(transform.traceTimeToPixel(1, 1)).toBeNull();
    expect(transform.pixelToTraceTime(1, 1)).toBeNull();
    expect(transform.isTraceTimeVisible(1, 1)).toBe(false);
    expect(transform.visibleTraceRange()).toBeNull();
    expect(transform.visibleTimeRange()).toBeNull();
  });
});
