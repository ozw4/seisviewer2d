import { afterEach, describe, expect, test, vi } from 'vitest';
import { createOverlayTransform } from '../../static/viewer/overlay_transform.js';
import {
  renderManualPickOverlay,
  updateManualPickOverlayState,
} from '../../static/viewer/manual_pick_overlay.js';

const BASE_TRANSFORM_INPUT = {
  containerRect: { left: 10, top: 20, width: 400, height: 300 },
  plotArea: { left: 40, top: 30, width: 200, height: 100 },
  xRange: [0, 100],
  yRange: [0, 2],
  renderedStart: 0,
  renderedEnd: 100,
  renderedTimeStart: 0,
  renderedTimeEnd: 2,
};

function makeContext() {
  return {
    beginPath: vi.fn(),
    clearRect: vi.fn(),
    clip: vi.fn(),
    closePath: vi.fn(),
    fill: vi.fn(),
    lineTo: vi.fn(),
    moveTo: vi.fn(),
    rect: vi.fn(),
    restore: vi.fn(),
    save: vi.fn(),
    setLineDash: vi.fn(),
    stroke: vi.fn(),
  };
}

function makeCanvas(ctx) {
  const canvas = document.createElement('canvas');
  vi.spyOn(canvas, 'getContext').mockReturnValue(ctx);
  return canvas;
}

afterEach(() => {
  updateManualPickOverlayState({}, { redraw: false });
  vi.restoreAllMocks();
});

describe('manual pick overlay', () => {
  test('clips pending line anchors to the Plotly plot area using container-relative coordinates', () => {
    const ctx = makeContext();
    const canvas = makeCanvas(ctx);
    const transform = createOverlayTransform(BASE_TRANSFORM_INPUT);

    updateManualPickOverlayState({
      pending: { kind: 'line', trace: 0, time: 0 },
    }, { redraw: false });

    const result = renderManualPickOverlay({
      canvas,
      transform,
      width: 400,
      height: 300,
    });

    expect(result).toEqual({ manual: 0, pending: true });
    expect(ctx.clearRect).toHaveBeenCalledWith(0, 0, 400, 300);
    expect(ctx.rect).toHaveBeenCalledWith(40, 30, 200, 100);
    expect(ctx.clip).toHaveBeenCalledTimes(1);
    expect(ctx.moveTo).toHaveBeenCalledWith(40, 122);
    expect(ctx.lineTo).toHaveBeenNthCalledWith(1, 48, 130);
    expect(ctx.lineTo).toHaveBeenNthCalledWith(2, 40, 138);
    expect(ctx.lineTo).toHaveBeenNthCalledWith(3, 32, 130);
    expect(ctx.stroke).toHaveBeenCalledTimes(1);
  });

  test('does not draw pending line anchors outside the visible viewport', () => {
    const ctx = makeContext();
    const canvas = makeCanvas(ctx);
    const transform = {
      valid: true,
      rect: { left: 0, top: 0, width: 400, height: 300 },
      plotArea: { left: 40, top: 30, width: 200, height: 100 },
      isTraceTimeVisible: vi.fn(() => false),
      traceTimeToPixel: vi.fn(() => ({ relativeX: 42, relativeY: 42 })),
    };

    updateManualPickOverlayState({
      pending: { kind: 'line', trace: 999, time: 9 },
    }, { redraw: false });

    const result = renderManualPickOverlay({
      canvas,
      transform,
      width: 400,
      height: 300,
    });

    expect(result).toEqual({ manual: 0, pending: true });
    expect(transform.isTraceTimeVisible).toHaveBeenCalledWith(999, 9);
    expect(transform.traceTimeToPixel).not.toHaveBeenCalled();
    expect(ctx.stroke).not.toHaveBeenCalled();
  });

  test('does not draw delete range anchors outside the visible viewport', () => {
    const ctx = makeContext();
    const canvas = makeCanvas(ctx);
    const transform = {
      valid: true,
      rect: { left: 0, top: 0, width: 400, height: 300 },
      plotArea: { left: 40, top: 30, width: 200, height: 100 },
      isTraceTimeVisible: vi.fn(() => false),
      traceTimeToPixel: vi.fn(() => ({ relativeX: 42, relativeY: 42 })),
      visibleTimeRange: vi.fn(() => [0.25, 1.25]),
    };

    updateManualPickOverlayState({
      pending: { kind: 'delete-range', trace: 999 },
    }, { redraw: false });

    const result = renderManualPickOverlay({
      canvas,
      transform,
      width: 400,
      height: 300,
    });

    expect(result).toEqual({ manual: 0, pending: true });
    expect(transform.isTraceTimeVisible).toHaveBeenCalledWith(999, 0.25);
    expect(transform.isTraceTimeVisible).toHaveBeenCalledWith(999, 1.25);
    expect(transform.traceTimeToPixel).not.toHaveBeenCalled();
    expect(ctx.stroke).not.toHaveBeenCalled();
  });
});
