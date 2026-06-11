import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import {
  isPredictionOverlayFresh,
  renderPredictionOverlay,
  updatePredictionOverlayState,
} from '../../static/viewer/prediction_overlay.js';

function makeTransform() {
  return {
    valid: true,
    isTraceTimeVisible: vi.fn(() => true),
    traceTimeToPixel: vi.fn((trace, time) => ({
      relativeX: trace * 2,
      relativeY: time * 100,
    })),
  };
}

beforeEach(() => {
  vi.stubGlobal('requestAnimationFrame', (callback) => {
    callback();
    return 1;
  });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('prediction overlay', () => {
  test('draws fresh prediction picks on the prediction canvas', () => {
    const canvas = document.createElement('canvas');
    const ctx = {
      beginPath: vi.fn(),
      clearRect: vi.fn(),
      lineTo: vi.fn(),
      moveTo: vi.fn(),
      setLineDash: vi.fn(),
      stroke: vi.fn(),
    };
    vi.spyOn(canvas, 'getContext').mockReturnValue(ctx);
    const metadata = {
      fileId: 'file-a',
      key1: 10,
      layer: 'raw',
      pipelineKey: null,
      modelId: 'fb.pt',
    };

    updatePredictionOverlayState({
      predictedPicks: [{ trace: 12, time: 0.25 }],
      source: metadata,
      current: metadata,
      show: true,
    }, { redraw: false });

    const result = renderPredictionOverlay({
      canvas,
      transform: makeTransform(),
      width: 200,
      height: 100,
    });

    expect(result).toEqual({ predicted: 1, stale: false });
    expect(ctx.clearRect).toHaveBeenCalledWith(0, 0, 200, 100);
    expect(ctx.stroke).toHaveBeenCalledTimes(1);
  });

  test('does not draw stale prediction metadata', () => {
    const canvas = document.createElement('canvas');
    const ctx = {
      beginPath: vi.fn(),
      clearRect: vi.fn(),
      lineTo: vi.fn(),
      moveTo: vi.fn(),
      setLineDash: vi.fn(),
      stroke: vi.fn(),
    };
    vi.spyOn(canvas, 'getContext').mockReturnValue(ctx);

    updatePredictionOverlayState({
      predictedPicks: [{ trace: 12, time: 0.25 }],
      source: {
        fileId: 'file-a',
        key1: 10,
        layer: 'raw',
        pipelineKey: null,
        modelId: 'fb-a.pt',
      },
      current: {
        fileId: 'file-a',
        key1: 11,
        layer: 'raw',
        pipelineKey: null,
        modelId: 'fb-a.pt',
      },
      show: true,
    }, { redraw: false });

    const result = renderPredictionOverlay({
      canvas,
      transform: makeTransform(),
      width: 200,
      height: 100,
    });

    expect(result).toEqual({ predicted: 0, stale: true });
    expect(ctx.clearRect).toHaveBeenCalledWith(0, 0, 200, 100);
    expect(ctx.stroke).not.toHaveBeenCalled();
  });

  test('requires key1 layer pipeline and model metadata to match', () => {
    const source = {
      fileId: 'file-a',
      key1: 10,
      layer: 'tap-a',
      pipelineKey: 'pipe-a',
      modelId: 'fb-a.pt',
    };

    expect(isPredictionOverlayFresh(source, { ...source })).toBe(true);
    expect(isPredictionOverlayFresh(source, { ...source, layer: 'raw' })).toBe(false);
    expect(isPredictionOverlayFresh(source, { ...source, pipelineKey: 'pipe-b' })).toBe(false);
    expect(isPredictionOverlayFresh(source, { ...source, modelId: 'fb-b.pt' })).toBe(false);
  });
});
