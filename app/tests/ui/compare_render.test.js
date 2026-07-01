import { beforeAll, expect, test } from 'vitest';

beforeAll(async () => {
  await import('../../static/viewer/compare/models.js');
  await import('../../static/viewer/compare/sources.js');
  await import('../../static/viewer/compare/data.js');
  await import('../../static/viewer/compare/render.js');
});

function renderHelpers() {
  return window.__svCompareRender;
}

function payload(overrides = {}) {
  return {
    key1: 101,
    shape: [2, 2],
    dt: 0.004,
    lmoKey: 'lmo-1',
    x0: 10,
    x1: 12,
    y0: 0,
    y1: 2,
    stepX: 1,
    stepY: 1,
    values: new Float32Array([1, 2, 3, 4]),
    ...overrides,
  };
}

function compareRender(overrides = {}) {
  return renderHelpers().buildCompareRender({
    aPayload: payload(),
    bPayload: payload({ values: new Float32Array([0.5, 1, 1.5, 2]) }),
    sources: {
      a: { fileId: 'file-a', layerId: 'raw', domain: 'amplitude', label: 'Line A / raw' },
      b: { fileId: 'file-b', layerId: 'raw', domain: 'amplitude', label: 'Line B / raw' },
    },
    decision: { mode: 'heatmap', panelCount: 3, stepX: 1, stepY: 1 },
    validation: { ok: true, message: '' },
    windowInfo: { x0: 10, x1: 12, y0: 0, y1: 2 },
    scaling: 'amax',
    ...overrides,
  });
}

function layoutFor(panels, render = compareRender()) {
  return renderHelpers().buildCompareLayout({
    render,
    panels,
    clickmode: 'event+select',
    dragmode: 'pan',
    uiRevision: 'ui-1',
  });
}

function withGlobalDefaultDt(value, fn) {
  const hadWindowDefault = Object.prototype.hasOwnProperty.call(window, 'defaultDt');
  const previousWindowDefault = window.defaultDt;
  const hadGlobalDefault = Object.prototype.hasOwnProperty.call(globalThis, 'defaultDt');
  const previousGlobalDefault = globalThis.defaultDt;
  delete window.defaultDt;
  globalThis.defaultDt = value;
  try {
    fn();
  } finally {
    if (hadWindowDefault) window.defaultDt = previousWindowDefault;
    else delete window.defaultDt;
    if (hadGlobalDefault) globalThis.defaultDt = previousGlobalDefault;
    else delete globalThis.defaultDt;
  }
}

test('diff enabled builds three compare panels and layout axes', () => {
  const render = compareRender();
  const panels = renderHelpers().buildComparePanels({ render, showDiff: true });
  const layout = layoutFor(panels, render);

  expect(panels.map((panel) => panel.kind)).toEqual(['source', 'source', 'diff']);
  expect(layout.annotations).toHaveLength(3);
  expect(layout.xaxis.domain).toEqual([0, expect.any(Number)]);
  expect(layout.xaxis2.domain).toEqual([expect.any(Number), expect.any(Number)]);
  expect(layout.xaxis3.domain).toEqual([expect.any(Number), 1]);
});

test('diff disabled builds two compare panels and layout axes', () => {
  const render = compareRender();
  const panels = renderHelpers().buildComparePanels({ render, showDiff: false });
  const layout = layoutFor(panels, render);

  expect(panels.map((panel) => panel.kind)).toEqual(['source', 'source']);
  expect(layout.annotations).toHaveLength(2);
  expect(layout.xaxis).toBeDefined();
  expect(layout.xaxis2).toBeDefined();
  expect(layout.xaxis3).toBeUndefined();
});

test('heatmap mode builds A, B, and Diff traces', () => {
  const render = compareRender();
  const panels = renderHelpers().buildComparePanels({ render, showDiff: true });
  const traces = panels.map((panel, axisIndex) => renderHelpers().buildCompareHeatmapTrace({
    panel,
    axisIndex,
    render,
    gain: 2,
    colormapName: 'Greys',
    reverse: false,
  }));

  expect(traces).toHaveLength(3);
  expect(traces.map((trace) => trace.type)).toEqual(['heatmap', 'heatmap', 'heatmap']);
  expect(traces.map((trace) => trace.xaxis)).toEqual(['x', 'x2', 'x3']);
  expect(traces.map((trace) => trace.yaxis)).toEqual(['y', 'y2', 'y3']);
});

test('wiggle mode preserves compare axis suffix assignment', () => {
  const render = { ...compareRender(), mode: 'wiggle' };
  const panels = renderHelpers().buildComparePanels({ render, showDiff: true });
  const traces = panels.flatMap((panel, axisIndex) => renderHelpers().buildCompareWiggleTraces({
    panel,
    axisIndex,
    render,
    gain: 1,
  }));

  expect(traces).toHaveLength(9);
  expect(traces.slice(0, 3).map((trace) => trace.xaxis)).toEqual(['x', 'x', 'x']);
  expect(traces.slice(3, 6).map((trace) => trace.xaxis)).toEqual(['x2', 'x2', 'x2']);
  expect(traces.slice(6, 9).map((trace) => trace.xaxis)).toEqual(['x3', 'x3', 'x3']);
  expect(renderHelpers().axisLayoutName('y', 2)).toBe('yaxis3');
});

test('payloads without dt use the default sample interval for layout and traces', () => {
  withGlobalDefaultDt(0.006, () => {
    const render = compareRender({
      aPayload: payload({ dt: undefined }),
      bPayload: payload({
        dt: undefined,
        values: new Float32Array([0.5, 1, 1.5, 2]),
      }),
    });
    const panels = renderHelpers().buildComparePanels({ render, showDiff: true });
    const layout = layoutFor(panels, render);
    const heatmapTrace = renderHelpers().buildCompareHeatmapTrace({
      panel: panels[0],
      axisIndex: 0,
      render,
    });
    const wiggleTraces = renderHelpers().buildCompareWiggleTraces({
      panel: panels[0],
      axisIndex: 0,
      render,
    });

    expect(layout.yaxis.range[0]).toBeCloseTo(0.012, 9);
    expect(layout.yaxis.range[1]).toBeCloseTo(0, 9);
    expect(heatmapTrace.y[0]).toBeCloseTo(0, 9);
    expect(heatmapTrace.y[1]).toBeCloseTo(0.006, 9);
    expect(wiggleTraces[2].y[0]).toBeCloseTo(0, 9);
    expect(wiggleTraces[2].y[1]).toBeCloseTo(0.006, 9);
  });
});

test('unavailable figure has no data and carries the message annotation', () => {
  const figure = renderHelpers().buildCompareUnavailableFigure('A-B unavailable: grids differ.');

  expect(figure.data).toEqual([]);
  expect(figure.layout.annotations[0].text).toBe('A-B unavailable: grids differ.');
  expect(figure.config).toMatchObject({
    responsive: true,
    doubleClick: false,
    doubleClickDelay: 300,
  });
});

test('panel titles include A, B, and Diff source labels', () => {
  const render = compareRender();
  const panels = renderHelpers().buildComparePanels({ render, showDiff: true });

  expect(panels.map((panel) => renderHelpers().panelTitle(panel))).toEqual([
    'A: Line A / raw',
    'B: Line B / raw',
    'A-B: Line A / raw - Line B / raw',
  ]);
});

test('compareHeatmapScale preserves symmetric amplitude and probability scaling', () => {
  expect(renderHelpers().compareHeatmapScale({ kind: 'source', domain: 'amplitude' }, 2)).toMatchObject({
    zmin: -1.5,
    zmax: 1.5,
    signed: true,
  });
  expect(renderHelpers().compareHeatmapScale({ kind: 'source', domain: 'probability' }, 2)).toMatchObject({
    zmin: 0,
    zmax: 0.5,
    signed: false,
  });
  expect(renderHelpers().compareHeatmapScale({ kind: 'diff', domain: 'probability' }, 2)).toMatchObject({
    zmin: -0.5,
    zmax: 0.5,
    signed: true,
  });
});
