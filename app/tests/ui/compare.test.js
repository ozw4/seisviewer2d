import { beforeAll, beforeEach, expect, test } from 'vitest';

beforeAll(async () => {
  await import('../../static/viewer/compare.js');
});

beforeEach(() => {
  document.body.innerHTML = `
    <input id="compareShowDiff" type="checkbox" checked>
    <input id="gain" value="2">
    <select id="colormap"><option value="Greys" selected>Greys</option></select>
    <input id="cmReverse" type="checkbox">
  `;
});

function scale(panel) {
  return window.__svCompare.compareHeatmapScale(panel, 2);
}

function expectF32Values(actual, expected) {
  expect(actual).toBeInstanceOf(Float32Array);
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    expect(actual[index]).toBeCloseTo(value, 6);
  });
}

test('compare heatmap scales mixed-domain source panels independently', () => {
  expect(scale({ kind: 'source', domain: 'amplitude' })).toMatchObject({
    zmin: -1.5,
    zmax: 1.5,
    signed: true,
  });
  expect(scale({ kind: 'source', domain: 'probability' })).toMatchObject({
    zmin: 0,
    zmax: 0.5,
    signed: false,
  });
});

test('compare heatmap allows signed probability diff values', () => {
  const render = {
    sources: {
      a: { domain: 'probability', label: 'prob-a' },
      b: { domain: 'probability', label: 'prob-b' },
    },
    a: { values: new Float32Array([0.1]) },
    b: { values: new Float32Array([0.8]) },
    diffAvailable: true,
    diffValues: new Float32Array([-0.7]),
  };

  const panels = window.__svCompare.buildComparePanels(render);
  const diffPanel = panels.find((panel) => panel.kind === 'diff');

  expect(diffPanel).toMatchObject({ role: 'A-B', domain: 'probability' });
  expect(scale(diffPanel)).toMatchObject({
    zmin: -0.5,
    zmax: 0.5,
    signed: true,
  });
});

test('compare heatmap uses signed amplitude scale for amplitude diff values', () => {
  expect(scale({ kind: 'diff', domain: 'amplitude' })).toMatchObject({
    zmin: -1.5,
    zmax: 1.5,
    signed: true,
  });
});

test('compare payload decode prefers quantized compute values over display backing', () => {
  const payload = {
    shape: [1, 2],
    valuesI8: new Int8Array([32, 64]),
    scale: 128,
    zBacking: new Float32Array([64, 128]),
  };

  const values = window.__svCompare.payloadToF32(payload, { domain: 'probability' });

  expectF32Values(values, [0.25, 0.5]);
});

test('compare payload decode rejects display backing for probability sources', () => {
  expect(window.__svCompare.payloadToF32({
    shape: [1, 2],
    zBacking: new Float32Array([64, 128]),
  }, { domain: 'probability' })).toBeNull();

  expect(window.__svCompare.payloadToF32({
    shape: [1, 2],
    zRows: [new Float32Array([64, 128])],
  }, { domain: 'probability' })).toBeNull();
});

test('compare payload decode keeps amplitude display backing fallback', () => {
  const values = window.__svCompare.payloadToF32({
    shape: [1, 2],
    zBacking: new Float32Array([1.5, -2.25]),
  }, { domain: 'amplitude' });

  expectF32Values(values, [1.5, -2.25]);
});

test('probability diff uses decoded probability values, not display-scaled backing', () => {
  const sourceA = window.__svCompare.payloadToF32({
    shape: [1, 1],
    valuesI8: new Int8Array([80]),
    scale: 100,
    zBacking: new Float32Array([204]),
  }, { domain: 'probability' });
  const sourceB = window.__svCompare.payloadToF32({
    shape: [1, 1],
    valuesI8: new Int8Array([20]),
    scale: 100,
    zBacking: new Float32Array([51]),
  }, { domain: 'probability' });

  const diff = window.__svCompare.subtractF32(sourceA, sourceB);

  expectF32Values(sourceA, [0.8]);
  expectF32Values(sourceB, [0.2]);
  expectF32Values(diff, [0.6]);
});
