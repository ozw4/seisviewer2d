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
