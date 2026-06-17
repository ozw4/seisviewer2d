import { afterEach, expect, test, vi } from 'vitest';
import { getCanvas2dContext } from '../../static/refraction-qc/render/canvas_helpers.js';
import { renderActiveView } from '../../static/refraction-qc/render/index.js';
import {
  clearPlotlyForTests,
  newPlot,
  plotlyUnavailableMessage,
  setPlotlyForTests,
} from '../../static/refraction-qc/render/plotly_helpers.js';

afterEach(() => {
  clearPlotlyForTests();
  vi.restoreAllMocks();
});

test('renderActiveView dispatches by view id', () => {
  const root = document.createElement('div');
  const bundle = { job_id: 'job-a', available_views: ['summary'] };
  const viewState = { artifactSearch: '', artifactTypeFilter: 'all' };
  const filterArtifactRows = vi.fn(() => ({
    rows: [{ type: 'csv', name: 'summary', path: 'summary.csv' }],
    filteredRows: [{ type: 'csv', name: 'summary', path: 'summary.csv' }],
  }));
  renderActiveView(
    { loading: false, qcBundle: bundle },
    { viewContents: new Map([['artifacts', root]]) },
    {
      appendText: (node, text) => node.appendChild(document.createTextNode(text)),
      clearNode: (node) => { node.textContent = ''; },
      context: {
        createKv: (items) => {
          const dl = document.createElement('dl');
          for (const [label, value] of items) {
            const dt = document.createElement('dt');
            dt.textContent = label;
            const dd = document.createElement('dd');
            dd.textContent = String(value ?? '');
            dl.append(dt, dd);
          }
          return dl;
        },
        createTable: ({ records }) => {
          const table = document.createElement('table');
          table.dataset.count = String(records.length);
          return table;
        },
        filterArtifactRows,
      },
      renderTabularView: vi.fn(),
      viewState,
      viewDef: { id: 'artifacts' },
    },
  );

  expect(root.className).toBe('');
  expect(filterArtifactRows).toHaveBeenCalledWith(bundle, viewState);
  expect(root.textContent).toContain('summary');
});

test('renderActiveView routes pick map views without requiring a QC bundle', () => {
  const root = document.createElement('div');
  const viewConfig = { label: 'RecNo-time' };
  const renderTabularView = vi.fn();
  renderActiveView(
    { loading: false, qcBundle: null },
    { viewContents: new Map([['pick_map', root]]) },
    {
      appendText: (node, text) => node.appendChild(document.createTextNode(text)),
      clearNode: (node) => { node.textContent = ''; },
      context: {
        controllerActions: {},
        createKv: () => document.createElement('dl'),
        render: vi.fn(),
      },
      pickMapViews: { pick_map: viewConfig },
      renderTabularView,
      viewState: {
        pickMap: null,
        pickMapCachedFile: null,
        pickMapCacheStatus: '',
        pickMapDisplayMode: 'before',
        pickMapError: null,
        pickMapGatherEnd: '',
        pickMapGatherStart: '',
        pickMapLoading: false,
        qcBundle: null,
        selectedJobId: '',
      },
      viewDef: { id: 'pick_map' },
    },
  );

  expect(root.className).toBe('');
  expect(renderTabularView).not.toHaveBeenCalled();
  expect(root.textContent).toContain('Load a completed job');
});

test('plotly helper supports explicit test injection and unavailable errors', () => {
  expect(() => newPlot(document.createElement('div'), [], {}, {})).toThrow(plotlyUnavailableMessage());

  const plotly = { newPlot: vi.fn(() => Promise.resolve()) };
  const plot = document.createElement('div');
  setPlotlyForTests(plotly);

  newPlot(plot, [{ name: 'trace' }], { title: 'Plot' }, { responsive: true });

  expect(plotly.newPlot).toHaveBeenCalledWith(
    plot,
    [{ name: 'trace' }],
    { title: 'Plot' },
    { responsive: true },
  );
});

test('canvas helper preserves the existing 2d context call shape', () => {
  const canvas = document.createElement('canvas');
  const context = {};
  const getContext = vi.spyOn(canvas, 'getContext').mockReturnValue(context);

  expect(getCanvas2dContext(canvas)).toBe(context);
  expect(getContext).toHaveBeenCalledWith('2d');
});
