import { renderArtifactsView } from './artifacts.js';
import { renderCellMapsView } from './cell_maps.js';
import { renderFirstBreakView } from './first_break.js';
import { renderGatherPreviewView } from './gather_preview.js';
import { renderPickMapView } from './pick_map.js';
import { renderProfilesView } from './profiles.js';
import { renderReducedTimeView } from './reduced_time.js';
import { renderStaticComponentsView } from './static_components.js';
import { renderStationStructureView } from './station_structure.js';
import { renderSummaryView } from './summary.js';

const VIEW_RENDERERS = {
  summary: renderSummaryView,
  first_break_residuals: renderFirstBreakView,
  reduced_time: renderReducedTimeView,
  profiles_2d: renderProfilesView,
  cell_maps_3d: renderCellMapsView,
  static_components: renderStaticComponentsView,
  station_structure: renderStationStructureView,
  gather_preview: renderGatherPreviewView,
  artifacts: renderArtifactsView,
};

export function renderActiveView(state, dom, options = {}) {
  const {
    appendText,
    clearNode,
    context = {},
    pickMapViews = {},
    renderDepsForView,
    renderTabularView,
    viewState,
    viewDef,
  } = options;
  if (!dom || !viewDef || !clearNode || !appendText) return;

  const root = dom.viewContents.get(viewDef.id);
  if (!root) return;
  clearNode(root);
  root.className = 'refraction-qc-placeholder';

  if (state.loading) {
    appendText(root, 'Loading QC bundle...');
    return;
  }

  const pickMapConfig = pickMapViews[viewDef.id];
  const depsForView = typeof renderDepsForView === 'function' ? renderDepsForView(viewDef.id) : {};
  const renderDeps = { ...depsForView, ...context };
  if (pickMapConfig) {
    root.className = '';
    renderPickMapView({
      root,
      viewConfig: pickMapConfig,
      viewState,
      ...renderDeps,
    });
    return;
  }

  if (!state.qcBundle) {
    appendText(root, state.error ? 'QC bundle is not loaded.' : 'No QC bundle loaded.');
    return;
  }

  root.className = '';
  const renderer = VIEW_RENDERERS[viewDef.id];
  if (!renderer) {
    renderTabularView(root, state.qcBundle, viewDef);
    return;
  }
  renderer({
    root,
    bundle: state.qcBundle,
    viewDef,
    viewState,
    ...renderDeps,
  });
}
