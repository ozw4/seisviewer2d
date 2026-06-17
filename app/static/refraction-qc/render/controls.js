import {
  ARTIFACT_TYPE_CONTROL_OPTIONS,
  CELL_MAP_QUANTITY_CONTROL_OPTIONS,
  ENDPOINT_KIND_CONTROL_OPTIONS,
  FIRST_BREAK_SORT_CONTROL_OPTIONS,
  LAYER_CONTROL_OPTIONS,
  PROFILE_GROUP_CONTROL_OPTIONS,
  PROFILE_UNITS_CONTROL_OPTIONS,
  STATUS_FILTER_CONTROL_OPTIONS,
  X_AXIS_CONTROL_OPTIONS,
} from '../constants.js';

export function renderActiveFilterChipsPanel({ state, dom, context }) {
  if (!dom?.activeFilters) return;
  const {
    clearGatherPreviewFilter,
    clearNode,
    clearSelectedCellFilter,
    clearSelectedObject,
    firstBreakResidualThresholdMs,
    formatNumber,
    layerLabel,
    optionLabel,
    render,
    selectedCellLabel,
  } = context;
  clearNode(dom.activeFilters);

  const label = document.createElement('span');
  label.className = 'refraction-qc-active-filters-label';
  label.textContent = 'Filters:';
  dom.activeFilters.appendChild(label);

  const chips = [];
  const addChip = (key, text, reset) => chips.push({ key, text, reset });
  if (state.selectedLayerKind !== 'all') {
    addChip('layer', `Layer ${layerLabel(state.selectedLayerKind)}`, () => {
      state.selectedLayerKind = 'all';
    });
  }
  if (state.firstBreakXAxis !== 'offset') {
    addChip('x-axis', `X axis ${optionLabel(X_AXIS_CONTROL_OPTIONS, state.firstBreakXAxis)}`, () => {
      state.firstBreakXAxis = 'offset';
    });
  }
  if (!state.showRejectedFirstBreaks) {
    addChip('used-only', 'Used only', () => {
      state.showRejectedFirstBreaks = true;
    });
  }
  const residualThreshold = firstBreakResidualThresholdMs();
  if (residualThreshold !== null) {
    addChip('residual-threshold', `Residual >= ${formatNumber(residualThreshold, 1)} ms`, () => {
      state.firstBreakResidualThresholdMs = '';
    });
  }
  if (state.selectedProfileGroup !== 'time_terms') {
    addChip('profile-group', `Profile group ${optionLabel(PROFILE_GROUP_CONTROL_OPTIONS, state.selectedProfileGroup)}`, () => {
      state.selectedProfileGroup = 'time_terms';
    });
  }
  if (state.selectedProfileUnits !== 'auto') {
    addChip('profile-units', `Profile units ${optionLabel(PROFILE_UNITS_CONTROL_OPTIONS, state.selectedProfileUnits)}`, () => {
      state.selectedProfileUnits = 'auto';
    });
  }
  if (state.profileStatusFilter !== 'all') {
    addChip('status', `Status ${optionLabel(STATUS_FILTER_CONTROL_OPTIONS, state.profileStatusFilter)}`, () => {
      state.profileStatusFilter = 'all';
    });
  }
  if (state.selectedCellMapQuantity !== 'velocity') {
    addChip('map-quantity', `Map quantity ${optionLabel(CELL_MAP_QUANTITY_CONTROL_OPTIONS, state.selectedCellMapQuantity)}`, () => {
      state.selectedCellMapQuantity = 'velocity';
    });
  }
  if (state.selectedEndpoint) {
    addChip('endpoint', `${state.selectedEndpointKind} ${state.selectedEndpoint}`, () => {
      state.selectedEndpoint = '';
      state.selectedProfileEndpoint = null;
      if (state.selectedObject?.kind === 'endpoint') clearSelectedObject();
    });
  }
  if (state.selectedTraceIndex) {
    addChip('trace', `Trace ${state.selectedTraceIndex}`, () => {
      state.selectedTraceIndex = '';
    });
  }
  if (state.selectedCell) {
    addChip('cell', `Cell ${selectedCellLabel(state.selectedCell)}`, () => {
      clearSelectedCellFilter();
    });
  }
  if (state.gatherEndpointKey) {
    addChip('gather-endpoint', `Gather ${state.gatherEndpointKey}`, () => {
      clearGatherPreviewFilter();
    });
  }
  if (state.artifactTypeFilter !== 'all') {
    addChip('artifact-type', `Artifact ${optionLabel(ARTIFACT_TYPE_CONTROL_OPTIONS, state.artifactTypeFilter)}`, () => {
      state.artifactTypeFilter = 'all';
    });
  }
  if (state.artifactSearch) {
    addChip('artifact-search', `Artifact search ${state.artifactSearch}`, () => {
      state.artifactSearch = '';
    });
  }

  if (!chips.length) {
    const empty = document.createElement('span');
    empty.className = 'refraction-qc-active-filters-empty';
    empty.textContent = 'none';
    dom.activeFilters.appendChild(empty);
    return;
  }

  for (const chipData of chips) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'refraction-qc-filter-chip';
    chip.dataset.testid = 'refraction-qc-filter-chip';
    chip.dataset.filter = chipData.key;
    chip.setAttribute('aria-label', `Clear ${chipData.text}`);
    chip.textContent = `${chipData.text} x`;
    chip.addEventListener('click', () => {
      chipData.reset();
      render();
    });
    dom.activeFilters.appendChild(chip);
  }
}

function renderFindProblemsControls(container, state, context, options = {}) {
  const {
    createCheckboxControl,
    createNumberControl,
    createSelectControl,
  } = context;
  container.append(
    createSelectControl('Layer', 'refraction-qc-layer-kind', state.selectedLayerKind, LAYER_CONTROL_OPTIONS, (value) => {
      state.selectedLayerKind = value;
    }),
    createSelectControl('X axis', 'refraction-qc-x-axis', state.firstBreakXAxis, X_AXIS_CONTROL_OPTIONS, (value) => {
      state.firstBreakXAxis = value;
    }),
    createCheckboxControl('Show rejected / unused', 'refraction-qc-show-rejected', state.showRejectedFirstBreaks, (checked) => {
      state.showRejectedFirstBreaks = checked;
    }),
  );
  if (!options.includeResidualControls) return;
  container.append(
    createNumberControl('Residual threshold (ms)', 'refraction-qc-residual-threshold', state.firstBreakResidualThresholdMs, (value) => {
      state.firstBreakResidualThresholdMs = value;
    }, { min: 0, step: '0.1' }),
    createSelectControl('Sort by', 'refraction-qc-residual-sort', state.firstBreakSortBy, FIRST_BREAK_SORT_CONTROL_OPTIONS, (value) => {
      state.firstBreakSortBy = value;
    }),
  );
}

function renderStationControls(container, state, context) {
  const { createSelectControl, createTextControl } = context;
  if (state.selectedView === 'profiles_2d') {
    container.append(
      createSelectControl('Layer', 'refraction-qc-layer-kind', state.selectedLayerKind, LAYER_CONTROL_OPTIONS, (value) => {
        state.selectedLayerKind = value;
      }),
      createSelectControl('Station kind', 'refraction-qc-endpoint-kind', state.selectedEndpointKind, ENDPOINT_KIND_CONTROL_OPTIONS, (value) => {
        state.selectedEndpointKind = value;
      }),
      createTextControl('Station search', 'refraction-qc-endpoint', state.selectedEndpoint, (value) => {
        state.selectedEndpoint = value;
      }),
      createSelectControl('Profile group', 'refraction-qc-profile-group', state.selectedProfileGroup, PROFILE_GROUP_CONTROL_OPTIONS, (value) => {
        state.selectedProfileGroup = value;
      }),
      createSelectControl('Profile units', 'refraction-qc-profile-units', state.selectedProfileUnits, PROFILE_UNITS_CONTROL_OPTIONS, (value) => {
        state.selectedProfileUnits = value;
      }),
      createSelectControl('Status', 'refraction-qc-status-filter', state.profileStatusFilter, STATUS_FILTER_CONTROL_OPTIONS, (value) => {
        state.profileStatusFilter = value;
      }),
    );
    return;
  }
  if (state.selectedView === 'static_components') {
    container.append(
      createSelectControl('Station kind', 'refraction-qc-endpoint-kind', state.selectedEndpointKind, ENDPOINT_KIND_CONTROL_OPTIONS, (value) => {
        state.selectedEndpointKind = value;
      }),
      createTextControl('Station search', 'refraction-qc-endpoint', state.selectedEndpoint, (value) => {
        state.selectedEndpoint = value;
      }),
      createTextControl('Trace', 'refraction-qc-trace', state.selectedTraceIndex, (value) => {
        state.selectedTraceIndex = value;
      }),
    );
  }
}

function renderCellControls(container, state, context) {
  const {
    clearSelectedCellFilter,
    createSelectControl,
    createTextControl,
    parseCell,
  } = context;
  container.append(
    createSelectControl('Layer', 'refraction-qc-layer-kind', state.selectedLayerKind, LAYER_CONTROL_OPTIONS, (value) => {
      state.selectedLayerKind = value;
    }),
    createSelectControl('Map quantity', 'refraction-qc-map-quantity', state.selectedCellMapQuantity, CELL_MAP_QUANTITY_CONTROL_OPTIONS, (value) => {
      state.selectedCellMapQuantity = value;
    }),
    createSelectControl('Status', 'refraction-qc-status-filter', state.profileStatusFilter, STATUS_FILTER_CONTROL_OPTIONS, (value) => {
      state.profileStatusFilter = value;
    }),
    createTextControl('Cell search', 'refraction-qc-cell', state.selectedCell?.cell_ix !== undefined
      ? `${state.selectedCell.cell_ix},${state.selectedCell.cell_iy}`
      : (state.selectedCell?.text || ''), (value) => {
      clearSelectedCellFilter();
      state.selectedCell = parseCell(value);
    }),
  );
}

function renderGatherControls(container, context) {
  container.appendChild(context.createGatherPreviewControls());
}

function renderArtifactsControls(container, state, context) {
  const { createSelectControl, createTextControl } = context;
  container.append(
    createSelectControl('Artifact type', 'refraction-qc-artifact-type', state.artifactTypeFilter, ARTIFACT_TYPE_CONTROL_OPTIONS, (value) => {
      state.artifactTypeFilter = value;
    }),
    createTextControl('Search table columns', 'refraction-qc-artifact-search', state.artifactSearch, (value) => {
      state.artifactSearch = value;
    }),
  );
}

export function renderControlsPanel({ state, dom, context }) {
  if (!dom?.viewControls) return;
  context.clearNode(dom.viewControls);
  dom.viewControls.hidden = false;
  if (state.activeTask === 'find_problems' && ['first_break_residuals', 'reduced_time'].includes(state.selectedView)) {
    renderFindProblemsControls(dom.viewControls, state, context, {
      includeResidualControls: state.selectedView === 'first_break_residuals',
    });
  } else if (state.activeTask === 'inspect_station') {
    renderStationControls(dom.viewControls, state, context);
  } else if (state.activeTask === 'inspect_cell' && state.selectedView === 'cell_maps_3d') {
    renderCellControls(dom.viewControls, state, context);
  } else if (state.activeTask === 'preview_gather' && state.selectedView === 'gather_preview') {
    renderGatherControls(dom.viewControls, context);
  } else if (state.activeTask === 'artifacts' && state.selectedView === 'artifacts') {
    renderArtifactsControls(dom.viewControls, state, context);
  }
  dom.viewControls.hidden = !dom.viewControls.childElementCount;
}
