function appendInspectorAction(parent, label, disabledReason, callback) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  if (disabledReason) {
    button.disabled = true;
    button.title = disabledReason;
  } else {
    button.addEventListener('click', callback);
  }
  parent.appendChild(button);
}

function pickInspectorItems(pick, context) {
  const { formatNumber, layerLabel } = context;
  const residual = Number.isFinite(pick.residualMs)
    ? `${pick.residualMs >= 0 ? '+' : ''}${formatNumber(pick.residualMs, 1)} ms`
    : '-';
  return [
    ['Layer', layerLabel(pick.layerKind)],
    ['Trace', pick.traceIndex],
    ['Offset', Number.isFinite(pick.offsetM) ? `${formatNumber(pick.offsetM, 1)} m` : '-'],
    ['Observed', Number.isFinite(pick.observedMs) ? `${formatNumber(pick.observedMs, 1)} ms` : '-'],
    ['Modeled', Number.isFinite(pick.modeledMs) ? `${formatNumber(pick.modeledMs, 1)} ms` : '-'],
    ['Residual', residual],
    ['Source', pick.source],
    ['Receiver', pick.receiver],
  ];
}

function endpointInspectorItems(endpoint, context) {
  const {
    cellEndpointDisplayName,
    firstDefined,
    formatNumber,
    textOrDash,
    toFiniteNumber,
  } = context;
  const kind = endpoint.endpointKind === 'receiver' ? 'receiver' : 'source';
  const raw = endpoint.raw && typeof endpoint.raw === 'object' ? endpoint.raw : {};
  const staticMs = toFiniteNumber(firstDefined(raw, [
    'total_static_ms',
    'total_applied_shift_ms',
    'static_ms',
    'weathering_correction_ms',
    'elevation_correction_ms',
    'field_shift_ms',
  ]));
  const pickCount = firstDefined(raw, ['pick_count', 'used_pick_count', 'fold', 'n_picks', 'observation_count']);
  const rms = toFiniteNumber(firstDefined(raw, ['residual_rms_ms', 'rms_ms', 'used_rms_ms']));
  const status = firstDefined(raw, ['static_status', 'solution_status', 'status']) || endpoint.staticStatus;
  return [
    ['Station', cellEndpointDisplayName(kind, endpoint.endpointKey)],
    ['Node', endpoint.nodeId],
    ['Picks', textOrDash(pickCount)],
    ['RMS', Number.isFinite(rms) ? `${formatNumber(rms, 1)} ms` : '-'],
    ['Static', Number.isFinite(staticMs) ? `${staticMs >= 0 ? '+' : ''}${formatNumber(staticMs, 1)} ms` : '-'],
    ['Status', textOrDash(status)],
  ];
}

function endpointInspectorPayloadFromRecord(record, endpointKind, endpointKey, context) {
  const {
    firstDefined,
    gatherEndpointStationId,
    staticEndpointKey,
    staticEndpointKind,
    textOrDash,
    toFiniteNumber,
  } = context;
  const kind = staticEndpointKind(record) || (endpointKind === 'receiver' ? 'receiver' : 'source');
  const key = staticEndpointKey(record) || String(endpointKey || '').trim();
  const raw = record && typeof record === 'object' ? record : {};
  return {
    endpointKind: kind,
    endpointKey: key,
    stationId: gatherEndpointStationId(raw, kind),
    nodeId: textOrDash(firstDefined(raw, ['node_id'])),
    inlineM: toFiniteNumber(firstDefined(raw, ['inline_m', 'x_m'])),
    staticStatus: textOrDash(firstDefined(raw, ['static_status'])),
    solutionStatus: textOrDash(firstDefined(raw, ['solution_status'])),
    raw,
  };
}

function selectedEndpointInspectorPayload(state, context) {
  const endpointKey = String(state.selectedEndpoint || '').trim();
  if (!endpointKey) return null;

  const {
    selectedStaticEndpointRecord,
    staticEndpointKey,
    staticEndpointRecords,
    STATIC_ENDPOINT_VIEW_KEY,
    viewByKey,
  } = context;
  const endpointRecords = [
    ...staticEndpointRecords(viewByKey(state.qcBundle, STATIC_ENDPOINT_VIEW_KEY)),
    ...staticEndpointRecords(viewByKey(state.qcBundle, 'static_components')),
    ...staticEndpointRecords(viewByKey(state.qcBundle, 'line_profiles')),
  ];
  const endpointRecord = selectedStaticEndpointRecord(endpointRecords);
  return endpointInspectorPayloadFromRecord(
    endpointRecord,
    state.selectedEndpointKind,
    endpointRecord ? staticEndpointKey(endpointRecord) : endpointKey,
    context,
  );
}

function selectedObjectForInspector(state, context) {
  const selected = state.selectedObject || {};
  if (selected.kind && selected.kind !== 'endpoint') return selected;

  const endpointPayload = selectedEndpointInspectorPayload(state, context);
  if (!endpointPayload) return selected;
  if (selected.kind !== 'endpoint') {
    return {
      kind: 'endpoint',
      key: `${endpointPayload.endpointKind}|${endpointPayload.endpointKey}`,
      payload: endpointPayload,
    };
  }

  const endpointFilter = context.normalizedText(state.selectedEndpoint);
  const selectedEndpointKey = context.normalizedText(selected.payload?.endpointKey);
  const selectedEndpointKind = selected.payload?.endpointKind === 'receiver' ? 'receiver' : 'source';
  const kindMatches = state.selectedEndpointKind === 'both' || state.selectedEndpointKind === selectedEndpointKind;
  if (kindMatches && selectedEndpointKey && selectedEndpointKey.includes(endpointFilter)) return selected;

  return {
    kind: 'endpoint',
    key: `${endpointPayload.endpointKind}|${endpointPayload.endpointKey}`,
    payload: endpointPayload,
  };
}

function cellInspectorItems(cell, state, context) {
  const {
    drilldownMetric,
    firstDefined,
    formatNumber,
    layerLabel,
    selectedCellLabel,
    textOrDash,
    toFiniteNumber,
  } = context;
  const payload = state.qcDrilldown || {};
  const drilldownCell = payload.cell || {};
  const velocity = payload.velocity || drilldownCell.velocity || {};
  const fold = payload.fold || drilldownCell.fold || {};
  const residual = payload.residual_summary || drilldownCell.residual_summary || {};
  const row = drilldownCell.row || {};
  const layerKind = cell.layer_kind || drilldownCell.layer_kind || payload.target?.layer_kind || state.qcDrilldownTarget?.layer_kind;
  const velocityValue = firstDefined(velocity, ['velocity_m_s', 'v2_m_s']) ?? cell.velocity_m_s;
  const foldValue = firstDefined(fold, ['n_observations', 'fold', 'observation_count']) ?? cell.fold;
  const residualRms = firstDefined(residual, ['cell_residual_rms_ms', 'used_rms_ms', 'all_rms_ms']) ?? cell.residual_rms_ms;
  const status = firstDefined(velocity, ['velocity_status', 'status'])
    || firstDefined(row, ['velocity_status', 'status'])
    || cell.status;
  const numericFold = toFiniteNumber(foldValue);
  return [
    ['Cell', selectedCellLabel(cell)],
    ['Layer', layerLabel(layerKind)],
    ['Velocity', drilldownMetric(velocityValue, 2, 'm/s')],
    ['Fold', Number.isFinite(numericFold) ? formatNumber(numericFold, 0) : textOrDash(foldValue)],
    ['Residual RMS', drilldownMetric(residualRms, 1, 'ms')],
    ['Status', textOrDash(status)],
  ];
}

function openCellDrilldown(cell, context) {
  if (!cell) return;
  const target = context.cellDrilldownTargetFromSelectedCell(cell);
  context.controllerActions.setSelectedView('cell_maps_3d');
  if (target) context.loadQcDrilldown(target);
}

export function renderInspectorPanel({ state, dom, context }) {
  if (!dom?.inspector) return;
  context.clearNode(dom.inspector);
  const title = document.createElement('h3');
  title.textContent = 'Inspector';
  dom.inspector.appendChild(title);

  const selected = selectedObjectForInspector(state, context);
  if (selected.kind === 'pick') {
    const pick = selected.payload || {};
    const section = document.createElement('section');
    section.className = 'refraction-qc-inspector-section';
    const heading = document.createElement('h4');
    heading.textContent = 'Selected pick';
    section.appendChild(heading);
    section.appendChild(context.createKv(pickInspectorItems(pick, context)));
    const actions = document.createElement('div');
    actions.className = 'refraction-qc-actions';
    appendInspectorAction(actions, 'Preview source gather', pick.sourceEndpointKey ? '' : 'Missing source endpoint key.', () => {
      context.previewGatherForEndpoint('source', pick.sourceEndpointKey);
    });
    appendInspectorAction(actions, 'Preview receiver gather', pick.receiverEndpointKey ? '' : 'Missing receiver endpoint key.', () => {
      context.previewGatherForEndpoint('receiver', pick.receiverEndpointKey);
    });
    appendInspectorAction(actions, 'Open source station', pick.sourceEndpointKey ? '' : 'Missing source endpoint key.', () => {
      context.openEndpointStaticDrilldown('source', pick.sourceEndpointKey);
    });
    appendInspectorAction(actions, 'Open receiver station', pick.receiverEndpointKey ? '' : 'Missing receiver endpoint key.', () => {
      context.openEndpointStaticDrilldown('receiver', pick.receiverEndpointKey);
    });
    appendInspectorAction(actions, 'Filter to this trace', pick.traceIndex && pick.traceIndex !== '-' ? '' : 'Missing trace index.', () => {
      state.selectedTraceIndex = String(pick.traceIndex);
      context.controllerActions.setSelectedView('static_components');
    });
    section.appendChild(actions);
    dom.inspector.appendChild(section);
    return;
  }

  if (selected.kind === 'endpoint') {
    const endpoint = selected.payload || {};
    const kind = endpoint.endpointKind === 'receiver' ? 'receiver' : 'source';
    const section = document.createElement('section');
    section.className = 'refraction-qc-inspector-section';
    const heading = document.createElement('h4');
    heading.textContent = `Selected ${kind} station`;
    section.appendChild(heading);
    section.appendChild(context.createKv(endpointInspectorItems(endpoint, context)));
    const actions = document.createElement('div');
    actions.className = 'refraction-qc-actions';
    appendInspectorAction(actions, 'Preview gather', '', () => context.previewGatherForEndpoint(kind, endpoint.endpointKey));
    appendInspectorAction(actions, 'Open endpoint drilldown', '', () => context.openEndpointStaticDrilldown(kind, endpoint.endpointKey));
    appendInspectorAction(actions, 'Show station profile', '', () => {
      context.setEndpointFilter(kind, endpoint.endpointKey);
      context.controllerActions.setSelectedView('profiles_2d');
    });
    appendInspectorAction(actions, 'Copy endpoint key', typeof navigator !== 'undefined' && navigator.clipboard?.writeText ? '' : 'Clipboard access is unavailable.', async () => {
      await navigator.clipboard.writeText(endpoint.endpointKey);
    });
    section.appendChild(actions);
    dom.inspector.appendChild(section);
    return;
  }

  if (selected.kind === 'cell') {
    const cell = selected.payload || {};
    const section = document.createElement('section');
    section.className = 'refraction-qc-inspector-section';
    const heading = document.createElement('h4');
    heading.textContent = 'Selected cell';
    section.appendChild(heading);
    section.appendChild(context.createKv(cellInspectorItems(cell, state, context)));
    const actions = document.createElement('div');
    actions.className = 'refraction-qc-actions';
    appendInspectorAction(actions, 'Open cell drilldown', '', () => openCellDrilldown(cell, context));
    appendInspectorAction(actions, 'Show contributing stations', '', () => openCellDrilldown(cell, context));
    section.appendChild(actions);
    if (state.qcDrilldownLoading) {
      const loading = document.createElement('p');
      loading.className = 'refraction-qc-note';
      loading.textContent = 'Loading cell drilldown...';
      section.appendChild(loading);
    }
    if (state.qcDrilldownError) {
      const error = document.createElement('p');
      error.className = 'refraction-qc-error';
      error.textContent = state.qcDrilldownError;
      section.appendChild(error);
    }
    dom.inspector.appendChild(section);
    return;
  }

  const empty = document.createElement('p');
  empty.className = 'refraction-qc-placeholder';
  empty.textContent = 'No selection. Click a pick, station, or cell to inspect it.';
  dom.inspector.appendChild(empty);
}
