import {
  firstDefined,
  isInvalidProfileRecord,
  isRejectedFirstBreakRecord,
  isUnusedReducedTimeRecord,
  normalizeCellMapRecord,
  normalizeFirstBreakRecord,
  normalizeProfileRecord,
  normalizeReducedTimeRecord,
  normalizedText,
  reducedTimeLayerKind,
  toFiniteNumber,
} from './normalizers.js';

function compareFiniteAscending(a, b) {
  const aFinite = Number.isFinite(a);
  const bFinite = Number.isFinite(b);
  if (aFinite && bFinite) return a - b;
  if (aFinite) return -1;
  if (bFinite) return 1;
  return 0;
}

function firstBreakLayerMatches(record, state) {
  if (state.selectedLayerKind === 'all') return true;
  const layerKind = String(firstDefined(record, ['layer_kind']) || '').trim();
  return !layerKind || layerKind === state.selectedLayerKind;
}

function reducedTimeLayerMatches(record, state) {
  if (state.selectedLayerKind === 'all') return true;
  return reducedTimeLayerKind(record) === state.selectedLayerKind;
}

export function firstBreakResidualThresholdMs(state) {
  const value = toFiniteNumber(state.firstBreakResidualThresholdMs);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function compareFirstBreakPoints(a, b, state) {
  const residualOrder = Math.abs(b.residualMs) - Math.abs(a.residualMs);
  const traceOrder = compareFiniteAscending(a.traceSortIndex, b.traceSortIndex);
  const offsetOrder = compareFiniteAscending(a.offsetM, b.offsetM);
  const endpointOrder = a.source.localeCompare(b.source) || a.receiver.localeCompare(b.receiver);
  if (state.firstBreakSortBy === 'trace') {
    return traceOrder || residualOrder || offsetOrder || endpointOrder;
  }
  if (state.firstBreakSortBy === 'offset') {
    return offsetOrder || residualOrder || traceOrder || endpointOrder;
  }
  return residualOrder || traceOrder || offsetOrder || endpointOrder;
}

export function filterFirstBreakPoints(view, state, xAxis) {
  const records = Array.isArray(view.records) ? view.records : [];
  const points = [];
  const residualThreshold = firstBreakResidualThresholdMs(state);
  for (const record of records) {
    if (!firstBreakLayerMatches(record, state)) continue;
    if (!state.showRejectedFirstBreaks && isRejectedFirstBreakRecord(record)) continue;
    const point = normalizeFirstBreakRecord(record, xAxis);
    if (point && residualThreshold !== null && Math.abs(point.residualMs) < residualThreshold) continue;
    if (point) points.push(point);
  }
  return points.sort((a, b) => compareFirstBreakPoints(a, b, state));
}

export function filterReducedTimePoints(view, state, xAxis) {
  const records = Array.isArray(view.records) ? view.records : [];
  const points = [];
  for (const record of records) {
    if (!reducedTimeLayerMatches(record, state)) continue;
    if (!state.showRejectedFirstBreaks && isUnusedReducedTimeRecord(record)) continue;
    const point = normalizeReducedTimeRecord(record, xAxis);
    if (point) points.push(point);
  }
  return points;
}

function profileStatusMatches(record, state, okStatuses) {
  const invalid = isInvalidProfileRecord(record, okStatuses);
  if (state.profileStatusFilter === 'valid') return !invalid;
  if (state.profileStatusFilter === 'invalid') return invalid;
  return true;
}

function profileEndpointMatches(record, state) {
  const kind = normalizedText(firstDefined(record, ['endpoint_kind']));
  if (state.selectedEndpointKind !== 'both' && kind !== state.selectedEndpointKind) return false;
  const endpointFilter = normalizedText(state.selectedEndpoint);
  if (!endpointFilter) return true;
  const endpointKey = normalizedText(firstDefined(record, ['endpoint_key']));
  return endpointKey.includes(endpointFilter);
}

export function filterProfileRecords(view, state, okStatuses) {
  const records = Array.isArray(view.records) ? view.records : [];
  const points = [];
  for (const record of records) {
    if (!profileEndpointMatches(record, state)) continue;
    if (!profileStatusMatches(record, state, okStatuses)) continue;
    const normalized = normalizeProfileRecord(record, okStatuses);
    if (normalized) points.push(normalized);
  }
  return points.sort((a, b) => (
    a.endpointKind.localeCompare(b.endpointKind)
    || a.inline - b.inline
    || a.endpointKey.localeCompare(b.endpointKey)
  ));
}

export function filterCellMapRecords(view, state, cellStatusCode) {
  const records = Array.isArray(view.records) ? view.records : [];
  const normalized = [];
  for (const record of records) {
    const point = normalizeCellMapRecord(record);
    if (point && state.profileStatusFilter === 'valid' && cellStatusCode(point.status) < 3) continue;
    if (point && state.profileStatusFilter === 'invalid' && cellStatusCode(point.status) >= 3) continue;
    if (point) normalized.push(point);
  }
  return normalized;
}

export function artifactRows(bundle) {
  const artifacts = bundle && typeof bundle.artifacts === 'object' && bundle.artifacts ? bundle.artifacts : {};
  const views = bundle && typeof bundle.views === 'object' && bundle.views ? bundle.views : {};
  const rows = Object.entries(artifacts).map(([name, path]) => ({ type: 'manifest', name, path }));
  for (const [name, view] of Object.entries(views)) {
    if (view?.artifact && !rows.some((row) => row.path === view.artifact)) {
      rows.push({ type: 'view', name, path: view.artifact });
    }
  }
  return rows;
}

export function filterArtifactRows(bundle, state) {
  const rows = artifactRows(bundle);
  const search = normalizedText(state.artifactSearch);
  return {
    rows,
    filteredRows: rows.filter((row) => {
      if (state.artifactTypeFilter !== 'all' && row.type !== state.artifactTypeFilter) return false;
      if (!search) return true;
      return [row.type, row.name, row.path].some((value) => normalizedText(value).includes(search));
    }),
  };
}
