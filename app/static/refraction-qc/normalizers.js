export const FIRST_BREAK_TRACE_COLUMNS = ['trace_index_sorted', 'sorted_trace_index', 'observation_index'];
export const FIRST_BREAK_OFFSET_COLUMNS = ['offset_m'];

export function firstDefined(record, columns) {
  if (!record || typeof record !== 'object') return undefined;
  for (const column of columns) {
    const value = record[column];
    if (value !== null && value !== undefined && value !== '') return value;
  }
  return undefined;
}

export function toFiniteNumber(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
  if (typeof value !== 'string') return NaN;
  const parsed = Number.parseFloat(value.trim());
  return Number.isFinite(parsed) ? parsed : NaN;
}

export function textOrDash(value) {
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function normalizedText(value) {
  return String(value ?? '').trim().toLowerCase();
}

export function positiveIntegerText(value) {
  const text = String(value ?? '').trim();
  const parsed = Number(text);
  if (!text || !Number.isInteger(parsed) || parsed < 1) return '';
  return text;
}

export function parsePositiveInteger(value, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed) || parsed < 1) return fallback;
  return parsed;
}

export function parseCell(value) {
  const text = String(value || '').trim();
  if (!text) return null;
  const parts = text.split(/[,\s]+/).filter(Boolean);
  if (parts.length !== 2) return { text };
  const ix = Number.parseInt(parts[0], 10);
  const iy = Number.parseInt(parts[1], 10);
  if (!Number.isInteger(ix) || !Number.isInteger(iy) || ix < 0 || iy < 0) {
    return { text };
  }
  return { cell_ix: ix, cell_iy: iy };
}

export function isRejectedFirstBreakRecord(record) {
  const usedValue = firstDefined(record, ['used_in_solve', 'used_for_inversion']);
  const usedText = normalizedText(usedValue);
  if (usedText === 'false' || usedText === '0' || usedText === 'no') return true;
  if (usedValue === false) return true;

  const statusText = normalizedText(firstDefined(record, ['status']));
  if (statusText === 'rejected' || statusText === 'unused') return true;

  const rejectText = normalizedText(firstDefined(record, ['reject_reason', 'rejection_reason']));
  return Boolean(rejectText && rejectText !== 'ok' && rejectText !== 'none' && rejectText !== 'nan');
}

export function isUnusedReducedTimeRecord(record) {
  const usedValue = firstDefined(record, ['used_for_inversion']);
  const usedText = normalizedText(usedValue);
  if (usedText === 'false' || usedText === '0' || usedText === 'no') return true;
  if (usedValue === false) return true;

  const statusText = normalizedText(firstDefined(record, ['status']));
  if (statusText === 'rejected' || statusText === 'unused') return true;

  const rejectText = normalizedText(firstDefined(record, ['reject_reason', 'rejection_reason']));
  return Boolean(rejectText && rejectText !== 'ok' && rejectText !== 'none' && rejectText !== 'nan');
}

export function reducedTimeLayerKind(record) {
  return String(firstDefined(record, ['layer_kind', 'layer_gate_kind']) || '').trim() || 'unknown';
}

function finiteReducedTimeMs(record) {
  const reducedS = toFiniteNumber(firstDefined(record, ['reduced_time_s']));
  if (Number.isFinite(reducedS)) return reducedS * 1000.0;
  return toFiniteNumber(firstDefined(record, ['reduced_time_ms']));
}

export function normalizeReducedTimeRecord(record, xAxis) {
  const x = toFiniteNumber(firstDefined(record, xAxis.columns));
  if (!Number.isFinite(x)) return null;

  const reducedMs = finiteReducedTimeMs(record);
  const observedS = toFiniteNumber(firstDefined(record, [
    'observed_time_s',
    'observed_first_break_time_s',
  ]));
  const velocity = toFiniteNumber(firstDefined(record, ['reduction_velocity_m_s']));
  const statusText = normalizedText(firstDefined(record, ['status'])) || 'ok';
  const used = !isUnusedReducedTimeRecord(record);
  const layerKind = reducedTimeLayerKind(record);
  const unavailableReason = Number.isFinite(reducedMs)
    ? ''
    : (statusText && statusText !== 'ok' ? statusText : 'missing_reduced_time');

  return {
    x,
    reducedMs,
    observedMs: Number.isFinite(observedS) ? observedS * 1000.0 : NaN,
    reductionVelocity: velocity,
    layerKind,
    used,
    status: unavailableReason ? 'unavailable' : (used ? 'used' : 'unused'),
    rawStatus: statusText,
    unavailableReason,
    traceIndex: textOrDash(firstDefined(record, ['trace_index_sorted', 'sorted_trace_index'])),
    source: textOrDash(firstDefined(record, ['source_endpoint_key', 'source_id'])),
    receiver: textOrDash(firstDefined(record, ['receiver_endpoint_key', 'receiver_id'])),
  };
}

export function normalizeFirstBreakRecord(record, xAxis) {
  const x = toFiniteNumber(firstDefined(record, xAxis.columns));
  const observedS = toFiniteNumber(firstDefined(record, [
    'observed_first_break_time_s',
    'observed_time_s',
  ]));
  const modeledS = toFiniteNumber(firstDefined(record, [
    'modeled_first_break_time_s',
    'modeled_time_s',
  ]));
  if (!Number.isFinite(x) || !Number.isFinite(observedS) || !Number.isFinite(modeledS)) {
    return null;
  }

  const layerKind = String(firstDefined(record, ['layer_kind']) || '').trim() || 'unknown';
  const rejected = isRejectedFirstBreakRecord(record);
  const sourceEndpointKey = String(firstDefined(record, ['source_endpoint_key']) || '').trim();
  const receiverEndpointKey = String(firstDefined(record, ['receiver_endpoint_key']) || '').trim();
  const traceIndex = textOrDash(firstDefined(record, ['trace_index_sorted', 'sorted_trace_index']));
  const traceSortIndex = toFiniteNumber(firstDefined(record, FIRST_BREAK_TRACE_COLUMNS));
  const offsetM = toFiniteNumber(firstDefined(record, FIRST_BREAK_OFFSET_COLUMNS));
  return {
    x,
    offsetM,
    observedMs: observedS * 1000.0,
    modeledMs: modeledS * 1000.0,
    residualMs: (observedS - modeledS) * 1000.0,
    layerKind,
    status: rejected ? 'rejected' : 'used',
    opacity: rejected ? 0.42 : 0.9,
    traceIndex,
    traceSortIndex,
    sourceEndpointKey,
    receiverEndpointKey,
    source: textOrDash(sourceEndpointKey || firstDefined(record, ['source_id'])),
    receiver: textOrDash(receiverEndpointKey || firstDefined(record, ['receiver_id'])),
  };
}

export function isInvalidProfileRecord(record, okStatuses) {
  const staticStatus = normalizedText(firstDefined(record, ['static_status']));
  const solutionStatus = normalizedText(firstDefined(record, ['solution_status']));
  return !okStatuses.has(staticStatus) || !okStatuses.has(solutionStatus);
}

export function normalizeProfileRecord(record, okStatuses) {
  const inline = toFiniteNumber(firstDefined(record, ['inline_m']));
  if (!Number.isFinite(inline)) return null;
  return {
    raw: record,
    inline,
    endpointKind: normalizedText(firstDefined(record, ['endpoint_kind'])) || 'unknown',
    endpointKey: textOrDash(firstDefined(record, ['endpoint_key'])),
    stationId: textOrDash(firstDefined(record, [
      'station_id',
      'endpoint_id',
      'source_id',
      'receiver_id',
    ])),
    nodeId: textOrDash(firstDefined(record, ['node_id'])),
    staticStatus: textOrDash(firstDefined(record, ['static_status'])),
    solutionStatus: textOrDash(firstDefined(record, ['solution_status'])),
    invalid: isInvalidProfileRecord(record, okStatuses),
  };
}

function finiteMetric(record, columns, secondColumns, scale = 1.0) {
  const direct = toFiniteNumber(firstDefined(record, columns || []));
  if (Number.isFinite(direct)) return direct;
  const seconds = toFiniteNumber(firstDefined(record, secondColumns || []));
  return Number.isFinite(seconds) ? seconds * scale : NaN;
}

export function cellMapLayerKind(record) {
  return String(firstDefined(record, ['layer_kind', 'cell_velocity_layer_kind']) || '').trim() || 'unknown';
}

export function normalizeCellMapRecord(record) {
  const cellIx = Number.parseInt(String(firstDefined(record, ['cell_ix', 'ix']) ?? ''), 10);
  const cellIy = Number.parseInt(String(firstDefined(record, ['cell_iy', 'iy']) ?? ''), 10);
  if (!Number.isInteger(cellIx) || !Number.isInteger(cellIy)) return null;

  const centerX = finiteMetric(record, [
    'cell_center_x_m',
    'x_center_m',
    'center_x_m',
    'cell_center_inline_m',
  ]);
  const centerY = finiteMetric(record, [
    'cell_center_y_m',
    'y_center_m',
    'center_y_m',
    'cell_center_crossline_m',
  ]);
  const velocity = finiteMetric(record, ['velocity_m_s', 'v2_m_s']);
  const initialVelocity = finiteMetric(record, ['initial_velocity_m_s', 'initial_v2_m_s']);
  const velocityUpdate = finiteMetric(record, [
    'velocity_update_from_initial_m_s',
    'v2_update_from_initial_m_s',
  ]);
  const fold = finiteMetric(record, ['n_observations', 'fold', 'observation_count']);
  const usedFold = finiteMetric(record, ['n_used_observations', 'used_fold']);
  const rejectedFold = finiteMetric(record, ['n_rejected_observations', 'rejected_fold']);
  const residualRmsMs = finiteMetric(record, ['residual_rms_ms'], ['residual_rms_s'], 1000.0);
  const residualMadMs = finiteMetric(record, ['residual_mad_ms'], ['residual_mad_s'], 1000.0);
  const rawStatus = normalizedText(firstDefined(record, ['status', 'velocity_status'])) || (
    Number.isFinite(fold) && fold <= 0 ? 'no_observations' : 'unknown'
  );

  return {
    raw: record,
    layerKind: cellMapLayerKind(record),
    cellIx,
    cellIy,
    centerX: Number.isFinite(centerX) ? centerX : cellIx,
    centerY: Number.isFinite(centerY) ? centerY : cellIy,
    velocity,
    initialVelocity,
    velocityUpdate,
    fold,
    usedFold,
    rejectedFold,
    residualRmsMs,
    residualMadMs,
    status: rawStatus,
    statusReason: textOrDash(firstDefined(record, ['status_reason', 'velocity_status_reason'])),
    component: textOrDash(firstDefined(record, ['cell_velocity_component'])),
  };
}
