(function () {
  const RECENT_JOBS_KEY = 'sv.refraction_qc.recent_jobs';
  const MAX_RECENT_JOBS = 8;
  const DEFAULT_MAX_POINTS = 20000;

  const VIEW_DEFS = [
    {
      id: 'summary',
      include: 'summary',
      viewKeys: [],
      unavailableKeys: ['summary'],
    },
    {
      id: 'first_break_residuals',
      include: 'first_break',
      viewKeys: ['first_break_fit', 'first_break_residual'],
      unavailableKeys: ['first_break'],
    },
    {
      id: 'reduced_time',
      include: 'reduced_time',
      viewKeys: ['reduced_time'],
      unavailableKeys: ['reduced_time'],
    },
    {
      id: 'profiles_2d',
      include: 'profiles',
      viewKeys: ['line_profiles'],
      unavailableKeys: ['profiles'],
    },
    {
      id: 'cell_maps_3d',
      include: 'cells',
      viewKeys: ['refractor_cells', 'v3_refractor_cells', 'vsub_refractor_cells'],
      unavailableKeys: ['cells'],
    },
    {
      id: 'static_components',
      include: 'static_components',
      viewKeys: ['static_components'],
      unavailableKeys: ['static_components'],
    },
    {
      id: 'gather_preview',
      include: 'gather_preview',
      viewKeys: [],
      unavailableKeys: ['gather_preview'],
    },
  ];

  const INCLUDE_ALL = Array.from(new Set(VIEW_DEFS.map((view) => view.include)));

  const state = {
    selectedJobId: '',
    qcBundle: null,
    selectedView: 'summary',
    selectedLayerKind: 'all',
    firstBreakXAxis: 'offset',
    showRejectedFirstBreaks: true,
    selectedEndpointKind: 'source',
    selectedCell: null,
    selectedEndpoint: '',
    maxPoints: DEFAULT_MAX_POINTS,
    error: null,
    loading: false,
  };

  let dom = null;
  let requestSerial = 0;

  const LAYER_LABELS = {
    v1_direct_arrival: 'V1 direct',
    v2_t1: 'V2/T1',
    v3_t2: 'V3/T2',
    vsub_t3: 'Vsub/T3',
  };

  const LAYER_COLORS = {
    v1_direct_arrival: '#7c3aed',
    v2_t1: '#2563eb',
    v3_t2: '#059669',
    vsub_t3: '#c2410c',
    unknown: '#64748b',
  };

  const REDUCED_TIME_GATE_KINDS = [
    'v1_direct_arrival',
    'v2_t1',
    'v3_t2',
    'vsub_t3',
  ];

  const FIRST_BREAK_X_AXES = {
    offset: {
      label: 'Offset (m)',
      columns: ['offset_m'],
    },
    inline: {
      label: 'Inline distance (m)',
      columns: ['inline_m', 'inline_distance_m'],
    },
    trace: {
      label: 'Trace index',
      columns: ['trace_index_sorted', 'sorted_trace_index', 'observation_index'],
    },
  };

  function readRecentJobs() {
    try {
      const raw = localStorage.getItem(RECENT_JOBS_KEY);
      const parsed = JSON.parse(raw || '[]');
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((item) => typeof item === 'string' && item).slice(0, MAX_RECENT_JOBS);
    } catch (_) {
      return [];
    }
  }

  function writeRecentJob(jobId) {
    const cleanJobId = String(jobId || '').trim();
    if (!cleanJobId) return;
    const recent = readRecentJobs().filter((item) => item !== cleanJobId);
    recent.unshift(cleanJobId);
    try {
      localStorage.setItem(RECENT_JOBS_KEY, JSON.stringify(recent.slice(0, MAX_RECENT_JOBS)));
    } catch (_) {
    }
    renderRecentJobs();
  }

  function parsePositiveInteger(value, fallback) {
    const parsed = Number.parseInt(String(value), 10);
    if (!Number.isFinite(parsed) || parsed < 1) return fallback;
    return parsed;
  }

  function parseCell(value) {
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

  function textOrDash(value) {
    if (value === null || value === undefined || value === '') return '-';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  function firstDefined(record, columns) {
    if (!record || typeof record !== 'object') return undefined;
    for (const column of columns) {
      const value = record[column];
      if (value !== null && value !== undefined && value !== '') return value;
    }
    return undefined;
  }

  function toFiniteNumber(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
    if (typeof value !== 'string') return NaN;
    const parsed = Number.parseFloat(value.trim());
    return Number.isFinite(parsed) ? parsed : NaN;
  }

  function normalizedText(value) {
    return String(value ?? '').trim().toLowerCase();
  }

  function formatNumber(value, digits = 3) {
    if (!Number.isFinite(value)) return '-';
    return value.toFixed(digits);
  }

  function layerLabel(layerKind) {
    return LAYER_LABELS[layerKind] || (layerKind ? String(layerKind) : 'Unlayered');
  }

  function firstBreakLayerMatches(record) {
    if (state.selectedLayerKind === 'all') return true;
    const layerKind = String(firstDefined(record, ['layer_kind']) || '').trim();
    return !layerKind || layerKind === state.selectedLayerKind;
  }

  function isRejectedFirstBreakRecord(record) {
    const usedValue = firstDefined(record, ['used_in_solve', 'used_for_inversion']);
    const usedText = normalizedText(usedValue);
    if (usedText === 'false' || usedText === '0' || usedText === 'no') return true;
    if (usedValue === false) return true;

    const statusText = normalizedText(firstDefined(record, ['status']));
    if (statusText === 'rejected' || statusText === 'unused') return true;

    const rejectText = normalizedText(firstDefined(record, ['reject_reason', 'rejection_reason']));
    return Boolean(rejectText && rejectText !== 'ok' && rejectText !== 'none' && rejectText !== 'nan');
  }

  function firstBreakXAxisDefinition() {
    return FIRST_BREAK_X_AXES[state.firstBreakXAxis] || FIRST_BREAK_X_AXES.offset;
  }

  function reducedTimeLayerKind(record) {
    return String(firstDefined(record, ['layer_kind', 'layer_gate_kind']) || '').trim() || 'unknown';
  }

  function reducedTimeLayerMatches(record) {
    if (state.selectedLayerKind === 'all') return true;
    return reducedTimeLayerKind(record) === state.selectedLayerKind;
  }

  function finiteReducedTimeMs(record) {
    const reducedS = toFiniteNumber(firstDefined(record, ['reduced_time_s']));
    if (Number.isFinite(reducedS)) return reducedS * 1000.0;
    return toFiniteNumber(firstDefined(record, ['reduced_time_ms']));
  }

  function normalizeReducedTimeRecord(record) {
    const xAxis = firstBreakXAxisDefinition();
    const x = toFiniteNumber(firstDefined(record, xAxis.columns));
    if (!Number.isFinite(x)) return null;

    const reducedMs = finiteReducedTimeMs(record);
    const observedS = toFiniteNumber(firstDefined(record, [
      'observed_time_s',
      'observed_first_break_time_s',
    ]));
    const velocity = toFiniteNumber(firstDefined(record, ['reduction_velocity_m_s']));
    const statusText = normalizedText(firstDefined(record, ['status'])) || 'ok';
    const used = !isRejectedFirstBreakRecord(record);
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

  function filteredReducedTimePoints(view) {
    const records = Array.isArray(view.records) ? view.records : [];
    const points = [];
    for (const record of records) {
      if (!reducedTimeLayerMatches(record)) continue;
      if (!state.showRejectedFirstBreaks && isRejectedFirstBreakRecord(record)) continue;
      const point = normalizeReducedTimeRecord(record);
      if (point) points.push(point);
    }
    return points;
  }

  function normalizeFirstBreakRecord(record) {
    const xAxis = firstBreakXAxisDefinition();
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
    return {
      x,
      observedMs: observedS * 1000.0,
      modeledMs: modeledS * 1000.0,
      residualMs: (observedS - modeledS) * 1000.0,
      layerKind,
      status: rejected ? 'rejected' : 'used',
      opacity: rejected ? 0.42 : 0.9,
      traceIndex: textOrDash(firstDefined(record, ['trace_index_sorted', 'sorted_trace_index'])),
      source: textOrDash(firstDefined(record, ['source_endpoint_key', 'source_id'])),
      receiver: textOrDash(firstDefined(record, ['receiver_endpoint_key', 'receiver_id'])),
    };
  }

  function filteredFirstBreakPoints(view) {
    const records = Array.isArray(view.records) ? view.records : [];
    const points = [];
    for (const record of records) {
      if (!firstBreakLayerMatches(record)) continue;
      if (!state.showRejectedFirstBreaks && isRejectedFirstBreakRecord(record)) continue;
      const point = normalizeFirstBreakRecord(record);
      if (point) points.push(point);
    }
    return points;
  }

  function clearNode(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function appendText(parent, text) {
    parent.appendChild(document.createTextNode(text));
  }

  function createKv(items) {
    const list = document.createElement('dl');
    list.className = 'refraction-qc-kv';
    for (const [label, value] of items) {
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = textOrDash(value);
      list.append(dt, dd);
    }
    return list;
  }

  function createTable(view) {
    const wrap = document.createElement('div');
    wrap.className = 'refraction-qc-table-wrap';
    const table = document.createElement('table');
    table.className = 'refraction-qc-table';

    const columns = Array.isArray(view.columns) ? view.columns.slice(0, 5) : [];
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const column of columns) {
      const th = document.createElement('th');
      th.textContent = column;
      headRow.appendChild(th);
    }
    head.appendChild(headRow);
    table.appendChild(head);

    const body = document.createElement('tbody');
    const records = Array.isArray(view.records) ? view.records.slice(0, 5) : [];
    for (const record of records) {
      const row = document.createElement('tr');
      for (const column of columns) {
        const td = document.createElement('td');
        td.textContent = textOrDash(record ? record[column] : '');
        row.appendChild(td);
      }
      body.appendChild(row);
    }
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function findViewData(bundle, viewDef) {
    const views = bundle && typeof bundle.views === 'object' && bundle.views ? bundle.views : {};
    for (const key of viewDef.viewKeys) {
      if (views[key]) return { key, view: views[key] };
    }
    return null;
  }

  function isUnavailable(bundle, viewDef) {
    const unavailable = Array.isArray(bundle?.unavailable_views) ? bundle.unavailable_views : [];
    return viewDef.unavailableKeys.some((key) => unavailable.includes(key));
  }

  function findDownsampling(bundle, key, view) {
    const downsampling = bundle && typeof bundle.downsampling === 'object' && bundle.downsampling
      ? bundle.downsampling
      : {};
    const entry = downsampling[key];
    if (entry && typeof entry === 'object') return entry;
    if (view && typeof view === 'object') {
      return {
        total_points: view.total_points,
        returned_points: view.returned_points,
        downsampled: view.downsampled,
        method: view.downsampling_method,
      };
    }
    return null;
  }

  function renderSummary(content, bundle) {
    const summary = bundle && typeof bundle.summary === 'object' && bundle.summary ? bundle.summary : {};
    content.appendChild(createKv([
      ['Job ID', bundle.job_id],
      ['State', summary.job_state || summary.status],
      ['Workflow', summary.workflow],
      ['Method', summary.method],
      ['Conversion', summary.conversion_mode],
      ['Layer count', summary.layer_count],
      ['Coordinate mode', bundle.coordinate_mode],
      ['Available views', Array.isArray(bundle.available_views) ? bundle.available_views.join(', ') : ''],
      ['Unavailable views', Array.isArray(bundle.unavailable_views) ? bundle.unavailable_views.join(', ') : ''],
    ]));
  }

  function renderTabular(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'This view is unavailable from the loaded QC bundle artifacts.'
        : 'No sampled records are present for this view.';
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Downsampled', view.downsampled ? 'yes' : 'no'],
      ['Columns', Array.isArray(view.columns) ? view.columns.join(', ') : ''],
    ]));
    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function createFirstBreakPlot(testId) {
    const plot = document.createElement('div');
    plot.className = 'refraction-qc-plot';
    plot.dataset.testid = testId;
    return plot;
  }

  function plotHoverText(point) {
    return [
      `Layer: ${layerLabel(point.layerKind)}`,
      `Status: ${point.status}`,
      `Trace: ${point.traceIndex}`,
      `Source: ${point.source}`,
      `Receiver: ${point.receiver}`,
      `Observed: ${formatNumber(point.observedMs, 2)} ms`,
      `Modeled: ${formatNumber(point.modeledMs, 2)} ms`,
      `Residual: ${formatNumber(point.residualMs, 2)} ms`,
    ].join('<br>');
  }

  function reducedTimeHoverText(point) {
    return [
      `Layer gate: ${layerLabel(point.layerKind)}`,
      `Status: ${point.status}`,
      `Trace: ${point.traceIndex}`,
      `Source: ${point.source}`,
      `Receiver: ${point.receiver}`,
      `Observed: ${formatNumber(point.observedMs, 2)} ms`,
      `Reduction velocity: ${formatNumber(point.reductionVelocity, 2)} m/s`,
      `Reduced time: ${formatNumber(point.reducedMs, 2)} ms`,
    ].join('<br>');
  }

  function reducedTimeVelocitySummary(points) {
    const velocities = points
      .map((point) => point.reductionVelocity)
      .filter((value) => Number.isFinite(value));
    if (!velocities.length) return 'unavailable';
    velocities.sort((a, b) => a - b);
    const min = velocities[0];
    const max = velocities[velocities.length - 1];
    const median = velocities[Math.floor((velocities.length - 1) / 2)];
    if (min === max) return `${formatNumber(min, 2)} m/s`;
    return `${formatNumber(min, 2)}-${formatNumber(max, 2)} m/s; median ${formatNumber(median, 2)} m/s`;
  }

  function statusCounts(points, field) {
    const counts = new Map();
    for (const point of points) {
      const key = point[field] || 'unknown';
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return Array.from(counts.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([key, count]) => `${key}: ${count}`)
      .join(', ');
  }

  function gateBound(value) {
    const parsed = toFiniteNumber(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function gateMinOffset(gate, kind) {
    if (kind === 'v1_direct_arrival') {
      const directBound = gateBound(gate.min_direct_offset_m);
      if (directBound !== null) return directBound;
    }
    return gateBound(gate.min_offset_m);
  }

  function gateMaxOffset(gate, kind) {
    if (kind === 'v1_direct_arrival') {
      const directBound = gateBound(gate.max_direct_offset_m);
      if (directBound !== null) return directBound;
    }
    return gateBound(gate.max_offset_m);
  }

  function gateHasBounds(gate, kind) {
    if (kind === 'v1_direct_arrival') {
      return (
        gate.min_direct_offset_m !== undefined
        || gate.max_direct_offset_m !== undefined
        || gate.min_offset_m !== undefined
        || gate.max_offset_m !== undefined
      );
    }
    return gate.min_offset_m !== undefined || gate.max_offset_m !== undefined;
  }

  function gateEnabled(gate, kind) {
    if (!gate || typeof gate !== 'object') return false;
    const enabled = gate.enabled;
    if (enabled === false) return false;
    const enabledText = normalizedText(enabled);
    if (enabledText === 'false' || enabledText === '0' || enabledText === 'no') return false;
    return gateHasBounds(gate, kind);
  }

  function collectReducedTimeGates(bundle) {
    const summary = bundle && typeof bundle.summary === 'object' && bundle.summary ? bundle.summary : {};
    const sources = [];
    if (summary.observation_gates && typeof summary.observation_gates === 'object') {
      sources.push(summary.observation_gates);
    }
    if (summary.layers && typeof summary.layers === 'object') {
      sources.push(summary.layers);
    }
    const gates = [];
    const seen = new Set();
    for (const source of sources) {
      for (const kind of REDUCED_TIME_GATE_KINDS) {
        if (seen.has(kind)) continue;
        const gate = source[kind];
        if (!gateEnabled(gate, kind)) continue;
        gates.push({
          kind,
          minOffset: gateMinOffset(gate, kind),
          maxOffset: gateMaxOffset(gate, kind),
        });
        seen.add(kind);
      }
    }
    return gates;
  }

  function gateRangeLabel(gate) {
    if (gate.minOffset !== null && gate.maxOffset !== null) {
      return `${formatNumber(gate.minOffset, 1)}-${formatNumber(gate.maxOffset, 1)} m`;
    }
    if (gate.minOffset !== null) return `>= ${formatNumber(gate.minOffset, 1)} m`;
    if (gate.maxOffset !== null) return `<= ${formatNumber(gate.maxOffset, 1)} m`;
    return 'all offsets';
  }

  function reducedTimeGateOverlays(bundle) {
    if (state.firstBreakXAxis !== 'offset') return { shapes: [], label: 'Offset gate overlays are shown when X axis is Offset.' };
    const shapes = [];
    const labels = [];
    for (const gate of collectReducedTimeGates(bundle)) {
      const color = LAYER_COLORS[gate.kind] || LAYER_COLORS.unknown;
      if (gate.minOffset !== null && gate.maxOffset !== null) {
        shapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: gate.minOffset,
          x1: gate.maxOffset,
          y0: 0,
          y1: 1,
          fillcolor: color,
          opacity: 0.08,
          line: { color, width: 1, dash: 'dot' },
          layer: 'below',
        });
      } else {
        const x = gate.minOffset !== null ? gate.minOffset : gate.maxOffset;
        if (x !== null) {
          shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: x,
            x1: x,
            y0: 0,
            y1: 1,
            line: { color, width: 1, dash: 'dot' },
            layer: 'below',
          });
        }
      }
      labels.push(`${layerLabel(gate.kind)} ${gateRangeLabel(gate)}`);
    }
    return {
      shapes,
      label: labels.length ? labels.join('; ') : 'No offset gate metadata is available for this QC bundle.',
    };
  }

  function renderFirstBreakPlots(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'This view is unavailable from the loaded QC bundle artifacts.'
        : 'No sampled first-break fit records are present for this view.';
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    const points = filteredFirstBreakPoints(view);
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';

    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Plotted points', `${points.length}`],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Rejected picks', state.showRejectedFirstBreaks ? 'shown' : 'hidden'],
    ]));

    const residualNote = document.createElement('p');
    residualNote.className = 'refraction-qc-note';
    residualNote.textContent = 'Residual = observed - modeled, shown in ms.';
    residualNote.dataset.testid = 'refraction-qc-first-break-residual-note';
    content.appendChild(residualNote);

    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-first-break-downsampling';
    content.appendChild(downsamplingNote);

    if (!points.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = 'No plottable observed/modeled first-break records match the current filters.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }

    const plotGrid = document.createElement('div');
    plotGrid.className = 'refraction-qc-plot-grid';
    const timePlot = createFirstBreakPlot('refraction-qc-first-break-time-plot');
    const residualPlot = createFirstBreakPlot('refraction-qc-first-break-residual-plot');
    timePlot.dataset.pointCount = String(points.length);
    residualPlot.dataset.pointCount = String(points.length);
    plotGrid.append(timePlot, residualPlot);
    content.appendChild(plotGrid);

    if (window.Plotly) {
      const xAxis = firstBreakXAxisDefinition();
      const hoverText = points.map(plotHoverText);
      const commonLayout = {
        height: 260,
        margin: { l: 58, r: 14, t: 34, b: 50 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: {
          title: { text: xAxis.label },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.16,
          xanchor: 'left',
          yanchor: 'top',
          font: { size: 10 },
        },
      };
      const config = { displayModeBar: false, responsive: true };

      window.Plotly.newPlot(timePlot, [
        {
          name: 'Observed',
          type: 'scatter',
          mode: 'markers',
          x: points.map((point) => point.x),
          y: points.map((point) => point.observedMs),
          text: hoverText,
          hovertemplate: '%{text}<extra>Observed</extra>',
          marker: {
            color: '#2563eb',
            size: 7,
            opacity: points.map((point) => point.opacity),
          },
        },
        {
          name: 'Modeled',
          type: 'scatter',
          mode: 'markers',
          x: points.map((point) => point.x),
          y: points.map((point) => point.modeledMs),
          text: hoverText,
          hovertemplate: '%{text}<extra>Modeled</extra>',
          marker: {
            color: '#f97316',
            symbol: 'x',
            size: 8,
            opacity: points.map((point) => point.opacity),
          },
        },
      ], {
        ...commonLayout,
        title: { text: 'Observed vs modeled first-break time', font: { size: 12 } },
        yaxis: {
          title: { text: 'First-break time (ms)' },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
      }, config);

      const residualGroups = new Map();
      for (const point of points) {
        const groupKey = `${point.layerKind}|${point.status}`;
        if (!residualGroups.has(groupKey)) {
          residualGroups.set(groupKey, {
            layerKind: point.layerKind,
            status: point.status,
            x: [],
            y: [],
            text: [],
          });
        }
        const group = residualGroups.get(groupKey);
        group.x.push(point.x);
        group.y.push(point.residualMs);
        group.text.push(plotHoverText(point));
      }
      const residualTraces = Array.from(residualGroups.values()).map((group) => ({
        name: `${layerLabel(group.layerKind)} ${group.status}`,
        type: 'scatter',
        mode: 'markers',
        x: group.x,
        y: group.y,
        text: group.text,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          color: LAYER_COLORS[group.layerKind] || LAYER_COLORS.unknown,
          symbol: group.status === 'rejected' ? 'x' : 'circle',
          size: group.status === 'rejected' ? 8 : 7,
          opacity: group.status === 'rejected' ? 0.55 : 0.9,
        },
      }));
      window.Plotly.newPlot(residualPlot, residualTraces, {
        ...commonLayout,
        title: { text: 'First-break residuals', font: { size: 12 } },
        yaxis: {
          title: { text: 'Residual (ms)' },
          zeroline: true,
          zerolinecolor: '#94a3b8',
          gridcolor: '#e5e7eb',
        },
      }, config);
    } else {
      timePlot.textContent = 'Plot library is unavailable.';
      residualPlot.textContent = 'Plot library is unavailable.';
    }

    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function renderReducedTimePlot(content, bundle, viewDef) {
    const found = findViewData(bundle, viewDef);
    if (!found) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = isUnavailable(bundle, viewDef)
        ? 'This view is unavailable from the loaded QC bundle artifacts.'
        : 'No sampled reduced-time records are present for this view.';
      content.appendChild(missing);
      return;
    }

    const { key, view } = found;
    const points = filteredReducedTimePoints(view);
    const plottedPoints = points.filter((point) => Number.isFinite(point.reducedMs));
    const unavailablePoints = points.filter((point) => !Number.isFinite(point.reducedMs));
    const downsampling = findDownsampling(bundle, key, view);
    const downsamplingText = downsampling
      ? `${downsampling.returned_points || 0} of ${downsampling.total_points || 0}; ${downsampling.downsampled ? 'downsampled' : 'not downsampled'}${downsampling.method ? ` (${downsampling.method})` : ''}`
      : 'not reported';
    const gateOverlay = reducedTimeGateOverlays(bundle);

    content.appendChild(createKv([
      ['Bundle view', key],
      ['Artifact', view.artifact],
      ['Rows', `${view.returned_points || 0} of ${view.total_points || 0}`],
      ['Plotted points', `${plottedPoints.length}`],
      ['Layer filter', state.selectedLayerKind === 'all' ? 'all' : layerLabel(state.selectedLayerKind)],
      ['Unused picks', state.showRejectedFirstBreaks ? 'shown' : 'hidden'],
      ['Reduction velocity', reducedTimeVelocitySummary(points)],
      ['Unavailable rows', unavailablePoints.length ? `${unavailablePoints.length}; ${statusCounts(unavailablePoints, 'unavailableReason')}` : '0'],
    ]));

    const formulaNote = document.createElement('p');
    formulaNote.className = 'refraction-qc-note';
    formulaNote.textContent = 'Reduced time = observed first-break time - offset / reduction velocity, shown in ms.';
    formulaNote.dataset.testid = 'refraction-qc-reduced-time-formula-note';
    content.appendChild(formulaNote);

    const gateNote = document.createElement('p');
    gateNote.className = 'refraction-qc-note';
    gateNote.textContent = `Gate overlays: ${gateOverlay.label}`;
    gateNote.dataset.testid = 'refraction-qc-reduced-time-gates';
    content.appendChild(gateNote);

    const downsamplingNote = document.createElement('p');
    downsamplingNote.className = 'refraction-qc-note';
    downsamplingNote.textContent = `Downsampling: ${downsamplingText}`;
    downsamplingNote.dataset.testid = 'refraction-qc-reduced-time-downsampling';
    content.appendChild(downsamplingNote);

    if (!plottedPoints.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = unavailablePoints.length
        ? 'Reduced-time rows matched the filters, but none have an available reduced-time value.'
        : 'No plottable reduced-time records match the current filters.';
      content.appendChild(missing);
      if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
        content.appendChild(createTable(view));
      }
      return;
    }

    const plot = createFirstBreakPlot('refraction-qc-reduced-time-plot');
    plot.dataset.pointCount = String(plottedPoints.length);
    content.appendChild(plot);

    if (window.Plotly) {
      const xAxis = firstBreakXAxisDefinition();
      const groups = new Map();
      for (const point of plottedPoints) {
        const groupKey = `${point.layerKind}|${point.used ? 'used' : 'unused'}`;
        if (!groups.has(groupKey)) {
          groups.set(groupKey, {
            layerKind: point.layerKind,
            used: point.used,
            x: [],
            y: [],
            text: [],
          });
        }
        const group = groups.get(groupKey);
        group.x.push(point.x);
        group.y.push(point.reducedMs);
        group.text.push(reducedTimeHoverText(point));
      }
      const traces = Array.from(groups.values()).map((group) => ({
        name: `${layerLabel(group.layerKind)} ${group.used ? 'used' : 'unused'}`,
        type: 'scatter',
        mode: 'markers',
        x: group.x,
        y: group.y,
        text: group.text,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          color: LAYER_COLORS[group.layerKind] || LAYER_COLORS.unknown,
          symbol: group.used ? 'circle' : 'x',
          size: group.used ? 7 : 8,
          opacity: group.used ? 0.9 : 0.55,
        },
      }));
      window.Plotly.newPlot(plot, traces, {
        height: 300,
        margin: { l: 62, r: 14, t: 34, b: 50 },
        font: { size: 10, color: '#334155' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        title: { text: 'Reduced-time first-break QC', font: { size: 12 } },
        xaxis: {
          title: { text: xAxis.label },
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        yaxis: {
          title: { text: 'Reduced time (ms)' },
          zeroline: true,
          zerolinecolor: '#94a3b8',
          gridcolor: '#e5e7eb',
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.16,
          xanchor: 'left',
          yanchor: 'top',
          font: { size: 10 },
        },
        shapes: gateOverlay.shapes,
      }, { displayModeBar: false, responsive: true });
    } else {
      plot.textContent = 'Plot library is unavailable.';
    }

    if (Array.isArray(view.records) && view.records.length && Array.isArray(view.columns) && view.columns.length) {
      content.appendChild(createTable(view));
    }
  }

  function renderGatherPreview(content, bundle) {
    const message = document.createElement('p');
    message.className = 'refraction-qc-placeholder';
    if (isUnavailable(bundle, VIEW_DEFS[6])) {
      message.textContent = 'Gather preview is not included in the compact QC bundle.';
    } else {
      message.textContent = 'No gather preview payload loaded.';
    }
    content.appendChild(message);
    content.appendChild(createKv([
      ['Endpoint kind', state.selectedEndpointKind],
      ['Endpoint', state.selectedEndpoint],
      ['Cell', state.selectedCell ? textOrDash(state.selectedCell) : '-'],
      ['Layer', state.selectedLayerKind],
    ]));
  }

  function renderViewContent(viewDef) {
    if (!dom) return;
    const content = dom.viewContents.get(viewDef.id);
    if (!content) return;
    clearNode(content);
    content.className = 'refraction-qc-placeholder';

    if (state.loading) {
      appendText(content, 'Loading QC bundle...');
      return;
    }
    if (!state.qcBundle) {
      appendText(content, state.error ? 'QC bundle is not loaded.' : 'No QC bundle loaded.');
      return;
    }

    content.className = '';
    if (viewDef.id === 'summary') {
      renderSummary(content, state.qcBundle);
    } else if (viewDef.id === 'first_break_residuals') {
      renderFirstBreakPlots(content, state.qcBundle, viewDef);
    } else if (viewDef.id === 'reduced_time') {
      renderReducedTimePlot(content, state.qcBundle, viewDef);
    } else if (viewDef.id === 'gather_preview') {
      renderGatherPreview(content, state.qcBundle);
    } else {
      renderTabular(content, state.qcBundle, viewDef);
    }
  }

  function renderRecentJobs() {
    if (!dom?.jobList) return;
    clearNode(dom.jobList);
    for (const jobId of readRecentJobs()) {
      const option = document.createElement('option');
      option.value = jobId;
      dom.jobList.appendChild(option);
    }
  }

  function render() {
    if (!dom) return;

    dom.loadButton.disabled = state.loading;
    dom.maxPoints.value = String(state.maxPoints);
    dom.layerKind.value = state.selectedLayerKind;
    dom.xAxisMode.value = state.firstBreakXAxis;
    dom.showRejected.checked = state.showRejectedFirstBreaks;
    dom.endpointKind.value = state.selectedEndpointKind;
    dom.endpoint.value = state.selectedEndpoint;
    dom.cell.value = state.selectedCell?.cell_ix !== undefined
      ? `${state.selectedCell.cell_ix},${state.selectedCell.cell_iy}`
      : (state.selectedCell?.text || '');

    if (state.loading) {
      dom.status.textContent = 'Loading QC bundle...';
    } else if (state.qcBundle) {
      const available = Array.isArray(state.qcBundle.available_views)
        ? state.qcBundle.available_views.length
        : 0;
      const unavailable = Array.isArray(state.qcBundle.unavailable_views)
        ? state.qcBundle.unavailable_views.length
        : 0;
      dom.status.textContent = `Loaded ${state.qcBundle.job_id}: ${available} available, ${unavailable} unavailable.`;
    } else {
      dom.status.textContent = 'No QC bundle loaded.';
    }

    dom.error.hidden = !state.error;
    dom.error.textContent = state.error || '';

    const signConvention = state.qcBundle?.sign_convention;
    dom.sign.hidden = !signConvention;
    dom.sign.textContent = signConvention ? `Sign: ${signConvention}` : '';

    for (const button of dom.viewButtons) {
      const active = button.dataset.view === state.selectedView;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
    }
    for (const panel of dom.viewPanels) {
      panel.hidden = panel.dataset.viewPanel !== state.selectedView;
    }
    const selectedViewDef = VIEW_DEFS.find((viewDef) => viewDef.id === state.selectedView);
    if (selectedViewDef) renderViewContent(selectedViewDef);
  }

  function setSelectedView(viewId) {
    if (!VIEW_DEFS.some((view) => view.id === viewId)) return;
    state.selectedView = viewId;
    render();
  }

  function activateSidebarTab(tabName) {
    if (!dom) return;
    const showQc = tabName === 'refraction_qc';
    dom.pipelineTab.classList.toggle('is-active', !showQc);
    dom.qcTab.classList.toggle('is-active', showQc);
    dom.pipelineTab.setAttribute('aria-selected', showQc ? 'false' : 'true');
    dom.qcTab.setAttribute('aria-selected', showQc ? 'true' : 'false');
    dom.pipelinePanel.hidden = showQc;
    dom.qcPanel.hidden = !showQc;
  }

  async function readError(response) {
    try {
      const text = await response.text();
      if (!text) return `QC bundle request failed with status ${response.status}`;
      try {
        const payload = JSON.parse(text);
        if (payload && typeof payload.detail === 'string') return payload.detail;
        if (payload && payload.detail) return JSON.stringify(payload.detail);
      } catch (_) {
      }
      return text;
    } catch (_) {
    }
    return `QC bundle request failed with status ${response.status}`;
  }

  async function loadBundle() {
    if (!dom) return;
    const jobId = String(dom.jobId.value || '').trim();
    const maxPoints = parsePositiveInteger(dom.maxPoints.value, DEFAULT_MAX_POINTS);
    state.selectedJobId = jobId;
    state.maxPoints = maxPoints;
    if (!jobId) {
      state.error = 'Job ID is required.';
      state.qcBundle = null;
      render();
      return;
    }

    const serial = ++requestSerial;
    state.loading = true;
    state.error = null;
    render();

    try {
      const response = await fetch('/statics/refraction/qc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          include: INCLUDE_ALL,
          max_points: maxPoints,
          coordinate_mode: 'auto',
        }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const bundle = await response.json();
      if (serial !== requestSerial) return;
      state.qcBundle = bundle;
      state.error = null;
      writeRecentJob(jobId);
    } catch (error) {
      if (serial !== requestSerial) return;
      state.qcBundle = null;
      state.error = error instanceof Error ? error.message : String(error);
    } finally {
      if (serial === requestSerial) {
        state.loading = false;
        render();
      }
    }
  }

  function init() {
    const pipelineTab = document.getElementById('pipelineSidebarTab');
    const qcTab = document.getElementById('refractionQcSidebarTab');
    const pipelinePanel = document.getElementById('pipelineTabPanel');
    const qcPanel = document.getElementById('refractionQcTabPanel');
    const form = document.getElementById('refractionQcForm');
    if (!pipelineTab || !qcTab || !pipelinePanel || !qcPanel || !form) return;

    dom = {
      pipelineTab,
      qcTab,
      pipelinePanel,
      qcPanel,
      form,
      jobId: document.getElementById('refractionQcJobId'),
      jobList: document.getElementById('refractionQcJobList'),
      maxPoints: document.getElementById('refractionQcMaxPoints'),
      loadButton: document.getElementById('refractionQcLoadButton'),
      status: document.getElementById('refractionQcStatus'),
      error: document.getElementById('refractionQcError'),
      sign: document.getElementById('refractionQcSign'),
      layerKind: document.getElementById('refractionQcLayerKind'),
      xAxisMode: document.getElementById('refractionQcXAxisMode'),
      showRejected: document.getElementById('refractionQcShowRejected'),
      endpointKind: document.getElementById('refractionQcEndpointKind'),
      endpoint: document.getElementById('refractionQcEndpoint'),
      cell: document.getElementById('refractionQcCell'),
      viewButtons: Array.from(document.querySelectorAll('.refraction-qc-view-button')),
      viewPanels: Array.from(document.querySelectorAll('.refraction-qc-view')),
      viewContents: new Map(Array.from(document.querySelectorAll('[data-view-content]')).map(
        (node) => [node.dataset.viewContent, node],
      )),
    };

    if (!dom.jobId || !dom.maxPoints || !dom.loadButton || !dom.status || !dom.error || !dom.sign) return;
    if (!dom.layerKind || !dom.xAxisMode || !dom.showRejected || !dom.endpointKind || !dom.endpoint || !dom.cell) return;

    pipelineTab.addEventListener('click', () => activateSidebarTab('pipeline'));
    qcTab.addEventListener('click', () => activateSidebarTab('refraction_qc'));
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      loadBundle();
    });
    dom.jobId.addEventListener('input', () => {
      state.selectedJobId = dom.jobId.value.trim();
    });
    dom.maxPoints.addEventListener('input', () => {
      state.maxPoints = parsePositiveInteger(dom.maxPoints.value, DEFAULT_MAX_POINTS);
    });
    dom.layerKind.addEventListener('change', () => {
      state.selectedLayerKind = dom.layerKind.value;
      render();
    });
    dom.xAxisMode.addEventListener('change', () => {
      state.firstBreakXAxis = dom.xAxisMode.value;
      render();
    });
    dom.showRejected.addEventListener('change', () => {
      state.showRejectedFirstBreaks = dom.showRejected.checked;
      render();
    });
    dom.endpointKind.addEventListener('change', () => {
      state.selectedEndpointKind = dom.endpointKind.value;
      render();
    });
    dom.endpoint.addEventListener('input', () => {
      state.selectedEndpoint = dom.endpoint.value.trim();
      render();
    });
    dom.cell.addEventListener('input', () => {
      state.selectedCell = parseCell(dom.cell.value);
      render();
    });
    for (const button of dom.viewButtons) {
      button.addEventListener('click', () => setSelectedView(button.dataset.view));
    }

    const params = new URLSearchParams(window.location.search || '');
    const jobId = params.get('refraction_job_id') || params.get('refraction_qc_job_id') || '';
    if (jobId) {
      state.selectedJobId = jobId;
      dom.jobId.value = jobId;
    }

    renderRecentJobs();
    render();
  }

  window.refractionQcState = state;
  window.refractionQcUI = {
    loadBundle,
    setSelectedView,
    activateSidebarTab,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
