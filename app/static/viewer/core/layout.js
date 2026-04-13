// /static/viewer/core/layout.js
// Build Plotly layout objects & pick overlay shapes

export function buildLayout({
  mode,
  x0,
  x1,
  y0,
  y1,
  stepX = 1,
  stepY = 1,
  totalSamples,
  dt,
  savedXRange,
  savedYRange,
  clickmode,
  dragmode,
  uirevision,
  fbTitle = null,
}) {
  const effectiveDt = typeof dt === 'number' ? dt : 0;
  const xaxis = {
    title: 'Trace',
    showgrid: false,
    tickfont: { color: '#000' },
    titlefont: { color: '#000' },
  };
  const yaxis = {
    title: 'Time (s)',
    showgrid: false,
    tickfont: { color: '#000' },
    titlefont: { color: '#000' },
  };

  if (mode === 'wiggle') {
    const defaultXRange = [x0, x1];
    const defaultYRange = [totalSamples * effectiveDt, 0];
    xaxis.autorange = false;
    xaxis.range = savedXRange ?? defaultXRange;
    yaxis.autorange = false;
    yaxis.range = savedYRange ?? defaultYRange;
  } else if (mode === 'heatmap') {
    const halfX = (stepX || 1) * 0.5;
    const halfYSec = (stepY || 1) * effectiveDt * 0.5;
    const defaultXRange = [x0 - halfX, x1 + halfX];
    const defaultYRange = [(y1 * effectiveDt) + halfYSec, (y0 * effectiveDt) + halfYSec];
    xaxis.autorange = !savedXRange;
    xaxis.range = savedXRange ?? defaultXRange;
    yaxis.autorange = false;
    yaxis.range = savedYRange ?? defaultYRange;
  }

  const layout = {
    xaxis,
    yaxis,
    clickmode,
    dragmode,
    uirevision,
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
    margin: { t: 10, r: 10, l: 60, b: 40 },
  };

  if (fbTitle !== null) layout.title = fbTitle;
  return layout;
}

export function buildPickShapes({
  manualPicks,
  predicted,
  xMin,
  xMax,
  showPredicted,
}) {
  const manualShapes = (manualPicks || [])
    .filter((p) => p && typeof p.trace === 'number' && p.trace >= xMin && p.trace <= xMax)
    .map((p) => ({
      xref: 'x',
      yref: 'y',
      type: 'line',
      x0: p.trace - 0.4,
      x1: p.trace + 0.4,
      y0: p.time,
      y1: p.time,
      line: { color: 'red', width: 2 },
    }));

  const predictedShapes = (predicted || [])
    .filter((p) => p && typeof p.trace === 'number' && p.trace >= xMin && p.trace <= xMax)
    .map((p) => ({
      xref: 'x',
      yref: 'y',
      type: 'line',
      x0: p.trace - 0.4,
      x1: p.trace + 0.4,
      y0: p.time,
      y1: p.time,
      line: { color: '#1f77b4', width: 5, dash: 'dot' },
    }));

  return [...manualShapes, ...(showPredicted ? predictedShapes : [])];
}

export function buildPickMarkerTraces({
  manualPicks,
  predicted,
  xMin,
  xMax,
  showPredicted,
}) {
  const manualInRange = (manualPicks || []).filter((p) => (
    p &&
    typeof p.trace === 'number' &&
    typeof p.time === 'number' &&
    p.trace >= xMin &&
    p.trace <= xMax
  ));
  const predInRange = (predicted || []).filter((p) => (
    p &&
    typeof p.trace === 'number' &&
    typeof p.time === 'number' &&
    p.trace >= xMin &&
    p.trace <= xMax
  ));

  const manualX = new Float32Array(manualInRange.length);
  const manualY = new Float32Array(manualInRange.length);
  for (let i = 0; i < manualInRange.length; i++) {
    manualX[i] = manualInRange[i].trace;
    manualY[i] = manualInRange[i].time;
  }

  const predX = new Float32Array(predInRange.length);
  const predY = new Float32Array(predInRange.length);
  for (let i = 0; i < predInRange.length; i++) {
    predX[i] = predInRange[i].trace;
    predY[i] = predInRange[i].time;
  }

  const manualTrace = {
    type: 'scattergl',
    mode: 'markers',
    x: manualX,
    y: manualY,
    marker: {
      symbol: 'line-ew',
      color: 'red',
      size: 14,
      line: { color: 'red', width: 2 },
    },
    hoverinfo: 'skip',
    showlegend: false,
    cliponaxis: false,
    meta: { svRole: 'pick', svKind: 'manual' },
  };

  const predTrace = {
    type: 'scattergl',
    mode: 'markers',
    x: predX,
    y: predY,
    marker: {
      symbol: 'line-ew',
      color: '#1f77b4',
      size: 18,
      line: { color: '#1f77b4', width: 4 },
    },
    visible: !!showPredicted,
    hoverinfo: 'skip',
    showlegend: false,
    cliponaxis: false,
    meta: { svRole: 'pick', svKind: 'pred' },
  };

  return [manualTrace, predTrace];
}

export function buildPendingPickMarkerTrace({
  pending,
  yMin,
  yMax,
}) {
  const hiddenTrace = {
    type: 'scatter',
    mode: 'markers',
    x: [],
    y: [],
    visible: false,
    hoverinfo: 'skip',
    showlegend: false,
    cliponaxis: false,
    meta: { svRole: 'pick', svKind: 'pending' },
  };

  if (!pending || typeof pending !== 'object') return hiddenTrace;

  if (
    pending.kind === 'line' &&
    Number.isFinite(pending.trace) &&
    Number.isFinite(pending.time)
  ) {
    return {
      type: 'scatter',
      mode: 'markers',
      x: [pending.trace],
      y: [pending.time],
      marker: {
        symbol: 'diamond-open',
        color: '#f59e0b',
        size: 16,
        line: { color: '#b45309', width: 2 },
      },
      hoverinfo: 'skip',
      showlegend: false,
      cliponaxis: false,
      visible: true,
      meta: { svRole: 'pick', svKind: 'pending' },
    };
  }

  if (
    pending.kind === 'delete-range' &&
    Number.isFinite(pending.trace) &&
    Number.isFinite(yMin) &&
    Number.isFinite(yMax)
  ) {
    return {
      type: 'scatter',
      mode: 'lines',
      x: [pending.trace, pending.trace],
      y: [yMin, yMax],
      line: {
        color: '#b91c1c',
        width: 2,
        dash: 'dash',
      },
      hoverinfo: 'skip',
      showlegend: false,
      cliponaxis: false,
      visible: true,
      meta: { svRole: 'pick', svKind: 'pending' },
    };
  }

  return hiddenTrace;
}
