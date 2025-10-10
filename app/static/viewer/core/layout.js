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
    const defaultYRange = [(y1 * effectiveDt) + halfYSec, (y0 * effectiveDt) - halfYSec];
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
