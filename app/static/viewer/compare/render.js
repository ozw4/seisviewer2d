(function () {
  'use strict';

  const compareData = window.__svCompareData || {};
  const compareSources = window.__svCompareSources || {};
  const {
    payloadToF32,
    subtractF32,
    rowsFromF32,
    compareHeatmapScale,
  } = compareData;
  const { sourcePairKey } = compareSources;

  for (const helper of [
    payloadToF32,
    subtractF32,
    rowsFromF32,
    compareHeatmapScale,
  ]) {
    if (typeof helper !== 'function') {
      throw new Error('compare/data.js must be loaded before compare/render.js');
    }
  }
  if (typeof sourcePairKey !== 'function') {
    throw new Error('compare/sources.js must be loaded before compare/render.js');
  }

  const AXIS_MARGIN = 0.035;
  const AMP_LIMIT = 3.0;

  function sampleInterval(payload) {
    const dt = Number(payload?.dt);
    if (Number.isFinite(dt) && dt > 0) return dt;
    const fallback = Number(window.defaultDt);
    return Number.isFinite(fallback) && fallback > 0 ? fallback : 0;
  }

  function axisSuffix(index) {
    return index === 0 ? '' : String(index + 1);
  }

  function axisRef(base, index) {
    return index === 0 ? base : `${base}${index + 1}`;
  }

  function axisLayoutName(base, index) {
    return `${base}axis${axisSuffix(index)}`;
  }

  function compareDomains(count) {
    const gapTotal = AXIS_MARGIN * Math.max(0, count - 1);
    const panelWidth = (1 - gapTotal) / count;
    const domains = [];
    let start = 0;
    for (let i = 0; i < count; i++) {
      const end = i === count - 1 ? 1 : start + panelWidth;
      domains.push([start, end]);
      start = end + AXIS_MARGIN;
    }
    return domains;
  }

  function panelTitle(panel) {
    if (panel.kind === 'diff') return `A-B: ${panel.left} - ${panel.right}`;
    return `${panel.role}: ${panel.label}`;
  }

  function buildCompareLayout(options) {
    const {
      render,
      panels,
      xRange = null,
      yRange = null,
      clickmode,
      dragmode,
      uiRevision,
    } = options;
    const domains = compareDomains(panels.length);
    const dt = sampleInterval(render.a.payload);
    const yDefault = [(render.windowInfo.y1 * dt), (render.windowInfo.y0 * dt)];
    const layout = {
      clickmode,
      dragmode,
      uirevision: `${uiRevision}:compare:${sourcePairKey(render.sources)}`,
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
      margin: { t: 38, r: 12, l: 58, b: 42 },
      annotations: [],
      showlegend: false,
    };
    for (let i = 0; i < panels.length; i++) {
      const xName = axisLayoutName('x', i);
      const yName = axisLayoutName('y', i);
      layout[xName] = {
        domain: domains[i],
        title: 'Trace',
        showgrid: false,
        tickfont: { color: '#000' },
        titlefont: { color: '#000' },
        autorange: false,
        range: xRange || [render.windowInfo.x0, render.windowInfo.x1],
      };
      layout[yName] = {
        domain: [0, 1],
        title: i === 0 ? 'Time (s)' : '',
        showgrid: false,
        tickfont: { color: '#000' },
        titlefont: { color: '#000' },
        autorange: false,
        range: yRange || yDefault,
      };
      layout.annotations.push({
        xref: 'paper',
        yref: 'paper',
        x: (domains[i][0] + domains[i][1]) / 2,
        y: 1.06,
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        text: panelTitle(panels[i]),
        font: { size: 13, color: '#111827' },
      });
    }
    return layout;
  }

  function buildCompareWiggleTraces(options) {
    const { panel, axisIndex, render, gain = 1.0 } = options;
    const { rows, cols, x0, stepX, y0, stepY } = render;
    const values = panel.values;
    const dt = sampleInterval(render.a.payload);
    const lineSegLen = rows + 1;
    const fillSegLen = (2 * rows) + 2;
    const baseX = new Float32Array(cols * lineSegLen);
    const baseY = new Float32Array(cols * lineSegLen);
    const lineX = new Float32Array(cols * lineSegLen);
    const lineY = new Float32Array(cols * lineSegLen);
    const fillX = new Float32Array(cols * fillSegLen);
    const fillY = new Float32Array(cols * fillSegLen);
    for (let c = 0; c < cols; c++) {
      const traceIndex = x0 + c * stepX;
      const lineStart = c * lineSegLen;
      const fillStart = c * fillSegLen;
      for (let r = 0; r < rows; r++) {
        const t = (y0 + r * stepY) * dt;
        const idx = r * cols + c;
        let val = values[idx] * gain;
        if (val > AMP_LIMIT) val = AMP_LIMIT;
        if (val < -AMP_LIMIT) val = -AMP_LIMIT;
        const posVal = val < 0 ? 0 : val;
        const lineIdx = lineStart + r;
        const fillBaseIdx = fillStart + r;
        const fillPosIdx = fillStart + rows + (rows - 1 - r);
        baseX[lineIdx] = traceIndex;
        baseY[lineIdx] = t;
        lineX[lineIdx] = traceIndex + val;
        lineY[lineIdx] = t;
        fillX[fillBaseIdx] = traceIndex;
        fillY[fillBaseIdx] = t;
        fillX[fillPosIdx] = traceIndex + posVal;
        fillY[fillPosIdx] = t;
      }
      const lineNanIdx = lineStart + rows;
      baseX[lineNanIdx] = NaN;
      baseY[lineNanIdx] = NaN;
      lineX[lineNanIdx] = NaN;
      lineY[lineNanIdx] = NaN;
      const fillCloseIdx = fillStart + (2 * rows);
      const fillNanIdx = fillCloseIdx + 1;
      fillX[fillCloseIdx] = traceIndex;
      fillY[fillCloseIdx] = (y0 * dt);
      fillX[fillNanIdx] = NaN;
      fillY[fillNanIdx] = NaN;
    }
    const xaxis = axisRef('x', axisIndex);
    const yaxis = axisRef('y', axisIndex);
    return [
      {
        type: 'scatter',
        mode: 'lines',
        x: baseX,
        y: baseY,
        xaxis,
        yaxis,
        line: { width: 0 },
        connectgaps: false,
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: fillX,
        y: fillY,
        xaxis,
        yaxis,
        fill: 'toself',
        fillcolor: 'black',
        line: { width: 0 },
        opacity: 0.6,
        connectgaps: false,
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: lineX,
        y: lineY,
        xaxis,
        yaxis,
        line: { color: 'black', width: 0.5 },
        connectgaps: false,
        hoverinfo: 'x+y',
        showlegend: false,
      },
    ];
  }

  function buildCompareHeatmapTrace(options) {
    const {
      panel,
      axisIndex,
      render,
      gain = 1.0,
      colormapName = 'Greys',
      reverse = false,
      colormaps = window.COLORMAPS,
    } = options;
    const { rows, cols, x0, stepX, y0, stepY } = render;
    const xVals = new Float32Array(cols);
    for (let c = 0; c < cols; c++) xVals[c] = x0 + c * stepX;
    const dt = sampleInterval(render.a.payload);
    const yVals = new Float32Array(rows);
    for (let r = 0; r < rows; r++) yVals[r] = (y0 + r * stepY) * dt;
    const cm = (colormaps && colormaps[colormapName]) || 'Greys';
    const scale = compareHeatmapScale(panel, gain);
    const isDiv = scale.signed && (colormapName === 'RdBu' || colormapName === 'BWR');
    return {
      type: 'heatmap',
      x: xVals,
      y: yVals,
      z: rowsFromF32(panel.values, rows, cols),
      xaxis: axisRef('x', axisIndex),
      yaxis: axisRef('y', axisIndex),
      colorscale: cm,
      reversescale: reverse,
      zmin: scale.zmin,
      zmax: scale.zmax,
      zmid: isDiv ? 0 : null,
      showscale: false,
      hoverinfo: 'x+y',
      hovertemplate: '',
    };
  }

  function buildComparePanels(options) {
    const { render, showDiff = false } = options;
    const panels = [
      {
        kind: 'source',
        role: 'A',
        domain: render.sources.a.domain,
        label: render.sources.a.label,
        values: render.a.values,
      },
      {
        kind: 'source',
        role: 'B',
        domain: render.sources.b.domain,
        label: render.sources.b.label,
        values: render.b.values,
      },
    ];
    if (showDiff && render.diffAvailable && render.diffValues) {
      panels.push({
        kind: 'diff',
        role: 'A-B',
        domain: render.sources.a.domain,
        label: `${render.sources.a.label} - ${render.sources.b.label}`,
        left: render.sources.a.label,
        right: render.sources.b.label,
        values: render.diffValues,
      });
    }
    return panels;
  }

  function buildCompareRender(options) {
    const {
      aPayload,
      bPayload,
      sources,
      decision,
      validation,
      windowInfo,
      scaling,
    } = options;
    const aValues = payloadToF32(aPayload, sources.a);
    const bValues = payloadToF32(bPayload, sources.b);
    if (!aValues || !bValues) {
      return null;
    }
    const rows = Number(aPayload.shape[0]);
    const cols = Number(aPayload.shape[1]);
    const diffValues = validation.ok ? subtractF32(aValues, bValues) : null;
    return {
      key1: aPayload.key1,
      sources,
      sourcePair: sourcePairKey(sources),
      scaling,
      lmoKey: aPayload.lmoKey,
      mode: decision.mode,
      panelCount: decision.panelCount,
      stepX: decision.stepX,
      stepY: decision.stepY,
      x0: aPayload.x0,
      x1: aPayload.x1,
      y0: aPayload.y0,
      y1: aPayload.y1,
      rows,
      cols,
      windowInfo,
      a: { payload: aPayload, values: aValues },
      b: { payload: bPayload, values: bValues },
      diffAvailable: validation.ok && !!diffValues,
      diffMessage: validation.message,
      diffValues,
    };
  }

  function buildCompareUnavailableFigure(message) {
    const text = String(message || 'A-B unavailable.').trim() || 'A-B unavailable.';
    return {
      data: [],
      layout: {
        margin: { t: 38, r: 12, l: 58, b: 42 },
        annotations: [{
          xref: 'paper',
          yref: 'paper',
          x: 0.5,
          y: 0.5,
          text,
          showarrow: false,
          font: { size: 14 },
        }],
        xaxis: { visible: false },
        yaxis: { visible: false },
      },
      config: {
        responsive: true,
        doubleClick: false,
        doubleClickDelay: 300,
      },
    };
  }

  window.__svCompareRender = Object.freeze({
    axisSuffix,
    axisRef,
    axisLayoutName,
    compareDomains,
    panelTitle,
    buildCompareLayout,
    buildCompareWiggleTraces,
    buildCompareHeatmapTrace,
    compareHeatmapScale,
    buildComparePanels,
    buildCompareRender,
    buildCompareUnavailableFigure,
  });
})();
