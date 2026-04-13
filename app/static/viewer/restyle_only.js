    (function enableRestyleOnly() {
      let raf = 0;
      function heatmapTraceIndex(plotDiv) {
        const data = plotDiv && Array.isArray(plotDiv.data) ? plotDiv.data : null;
        if (!data) return -1;
        for (let i = 0; i < data.length; i++) {
          const tr = data[i];
          if (tr && tr.type === 'heatmap') return i;
        }
        return -1;
      }
      window.heatmapTraceIndex = heatmapTraceIndex;

      window.restyleColorAndGain = function () {
        const plotDiv = document.getElementById('plot');
        if (!plotDiv) throw new Error('plot div not found');
        const idx = heatmapTraceIndex(plotDiv);
        if (idx < 0) {
            // ウィグルのみ: 幾何にgainを掛け直すため再描画へ
              renderLatestView();
            return;
          }
        const gainEl = document.getElementById('gain');
        const gain = parseFloat(gainEl && gainEl.value) || 1.0;
        const cmSelect = document.getElementById('colormap');
        const cmName = (cmSelect && cmSelect.value) || 'Greys';
        const reverse = !!(document.getElementById('cmReverse') && document.getElementById('cmReverse').checked);

        const AMP_LIMIT = 3.0;
        const fbMode = !!(window.latestWindowRender && window.latestWindowRender.effectiveLayer === 'fbprob');
        const g = Math.max(gain, 1e-9);
        const zmin = fbMode ? 0 : -AMP_LIMIT / g;
        const zmax = fbMode ? 255 : AMP_LIMIT / g;
        const cm = (window.COLORMAPS && window.COLORMAPS[cmName]) || 'Greys';
        const isDiv = !fbMode && (cmName === 'RdBu' || cmName === 'BWR');
        const props = { colorscale: [cm], reversescale: [reverse], zmin: [zmin], zmax: [zmax] };
        if (isDiv) props.zmid = [0];   // 発散CMは中央0を維持
        Plotly.restyle(plotDiv, props, [idx]);
      };

      window.scheduleRestyle = function () {
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(function () {
          raf = 0;
          window.restyleColorAndGain();
        });
      };
    })();
