(function () {
  function computeDiff(a, b, mode) {
    const len = Math.min(a?.length || 0, b?.length || 0);
    const diff = new Float32Array(len);
    const subtractBMinusA = mode !== 'a_minus_b';
    for (let i = 0; i < len; i += 1) {
      diff[i] = subtractBMinusA ? (b[i] - a[i]) : (a[i] - b[i]);
    }
    return diff;
  }

  function computeSummaryStats(diff) {
    const len = diff?.length || 0;
    if (!len) {
      return {
        mean: 0,
        std: 0,
        rms: 0,
        maxAbs: 0,
      };
    }

    let sum = 0;
    let sumSq = 0;
    let maxAbs = 0;
    for (let i = 0; i < len; i += 1) {
      const value = Number(diff[i]) || 0;
      const absValue = Math.abs(value);
      sum += value;
      sumSq += value * value;
      if (absValue > maxAbs) maxAbs = absValue;
    }

    const mean = sum / len;
    const variance = Math.max(0, (sumSq / len) - (mean * mean));
    return {
      mean,
      std: Math.sqrt(variance),
      rms: Math.sqrt(sumSq / len),
      maxAbs,
    };
  }

  function computeRmsByTrace(a, b, height, width) {
    const rows = Math.max(0, Number(height) | 0);
    const cols = Math.max(0, Number(width) | 0);
    const out = new Float32Array(cols);
    if (!rows || !cols) return out;

    for (let ix = 0; ix < cols; ix += 1) {
      let sumSq = 0;
      for (let iy = 0; iy < rows; iy += 1) {
        const k = (iy * cols) + ix;
        const d = (Number(b[k]) || 0) - (Number(a[k]) || 0);
        sumSq += d * d;
      }
      out[ix] = Math.sqrt(sumSq / rows);
    }
    return out;
  }

  function percentileAbs(values, percentile) {
    const len = values?.length || 0;
    if (!len) return 0;

    const pRaw = Number(percentile);
    const p = Number.isFinite(pRaw) ? Math.max(0, Math.min(100, pRaw)) : 99;
    const sampleLimit = 16384;
    const stride = Math.max(1, Math.ceil(len / sampleLimit));
    const sampled = [];
    for (let i = 0; i < len; i += stride) {
      sampled.push(Math.abs(Number(values[i]) || 0));
    }
    if (!sampled.length) return 0;
    sampled.sort((a, b) => a - b);
    const rank = Math.min(sampled.length - 1, Math.floor((sampled.length - 1) * (p / 100)));
    return sampled[rank];
  }

  window.compareMetrics = {
    computeDiff,
    computeSummaryStats,
    computeRmsByTrace,
    percentileAbs,
  };
})();
