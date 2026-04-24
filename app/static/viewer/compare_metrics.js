(function () {
  function assertFiniteArray(values, name) {
    if (!(values instanceof Float32Array)) {
      throw new Error(`${name} must be Float32Array.`);
    }
    for (let i = 0; i < values.length; i += 1) {
      if (!Number.isFinite(values[i])) {
        throw new Error(`${name} contains non-finite value at index ${i}.`);
      }
    }
  }

  function assertComparableArrays(a, b) {
    assertFiniteArray(a, 'a');
    assertFiniteArray(b, 'b');
    if (a.length !== b.length) {
      throw new Error('Compare inputs must have the same length.');
    }
  }

  function computeDiff(a, b, mode) {
    assertComparableArrays(a, b);
    const len = a.length;
    const diff = new Float32Array(len);
    const subtractBMinusA = mode !== 'a_minus_b';
    for (let i = 0; i < len; i += 1) {
      diff[i] = subtractBMinusA ? (b[i] - a[i]) : (a[i] - b[i]);
    }
    return diff;
  }

  function computeSummaryStats(diff) {
    assertFiniteArray(diff, 'diff');
    const len = diff.length;
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
      const value = diff[i];
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
    assertComparableArrays(a, b);
    const rows = Number(height);
    const cols = Number(width);
    if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(cols) || cols <= 0) {
      throw new Error('RMS shape must be positive integers.');
    }
    if ((rows * cols) !== a.length) {
      throw new Error('RMS shape does not match input length.');
    }
    const out = new Float32Array(cols);

    for (let ix = 0; ix < cols; ix += 1) {
      let sumSq = 0;
      for (let iy = 0; iy < rows; iy += 1) {
        const k = (iy * cols) + ix;
        const d = b[k] - a[k];
        sumSq += d * d;
      }
      out[ix] = Math.sqrt(sumSq / rows);
    }
    return out;
  }

  function percentileAbs(values, percentile) {
    assertFiniteArray(values, 'values');
    const len = values.length;
    if (!len) return 0;

    const pRaw = Number(percentile);
    const p = Number.isFinite(pRaw) ? Math.max(0, Math.min(100, pRaw)) : 99;
    const sampleLimit = 16384;
    const stride = Math.max(1, Math.ceil(len / sampleLimit));
    const sampled = [];
    for (let i = 0; i < len; i += stride) {
      sampled.push(Math.abs(values[i]));
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
