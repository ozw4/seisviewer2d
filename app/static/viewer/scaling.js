(function () {
  var ALLOWED = { section: 1, trace: 1 };

  function _mode() {
    var el = document.getElementById('scale_mode');
    var v = el ? el.value : 'section';
    if (!ALLOWED[v]) v = 'section';
    return v;
  }

  function Scale_onModeChange() {
    var el = document.getElementById('scale_mode');
    var v = el ? el.value : 'section';
    if (!ALLOWED[v]) v = 'section';
    localStorage.setItem('scale_mode', v);
    if (typeof window.renderLatestView === 'function') window.renderLatestView();
  }

  async function Scale_fetchBaselineForCurrentSection() {
    var elSlider = document.getElementById('key1_idx_slider');
    var idx = elSlider ? parseInt(elSlider.value, 10) : 0;
    var key1Val = Array.isArray(window.key1Values) ? window.key1Values[idx] : null;
    if (key1Val == null) throw new Error('key1 value is not available');

    var q = new URLSearchParams({
      file_id: window.currentFileId,
      baseline: 'raw',
      key1_idx: String(key1Val),
      key1_byte: String(window.currentKey1Byte),
      key2_byte: String(window.currentKey2Byte)
    });
    var res = await fetch('/get_section_stats?' + q.toString());
    if (!res.ok) throw new Error('fetch baseline failed: ' + res.status);
    var j = await res.json();
    window.currentBaseline = j;
    return j;
  }

  function _sectionStats() {
    var bl = window.currentBaseline;
    var elSlider = document.getElementById('key1_idx_slider');
    if (!bl) throw new Error('baseline is not loaded');
    var idx = elSlider ? parseInt(elSlider.value, 10) : 0;
    var key1Val = Array.isArray(window.key1Values) ? window.key1Values[idx] : null;
    if (key1Val == null) throw new Error('key1 value is not available');

    var key1Str = String(key1Val);
    var spans = null;
    var selected = bl.selected_key1;
    if (selected && Number(selected.key1_value) === Number(key1Val) && Array.isArray(selected.trace_spans)) {
      spans = selected.trace_spans;
    }
    if (!Array.isArray(spans) || spans.length === 0) {
      var spanMap = bl.trace_spans_by_key1;
      if (spanMap && typeof spanMap === 'object') spans = spanMap[key1Str];
    }
    if (!Array.isArray(spans) || spans.length === 0) {
      var legacyRange = null;
      if (selected && Array.isArray(selected.trace_range) && selected.trace_range.length === 2) {
        legacyRange = selected.trace_range;
      } else if (
        bl.trace_index_map &&
        typeof bl.trace_index_map === 'object' &&
        Array.isArray(bl.trace_index_map[key1Str]) &&
        bl.trace_index_map[key1Str].length === 2
      ) {
        legacyRange = bl.trace_index_map[key1Str];
      }
      if (legacyRange) spans = [legacyRange];
    }

    if (!Array.isArray(spans) || spans.length === 0) {
      throw new Error('trace spans missing for selected key1=' + key1Str);
    }

    var normalized = [];
    for (var i = 0; i < spans.length; i++) {
      var raw = spans[i];
      if (!Array.isArray(raw) || raw.length !== 2) continue;
      var start = raw[0];
      var stop = raw[1];
      if (!Number.isFinite(start) || !Number.isFinite(stop)) continue;
      start = start | 0;
      stop = stop | 0;
      if (stop < start) {
        var tmp = start;
        start = stop;
        stop = tmp;
      }
      if (stop === start) continue;
      normalized.push([start, stop]);
    }

    if (!normalized.length) {
      throw new Error('trace spans empty after normalization for key1=' + key1Str);
    }

    var spanOffsets = new Array(normalized.length);
    var spanLengths = new Array(normalized.length);
    var total = 0;
    for (var j = 0; j < normalized.length; j++) {
      var span = normalized[j];
      spanOffsets[j] = total;
      var len = (span[1] | 0) - (span[0] | 0);
      spanLengths[j] = len;
      total += len;
    }
    if (!(total > 0)) throw new Error('section trace span total length invalid for key1=' + key1Str);

    var muSec = Array.isArray(bl.mu_section_by_key1) ? bl.mu_section_by_key1[idx] : null;
    var sgSec = Array.isArray(bl.sigma_section_by_key1) ? bl.sigma_section_by_key1[idx] : null;
    return {
      spans: normalized,
      spanOffsets: spanOffsets,
      spanLengths: spanLengths,
      traceCount: total,
      muSec: muSec,
      sgSec: sgSec
    };
  }

  function _resolveGlobalTraceIndex(sectionStats, localIndex) {
    if (!sectionStats) throw new Error('section stats missing');
    var spans = sectionStats.spans;
    var offsets = sectionStats.spanOffsets;
    var lengths = sectionStats.spanLengths;
    if (!Array.isArray(spans) || !Array.isArray(offsets) || !Array.isArray(lengths)) {
      throw new Error('section trace spans not available');
    }
    var local = Number(localIndex);
    if (!Number.isFinite(local)) local = 0;
    local = local | 0;
    if (local < 0) local = 0;
    var total = sectionStats.traceCount | 0;
    if (!(total > 0)) throw new Error('section trace count invalid');
    if (local >= total) local = total - 1;
    for (var i = 0; i < spans.length; i++) {
      var offset = offsets[i] | 0;
      var len = lengths[i] | 0;
      if (local < offset) continue;
      if (local < offset + len) {
        return (spans[i][0] | 0) + (local - offset);
      }
    }
    var last = spans[spans.length - 1];
    return (last[1] | 0) - 1;
  }

  function Scale_value(val, traceIndexInSection) {
    var bl = window.currentBaseline;
    if (!bl) throw new Error('baseline is not loaded');
    var mode = _mode();
    var EPS = 1e-12;

    if (mode === 'section') {
      var ss = _sectionStats();
      if (!Number.isFinite(ss.muSec) || !Number.isFinite(ss.sgSec)) throw new Error('section stats not finite');
      var sig = Math.abs(ss.sgSec) < EPS ? 1 : ss.sgSec;
      return (val - ss.muSec) / sig;
    }

    // mode === 'trace'
    var ss2 = _sectionStats();
    var globalIdx = _resolveGlobalTraceIndex(ss2, traceIndexInSection);
    var muArr = bl.mu_traces, sgArr = bl.sigma_traces, zm = bl.zero_var_mask;
    var mu = Array.isArray(muArr) ? muArr[globalIdx] : null;
    var sg = Array.isArray(sgArr) ? sgArr[globalIdx] : null;
    var zero = Array.isArray(zm) ? !!zm[globalIdx] : false;
    var sig2 = (!Number.isFinite(sg) || sg === 0 || zero) ? 1 : sg;
    var m = Number.isFinite(mu) ? mu : 0;
    return (val - m) / sig2;
  }

  function Scale_applyToZ(zRows, x0, stepX) {
    var bl = window.currentBaseline;
    if (!bl) throw new Error('baseline is not loaded');

    var mode = _mode();
    var rows = zRows.length;
    var cols = rows ? zRows[0].length : 0;
    var ss = _sectionStats();

    if (mode === 'section') {
      var EPS = 1e-12;
      var mu = ss.muSec, sg = ss.sgSec;
      if (!Number.isFinite(mu) || !Number.isFinite(sg)) throw new Error('section stats not finite');
      var inv = 1 / (Math.abs(sg) < EPS ? 1 : sg);
      for (var r = 0; r < rows; r++) {
        var row = zRows[r];
        for (var c = 0; c < cols; c++) row[c] = (row[c] - mu) * inv;
      }
      return;
    }

    // mode === 'trace'
    var muArr = bl.mu_traces, sgArr = bl.sigma_traces, zm = bl.zero_var_mask;
    if (!Array.isArray(muArr) || !Array.isArray(sgArr)) throw new Error('trace stats missing');
    var st = x0 | 0,
      sx = (stepX | 0) || 1;
    for (var c = 0; c < cols; c++) {
      var localIdx = st + c * sx;
      var global = _resolveGlobalTraceIndex(ss, localIdx);
      var mu = muArr[global], sg = sgArr[global];
      var zero = Array.isArray(zm) ? !!zm[global] : false;
      var inv = (!Number.isFinite(sg) || sg === 0 || zero) ? 1 : (1 / sg);
      var m = Number.isFinite(mu) ? mu : 0;
      for (var r = 0; r < rows; r++) {
        var row = zRows[r];
        row[c] = (row[c] - m) * inv;
      }
    }
  }

  window.Scale_onModeChange = Scale_onModeChange;
  window.Scale_fetchBaselineForCurrentSection = Scale_fetchBaselineForCurrentSection;
  window.Scale_applyToZ = Scale_applyToZ;
  window.Scale_value = Scale_value;
})();
