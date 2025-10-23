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

    var range = bl.trace_index_map && bl.trace_index_map[String(key1Val)];
    if (!Array.isArray(range) || range.length !== 2) throw new Error('trace range missing for selected key1');

    var muSec = Array.isArray(bl.mu_section_by_key1) ? bl.mu_section_by_key1[idx] : null;
    var sgSec = Array.isArray(bl.sigma_section_by_key1) ? bl.sigma_section_by_key1[idx] : null;
    return { start: range[0] | 0, stop: range[1] | 0, muSec: muSec, sgSec: sgSec };
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
    var globalIdx = (ss2.start | 0) + (traceIndexInSection | 0);
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
    var st = x0 | 0, sx = (stepX | 0) || 1;
    for (var c = 0; c < cols; c++) {
      var global = (ss.start | 0) + (st + c * sx);
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
