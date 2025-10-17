// /static/viewer/settings/prefs.js
// Centralized preferences <-> DOM <-> localStorage
export const PREFS = {
  'gain':           { sel: '#gain',           type: 'number',  def: 1 },
  'colormap':       { sel: '#colormap',       type: 'string',  def: 'Greys' },
  'cmReverse':      { sel: '#cmReverse',      type: 'bool',    def: false },
  'wiggle_density': { sel: '#wiggle_density', type: 'number',  def: 0.20 },
  'sigma_ms_max':   { sel: '#sigma_ms_max',   type: 'number',  def: 20 },
  'pick_method':    { sel: '#pick_method',    type: 'string',  def: 'argmax' },
  'snap_mode':      { sel: '#snap_mode',      type: 'string',  def: 'none' },
  'snap_ms':        { sel: '#snap_ms',        type: 'number',  def: 4 },
  'snap_refine':    { sel: '#snap_refine',    type: 'string',  def: 'none' },
  'showFbPred':     { sel: '#showFbPred',     type: 'bool',    def: false },
  // server-provided dt is persisted here by existing code; no bound control
  'segy.dt':        { sel: null,              type: 'number',  def: 0.002 },
};

function parseByType(t, v) {
  if (v == null) return null;
  if (t === 'bool')   return v === true || v === 'true';
  if (t === 'number') { const x = parseFloat(v); return Number.isFinite(x) ? x : null; }
  return String(v);
}
function formatByType(t, v) {
  if (t === 'bool')   return v ? 'true' : 'false';
  if (t === 'number') return String(v);
  return String(v);
}

export function getPref(key) {
  const spec = PREFS[key];
  if (!spec) throw new Error(`Unknown pref: ${key}`);
  const raw = localStorage.getItem(key);
  const parsed = parseByType(spec.type, raw);
  return parsed == null ? spec.def : parsed;
}

export function setPref(key, value, { applyDom = true, emit = true } = {}) {
  const spec = PREFS[key];
  if (!spec) throw new Error(`Unknown pref: ${key}`);
  localStorage.setItem(key, formatByType(spec.type, value));
  if (applyDom && spec.sel) applyDomValue(spec, value);
  if (emit && typeof window.dispatchEvent === 'function') {
    window.dispatchEvent(new CustomEvent('prefs:change', { detail: { key, value } }));
  }
}

function applyDomValue(spec, value) {
  const el = spec.sel ? document.querySelector(spec.sel) : null;
  if (!el) return;
  if (spec.type === 'bool') el.checked = !!value;
  else el.value = String(value);
}

function bindControl(key, spec, onChange) {
  if (!spec.sel) return;
  const el = document.querySelector(spec.sel);
  if (!el) return;
  const evt = (el.tagName === 'SELECT' || el.type === 'checkbox') ? 'change' : 'input';
  el.addEventListener(evt, () => {
    const next =
      spec.type === 'bool'    ? !!el.checked :
      spec.type === 'number'  ? parseFloat(el.value) :
                                String(el.value);
    setPref(key, next, { applyDom: false, emit: false });
    if (onChange) onChange(key, next);
  });
}

// Initialize: load all prefs, apply to DOM, and attach listeners
export function initPrefs({ onChange } = {}) {
  const snapshot = {};
  for (const key of Object.keys(PREFS)) {
    const spec = PREFS[key];
    const val = getPref(key);
    snapshot[key] = val;
    if (spec.sel) applyDomValue(spec, val);
    bindControl(key, spec, onChange);
  }
  // Also allow programmatic updates to notify
  if (onChange) {
    window.addEventListener('prefs:change', (ev) => {
      const { key, value } = ev.detail || {};
      if (key in PREFS) onChange(key, value);
    });
  }
  return snapshot;
}
