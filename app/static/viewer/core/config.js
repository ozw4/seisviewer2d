export const cfg = {
  PREFETCH_WIDTH: 3,
  FALLBACK_MAX: 8,
  HARD_LIMIT_BYTES: 512 * 1024 * 1024,
  WINDOW_FETCH_DEBOUNCE_MS: 120,
  FETCH_DEBOUNCE_MS: 200,
  WINDOW_MAX_POINTS: 5_000_000,
  LS_KEYS: { DT: 'segy.dt', WIGGLE_DENSITY: 'wiggle_density', DRAG_BASE: 'drag_base' },

  getDefaultDt() {
    const fallback = 0.002;
    try {
      const stored = localStorage.getItem(this.LS_KEYS.DT);
      if (stored != null) {
        const parsed = parseFloat(stored);
        if (Number.isFinite(parsed) && parsed > 0) {
          return parsed;
        }
      }
    } catch (_) { /* ignore */ }
    return fallback;
  },

  setDefaultDt(v) {
    if (Number.isFinite(v) && v > 0) {
      try {
        localStorage.setItem(this.LS_KEYS.DT, String(v));
      } catch (_) { /* ignore */ }
      return v;
    }
    return this.getDefaultDt();
  },

  getWiggleDensity() {
    const fallback = 0.20;
    const min = 0.02;
    const max = 0.30;
    try {
      const stored = localStorage.getItem(this.LS_KEYS.WIGGLE_DENSITY);
      if (stored != null) {
        const parsed = parseFloat(stored);
        if (Number.isFinite(parsed)) {
          return Math.min(max, Math.max(min, parsed));
        }
      }
    } catch (_) { /* ignore */ }
    return fallback;
  },

  setWiggleDensity(v) {
    const min = 0.02;
    const max = 0.30;
    let next = Number(v);
    if (!Number.isFinite(next)) {
      next = this.getWiggleDensity();
    }
    next = Math.min(max, Math.max(min, next));
    try {
      localStorage.setItem(this.LS_KEYS.WIGGLE_DENSITY, String(next));
    } catch (_) { /* ignore */ }
    return next;
  },

  get limits() {
    return {
      HARD_LIMIT_BYTES: this.HARD_LIMIT_BYTES,
      WINDOW_MAX_POINTS: this.WINDOW_MAX_POINTS,
    };
  },
};
