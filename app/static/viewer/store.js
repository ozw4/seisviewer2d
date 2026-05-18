// /viewer/store.js
export function createStore(initial) {
  let state = { ...initial };
  const subs = new Set();

  return {
    get() { return state; },
    set(next) { state = { ...next }; subs.forEach(fn => fn(state)); },
    patch(patch) { state = { ...state, ...patch }; subs.forEach(fn => fn(state)); },
    subscribe(fn) { subs.add(fn); return () => subs.delete(fn); },
  };
}

function toFiniteInteger(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

export function getActiveFileTargetFromState(state) {
  if (!state || typeof state !== 'object') return null;

  const fileId = String(state.fileId || '').trim();
  if (!fileId) return null;
  if (state.isFileLoaded !== true) return null;

  const key1Byte = toFiniteInteger(state.key1Byte);
  const key2Byte = toFiniteInteger(state.key2Byte);
  if (key1Byte === null || key2Byte === null) return null;

  const label = String(state.displayName || state.fileName || fileId).trim() || fileId;
  return {
    fileId,
    displayName: label,
    key1Byte,
    key2Byte,
  };
}

export function createSeisViewerState(store) {
  return {
    getActiveFileTargetState() {
      const state = store.get();
      return {
        fileId: state.fileId,
        displayName: state.displayName,
        key1Byte: state.key1Byte,
        key2Byte: state.key2Byte,
        isFileLoaded: state.isFileLoaded === true,
      };
    },
    getActiveFileTarget() {
      return getActiveFileTargetFromState(store.get());
    },
  };
}
