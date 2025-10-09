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
