export function debounce(fn, wait, { leading = false, trailing = true } = {}) {
  let t = null;
  let lastArgs;
  let lastThis;
  let invoked = false;
  return function debounced(...args) {
    lastArgs = args;
    lastThis = this;
    if (t) clearTimeout(t);
    if (leading && !invoked) {
      fn.apply(lastThis, lastArgs);
      invoked = true;
    }
    t = setTimeout(() => {
      if (trailing) {
        fn.apply(lastThis, lastArgs);
      }
      invoked = false;
      t = null;
    }, wait);
  };
}

export function throttle(fn, wait) {
  let last = 0;
  let t = null;
  let lastArgs;
  let lastThis;
  return function throttled(...args) {
    const now = Date.now();
    lastArgs = args;
    lastThis = this;
    const remaining = wait - (now - last);
    if (remaining <= 0) {
      if (t) {
        clearTimeout(t);
        t = null;
      }
      last = now;
      fn.apply(lastThis, lastArgs);
    } else if (!t) {
      t = setTimeout(() => {
        last = Date.now();
        t = null;
        fn.apply(lastThis, lastArgs);
      }, remaining);
    }
  };
}

export function rafDebounce(fn) {
  let scheduled = false;
  return function (...args) {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      fn.apply(this, args);
    });
  };
}
