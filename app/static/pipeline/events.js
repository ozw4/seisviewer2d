(function () {
  function createPipelineEventBus() {
    const listeners = {};

    function on(eventName, handler) {
      if (typeof handler !== 'function') return;
      (listeners[eventName] ||= new Set()).add(handler);
    }

    function off(eventName, handler) {
      const bucket = listeners[eventName];
      if (!bucket) return;
      if (!handler) {
        bucket.clear();
        return;
      }
      bucket.delete(handler);
    }

    function emit(eventName, payload) {
      const bucket = listeners[eventName];
      if (!bucket) return;
      for (const handler of bucket) {
        try {
          handler(payload);
        } catch (err) {
          console.warn('[pipeline] listener for', eventName, 'threw', err);
        }
      }
    }

    return { on, off, emit };
  }

  window.createPipelineEventBus = createPipelineEventBus;
})();
