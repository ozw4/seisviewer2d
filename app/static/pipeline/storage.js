(function () {
  function createPipelineStorage(options = {}) {
    const normalizeStep = options.normalizeStep;
    const graphToSpec = options.graphToSpec;
    const specToGraph = options.specToGraph;

    function loadPipelineFromLocalStorage() {
      if (typeof normalizeStep !== 'function' || typeof specToGraph !== 'function') return [];

      try {
        const stored = localStorage.getItem('viewer.pipelineGraph');
        if (stored) {
          const parsed = JSON.parse(stored);
          if (Array.isArray(parsed)) {
            const steps = parsed.map(normalizeStep).filter(Boolean);
            if (steps.length) return steps;
          }
        }
      } catch (e) {
        console.warn('Failed to parse viewer.pipelineGraph', e);
      }

      try {
        const specRaw = localStorage.getItem('viewer.pipelineSpec');
        const tapsRaw = localStorage.getItem('viewer.pipelineTaps');
        if (specRaw) {
          const spec = JSON.parse(specRaw);
          const taps = tapsRaw ? JSON.parse(tapsRaw) : [];
          const steps = specToGraph(spec, taps);
          if (steps.length) return steps;
        }
      } catch (e) {
        console.warn('Failed to parse stored pipeline spec/taps', e);
      }

      return [];
    }

    function savePipelineToLocalStorage(graph) {
      if (typeof graphToSpec !== 'function') return;

      try {
        localStorage.setItem('viewer.pipelineGraph', JSON.stringify(graph));
      } catch (e) {
        console.warn('Failed to persist pipeline graph', e);
      }
      try {
        const converted = graphToSpec(graph);
        const spec = converted && converted.spec ? converted.spec : { steps: [] };
        const taps = converted && Array.isArray(converted.taps) ? converted.taps : [];
        localStorage.setItem('viewer.pipelineSpec', JSON.stringify(spec));
        localStorage.setItem('viewer.pipelineTaps', JSON.stringify(taps));
      } catch (e) {
        console.warn('Failed to persist pipeline spec/taps', e);
      }
    }

    return {
      loadPipelineFromLocalStorage,
      savePipelineToLocalStorage,
    };
  }

  window.createPipelineStorage = createPipelineStorage;
})();
