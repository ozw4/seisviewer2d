(function () {
  if (window.pipelineIndex && typeof window.pipelineIndex.bootstrap === 'function') {
    window.pipelineIndex.bootstrap();
    return;
  }
  window.__pipelineLegacyPipelineUIBootstrapRequested = true;
})();
