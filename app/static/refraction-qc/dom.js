export function collectRefractionQcDom() {
  return {
    pipelineTab: document.getElementById('pipelineSidebarTab'),
    staticCorrectionTab: document.getElementById('staticCorrectionSidebarTab'),
    qcTab: document.getElementById('refractionQcSidebarTab'),
    pipelinePanel: document.getElementById('pipelineTabPanel'),
    staticCorrectionPanel: document.getElementById('staticCorrectionTabPanel'),
    qcPanel: document.getElementById('refractionQcTabPanel'),
    form: document.getElementById('refractionQcForm'),
    jobId: document.getElementById('refractionQcJobId'),
    jobList: document.getElementById('refractionQcJobList'),
    maxPoints: document.getElementById('refractionQcMaxPoints'),
    loadButton: document.getElementById('refractionQcLoadButton'),
    jobSummary: document.getElementById('refractionQcJobSummary'),
    status: document.getElementById('refractionQcStatus'),
    error: document.getElementById('refractionQcError'),
    sign: document.getElementById('refractionQcSign'),
    activeFilters: document.getElementById('refractionQcActiveFilters'),
    viewControls: document.getElementById('refractionQcViewControls'),
    taskButtons: Array.from(document.querySelectorAll('.refraction-qc-task-button')),
    inspector: document.getElementById('refractionQcInspector'),
    viewButtonsContainer: document.getElementById('refractionQcViewButtons'),
    viewButtons: Array.from(document.querySelectorAll('.refraction-qc-view-button')),
    viewPanels: Array.from(document.querySelectorAll('.refraction-qc-view')),
    viewContents: new Map(Array.from(document.querySelectorAll('[data-view-content]')).map(
      (node) => [node.dataset.viewContent, node],
    )),
  };
}

export function isStandaloneRefractionQcPage() {
  return Boolean(document.body && document.body.classList.contains('refraction-qc-page'));
}
