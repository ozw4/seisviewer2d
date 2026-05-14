(function () {
  const DEFAULTS = {
    key1Byte: '189',
    key2Byte: '193',
    pickKind: 'batch_predicted_npz',
    pickArtifactName: 'predicted_picks_time_s.npz',
  };
  const ARTIFACT_PICK_KINDS = new Set(['batch_predicted_npz', 'manual_npz_artifact']);

  const state = {
    ready: false,
    message: [
      'Enter a SEG-Y/TraceStore file_id and a first-break pick artifact usable by refraction statics.',
      'Static correction job submission is not enabled in this milestone.',
    ].join(' '),
    error: '',
    loadingPickArtifacts: false,
    pickArtifacts: [],
    lastRequest: null,
  };

  let dom = null;

  function trimValue(value) {
    return String(value || '').trim();
  }

  function setDefaultValue(element, value) {
    if (element && trimValue(element.value) === '') {
      element.value = value;
    }
  }

  function isLikelyPickArtifact(name) {
    const normalized = trimValue(name);
    return (
      normalized === DEFAULTS.pickArtifactName
      || /^manual_picks_time.*\.npz$/i.test(normalized)
    );
  }

  function collectInputs() {
    if (!dom) return null;
    return {
      file_id: trimValue(dom.fileId.value),
      key1_byte: trimValue(dom.key1Byte.value),
      key2_byte: trimValue(dom.key2Byte.value),
      pick_source: {
        kind: trimValue(dom.pickKind.value),
        job_id: trimValue(dom.pickJobId.value),
        artifact_name: trimValue(dom.pickArtifactName.value),
      },
    };
  }

  function parsePositiveInteger(value, label, errors) {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      errors.push(`${label} must be a positive integer.`);
      return null;
    }
    return parsed;
  }

  function buildStaticCorrectionRequest() {
    const values = collectInputs();
    const errors = [];
    if (!values) {
      return { payload: null, errors: ['Static correction form is not available.'] };
    }

    if (!values.file_id) {
      errors.push('file_id is required.');
    }
    const key1Byte = parsePositiveInteger(values.key1_byte, 'key1_byte', errors);
    const key2Byte = parsePositiveInteger(values.key2_byte, 'key2_byte', errors);
    const pickKind = values.pick_source.kind;
    if (!pickKind) {
      errors.push('pick_source.kind is required.');
    }
    if (ARTIFACT_PICK_KINDS.has(pickKind)) {
      if (!values.pick_source.job_id) {
        errors.push('pick_source.job_id is required for artifact-backed pick sources.');
      }
      if (!values.pick_source.artifact_name) {
        errors.push('pick_source.artifact_name is required for artifact-backed pick sources.');
      }
    }
    if (
      values.pick_source.artifact_name
      && !values.pick_source.artifact_name.toLowerCase().endsWith('.npz')
    ) {
      errors.push('pick_source.artifact_name must be an .npz artifact.');
    }

    if (errors.length) {
      return { payload: null, errors };
    }
    return {
      payload: {
        file_id: values.file_id,
        key1_byte: key1Byte,
        key2_byte: key2Byte,
        pick_source: values.pick_source,
      },
      errors,
    };
  }

  function sortedArtifactFiles(files) {
    return [...files].sort((left, right) => {
      const leftName = trimValue(left && left.name ? left.name : left);
      const rightName = trimValue(right && right.name ? right.name : right);
      const likelihood = Number(isLikelyPickArtifact(rightName)) - Number(isLikelyPickArtifact(leftName));
      return likelihood || leftName.localeCompare(rightName);
    });
  }

  async function readResponseError(response) {
    let detail = '';
    try {
      const contentType = response.headers && response.headers.get
        ? response.headers.get('content-type') || ''
        : '';
      if (contentType.includes('application/json')) {
        const payload = await response.json();
        detail = payload && typeof payload.detail === 'string'
          ? payload.detail
          : JSON.stringify(payload);
      } else {
        detail = await response.text();
      }
    } catch {
      detail = '';
    }
    return `batch job files ${response.status}${detail ? `: ${detail}` : ''}`;
  }

  function renderPickArtifactList() {
    if (!dom || !dom.pickArtifactList) return;
    dom.pickArtifactList.innerHTML = '';

    if (state.loadingPickArtifacts) {
      dom.pickArtifactList.hidden = false;
      dom.pickArtifactList.textContent = 'Loading pick artifacts...';
      return;
    }

    if (!state.pickArtifacts.length) {
      dom.pickArtifactList.hidden = true;
      return;
    }

    const list = document.createElement('ul');
    for (const file of state.pickArtifacts) {
      const name = trimValue(file && file.name ? file.name : file);
      if (!name) continue;
      const item = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = name;
      button.dataset.artifactName = name;
      button.dataset.testid = `static-correction-pick-artifact-${name}`;
      if (isLikelyPickArtifact(name)) {
        button.classList.add('is-likely');
      }
      button.addEventListener('click', () => {
        dom.pickArtifactName.value = name;
        state.error = '';
        state.message = `Selected pick artifact ${name}.`;
        render();
      });
      item.appendChild(button);
      if (isLikelyPickArtifact(name)) {
        const tag = document.createElement('span');
        tag.className = 'static-correction-artifact-tag';
        tag.textContent = 'first-break candidate';
        item.appendChild(tag);
      }
      list.appendChild(item);
    }
    dom.pickArtifactList.hidden = list.childNodes.length === 0;
    if (!dom.pickArtifactList.hidden) {
      dom.pickArtifactList.appendChild(list);
    }
  }

  function render() {
    if (!dom) return;
    dom.status.textContent = state.message;
    dom.error.hidden = !state.error;
    dom.error.textContent = state.error;
    dom.runButton.disabled = false;
    if (dom.loadPickArtifactsButton) {
      dom.loadPickArtifactsButton.disabled = state.loadingPickArtifacts;
    }
    renderPickArtifactList();
  }

  async function loadPickArtifacts() {
    if (!dom) return;
    const jobId = trimValue(dom.pickJobId.value);
    if (!jobId) {
      state.error = 'pick_source.job_id is required before loading pick artifacts.';
      state.message = 'Enter the batch job ID that produced the first-break pick artifact.';
      state.pickArtifacts = [];
      render();
      return;
    }

    state.loadingPickArtifacts = true;
    state.error = '';
    state.message = 'Loading pick artifacts...';
    state.pickArtifacts = [];
    render();

    try {
      const response = await fetch(`/batch/job/${encodeURIComponent(jobId)}/files`);
      if (!response.ok) {
        throw new Error(await readResponseError(response));
      }
      const payload = await response.json();
      const files = Array.isArray(payload.files) ? payload.files : [];
      state.pickArtifacts = sortedArtifactFiles(files);
      state.message = files.length
        ? `Loaded ${files.length} artifact file${files.length === 1 ? '' : 's'} for ${jobId}.`
        : `No artifact files returned for ${jobId}.`;
    } catch (error) {
      state.pickArtifacts = [];
      state.error = error instanceof Error ? error.message : String(error);
      state.message = 'Unable to load pick artifacts.';
    } finally {
      state.loadingPickArtifacts = false;
      render();
    }
  }

  function handleRun(event) {
    if (event) {
      event.preventDefault();
    }
    const { payload, errors } = buildStaticCorrectionRequest();
    state.lastRequest = payload;
    if (errors.length) {
      state.ready = false;
      state.error = errors.join(' ');
      state.message = 'Fix input errors before running refraction statics.';
      render();
      return;
    }

    state.ready = true;
    state.error = '';
    state.message = 'Inputs are valid. Static correction job submission is not enabled in this milestone.';
    render();
  }

  function init() {
    const form = document.getElementById('staticCorrectionForm');
    const status = document.getElementById('staticCorrectionStatus');
    const error = document.getElementById('staticCorrectionError');
    const runButton = document.getElementById('staticCorrectionRunButton');
    const fileId = document.getElementById('staticCorrectionFileId');
    const key1Byte = document.getElementById('staticCorrectionKey1Byte');
    const key2Byte = document.getElementById('staticCorrectionKey2Byte');
    const pickKind = document.getElementById('staticCorrectionPickKind');
    const pickJobId = document.getElementById('staticCorrectionPickJobId');
    const pickArtifactName = document.getElementById('staticCorrectionPickArtifactName');
    const loadPickArtifactsButton = document.getElementById('staticCorrectionLoadPickArtifactsButton');
    const pickArtifactList = document.getElementById('staticCorrectionPickArtifactList');
    if (
      !form || !status || !error || !runButton || !fileId || !key1Byte || !key2Byte
      || !pickKind || !pickJobId || !pickArtifactName || !loadPickArtifactsButton
      || !pickArtifactList
    ) {
      return;
    }

    setDefaultValue(key1Byte, DEFAULTS.key1Byte);
    setDefaultValue(key2Byte, DEFAULTS.key2Byte);
    setDefaultValue(pickKind, DEFAULTS.pickKind);
    setDefaultValue(pickArtifactName, DEFAULTS.pickArtifactName);

    dom = {
      form,
      status,
      error,
      runButton,
      fileId,
      key1Byte,
      key2Byte,
      pickKind,
      pickJobId,
      pickArtifactName,
      loadPickArtifactsButton,
      pickArtifactList,
    };

    form.addEventListener('submit', handleRun);
    runButton.addEventListener('click', handleRun);
    loadPickArtifactsButton.addEventListener('click', (event) => {
      event.preventDefault();
      loadPickArtifacts();
    });

    render();
  }

  window.refractionStaticRunState = state;
  window.refractionStaticRunUI = {
    buildStaticCorrectionRequest,
    collectInputs,
    isLikelyPickArtifact,
    loadPickArtifacts,
    render,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
