const POLL_INTERVAL_MS = 1000;
const LOCAL_STORAGE_KEYS = {
  fileId: 'file_id',
  key1Byte: 'key1_byte',
  key2Byte: 'key2_byte',
  fbpickModel: 'fbpick_model_id',
};

function toFiniteNumber(value, fieldName) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    throw new Error(`${fieldName} must be a finite number`);
  }
  return num;
}

function toFiniteInteger(value, fieldName) {
  const num = Number(value);
  if (!Number.isFinite(num) || !Number.isInteger(num)) {
    throw new Error(`${fieldName} must be an integer`);
  }
  return num;
}

function normalizeSigmaMsMax(value) {
  if (value === null || value === undefined || String(value).trim() === '') {
    return null;
  }
  const sigma = toFiniteNumber(value, 'sigma_ms_max');
  if (sigma <= 0) {
    throw new Error('sigma_ms_max must be > 0 when provided');
  }
  return sigma;
}

export function buildBatchApplyRequest(state) {
  const fileId = String(state.fileId || '').trim();
  if (!fileId) {
    throw new Error('file_id is required');
  }

  const key1Byte = toFiniteInteger(state.key1Byte, 'key1_byte');
  const key2Byte = toFiniteInteger(state.key2Byte, 'key2_byte');
  const steps = [];

  if (state.enableBandpass) {
    const lowHz = toFiniteNumber(state.bandpass.lowHz, 'bandpass.low_hz');
    const highHz = toFiniteNumber(state.bandpass.highHz, 'bandpass.high_hz');
    const taper = toFiniteNumber(state.bandpass.taper, 'bandpass.taper');
    if (lowHz < 0 || highHz <= 0 || taper < 0) {
      throw new Error('Bandpass params must satisfy low_hz>=0, high_hz>0, taper>=0');
    }
    if (lowHz >= highHz) {
      throw new Error('bandpass.low_hz must be less than bandpass.high_hz');
    }
    steps.push({
      kind: 'transform',
      name: 'bandpass',
      params: {
        low_hz: lowHz,
        high_hz: highHz,
        taper,
      },
    });
  }

  if (state.enableDenoise) {
    const chunkH = toFiniteInteger(state.denoise.chunkH, 'denoise.chunk_h');
    const overlap = toFiniteInteger(state.denoise.overlap, 'denoise.overlap');
    const maskRatio = toFiniteNumber(state.denoise.maskRatio, 'denoise.mask_ratio');
    const noiseStd = toFiniteNumber(state.denoise.noiseStd, 'denoise.noise_std');
    const passesBatch = toFiniteInteger(
      state.denoise.passesBatch,
      'denoise.passes_batch'
    );
    const maskNoiseMode = String(state.denoise.maskNoiseMode || '').trim();
    if (chunkH < 1 || overlap < 0 || passesBatch < 1) {
      throw new Error('Denoise params must satisfy chunk_h>=1, overlap>=0, passes_batch>=1');
    }
    if (maskRatio < 0 || maskRatio > 1) {
      throw new Error('denoise.mask_ratio must be within [0, 1]');
    }
    if (noiseStd < 0) {
      throw new Error('denoise.noise_std must be >= 0');
    }
    if (!['replace', 'add'].includes(maskNoiseMode)) {
      throw new Error('denoise.mask_noise_mode must be replace or add');
    }
    steps.push({
      kind: 'transform',
      name: 'denoise',
      params: {
        chunk_h: chunkH,
        overlap,
        mask_ratio: maskRatio,
        noise_std: noiseStd,
        mask_noise_mode: maskNoiseMode,
        passes_batch: passesBatch,
      },
    });
  }

  if (state.enableFbpick) {
    const params = {};
    const modelId = String(state.fbpick.modelId || '').trim();
    if (modelId) {
      params.model_id = modelId;
    }
    steps.push({
      kind: 'analyzer',
      name: 'fbpick',
      params,
    });
  }

  if (steps.length === 0) {
    throw new Error('At least one pipeline step must be enabled');
  }

  const method = String(state.pick.method || '').trim();
  if (!['expectation', 'argmax'].includes(method)) {
    throw new Error('pick_options.method must be expectation or argmax');
  }
  const snapMode = String(state.pick.snapMode || '').trim();
  if (!['peak', 'trough', 'rise'].includes(snapMode)) {
    throw new Error('pick_options.snap.mode is invalid');
  }
  const snapRefine = String(state.pick.snapRefine || '').trim();
  if (!['none', 'parabolic', 'zc'].includes(snapRefine)) {
    throw new Error('pick_options.snap.refine is invalid');
  }
  const snapWindowMs = toFiniteNumber(
    state.pick.snapWindowMs,
    'pick_options.snap.window_ms'
  );
  if (snapWindowMs < 0) {
    throw new Error('pick_options.snap.window_ms must be >= 0');
  }

  const payload = {
    file_id: fileId,
    key1_byte: key1Byte,
    key2_byte: key2Byte,
    pipeline_spec: { steps },
    pick_options: {
      method,
      subsample: Boolean(state.pick.subsample),
      sigma_ms_max: normalizeSigmaMsMax(state.pick.sigmaMsMax),
      snap: {
        enabled: Boolean(state.pick.snapEnabled),
        mode: snapMode,
        refine: snapRefine,
        window_ms: snapWindowMs,
      },
    },
    save_picks: state.enableFbpick ? Boolean(state.savePicks) : false,
  };

  return payload;
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) {
    return '-';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  let amount = value;
  let idx = 0;
  while (amount >= 1024 && idx < units.length - 1) {
    amount /= 1024;
    idx += 1;
  }
  return `${amount.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function parseSearchOrStorage(paramKey, storageKey, fallback = '') {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get(paramKey);
  if (fromQuery !== null && fromQuery !== '') {
    return fromQuery;
  }
  const fromStorage = localStorage.getItem(storageKey);
  if (fromStorage !== null && fromStorage !== '') {
    return fromStorage;
  }
  return fallback;
}

function createUi() {
  return {
    form: document.getElementById('batchApplyForm'),
    formError: document.getElementById('formError'),
    fileIdInput: document.getElementById('fileIdInput'),
    fileNameHint: document.getElementById('fileNameHint'),
    key1ByteInput: document.getElementById('key1ByteInput'),
    key2ByteInput: document.getElementById('key2ByteInput'),
    enableBandpass: document.getElementById('enableBandpass'),
    enableDenoise: document.getElementById('enableDenoise'),
    enableFbpick: document.getElementById('enableFbpick'),
    bandpassLowHz: document.getElementById('bandpassLowHz'),
    bandpassHighHz: document.getElementById('bandpassHighHz'),
    bandpassTaper: document.getElementById('bandpassTaper'),
    denoiseChunkH: document.getElementById('denoiseChunkH'),
    denoiseOverlap: document.getElementById('denoiseOverlap'),
    denoiseMaskRatio: document.getElementById('denoiseMaskRatio'),
    denoiseNoiseStd: document.getElementById('denoiseNoiseStd'),
    denoiseMaskNoiseMode: document.getElementById('denoiseMaskNoiseMode'),
    denoisePassesBatch: document.getElementById('denoisePassesBatch'),
    fbpickModelSelect: document.getElementById('fbpickModelSelect'),
    pickMethod: document.getElementById('pickMethod'),
    sigmaMsMax: document.getElementById('sigmaMsMax'),
    subsampleCheck: document.getElementById('subsampleCheck'),
    snapEnabledCheck: document.getElementById('snapEnabledCheck'),
    snapMode: document.getElementById('snapMode'),
    snapRefine: document.getElementById('snapRefine'),
    snapWindowMs: document.getElementById('snapWindowMs'),
    savePicksCheck: document.getElementById('savePicksCheck'),
    runBtn: document.getElementById('runBtn'),
    cancelBtn: document.getElementById('cancelBtn'),
    refreshFilesBtn: document.getElementById('refreshFilesBtn'),
    currentJobId: document.getElementById('currentJobId'),
    jobStateBadge: document.getElementById('jobStateBadge'),
    jobProgress: document.getElementById('jobProgress'),
    jobMessage: document.getElementById('jobMessage'),
    jobsEmpty: document.getElementById('jobsEmpty'),
    jobHistory: document.getElementById('jobHistory'),
    filesTable: document.getElementById('filesTable'),
    filesBody: document.getElementById('filesBody'),
    filesEmpty: document.getElementById('filesEmpty'),
  };
}

function collectState(ui) {
  return {
    fileId: ui.fileIdInput.value,
    key1Byte: ui.key1ByteInput.value,
    key2Byte: ui.key2ByteInput.value,
    enableBandpass: ui.enableBandpass.checked,
    enableDenoise: ui.enableDenoise.checked,
    enableFbpick: ui.enableFbpick.checked,
    bandpass: {
      lowHz: ui.bandpassLowHz.value,
      highHz: ui.bandpassHighHz.value,
      taper: ui.bandpassTaper.value,
    },
    denoise: {
      chunkH: ui.denoiseChunkH.value,
      overlap: ui.denoiseOverlap.value,
      maskRatio: ui.denoiseMaskRatio.value,
      noiseStd: ui.denoiseNoiseStd.value,
      maskNoiseMode: ui.denoiseMaskNoiseMode.value,
      passesBatch: ui.denoisePassesBatch.value,
    },
    fbpick: {
      modelId: ui.fbpickModelSelect.value,
    },
    pick: {
      method: ui.pickMethod.value,
      sigmaMsMax: ui.sigmaMsMax.value,
      subsample: ui.subsampleCheck.checked,
      snapEnabled: ui.snapEnabledCheck.checked,
      snapMode: ui.snapMode.value,
      snapRefine: ui.snapRefine.value,
      snapWindowMs: ui.snapWindowMs.value,
    },
    savePicks: ui.savePicksCheck.checked,
  };
}

function setFormError(ui, message) {
  ui.formError.textContent = message || '';
}

function normalizeJobState(state) {
  const value = String(state || 'idle').trim().toLowerCase();
  if (value === 'done') return 'completed';
  if (value === 'error') return 'failed';
  return value || 'idle';
}

function formatBatchMessage(state, message) {
  const normalizedState = normalizeJobState(state);
  const text = String(message || '').trim();
  const lower = text.toLowerCase();
  if (normalizedState === 'cancelled') {
    return text || 'The job was cancelled by the user.';
  }
  if (normalizedState === 'cancel_requested') {
    return text || 'Cancel requested. The job will stop at the next safe point.';
  }
  if (!text) {
    if (normalizedState === 'completed') {
      return 'Batch job completed.';
    }
    if (normalizedState === 'queued') {
      return 'The batch job is queued.';
    }
    if (normalizedState === 'running') {
      return 'The batch job is running.';
    }
    return '';
  }
  if (lower.includes('fb pick model weights not found') || lower.includes('fbpick model not found')) {
    return 'FB model file is missing. Place the model weights under model/ and retry.';
  }
  if (lower.includes('file_id is required') || lower.includes('file id not found')) {
    return 'No dataset is loaded yet. Open data before starting a batch job.';
  }
  if (lower.includes('must be') || lower.includes('invalid')) {
    return text;
  }
  if (lower.includes('failed to fetch') || lower.includes('networkerror')) {
    return 'Network error while contacting the server. Check the connection and retry.';
  }
  if (normalizedState === 'failed' && !lower.includes('server')) {
    return `The batch job failed. ${text}`;
  }
  return text;
}

function formatBatchUiError(error) {
  const text = String((error && error.message) || error || '').trim();
  if (!text) {
    return 'The server returned an unexpected error. Check server logs.';
  }
  if (text.includes('file_id is required')) {
    return 'No dataset is loaded yet. Open data before starting a batch job.';
  }
  if (text.includes('At least one pipeline step must be enabled')) {
    return 'Enable at least one pipeline step before running batch apply.';
  }
  if (text.includes('Failed to fetch') || text.includes('NetworkError')) {
    return 'Network error while contacting the server. Check the connection and retry.';
  }
  return text;
}

function setJobState(ui, { jobId, state, progress, message }) {
  const normalizedState = normalizeJobState(state);
  const formattedMessage = formatBatchMessage(normalizedState, message);
  ui.currentJobId.textContent = jobId || '-';
  ui.jobStateBadge.textContent = normalizedState.replaceAll('_', ' ');
  ui.jobStateBadge.className = '';
  ui.jobStateBadge.id = 'jobStateBadge';
  ui.jobStateBadge.classList.add(`state-${normalizedState}`);
  const progressValue = Number(progress);
  ui.jobProgress.value = Number.isFinite(progressValue)
    ? Math.max(0, Math.min(1, progressValue))
    : 0;
  ui.jobMessage.textContent = formattedMessage;
  if (normalizedState === 'failed' || normalizedState === 'cancelled') {
    ui.jobMessage.style.borderColor = '#fecaca';
    ui.jobMessage.style.background = '#fef2f2';
    ui.jobMessage.style.color = '#991b1b';
  } else if (normalizedState === 'completed') {
    ui.jobMessage.style.borderColor = '#bbf7d0';
    ui.jobMessage.style.background = '#f0fdf4';
    ui.jobMessage.style.color = '#166534';
  } else {
    ui.jobMessage.style.borderColor = '#f3d8bd';
    ui.jobMessage.style.background = '#fff7ed';
    ui.jobMessage.style.color = '#b45309';
  }
}

function renderJobHistory(ui, jobs, activeJobId) {
  const list = Array.isArray(jobs) ? jobs : [];
  ui.jobHistory.innerHTML = '';
  ui.jobsEmpty.hidden = list.length > 0;
  for (const job of list) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'job-history-item';
    if (job.jobId === activeJobId) {
      item.classList.add('is-active');
    }
    item.dataset.jobId = job.jobId;

    const head = document.createElement('div');
    head.className = 'job-history-head';
    const idNode = document.createElement('span');
    idNode.className = 'job-history-id';
    idNode.textContent = job.jobId;
    const badge = document.createElement('span');
    badge.className = `job-history-badge state-${job.state}`;
    badge.textContent = job.state.replaceAll('_', ' ');
    head.appendChild(idNode);
    head.appendChild(badge);

    const meta = document.createElement('div');
    meta.className = 'job-history-meta';
    meta.textContent = `Progress ${(Math.max(0, Math.min(1, Number(job.progress) || 0)) * 100).toFixed(0)}%`;

    const messageNode = document.createElement('div');
    messageNode.className = 'job-history-message';
    messageNode.textContent = formatBatchMessage(job.state, job.message);

    item.appendChild(head);
    item.appendChild(meta);
    item.appendChild(messageNode);
    ui.jobHistory.appendChild(item);
  }
}

function renderFiles(ui, files, jobId) {
  const list = Array.isArray(files) ? files : [];
  ui.filesBody.innerHTML = '';
  if (!list.length) {
    ui.filesTable.hidden = true;
    ui.filesEmpty.textContent = 'No files generated yet.';
    ui.filesEmpty.hidden = false;
    return;
  }

  const safeJobId = encodeURIComponent(jobId);
  for (const file of list) {
    const tr = document.createElement('tr');
    const tdName = document.createElement('td');
    const tdSize = document.createElement('td');
    const tdDl = document.createElement('td');
    const fileName = String(file.name || '');
    tdName.textContent = fileName;
    tdSize.textContent = formatBytes(file.size_bytes);

    const link = document.createElement('a');
    link.className = 'download-link';
    link.textContent = 'Download';
    link.href = `/batch/job/${safeJobId}/download?name=${encodeURIComponent(fileName)}`;
    link.target = '_blank';
    link.rel = 'noopener';
    tdDl.appendChild(link);

    tr.appendChild(tdName);
    tr.appendChild(tdSize);
    tr.appendChild(tdDl);
    ui.filesBody.appendChild(tr);
  }

  ui.filesEmpty.hidden = true;
  ui.filesTable.hidden = false;
}

async function fetchJsonOrThrow(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(formatBatchUiError(error));
  }
  if (!response.ok) {
    let text = '';
    try {
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        const payload = await response.json();
        text = payload && typeof payload.detail === 'string' ? payload.detail : '';
      } else {
        text = await response.text();
      }
    } catch {
      text = '';
    }
    throw new Error(formatBatchUiError(
      `HTTP ${response.status} ${response.statusText}${text ? `: ${text}` : ''}`
    ));
  }
  return response.json();
}

function syncSavePicksControl(ui) {
  if (ui.enableFbpick.checked) {
    if (ui.savePicksCheck.disabled) {
      ui.savePicksCheck.checked = true;
    }
    ui.savePicksCheck.disabled = false;
  } else {
    ui.savePicksCheck.checked = false;
    ui.savePicksCheck.disabled = true;
  }
}

async function initModels(ui) {
  ui.fbpickModelSelect.innerHTML = '';
  ui.fbpickModelSelect.disabled = true;
  try {
    const payload = await fetchJsonOrThrow('/fbpick_models');
    const models = Array.isArray(payload.models) ? payload.models : [];
    if (!models.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(no models)';
      ui.fbpickModelSelect.appendChild(opt);
      return;
    }

    for (const model of models) {
      const modelId = String(model.id || '');
      if (!modelId) {
        continue;
      }
      const option = document.createElement('option');
      option.value = modelId;
      option.textContent =
        model.uses_offset === true ? `${modelId} (offset)` : modelId;
      ui.fbpickModelSelect.appendChild(option);
    }

    if (!ui.fbpickModelSelect.options.length) {
      return;
    }
    const saved = localStorage.getItem(LOCAL_STORAGE_KEYS.fbpickModel);
    const defaultId =
      typeof payload.default_model_id === 'string' ? payload.default_model_id : '';
    const allValues = Array.from(ui.fbpickModelSelect.options).map((opt) => opt.value);
    if (saved && allValues.includes(saved)) {
      ui.fbpickModelSelect.value = saved;
    } else if (defaultId && allValues.includes(defaultId)) {
      ui.fbpickModelSelect.value = defaultId;
    } else {
      ui.fbpickModelSelect.selectedIndex = 0;
    }
    ui.fbpickModelSelect.disabled = false;
  } catch (error) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(unavailable)';
    ui.fbpickModelSelect.appendChild(opt);
  }
}

async function initFileHint(ui) {
  const fileId = String(ui.fileIdInput.value || '').trim();
  if (!fileId) {
    ui.fileNameHint.textContent = 'Dataset: missing file_id';
    return;
  }
  try {
    const payload = await fetchJsonOrThrow(
      `/file_info?file_id=${encodeURIComponent(fileId)}`
    );
    const fileName = String(payload.file_name || 'unknown');
    ui.fileNameHint.textContent = `Dataset: ${fileName}`;
  } catch {
    ui.fileNameHint.textContent = `Dataset: ${fileId}`;
  }
}

function startBatchUi() {
  const ui = createUi();
  if (!ui.form) {
    return;
  }

  const initialFileId = parseSearchOrStorage('file_id', LOCAL_STORAGE_KEYS.fileId, '');
  const initialKey1 = parseSearchOrStorage('key1_byte', LOCAL_STORAGE_KEYS.key1Byte, '189');
  const initialKey2 = parseSearchOrStorage('key2_byte', LOCAL_STORAGE_KEYS.key2Byte, '193');
  ui.fileIdInput.value = initialFileId;
  ui.key1ByteInput.value = initialKey1;
  ui.key2ByteInput.value = initialKey2;

  syncSavePicksControl(ui);
  setJobState(ui, {
    jobId: '',
    state: 'idle',
    progress: 0,
    message: 'Ready',
  });
  renderFiles(ui, [], '');
  initFileHint(ui);
  initModels(ui);

  let pollingTimer = null;
  let activeJobId = '';
  let selectedJobId = '';
  const recentJobs = [];

  function upsertRecentJob(snapshot) {
    const jobId = String(snapshot.jobId || '').trim();
    if (!jobId) return;
    const normalized = {
      jobId,
      state: normalizeJobState(snapshot.state),
      progress: Number(snapshot.progress) || 0,
      message: snapshot.message || '',
    };
    const index = recentJobs.findIndex((entry) => entry.jobId === jobId);
    if (index >= 0) {
      recentJobs[index] = { ...recentJobs[index], ...normalized };
    } else {
      recentJobs.unshift(normalized);
    }
    renderJobHistory(ui, recentJobs, selectedJobId || activeJobId);
  }

  function syncActionButtons(stateValue = ui.jobStateBadge.textContent) {
    const normalized = normalizeJobState(stateValue);
    const busy = ['queued', 'running', 'cancel_requested'].includes(normalized);
    ui.runBtn.disabled = busy || !String(ui.fileIdInput.value || '').trim();
    ui.cancelBtn.disabled = !activeJobId || !busy;
    ui.refreshFilesBtn.disabled = !activeJobId || busy;
  }

  function selectJob(jobId) {
    selectedJobId = jobId;
    const snapshot = recentJobs.find((entry) => entry.jobId === jobId);
    renderJobHistory(ui, recentJobs, selectedJobId || activeJobId);
    if (!snapshot) {
      return;
    }
    setJobState(ui, snapshot);
    if (['queued', 'running', 'cancel_requested'].includes(snapshot.state)) {
      activeJobId = jobId;
      startPolling(jobId);
      return;
    }
    stopPolling();
    syncActionButtons(snapshot.state);
    if (snapshot.state === 'completed') {
      loadFiles(jobId);
    }
  }

  function enforceFileIdGuard() {
    const missing = !String(ui.fileIdInput.value || '').trim();
    if (missing) {
      ui.runBtn.disabled = true;
      setFormError(
        ui,
        'file_id is missing. Open this page via Viewer Batch link or fill file_id.'
      );
      syncActionButtons('idle');
      return false;
    }
    if (
      ui.formError.textContent ===
      'file_id is missing. Open this page via Viewer Batch link or fill file_id.'
    ) {
      setFormError(ui, '');
    }
    if (pollingTimer === null) {
      syncActionButtons('idle');
    }
    return true;
  }

  function stopPolling() {
    if (pollingTimer !== null) {
      clearInterval(pollingTimer);
      pollingTimer = null;
    }
  }

  async function loadFiles(jobId) {
    if (!jobId) {
      renderFiles(ui, [], '');
      return;
    }
    try {
      const payload = await fetchJsonOrThrow(
        `/batch/job/${encodeURIComponent(jobId)}/files`
      );
      renderFiles(ui, payload.files || [], jobId);
    } catch (error) {
      setJobState(ui, {
        jobId,
        state: 'failed',
        progress: ui.jobProgress.value,
        message: `Failed to fetch files: ${error.message}`,
      });
      upsertRecentJob({ jobId, state: 'failed', progress: ui.jobProgress.value, message: `Failed to fetch files: ${error.message}` });
    }
  }

  async function pollStatus(jobId) {
    try {
      const payload = await fetchJsonOrThrow(
        `/batch/job/${encodeURIComponent(jobId)}/status`
      );
      setJobState(ui, {
        jobId,
        state: payload.state || 'unknown',
        progress: payload.progress,
        message: payload.message || '',
      });
      upsertRecentJob({
        jobId,
        state: payload.state || 'unknown',
        progress: payload.progress,
        message: payload.message || '',
      });

      const normalizedState = normalizeJobState(payload.state);
      if (normalizedState === 'completed') {
        stopPolling();
        syncActionButtons(normalizedState);
        loadFiles(jobId);
      } else if (['failed', 'cancelled'].includes(normalizedState)) {
        stopPolling();
        syncActionButtons(normalizedState);
      } else {
        syncActionButtons(normalizedState);
      }
    } catch (error) {
      stopPolling();
      setJobState(ui, {
        jobId,
        state: 'failed',
        progress: ui.jobProgress.value,
        message: `Status polling failed: ${error.message}`,
      });
      upsertRecentJob({ jobId, state: 'failed', progress: ui.jobProgress.value, message: `Status polling failed: ${error.message}` });
      syncActionButtons('failed');
    }
  }

  function startPolling(jobId) {
    stopPolling();
    pollStatus(jobId);
    pollingTimer = window.setInterval(() => {
      pollStatus(jobId);
    }, POLL_INTERVAL_MS);
  }

  ui.enableFbpick.addEventListener('change', () => {
    syncSavePicksControl(ui);
  });

  ui.fileIdInput.addEventListener('change', () => {
    initFileHint(ui);
    enforceFileIdGuard();
  });
  ui.fileIdInput.addEventListener('input', () => {
    enforceFileIdGuard();
  });

  ui.fbpickModelSelect.addEventListener('change', () => {
    localStorage.setItem(LOCAL_STORAGE_KEYS.fbpickModel, ui.fbpickModelSelect.value);
  });

  ui.refreshFilesBtn.addEventListener('click', () => {
    if (activeJobId) {
      loadFiles(activeJobId);
    }
  });

  ui.cancelBtn.addEventListener('click', async () => {
    if (!activeJobId) {
      return;
    }
    try {
      const payload = await fetchJsonOrThrow(
        `/batch/job/${encodeURIComponent(activeJobId)}/cancel`,
        { method: 'POST' }
      );
      setJobState(ui, {
        jobId: activeJobId,
        state: payload.state,
        progress: payload.progress,
        message: payload.message,
      });
      upsertRecentJob({
        jobId: activeJobId,
        state: payload.state,
        progress: payload.progress,
        message: payload.message,
      });
      syncActionButtons(payload.state);
    } catch (error) {
      setJobState(ui, {
        jobId: activeJobId,
        state: 'failed',
        progress: ui.jobProgress.value,
        message: `Failed to cancel job: ${error.message}`,
      });
      upsertRecentJob({ jobId: activeJobId, state: 'failed', progress: ui.jobProgress.value, message: `Failed to cancel job: ${error.message}` });
      syncActionButtons('failed');
    }
  });

  ui.jobHistory.addEventListener('click', (event) => {
    const button = event.target.closest('.job-history-item');
    if (!button) {
      return;
    }
    selectJob(button.dataset.jobId || '');
  });

  ui.runBtn.addEventListener('click', async () => {
    setFormError(ui, '');
    syncSavePicksControl(ui);
    if (!enforceFileIdGuard()) {
      return;
    }
    let payload;
    try {
      payload = buildBatchApplyRequest(collectState(ui));
    } catch (error) {
      setFormError(ui, formatBatchUiError(error));
      return;
    }

    localStorage.setItem(LOCAL_STORAGE_KEYS.fileId, payload.file_id);
    localStorage.setItem(LOCAL_STORAGE_KEYS.key1Byte, String(payload.key1_byte));
    localStorage.setItem(LOCAL_STORAGE_KEYS.key2Byte, String(payload.key2_byte));

    syncActionButtons('queued');
    setJobState(ui, {
      jobId: activeJobId,
      state: 'queued',
      progress: 0,
      message: 'Submitting batch job...',
    });
    try {
      const response = await fetchJsonOrThrow('/batch/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      activeJobId = String(response.job_id || '');
      if (!activeJobId) {
        throw new Error('Server did not return job_id');
      }
      selectedJobId = activeJobId;
      renderFiles(ui, [], activeJobId);
      setJobState(ui, {
        jobId: activeJobId,
        state: response.state || 'queued',
        progress: 0,
        message: '',
      });
      upsertRecentJob({ jobId: activeJobId, state: response.state || 'queued', progress: 0, message: '' });
      startPolling(activeJobId);
    } catch (error) {
      setJobState(ui, {
        jobId: activeJobId,
        state: 'failed',
        progress: 0,
        message: `Failed to start job: ${error.message}`,
      });
      upsertRecentJob({ jobId: activeJobId, state: 'failed', progress: 0, message: `Failed to start job: ${error.message}` });
      syncActionButtons('failed');
    }
  });

  enforceFileIdGuard();
  syncActionButtons('idle');
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  startBatchUi();
}
