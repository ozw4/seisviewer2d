import { LINKAGE_FAILED_STATES, LINKAGE_READY_STATES, STATIC_ACTIVE_STATES, STATIC_READY_STATES, STATIC_TERMINAL_STATES } from './constants.js';
import { loadStaticArtifacts } from './artifacts_view.js';
import { saveStaticCorrectionDraft } from './draft.js';
import {
  buildStaticCorrectionFormData,
  buildStaticCorrectionLinkage,
  buildStaticCorrectionRequest,
  buildStaticLinkageBuildRequest,
} from './request_builder.js';
import { getStaticCorrectionDom, requestStaticCorrectionRender } from './runtime.js';
import { state } from './state.js';
import { buildRefractionQcUrl } from './target.js';
import { delay, readResponseError, trimValue } from './utils.js';
import {
  initRefractionQcPage,
  loadJob as loadRefractionQcJob,
} from '../refraction-qc/main.js';

let staticPollToken = 0;

export function normalizeStaticJobState(value) {
  const normalized = trimValue(value).toLowerCase();
  if (normalized === 'completed') return 'done';
  if (normalized === 'failed') return 'error';
  if (normalized === 'canceled') return 'cancelled';
  return normalized || 'unknown';
}
export function isStaticJobActive(value = state.lastStaticCorrectionState) {
  return STATIC_ACTIVE_STATES.has(normalizeStaticJobState(value));
}
function isStaticJobTerminal(value = state.lastStaticCorrectionState) {
  return STATIC_TERMINAL_STATES.has(normalizeStaticJobState(value));
}
async function postJson(url, payload, operation) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await readResponseError(response, operation));
  }
  return response.json();
}
async function postMultipart(url, formData, operation) {
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readResponseError(response, operation));
  }
  return response.json();
}
function setStaticJobSnapshot(payload, fallbackJobId = '') {
  const jobId = trimValue((payload && payload.job_id) || fallbackJobId || state.lastStaticCorrectionJobId);
  const jobState = normalizeStaticJobState(payload && payload.state);
  const progress = Number(payload && payload.progress);
  const message = trimValue(payload && payload.message);
  state.lastStaticCorrectionJobId = jobId;
  state.lastStaticCorrectionState = jobState;
  state.lastStaticCorrectionProgress = Number.isFinite(progress) ? progress : state.lastStaticCorrectionProgress;
  state.lastStaticCorrectionMessage = message;
  return {
    jobId,
    state: jobState,
    progress: state.lastStaticCorrectionProgress,
    message,
  };
}
function formatStaticJobStatus(snapshot) {
  const jobId = snapshot.jobId || state.lastStaticCorrectionJobId;
  const jobState = normalizeStaticJobState(snapshot.state);
  const message = trimValue(snapshot.message);
  if (jobState === 'done') {
    return `Static correction job ${jobId} is done. Loading artifacts...`;
  }
  if (jobState === 'error') {
    return `Static correction job ${jobId} failed${message ? `: ${message}` : '.'}`;
  }
  if (jobState === 'expired') {
    return `Static correction job ${jobId} expired${message ? `: ${message}` : '.'}`;
  }
  if (jobState === 'cancelled') {
    return `Static correction job ${jobId} was cancelled${message ? `: ${message}` : '.'}`;
  }
  if (jobState === 'cancel_requested') {
    return `Cancel requested for static correction job ${jobId}.`;
  }
  return `Static correction job ${jobId} is ${jobState || 'unknown'}${message ? `: ${message}` : '.'}`;
}
async function autoLoadRefractionQc(jobId) {
  const safeJobId = trimValue(jobId);
  if (!safeJobId) {
    return null;
  }
  if (document.getElementById('refractionQcForm')) {
    initRefractionQcPage();
    return loadRefractionQcJob(safeJobId, { activateTab: true });
  }
  const qcUrl = buildRefractionQcUrl(safeJobId);
  const dom = getStaticCorrectionDom();
  if (dom && dom.qcLink) {
    dom.qcLink.href = qcUrl;
  }
  window.location.assign(qcUrl);
  return null;
}
export async function pollStaticCorrectionStatus(jobId) {
  const response = await fetch(`/statics/job/${encodeURIComponent(jobId)}/status`);
  if (!response.ok) {
    throw new Error(await readResponseError(response, 'static job status'));
  }
  const payload = await response.json();
  const snapshot = setStaticJobSnapshot(payload, jobId);
  state.message = formatStaticJobStatus(snapshot);
  if (snapshot.state === 'error' || snapshot.state === 'expired') {
    state.error = state.message;
  } else if (snapshot.state === 'cancelled') {
    state.error = '';
  }
  requestStaticCorrectionRender();
  return snapshot;
}
export function stopStaticCorrectionPolling() {
  staticPollToken += 1;
}
export async function pollStaticCorrectionJobUntilTerminal(jobId) {
  const token = staticPollToken + 1;
  staticPollToken = token;
  state.staticArtifacts = [];
  requestStaticCorrectionRender();

  while (token === staticPollToken) {
    try {
      const snapshot = await pollStaticCorrectionStatus(jobId);
      if (STATIC_READY_STATES.has(snapshot.state)) {
        await loadStaticArtifacts(jobId);
        if (token !== staticPollToken) {
          return null;
        }
        if (state.autoOpenQcOnCompletion) {
          state.autoOpenQcOnCompletion = false;
          await autoLoadRefractionQc(jobId);
        }
        return snapshot;
      }
      if (isStaticJobTerminal(snapshot.state)) {
        state.autoOpenQcOnCompletion = false;
        if (snapshot.state === 'error') {
          await loadStaticArtifacts(jobId, {
            preserveMessage: true,
            preserveError: true,
          });
        }
        return snapshot;
      }
    } catch (error) {
      if (token !== staticPollToken) {
        return null;
      }
      state.error = error instanceof Error ? error.message : String(error);
      state.message = `Static correction status polling failed: ${state.error}`;
      requestStaticCorrectionRender();
      return null;
    }
    await delay(Math.max(0, Number(state.pollIntervalMs) || 0));
  }
  return null;
}
export async function pollStaticJobUntilReady(jobId) {
  const encodedJobId = encodeURIComponent(jobId);
  while (true) {
    const response = await fetch(`/statics/job/${encodedJobId}/status`);
    if (!response.ok) {
      throw new Error(await readResponseError(response, 'static job status'));
    }
    const payload = await response.json();
    const stateValue = normalizeStaticJobState(payload && payload.state);
    const message = trimValue(payload && payload.message);
    if (LINKAGE_READY_STATES.has(stateValue)) {
      return payload;
    }
    if (LINKAGE_FAILED_STATES.has(stateValue)) {
      const detail = message ? `: ${message}` : '';
      throw new Error(`Linkage job ${jobId} ${stateValue}${detail}`);
    }
    state.message = `Linkage job ${jobId} is ${stateValue || 'pending'}; waiting for geometry linkage.`;
    requestStaticCorrectionRender();
    await delay(Math.max(0, Number(state.pollIntervalMs) || 0));
  }
}
async function submitStaticCorrection(payload) {
  state.phase = 'submitting_static_correction';
  state.message = state.lastLinkageJobId
    ? `Linkage job ${state.lastLinkageJobId} is ready. Submitting static correction...`
    : 'Submitting static correction...';
  requestStaticCorrectionRender();
  const formData = buildStaticCorrectionFormData(payload);
  const responsePayload = await postMultipart(
    '/statics/refraction/apply-with-picks',
    formData,
    'refraction static apply with picks'
  );
  state.lastStaticCorrectionJobId = trimValue(responsePayload && responsePayload.job_id);
  setStaticJobSnapshot(responsePayload);
  state.lastResponse = responsePayload;
  state.autoOpenQcOnCompletion = Boolean(state.lastStaticCorrectionJobId);
  state.phase = 'idle';
  const initialState = state.lastStaticCorrectionState
    ? ` Initial state: ${state.lastStaticCorrectionState}.`
    : '';
  state.message = state.lastStaticCorrectionJobId
    ? `Static correction job ${state.lastStaticCorrectionJobId} submitted.${initialState}`
    : 'Static correction job submitted.';
  requestStaticCorrectionRender();
  if (state.lastStaticCorrectionJobId) {
    pollStaticCorrectionJobUntilTerminal(state.lastStaticCorrectionJobId);
  }
  return responsePayload;
}
export async function validateStaticCorrectionInputs() {
  saveStaticCorrectionDraft();
  state.lastResponse = null;
  state.lastLinkageBuildRequest = null;
  state.lastLinkageJobId = '';
  state.validationDiagnostics = null;
  state.showValidationSummary = false;
  state.validationErrors = [];
  const { payload, errors } = buildStaticCorrectionRequest();
  state.lastRequest = payload;
  if (errors.length) {
    state.error = errors.join(' ');
    state.validationErrors = errors;
    state.showValidationSummary = true;
    state.message = 'Fix input errors before validating refraction statics.';
    requestStaticCorrectionRender();
    return null;
  }

  try {
    state.error = '';
    state.phase = 'validating_static_correction';
    state.message = 'Validating refraction static inputs...';
    requestStaticCorrectionRender();
    let validationPayload = payload;
    if (getStaticCorrectionDom().enableLinkage.checked) {
      state.phase = 'building_linkage';
      state.message = 'Building endpoint geometry linkage for validation...';
      requestStaticCorrectionRender();
      const linkageBuildPayload = buildStaticLinkageBuildRequest(getStaticCorrectionDom());
      state.lastLinkageBuildRequest = linkageBuildPayload;

      const linkageResponse = await postJson(
        '/statics/linkage/build',
        linkageBuildPayload,
        'geometry linkage build'
      );
      const linkageJobId = trimValue(linkageResponse && linkageResponse.job_id);
      if (!linkageJobId) {
        throw new Error('Geometry linkage build did not return a job_id.');
      }
      state.lastLinkageJobId = linkageJobId;
      state.message = `Linkage job ${linkageJobId} created. Waiting for geometry linkage...`;
      requestStaticCorrectionRender();

      await pollStaticJobUntilReady(linkageJobId);
      validationPayload = {
        ...payload,
        linkage: buildStaticCorrectionLinkage(getStaticCorrectionDom(), linkageJobId),
      };
      state.lastRequest = validationPayload;
    }
    const formData = buildStaticCorrectionFormData(validationPayload);
    const responsePayload = await postMultipart(
      '/statics/refraction/validate-with-picks',
      formData,
      'refraction static validation with picks'
    );
    state.validationDiagnostics = responsePayload;
    state.error = Array.isArray(responsePayload.errors) && responsePayload.errors.length
      ? responsePayload.errors.join(' ')
      : '';
    state.message = state.error
      ? 'Validation found input issues. Static correction was not submitted.'
      : 'Validation completed. Inputs are ready for static correction.';
    return responsePayload;
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    state.message = 'Validation failed. Static correction was not submitted.';
    return null;
  } finally {
    state.phase = 'idle';
    requestStaticCorrectionRender();
  }
}
export async function runStaticCorrection() {
  saveStaticCorrectionDraft();
  const { payload, errors } = buildStaticCorrectionRequest();
  state.lastRequest = payload;
  state.lastResponse = null;
  state.lastLinkageBuildRequest = null;
  state.lastLinkageJobId = '';
  state.lastStaticCorrectionJobId = '';
  state.lastStaticCorrectionState = '';
  state.lastStaticCorrectionMessage = '';
  state.lastStaticCorrectionProgress = 0;
  state.autoOpenQcOnCompletion = false;
  state.staticArtifacts = [];
  stopStaticCorrectionPolling();
  if (errors.length) {
    state.ready = false;
    state.error = errors.join(' ');
    state.validationErrors = errors;
    state.showValidationSummary = true;
    state.message = 'Fix input errors before running refraction statics.';
    state.phase = 'idle';
    requestStaticCorrectionRender();
    return;
  }

  state.ready = true;
  state.error = '';
  state.validationErrors = [];
  state.showValidationSummary = false;
  state.phase = 'submitting_static_correction';
  requestStaticCorrectionRender();

  try {
    let applyPayload = payload;
    if (getStaticCorrectionDom().enableLinkage.checked) {
      state.phase = 'building_linkage';
      state.message = 'Building endpoint geometry linkage...';
      requestStaticCorrectionRender();
      const linkageBuildPayload = buildStaticLinkageBuildRequest(getStaticCorrectionDom());
      state.lastLinkageBuildRequest = linkageBuildPayload;

      const linkageResponse = await postJson(
        '/statics/linkage/build',
        linkageBuildPayload,
        'geometry linkage build'
      );
      const linkageJobId = trimValue(linkageResponse && linkageResponse.job_id);
      if (!linkageJobId) {
        throw new Error('Geometry linkage build did not return a job_id.');
      }
      state.lastLinkageJobId = linkageJobId;
      state.message = `Linkage job ${linkageJobId} created. Waiting for geometry linkage...`;
      requestStaticCorrectionRender();

      await pollStaticJobUntilReady(linkageJobId);
      state.phase = 'linkage_ready';
      applyPayload = {
        ...payload,
        linkage: buildStaticCorrectionLinkage(getStaticCorrectionDom(), linkageJobId),
      };
      state.lastRequest = applyPayload;
    }

    await submitStaticCorrection(applyPayload);
  } catch (error) {
    const failedDuringLinkage = state.phase === 'building_linkage';
    state.ready = false;
    state.error = error instanceof Error ? error.message : String(error);
    state.message = failedDuringLinkage
      ? 'Geometry linkage failed. Static correction was not submitted.'
      : 'Static correction submission failed.';
    state.phase = 'idle';
    requestStaticCorrectionRender();
  }
}
export async function cancelStaticCorrectionJob() {
  const jobId = trimValue(state.lastStaticCorrectionJobId);
  if (!jobId || !isStaticJobActive()) {
    return null;
  }

  try {
    state.autoOpenQcOnCompletion = false;
    state.message = `Cancelling static correction job ${jobId}...`;
    state.error = '';
    requestStaticCorrectionRender();
    const response = await fetch(`/statics/job/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error(await readResponseError(response, 'static job cancel'));
    }
    const payload = await response.json();
    const snapshot = setStaticJobSnapshot(payload, jobId);
    state.message = formatStaticJobStatus(snapshot);
    if (isStaticJobTerminal(snapshot.state)) {
      stopStaticCorrectionPolling();
    }
    requestStaticCorrectionRender();
    return snapshot;
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    state.message = `Failed to cancel static correction job ${jobId}.`;
    requestStaticCorrectionRender();
    return null;
  }
}
