import { getStaticCorrectionDom, requestStaticCorrectionRender } from './runtime.js';
import { state } from './state.js';
import { readResponseError, trimValue } from './utils.js';

function sortedArtifactFiles(files) {
  return [...files].sort((left, right) => {
    const leftName = trimValue(left && left.name ? left.name : left);
    const rightName = trimValue(right && right.name ? right.name : right);
    return leftName.localeCompare(rightName);
  });
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
export function renderStaticArtifacts() {
  const dom = getStaticCorrectionDom();
  if (!dom || !dom.staticArtifactTable || !dom.staticArtifactBody || !dom.staticArtifactEmpty) {
    return;
  }
  const files = Array.isArray(state.staticArtifacts) ? state.staticArtifacts : [];
  dom.staticArtifactBody.innerHTML = '';

  if (state.loadingStaticArtifacts) {
    dom.staticArtifactTable.hidden = true;
    dom.staticArtifactEmpty.hidden = false;
    dom.staticArtifactEmpty.textContent = 'Loading static correction artifacts...';
    return;
  }

  if (!files.length) {
    dom.staticArtifactTable.hidden = true;
    dom.staticArtifactEmpty.hidden = false;
    dom.staticArtifactEmpty.textContent = state.lastStaticCorrectionJobId
      ? 'No static correction artifacts loaded.'
      : 'Run a static correction job to list generated artifacts.';
    return;
  }

  const safeJobId = encodeURIComponent(state.lastStaticCorrectionJobId);
  for (const file of files) {
    const name = trimValue(file && file.name ? file.name : file);
    if (!name) continue;
    const row = document.createElement('tr');
    const nameCell = document.createElement('td');
    const sizeCell = document.createElement('td');
    const downloadCell = document.createElement('td');
    const link = document.createElement('a');

    nameCell.textContent = name;
    sizeCell.textContent = formatBytes(file && file.size_bytes);
    link.href = `/statics/job/${safeJobId}/download?name=${encodeURIComponent(name)}`;
    link.textContent = 'Download';
    link.target = '_blank';
    link.rel = 'noopener';
    link.dataset.testid = `static-correction-artifact-download-${name}`;
    downloadCell.appendChild(link);

    row.appendChild(nameCell);
    row.appendChild(sizeCell);
    row.appendChild(downloadCell);
    dom.staticArtifactBody.appendChild(row);
  }

  dom.staticArtifactEmpty.hidden = true;
  dom.staticArtifactTable.hidden = false;
}
export async function loadStaticArtifacts(jobId = state.lastStaticCorrectionJobId, options = {}) {
  const safeJobId = trimValue(jobId);
  if (!safeJobId) {
    state.staticArtifacts = [];
    requestStaticCorrectionRender();
    return [];
  }

  const preserveMessage = Boolean(options && options.preserveMessage);
  const preserveError = Boolean(options && options.preserveError);
  const previousMessage = state.message;
  const previousError = state.error;
  state.loadingStaticArtifacts = true;
  state.staticArtifacts = [];
  requestStaticCorrectionRender();
  try {
    const response = await fetch(`/statics/job/${encodeURIComponent(safeJobId)}/files`);
    if (!response.ok) {
      throw new Error(await readResponseError(response, 'static job files'));
    }
    const payload = await response.json();
    const files = Array.isArray(payload.files) ? payload.files : [];
    state.staticArtifacts = sortedArtifactFiles(files);
    if (!preserveMessage) {
      state.message = files.length
        ? `Loaded ${files.length} static correction artifact${files.length === 1 ? '' : 's'} for ${safeJobId}.`
        : `Static correction job ${safeJobId} finished without generated artifact files.`;
    } else {
      state.message = previousMessage;
    }
    state.error = preserveError ? previousError : '';
    return state.staticArtifacts;
  } catch (error) {
    state.staticArtifacts = [];
    if (preserveError) {
      state.error = previousError;
    } else {
      state.error = error instanceof Error ? error.message : String(error);
    }
    state.message = preserveMessage
      ? previousMessage
      : `Static correction job ${safeJobId} finished, but artifacts could not be loaded.`;
    return [];
  } finally {
    state.loadingStaticArtifacts = false;
    requestStaticCorrectionRender();
  }
}
