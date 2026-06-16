import {
  CELL_MODEL_PRESETS,
  DEFAULTS,
  GEOMETRY_DEFAULTS,
  GEOMETRY_PRESET_CUSTOM,
  GEOMETRY_PRESET_SEG_Y_DEFAULT,
  LINKAGE_THRESHOLD_MODES,
  MODEL_PRESETS,
  MULTILAYER_MODEL_PRESETS,
  NO_ACTIVE_TARGET_ERROR,
  STATIC_READY_STATES,
} from './constants.js';
import { renderStaticArtifacts } from './artifacts_view.js';
import { formatFileSize, selectedPickNpzFile } from './form_collectors.js';
import { normalizeStaticJobState, isStaticJobActive } from './job_runner.js';
import { getStaticCorrectionValidationSnapshot } from './request_builder.js';
import { getStaticCorrectionDom } from './runtime.js';
import { state } from './state.js';
import { buildRefractionQcUrl, getStaticCorrectionTarget, validateStaticCorrectionTarget } from './target.js';
import { formatSavedAt, setDisabled, setHidden, trimValue } from './utils.js';

export function geometryValueElements(targetDom) {
  if (!targetDom) return [];
  return [
    targetDom.sourceIdByte,
    targetDom.receiverIdByte,
    targetDom.sourceXByte,
    targetDom.sourceYByte,
    targetDom.receiverXByte,
    targetDom.receiverYByte,
    targetDom.sourceElevationByte,
    targetDom.receiverElevationByte,
    targetDom.coordinateScalarByte,
    targetDom.elevationScalarByte,
    targetDom.sourceDepthByte,
    targetDom.coordinateUnit,
    targetDom.elevationUnit,
    targetDom.offsetByte,
  ].filter(Boolean);
}
export function setGeometryDefaults(targetDom) {
  if (!targetDom) return;
  for (const [key, value] of Object.entries(GEOMETRY_DEFAULTS)) {
    if (key === 'preset') continue;
    if (targetDom[key]) {
      targetDom[key].value = value;
    }
  }
}
export function applyStaticCorrectionGeometryPreset(targetDom = getStaticCorrectionDom(), presetName = null) {
  if (!targetDom || !targetDom.geometryPreset) return;
  const preset = presetName || trimValue(targetDom.geometryPreset.value) || GEOMETRY_DEFAULTS.preset;
  const normalized = preset === GEOMETRY_PRESET_CUSTOM
    ? GEOMETRY_PRESET_CUSTOM
    : GEOMETRY_PRESET_SEG_Y_DEFAULT;

  targetDom.geometryPreset.value = normalized;
  if (normalized === GEOMETRY_PRESET_SEG_Y_DEFAULT) {
    setGeometryDefaults(targetDom);
  }

  const disabled = normalized !== GEOMETRY_PRESET_CUSTOM;
  for (const element of geometryValueElements(targetDom)) {
    element.disabled = disabled;
  }
}
export function renderPickNpzSummary(targetDom = getStaticCorrectionDom()) {
  if (!targetDom || !targetDom.pickNpzSummary) return;
  const pickFile = selectedPickNpzFile(targetDom);
  if (!pickFile) {
    if (state.pickNpzRestoreStatus === 'warning' && state.pickNpzRestoreMessage) {
      targetDom.pickNpzSummary.textContent = state.pickNpzRestoreMessage;
      return;
    }
    targetDom.pickNpzSummary.textContent = 'No NPZ file selected.';
    return;
  }
  const size = formatFileSize(pickFile.size);
  if (state.pickNpzRestoreStatus === 'restored') {
    const savedAt = formatSavedAt(state.pickNpzRestoredSavedAt);
    const savedText = savedAt ? `, saved ${savedAt}` : '';
    targetDom.pickNpzSummary.textContent = (
      `Restored NPZ: ${pickFile.name}${size ? ` (${size})` : ''}${savedText}. `
      + 'This restored NPZ is loaded into the file input and will be submitted.'
    );
    return;
  }
  targetDom.pickNpzSummary.textContent = size ? `${pickFile.name} (${size})` : pickFile.name;
}
export function linkageValueElements(targetDom) {
  if (!targetDom) return [];
  return [
    targetDom.linkageMode,
    targetDom.linkageThresholdM,
    targetDom.receiverLocationIntervalM,
    targetDom.preferReceiverAnchor,
  ].filter(Boolean);
}
export function updateStaticCorrectionLinkageOptions(targetDom = getStaticCorrectionDom()) {
  if (!targetDom || !targetDom.enableLinkage || !targetDom.linkageOptions) return;
  const enabled = Boolean(targetDom.enableLinkage.checked);
  const mode = trimValue(targetDom.linkageMode && targetDom.linkageMode.value);
  targetDom.linkageOptions.hidden = !enabled;
  for (const element of linkageValueElements(targetDom)) {
    element.disabled = !enabled;
  }
  if (targetDom.linkageThresholdM) {
    targetDom.linkageThresholdM.disabled = !enabled || !LINKAGE_THRESHOLD_MODES.has(mode);
  }
}
export function fieldCorrectionValueElements(targetDom) {
  if (!targetDom) return [];
  return [
    targetDom.fieldSourceDepthMode,
    targetDom.fieldSourceDepthByte,
    targetDom.fieldUpholeMode,
    targetDom.fieldUpholeTimeByte,
    targetDom.fieldManualStaticMode,
    targetDom.fieldManualStaticSignConvention,
    targetDom.fieldManualSourceJobId,
    targetDom.fieldManualSourceArtifactName,
    targetDom.fieldManualReceiverJobId,
    targetDom.fieldManualReceiverArtifactName,
    targetDom.fieldApplyToTraceShift,
  ].filter(Boolean);
}
export function updateStaticCorrectionFieldCorrectionOptions(targetDom = getStaticCorrectionDom()) {
  if (!targetDom || !targetDom.fieldCorrectionsEnabled) return;
  const enabled = Boolean(targetDom.fieldCorrectionsEnabled.checked);
  const sourceDepthMode = trimValue(targetDom.fieldSourceDepthMode && targetDom.fieldSourceDepthMode.value);
  const upholeMode = trimValue(targetDom.fieldUpholeMode && targetDom.fieldUpholeMode.value);
  const manualMode = trimValue(targetDom.fieldManualStaticMode && targetDom.fieldManualStaticMode.value);

  if (targetDom.fieldCorrectionOptions) {
    targetDom.fieldCorrectionOptions.hidden = !enabled;
  }
  if (targetDom.fieldSourceDepthByte) {
    targetDom.fieldSourceDepthByte.disabled = !enabled || sourceDepthMode !== 'weathering_velocity_time';
  }
  if (targetDom.fieldUpholeTimeByte) {
    targetDom.fieldUpholeTimeByte.disabled = !enabled || upholeMode !== 'header_time';
  }
  const manualEnabled = enabled && manualMode === 'artifact_table';
  if (targetDom.fieldManualArtifactFields) {
    targetDom.fieldManualArtifactFields.hidden = !manualEnabled;
  }
  for (const element of fieldCorrectionValueElements(targetDom)) {
    if (
      element === targetDom.fieldSourceDepthByte
      || element === targetDom.fieldUpholeTimeByte
    ) {
      continue;
    }
    element.disabled = !enabled;
  }
  setDisabled(
    [
      targetDom.fieldManualStaticSignConvention,
      targetDom.fieldManualSourceJobId,
      targetDom.fieldManualSourceArtifactName,
      targetDom.fieldManualReceiverJobId,
      targetDom.fieldManualReceiverArtifactName,
    ],
    !manualEnabled
  );
}
export function updateBedrockVelocityControls(targetDom = getStaticCorrectionDom()) {
  if (!targetDom || !targetDom.bedrockVelocityMode) return;
  const preset = trimValue(targetDom.modelKind && targetDom.modelKind.value);
  const fixedMode = targetDom.bedrockVelocityMode.value === 'fixed_global';
  const cellMode = CELL_MODEL_PRESETS.has(preset);
  if (targetDom.initialBedrockVelocityMS) {
    targetDom.initialBedrockVelocityMS.disabled = fixedMode;
  }
  if (targetDom.fixedBedrockVelocityMS) {
    targetDom.fixedBedrockVelocityMS.disabled = !fixedMode || cellMode;
  }
  if (targetDom.bedrockVelocityMode) {
    targetDom.bedrockVelocityMode.disabled = cellMode;
  }
}
export function updateModelPresetControls(targetDom = getStaticCorrectionDom()) {
  if (!targetDom || !targetDom.modelKind) return;
  const preset = MODEL_PRESETS.has(targetDom.modelKind.value)
    ? targetDom.modelKind.value
    : DEFAULTS.modelPreset;
  targetDom.modelKind.value = preset;
  const isMultilayer = MULTILAYER_MODEL_PRESETS.has(preset);
  const isThreeLayer = preset === 'three_layer_global';
  const isCell = CELL_MODEL_PRESETS.has(preset);
  const isLine2D = preset === 'cell_v2_t1_line_2d';

  setHidden(targetDom.v3LayerFields, !isMultilayer);
  setHidden(targetDom.vsubLayerFields, !isThreeLayer);
  setHidden(targetDom.cellFields, !isCell);
  setHidden(targetDom.line2DFields, !isLine2D);

  setDisabled(
    [targetDom.v3MinOffsetM, targetDom.v3MaxOffsetM, targetDom.initialV3VelocityMS],
    !isMultilayer
  );
  setDisabled(
    [targetDom.vsubMinOffsetM, targetDom.initialVsubVelocityMS],
    !isThreeLayer
  );
  setDisabled(
    [
      targetDom.cellXOriginM,
      targetDom.cellYOriginM,
      targetDom.cellCountX,
      targetDom.cellCountY,
      targetDom.cellSizeXM,
      targetDom.cellSizeYM,
      targetDom.cellMinObservations,
      targetDom.cellSmoothingWeight,
    ],
    !isCell
  );
  setDisabled(
    [targetDom.lineOriginXM, targetDom.lineOriginYM, targetDom.lineAzimuthDeg],
    !isLine2D
  );

  if (targetDom.conversionMode) {
    targetDom.conversionMode.value = isMultilayer ? 't1lsst_multilayer' : 't1lsst_1layer';
    targetDom.conversionMode.disabled = true;
  }
  if (targetDom.bedrockVelocityMode) {
    if (isCell) {
      targetDom.bedrockVelocityMode.value = 'solve_cell';
    } else if (targetDom.bedrockVelocityMode.value === 'solve_cell') {
      targetDom.bedrockVelocityMode.value = DEFAULTS.bedrockVelocityMode;
    }
  }
  if (isLine2D && targetDom.cellCountY) {
    targetDom.cellCountY.value = '1';
  }
  updateBedrockVelocityControls(targetDom);
}
export function renderStaticJobPanel() {
  const dom = getStaticCorrectionDom();
  if (!dom || !dom.staticJobPanel) return;
  const jobId = state.lastStaticCorrectionJobId;
  const jobState = normalizeStaticJobState(state.lastStaticCorrectionState);
  const hasJob = Boolean(jobId || state.lastStaticCorrectionState || state.lastStaticCorrectionMessage);
  const progress = Number(state.lastStaticCorrectionProgress);
  const progressValue = Number.isFinite(progress) ? Math.max(0, Math.min(1, progress)) : 0;

  dom.staticJobPanel.hidden = !hasJob;
  if (dom.staticJobIdValue) {
    dom.staticJobIdValue.textContent = jobId || '-';
  }
  if (dom.staticJobStateValue) {
    dom.staticJobStateValue.textContent = jobState || '-';
  }
  if (dom.staticJobMessageValue) {
    dom.staticJobMessageValue.textContent = state.lastStaticCorrectionMessage || '-';
  }
  if (dom.staticJobProgressValue) {
    dom.staticJobProgressValue.textContent = hasJob ? `${Math.round(progressValue * 100)}%` : '-';
  }
  if (dom.staticJobProgress) {
    dom.staticJobProgress.value = progressValue;
  }
  if (dom.cancelButton) {
    dom.cancelButton.hidden = !jobId || !isStaticJobActive(jobState);
    dom.cancelButton.disabled = !jobId || jobState === 'cancel_requested';
  }
  if (dom.qcLinkRow && dom.qcLink) {
    const showQcLink = Boolean(jobId && STATIC_READY_STATES.has(jobState));
    dom.qcLinkRow.hidden = !showQcLink;
    if (showQcLink) {
      dom.qcLink.href = buildRefractionQcUrl(jobId);
    }
  }
}
export function renderValidationSummary() {
  const dom = getStaticCorrectionDom();
  if (!dom || !dom.validationSummary) return;
  const errors = state.validationErrors || [];
  dom.validationSummary.innerHTML = '';
  dom.validationSummary.hidden = !state.showValidationSummary || errors.length === 0;
  if (dom.validationSummary.hidden) {
    return;
  }
  const heading = document.createElement('div');
  heading.textContent = 'Fix these fields before submitting:';
  const list = document.createElement('ul');
  for (const error of errors) {
    const item = document.createElement('li');
    item.textContent = error;
    list.appendChild(item);
  }
  dom.validationSummary.appendChild(heading);
  dom.validationSummary.appendChild(list);
}
export function formatDiagnosticsValue(value) {
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(', ') : '-';
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : '-';
  }
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}
export function appendDiagnosticsRow(tableBody, label, value) {
  const row = document.createElement('tr');
  const labelCell = document.createElement('th');
  const valueCell = document.createElement('td');
  labelCell.scope = 'row';
  labelCell.textContent = label;
  valueCell.textContent = formatDiagnosticsValue(value);
  row.appendChild(labelCell);
  row.appendChild(valueCell);
  tableBody.appendChild(row);
}
export function renderValidationDiagnostics() {
  const dom = getStaticCorrectionDom();
  if (!dom || !dom.validationDiagnostics) return;
  const payload = state.validationDiagnostics;
  dom.validationDiagnostics.innerHTML = '';
  dom.validationDiagnostics.hidden = !payload;
  if (!payload) return;

  const target = payload.target || {};
  const pick = payload.pick_npz || {};
  const diagnostics = payload.diagnostics || {};
  const offset = diagnostics.offset_m || {};
  const filterCounts = diagnostics.filter_reason_counts || {};
  const errors = Array.isArray(payload.errors) ? payload.errors : [];
  const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];
  const targetState = getStaticCorrectionTarget();

  const heading = document.createElement('div');
  heading.textContent = payload.status === 'ok'
    ? 'Validation passed.'
    : 'Validation found input issues.';
  dom.validationDiagnostics.appendChild(heading);

  const table = document.createElement('table');
  const body = document.createElement('tbody');
  appendDiagnosticsRow(
    body,
    'Target file',
    targetState && targetState.display_name ? targetState.display_name : target.file_id
  );
  appendDiagnosticsRow(body, 'key1/key2', `${target.key1_byte}/${target.key2_byte}`);
  appendDiagnosticsRow(body, 'Pick NPZ key', pick.selected_key);
  appendDiagnosticsRow(body, 'Pick NPZ shape', pick.shape);
  appendDiagnosticsRow(body, 'n_total_traces', diagnostics.n_total_traces);
  appendDiagnosticsRow(body, 'n_finite_picks', diagnostics.n_finite_picks);
  appendDiagnosticsRow(body, 'n_used_for_inversion', diagnostics.n_used_for_inversion);
  appendDiagnosticsRow(body, 'n_unique_source_endpoints', diagnostics.n_unique_source_endpoints);
  appendDiagnosticsRow(body, 'n_unique_receiver_endpoints', diagnostics.n_unique_receiver_endpoints);
  appendDiagnosticsRow(
    body,
    'offset min/median/max',
    `${formatDiagnosticsValue(offset.min)} / ${formatDiagnosticsValue(offset.median)} / ${formatDiagnosticsValue(offset.max)}`
  );
  appendDiagnosticsRow(body, 'filter reason counts', filterCounts);
  appendDiagnosticsRow(body, 'warnings', warnings);
  appendDiagnosticsRow(body, 'errors', errors);
  table.appendChild(body);
  dom.validationDiagnostics.appendChild(table);
}
export function renderPresetSelect() {
  const dom = getStaticCorrectionDom();
  if (!dom || !dom.presetSelect) return;
  const currentValue = trimValue(dom.presetSelect.value);
  dom.presetSelect.innerHTML = '';

  if (!state.presets.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No saved presets';
    dom.presetSelect.appendChild(option);
  } else {
    for (const preset of state.presets) {
      const option = document.createElement('option');
      option.value = preset.name;
      option.textContent = preset.name;
      dom.presetSelect.appendChild(option);
    }
    if (state.presets.some((entry) => entry.name === currentValue)) {
      dom.presetSelect.value = currentValue;
    }
  }

  if (dom.loadPresetButton) {
    dom.loadPresetButton.disabled = state.presets.length === 0;
  }
  if (dom.deletePresetButton) {
    dom.deletePresetButton.disabled = state.presets.length === 0;
  }
}
export function renderTargetSummary() {
  const dom = getStaticCorrectionDom();
  if (!dom || !dom.targetEmpty || !dom.targetDetails) return;
  const target = getStaticCorrectionTarget();
  dom.targetEmpty.hidden = Boolean(target);
  dom.targetDetails.hidden = !target;
  if (!target) {
    const targetErrors = validateStaticCorrectionTarget().errors;
    dom.targetEmpty.textContent = targetErrors[0] || NO_ACTIVE_TARGET_ERROR;
    return;
  }
  if (dom.targetFile) {
    dom.targetFile.textContent = target.display_name;
    dom.targetFile.title = target.file_id;
  }
  if (dom.targetKeys) {
    dom.targetKeys.textContent = `key1=${target.key1_byte}, key2=${target.key2_byte}`;
  }
  if (dom.targetStatus) {
    dom.targetStatus.textContent = 'Ready';
  }
}
export function render() {
  const dom = getStaticCorrectionDom();
  if (!dom) return;
  const preview = getStaticCorrectionValidationSnapshot(dom);
  const hasTarget = Boolean(getStaticCorrectionTarget());
  if (state.showValidationSummary) {
    state.validationErrors = preview.errors;
  }
  updateStaticCorrectionLinkageOptions(dom);
  renderTargetSummary();
  renderPickNpzSummary(dom);
  dom.status.textContent = state.message;
  dom.error.hidden = !state.error;
  dom.error.textContent = state.error;
  dom.runButton.disabled = (
    !hasTarget
    || preview.errors.length > 0
    || state.phase !== 'idle'
    || isStaticJobActive()
  );
  if (dom.validateButton) {
    dom.validateButton.disabled = (
      !hasTarget
      || preview.errors.length > 0
      || state.phase !== 'idle'
      || isStaticJobActive()
    );
  }
  if (dom.requestPreview) {
    dom.requestPreview.textContent = preview.payload
      ? JSON.stringify(preview.payload, null, 2)
      : JSON.stringify({ validation_errors: preview.errors }, null, 2);
  }
  renderValidationSummary();
  renderValidationDiagnostics();
  renderPresetSelect();
  renderStaticJobPanel();
  renderStaticArtifacts();
}
