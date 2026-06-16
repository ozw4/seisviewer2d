import {
  ACTIVE_VIEWER_TARGET_STORAGE_KEY,
  DEFAULTS,
  GEOMETRY_DEFAULTS,
  GEOMETRY_HEADER_FIELDS,
  GEOMETRY_PRESET_CUSTOM,
  PRESET_STORAGE_KEY,
  STATIC_DRAFT_STORAGE_KEY,
  STATIC_PICK_DB_NAME,
  STATIC_PICK_STORE,
} from './constants.js';
import { loadStaticArtifacts } from './artifacts_view.js';
import {
  clearRestoredPickNpz,
  clearStaticCorrectionDraft,
  readStaticCorrectionDraft,
  restoreStaticCorrectionDraftIfAvailable,
  saveStaticCorrectionDraft,
} from './draft.js';
import {
  collectFieldCorrectionInputs,
  collectGeometryInputs,
  collectInputs,
  collectLinkageInputs,
  collectModelInputs,
  collectOutputInputs,
  collectPresetInputs,
  hasNpzExtension,
  selectedPickNpzFile,
} from './form_collectors.js';
import {
  cancelStaticCorrectionJob,
  normalizeStaticJobState,
  pollStaticCorrectionJobUntilTerminal,
  pollStaticCorrectionStatus,
  pollStaticJobUntilReady,
  runStaticCorrection,
  stopStaticCorrectionPolling,
  validateStaticCorrectionInputs,
} from './job_runner.js';
import {
  deletePickNpzFromIndexedDb,
  loadPickNpzFromIndexedDb,
  pickRecordMetadata,
  savePickNpzToIndexedDb,
  staticPickRecordId,
} from './picks_db.js';
import {
  applyPresetValues,
  deleteSelectedPreset,
  loadSelectedPreset,
  readStoredPresets,
  saveCurrentPreset,
  writeStoredPresets,
} from './presets.js';
import {
  buildOneLayerRefractionModel,
  buildRefractionStaticApplyRequest,
  buildStaticCorrectionFormData,
  buildStaticCorrectionGeometry,
  buildStaticCorrectionGeometryRequest,
  buildStaticCorrectionLinkage,
  buildStaticCorrectionLinkageRequest,
  buildStaticCorrectionPickSource,
  buildStaticCorrectionRequest,
  buildStaticLinkageBuildRequest,
  getStaticCorrectionValidationSnapshot,
} from './request_builder.js';
import { setStaticCorrectionDom, setStaticCorrectionRender } from './runtime.js';
import { state } from './state.js';
import {
  buildRefractionQcUrl,
  getStandaloneStaticCorrectionTargetState,
  getStaticCorrectionTarget,
  getStaticCorrectionTargetState,
  normalizeStandaloneTargetCandidate,
  validateStaticCorrectionTarget,
} from './target.js';
import {
  applyStaticCorrectionGeometryPreset,
  render,
  updateBedrockVelocityControls,
  updateModelPresetControls,
  updateStaticCorrectionFieldCorrectionOptions,
  updateStaticCorrectionLinkageOptions,
} from './ui.js';
import {
  validateCellRefractionModel,
  validateMultilayerRefractionModel,
  validateOneLayerRefractionModel,
  validateRefractionStaticModel,
  validateStaticCorrectionFieldCorrections,
  validateStaticCorrectionGeometryRequest,
  validateStaticCorrectionLinkageRequest,
} from './validation.js';
import { setDefaultValue, trimValue } from './utils.js';

let dom = null;
let viewerTargetUnsubscribe = null;

function handleRun(event) {
  if (event) {
    event.preventDefault();
  }
  runStaticCorrection();
}

function handleValidate(event) {
  if (event) {
    event.preventDefault();
  }
  validateStaticCorrectionInputs();
}

function subscribeToViewerTargetUpdates() {
  if (viewerTargetUnsubscribe || !window.store || typeof window.store.subscribe !== 'function') {
    return;
  }
  viewerTargetUnsubscribe = window.store.subscribe(() => {
    restoreStaticCorrectionDraftIfAvailable();
    render();
  });
}

function init() {
  const form = document.getElementById('staticCorrectionForm');
  const status = document.getElementById('staticCorrectionStatus');
  const error = document.getElementById('staticCorrectionError');
  const validateButton = document.getElementById('staticCorrectionValidateButton');
  const runButton = document.getElementById('staticCorrectionRunButton');
  const targetEmpty = document.getElementById('staticCorrectionTargetEmpty');
  const targetDetails = document.getElementById('staticCorrectionTargetDetails');
  const targetFile = document.getElementById('staticCorrectionTargetFile');
  const targetKeys = document.getElementById('staticCorrectionTargetKeys');
  const targetStatus = document.getElementById('staticCorrectionTargetStatus');
  const presetSelect = document.getElementById('staticCorrectionPresetSelect');
  const presetName = document.getElementById('staticCorrectionPresetName');
  const savePresetButton = document.getElementById('staticCorrectionSavePresetButton');
  const loadPresetButton = document.getElementById('staticCorrectionLoadPresetButton');
  const deletePresetButton = document.getElementById('staticCorrectionDeletePresetButton');
  const pickNpzFile = document.getElementById('staticCorrectionPickNpz');
  const pickNpzSummary = document.getElementById('staticCorrectionPickNpzSummary');
  const replacePickNpzButton = document.getElementById('staticCorrectionReplacePickNpzButton');
  const clearPickNpzButton = document.getElementById('staticCorrectionClearPickNpzButton');
  const clearDraftButton = document.getElementById('staticCorrectionClearDraftButton');
  const geometryPreset = document.getElementById('staticCorrectionGeometryPreset');
  const sourceIdByte = document.getElementById('staticCorrectionSourceIdByte');
  const receiverIdByte = document.getElementById('staticCorrectionReceiverIdByte');
  const sourceXByte = document.getElementById('staticCorrectionSourceXByte');
  const sourceYByte = document.getElementById('staticCorrectionSourceYByte');
  const receiverXByte = document.getElementById('staticCorrectionReceiverXByte');
  const receiverYByte = document.getElementById('staticCorrectionReceiverYByte');
  const sourceElevationByte = document.getElementById('staticCorrectionSourceElevationByte');
  const receiverElevationByte = document.getElementById('staticCorrectionReceiverElevationByte');
  const coordinateScalarByte = document.getElementById('staticCorrectionCoordinateScalarByte');
  const elevationScalarByte = document.getElementById('staticCorrectionElevationScalarByte');
  const sourceDepthByte = document.getElementById('staticCorrectionSourceDepthByte');
  const coordinateUnit = document.getElementById('staticCorrectionCoordinateUnit');
  const elevationUnit = document.getElementById('staticCorrectionElevationUnit');
  const offsetByte = document.getElementById('staticCorrectionOffsetByte');
  const enableLinkage = document.getElementById('staticCorrectionEnableLinkage');
  const linkageOptions = document.getElementById('staticCorrectionLinkageOptions');
  const linkageMode = document.getElementById('staticCorrectionLinkageMode');
  const linkageThresholdM = document.getElementById('staticCorrectionLinkageThresholdM');
  const receiverLocationIntervalM = document.getElementById(
    'staticCorrectionReceiverLocationIntervalM'
  );
  const preferReceiverAnchor = document.getElementById('staticCorrectionPreferReceiverAnchor');
  const modelKind = document.getElementById('staticCorrectionModelKind');
  const weatheringVelocityMS = document.getElementById('staticCorrectionWeatheringVelocityMS');
  const bedrockVelocityMode = document.getElementById('staticCorrectionBedrockVelocityMode');
  const initialBedrockVelocityMS = document.getElementById('staticCorrectionInitialBedrockVelocityMS');
  const fixedBedrockVelocityMS = document.getElementById('staticCorrectionFixedBedrockVelocityMS');
  const minOffsetM = document.getElementById('staticCorrectionMinOffsetM');
  const maxOffsetM = document.getElementById('staticCorrectionMaxOffsetM');
  const conversionMode = document.getElementById('staticCorrectionConversionMode');
  const v3LayerFields = document.getElementById('staticCorrectionV3LayerFields');
  const v3MinOffsetM = document.getElementById('staticCorrectionV3MinOffsetM');
  const v3MaxOffsetM = document.getElementById('staticCorrectionV3MaxOffsetM');
  const initialV3VelocityMS = document.getElementById('staticCorrectionInitialV3VelocityMS');
  const vsubLayerFields = document.getElementById('staticCorrectionVsubLayerFields');
  const vsubMinOffsetM = document.getElementById('staticCorrectionVsubMinOffsetM');
  const initialVsubVelocityMS = document.getElementById('staticCorrectionInitialVsubVelocityMS');
  const cellFields = document.getElementById('staticCorrectionCellFields');
  const cellXOriginM = document.getElementById('staticCorrectionCellXOriginM');
  const cellYOriginM = document.getElementById('staticCorrectionCellYOriginM');
  const cellCountX = document.getElementById('staticCorrectionCellCountX');
  const cellCountY = document.getElementById('staticCorrectionCellCountY');
  const cellSizeXM = document.getElementById('staticCorrectionCellSizeXM');
  const cellSizeYM = document.getElementById('staticCorrectionCellSizeYM');
  const cellMinObservations = document.getElementById('staticCorrectionCellMinObservations');
  const cellSmoothingWeight = document.getElementById('staticCorrectionCellSmoothingWeight');
  const line2DFields = document.getElementById('staticCorrectionLine2DFields');
  const lineOriginXM = document.getElementById('staticCorrectionLineOriginXM');
  const lineOriginYM = document.getElementById('staticCorrectionLineOriginYM');
  const lineAzimuthDeg = document.getElementById('staticCorrectionLineAzimuthDeg');
  const fieldCorrectionsEnabled = document.getElementById('staticCorrectionFieldCorrectionsEnabled');
  const fieldCorrectionOptions = document.getElementById('staticCorrectionFieldCorrectionOptions');
  const fieldSourceDepthMode = document.getElementById('staticCorrectionFieldSourceDepthMode');
  const fieldSourceDepthByte = document.getElementById('staticCorrectionFieldSourceDepthByte');
  const fieldUpholeMode = document.getElementById('staticCorrectionFieldUpholeMode');
  const fieldUpholeTimeByte = document.getElementById('staticCorrectionFieldUpholeTimeByte');
  const fieldManualStaticMode = document.getElementById('staticCorrectionFieldManualStaticMode');
  const fieldManualArtifactFields = document.getElementById('staticCorrectionFieldManualArtifactFields');
  const fieldManualStaticSignConvention = document.getElementById(
    'staticCorrectionFieldManualStaticSignConvention'
  );
  const fieldManualSourceJobId = document.getElementById('staticCorrectionFieldManualSourceJobId');
  const fieldManualSourceArtifactName = document.getElementById(
    'staticCorrectionFieldManualSourceArtifactName'
  );
  const fieldManualReceiverJobId = document.getElementById('staticCorrectionFieldManualReceiverJobId');
  const fieldManualReceiverArtifactName = document.getElementById(
    'staticCorrectionFieldManualReceiverArtifactName'
  );
  const fieldApplyToTraceShift = document.getElementById('staticCorrectionFieldApplyToTraceShift');
  const registerCorrectedFile = document.getElementById('staticCorrectionRegisterCorrectedFile');
  const exportEnabled = document.getElementById('staticCorrectionExportEnabled');
  const exportFormatInputs = Array.from(document.querySelectorAll('[data-static-correction-export-format]'));
  const validationSummary = document.getElementById('staticCorrectionValidationSummary');
  const validationDiagnostics = document.getElementById('staticCorrectionValidationDiagnostics');
  const requestPreview = document.getElementById('staticCorrectionRequestPreview');
  const cancelButton = document.getElementById('staticCorrectionCancelButton');
  const staticJobPanel = document.getElementById('staticCorrectionJobPanel');
  const staticJobIdValue = document.getElementById('staticCorrectionJobIdValue');
  const staticJobStateValue = document.getElementById('staticCorrectionJobStateValue');
  const staticJobMessageValue = document.getElementById('staticCorrectionJobMessageValue');
  const staticJobProgress = document.getElementById('staticCorrectionJobProgress');
  const staticJobProgressValue = document.getElementById('staticCorrectionJobProgressValue');
  const staticArtifactTable = document.getElementById('staticCorrectionArtifactTable');
  const staticArtifactBody = document.getElementById('staticCorrectionArtifactBody');
  const staticArtifactEmpty = document.getElementById('staticCorrectionArtifactEmpty');
  const qcLinkRow = document.getElementById('staticCorrectionQcLinkRow');
  const qcLink = document.getElementById('staticCorrectionQcLink');
  if (
    !form || !status || !error || !validateButton || !runButton || !targetEmpty || !targetDetails
    || !targetFile || !targetKeys || !targetStatus
    || !presetSelect || !presetName || !savePresetButton || !loadPresetButton
    || !deletePresetButton
    || !pickNpzFile || !pickNpzSummary || !replacePickNpzButton || !clearPickNpzButton
    || !clearDraftButton || !geometryPreset || !sourceIdByte || !receiverIdByte
    || !sourceXByte || !sourceYByte || !receiverXByte || !receiverYByte
    || !sourceElevationByte || !receiverElevationByte || !coordinateScalarByte
    || !elevationScalarByte || !sourceDepthByte || !coordinateUnit || !elevationUnit
    || !offsetByte || !enableLinkage || !linkageOptions || !linkageMode
    || !linkageThresholdM || !receiverLocationIntervalM || !preferReceiverAnchor
    || !modelKind || !weatheringVelocityMS || !bedrockVelocityMode || !initialBedrockVelocityMS
    || !fixedBedrockVelocityMS || !minOffsetM || !maxOffsetM || !conversionMode
    || !v3LayerFields || !v3MinOffsetM || !v3MaxOffsetM || !initialV3VelocityMS
    || !vsubLayerFields || !vsubMinOffsetM || !initialVsubVelocityMS
    || !cellFields || !cellXOriginM || !cellYOriginM || !cellCountX || !cellCountY
    || !cellSizeXM || !cellSizeYM || !cellMinObservations || !cellSmoothingWeight
    || !line2DFields || !lineOriginXM || !lineOriginYM || !lineAzimuthDeg
    || !fieldCorrectionsEnabled || !fieldCorrectionOptions || !fieldSourceDepthMode
    || !fieldSourceDepthByte || !fieldUpholeMode || !fieldUpholeTimeByte
    || !fieldManualStaticMode || !fieldManualArtifactFields
    || !fieldManualStaticSignConvention || !fieldManualSourceJobId
    || !fieldManualSourceArtifactName || !fieldManualReceiverJobId
    || !fieldManualReceiverArtifactName || !fieldApplyToTraceShift
    || !registerCorrectedFile || !exportEnabled || !validationSummary
    || !validationDiagnostics || !requestPreview
    || !cancelButton || !staticJobPanel || !staticJobIdValue || !staticJobStateValue
    || !staticJobMessageValue || !staticJobProgress || !staticJobProgressValue
    || !staticArtifactTable || !staticArtifactBody || !staticArtifactEmpty
    || exportFormatInputs.length === 0
  ) {
    return;
  }

  setDefaultValue(linkageMode, DEFAULTS.linkageMode);
  setDefaultValue(linkageThresholdM, DEFAULTS.linkageThresholdM);
  setDefaultValue(modelKind, DEFAULTS.modelPreset);
  setDefaultValue(weatheringVelocityMS, DEFAULTS.weatheringVelocityMS);
  setDefaultValue(bedrockVelocityMode, DEFAULTS.bedrockVelocityMode);
  setDefaultValue(initialBedrockVelocityMS, DEFAULTS.initialBedrockVelocityMS);
  setDefaultValue(fixedBedrockVelocityMS, DEFAULTS.fixedBedrockVelocityMS);
  setDefaultValue(minOffsetM, DEFAULTS.minOffsetM);
  setDefaultValue(maxOffsetM, DEFAULTS.maxOffsetM);
  setDefaultValue(conversionMode, DEFAULTS.conversionMode);
  setDefaultValue(v3MinOffsetM, DEFAULTS.v3MinOffsetM);
  setDefaultValue(v3MaxOffsetM, DEFAULTS.v3MaxOffsetM);
  setDefaultValue(initialV3VelocityMS, DEFAULTS.initialV3VelocityMS);
  setDefaultValue(vsubMinOffsetM, DEFAULTS.vsubMinOffsetM);
  setDefaultValue(initialVsubVelocityMS, DEFAULTS.initialVsubVelocityMS);
  setDefaultValue(cellXOriginM, DEFAULTS.cellXOriginM);
  setDefaultValue(cellYOriginM, DEFAULTS.cellYOriginM);
  setDefaultValue(cellCountX, DEFAULTS.cellCountX);
  setDefaultValue(cellCountY, DEFAULTS.cellCountY);
  setDefaultValue(cellSizeXM, DEFAULTS.cellSizeXM);
  setDefaultValue(cellSizeYM, DEFAULTS.cellSizeYM);
  setDefaultValue(cellMinObservations, DEFAULTS.cellMinObservations);
  setDefaultValue(cellSmoothingWeight, DEFAULTS.cellSmoothingWeight);
  setDefaultValue(lineOriginXM, DEFAULTS.lineOriginXM);
  setDefaultValue(lineOriginYM, DEFAULTS.lineOriginYM);
  setDefaultValue(lineAzimuthDeg, DEFAULTS.lineAzimuthDeg);

  dom = {
    form,
    status,
    error,
    validateButton,
    runButton,
    targetEmpty,
    targetDetails,
    targetFile,
    targetKeys,
    targetStatus,
    presetSelect,
    presetName,
    savePresetButton,
    loadPresetButton,
    deletePresetButton,
    pickNpzFile,
    pickNpzSummary,
    replacePickNpzButton,
    clearPickNpzButton,
    clearDraftButton,
    geometryPreset,
    sourceIdByte,
    receiverIdByte,
    sourceXByte,
    sourceYByte,
    receiverXByte,
    receiverYByte,
    sourceElevationByte,
    receiverElevationByte,
    coordinateScalarByte,
    elevationScalarByte,
    sourceDepthByte,
    coordinateUnit,
    elevationUnit,
    offsetByte,
    enableLinkage,
    linkageOptions,
    linkageMode,
    linkageThresholdM,
    receiverLocationIntervalM,
    preferReceiverAnchor,
    modelKind,
    weatheringVelocityMS,
    bedrockVelocityMode,
    initialBedrockVelocityMS,
    fixedBedrockVelocityMS,
    minOffsetM,
    maxOffsetM,
    conversionMode,
    v3LayerFields,
    v3MinOffsetM,
    v3MaxOffsetM,
    initialV3VelocityMS,
    vsubLayerFields,
    vsubMinOffsetM,
    initialVsubVelocityMS,
    cellFields,
    cellXOriginM,
    cellYOriginM,
    cellCountX,
    cellCountY,
    cellSizeXM,
    cellSizeYM,
    cellMinObservations,
    cellSmoothingWeight,
    line2DFields,
    lineOriginXM,
    lineOriginYM,
    lineAzimuthDeg,
    fieldCorrectionsEnabled,
    fieldCorrectionOptions,
    fieldSourceDepthMode,
    fieldSourceDepthByte,
    fieldUpholeMode,
    fieldUpholeTimeByte,
    fieldManualStaticMode,
    fieldManualArtifactFields,
    fieldManualStaticSignConvention,
    fieldManualSourceJobId,
    fieldManualSourceArtifactName,
    fieldManualReceiverJobId,
    fieldManualReceiverArtifactName,
    fieldApplyToTraceShift,
    registerCorrectedFile,
    exportEnabled,
    exportFormatInputs,
    validationSummary,
    validationDiagnostics,
    requestPreview,
    cancelButton,
    staticJobPanel,
    staticJobIdValue,
    staticJobStateValue,
    staticJobMessageValue,
    staticJobProgress,
    staticJobProgressValue,
    staticArtifactTable,
    staticArtifactBody,
    staticArtifactEmpty,
    qcLinkRow,
    qcLink,
  };
  setStaticCorrectionDom(dom);
  setStaticCorrectionRender(render);
  state.presets = readStoredPresets();
  subscribeToViewerTargetUpdates();
  if (window.viewerBootstrapReady && typeof window.viewerBootstrapReady.then === 'function') {
    window.viewerBootstrapReady.then(() => {
      subscribeToViewerTargetUpdates();
      restoreStaticCorrectionDraftIfAvailable();
      render();
    });
  }
  applyStaticCorrectionGeometryPreset(dom, trimValue(geometryPreset.value) || GEOMETRY_DEFAULTS.preset);
  updateModelPresetControls(dom);
  updateStaticCorrectionLinkageOptions(dom);
  updateStaticCorrectionFieldCorrectionOptions(dom);

  form.addEventListener('submit', handleRun);
  form.addEventListener('input', () => {
    if (!state.suppressPickChangeHandler) {
      state.draftCleared = false;
      saveStaticCorrectionDraft();
    }
    state.validationDiagnostics = null;
    if (state.showValidationSummary) {
      state.validationErrors = getStaticCorrectionValidationSnapshot(dom).errors;
    }
    render();
  });
  form.addEventListener('change', () => {
    if (!state.suppressPickChangeHandler) {
      state.draftCleared = false;
      saveStaticCorrectionDraft();
    }
    state.validationDiagnostics = null;
    if (state.showValidationSummary) {
      state.validationErrors = getStaticCorrectionValidationSnapshot(dom).errors;
    }
    render();
  });
  validateButton.addEventListener('click', handleValidate);
  runButton.addEventListener('click', handleRun);
  savePresetButton.addEventListener('click', (event) => {
    event.preventDefault();
    saveCurrentPreset();
  });
  loadPresetButton.addEventListener('click', (event) => {
    event.preventDefault();
    loadSelectedPreset();
  });
  deletePresetButton.addEventListener('click', (event) => {
    event.preventDefault();
    deleteSelectedPreset();
  });
  cancelButton.addEventListener('click', (event) => {
    event.preventDefault();
    cancelStaticCorrectionJob();
  });
  replacePickNpzButton.addEventListener('click', (event) => {
    event.preventDefault();
    pickNpzFile.click();
  });
  clearPickNpzButton.addEventListener('click', (event) => {
    event.preventDefault();
    clearRestoredPickNpz();
  });
  clearDraftButton.addEventListener('click', (event) => {
    event.preventDefault();
    clearStaticCorrectionDraft();
  });
  pickNpzFile.addEventListener('change', async () => {
    const pickFile = selectedPickNpzFile(dom);
    if (state.suppressPickChangeHandler) {
      render();
      return;
    }
    if (state.restoringPickInput) {
      render();
      return;
    }
    state.draftCleared = false;
    state.error = '';
    state.validationDiagnostics = null;
    state.showValidationSummary = false;
    state.validationErrors = [];
    state.pickNpzRestoreStatus = '';
    state.pickNpzRestoreMessage = '';
    if (pickFile) {
      const target = getStaticCorrectionTarget();
      if (target) {
        try {
          const record = await savePickNpzToIndexedDb(staticPickRecordId(target), pickFile, target);
          state.pickNpzDraftMeta = pickRecordMetadata(record);
          saveStaticCorrectionDraft();
          state.message = `Selected first-break pick NPZ ${pickFile.name}.`;
        } catch (error) {
          state.pickNpzDraftMeta = null;
          saveStaticCorrectionDraft({ pickNpz: null });
          state.error = error instanceof Error ? error.message : String(error);
          state.message = 'Selected NPZ could not be saved for restore.';
        }
      } else {
        state.pickNpzDraftMeta = null;
        state.message = `Selected first-break pick NPZ ${pickFile.name}.`;
      }
    } else {
      state.pickNpzDraftMeta = null;
      saveStaticCorrectionDraft({ pickNpz: null });
      state.message = 'Choose a first-break pick NPZ before running refraction statics.';
    }
    render();
  });
  geometryPreset.addEventListener('change', () => {
    applyStaticCorrectionGeometryPreset(dom, geometryPreset.value);
    state.error = '';
    state.message = geometryPreset.value === GEOMETRY_PRESET_CUSTOM
      ? 'Custom geometry header bytes can be edited.'
      : 'SEG-Y default geometry header bytes restored.';
    render();
  });
  enableLinkage.addEventListener('change', () => {
    updateStaticCorrectionLinkageOptions(dom);
    state.error = '';
    state.message = enableLinkage.checked
      ? 'Endpoint linkage options are enabled.'
      : 'Endpoint linkage is disabled for static correction.';
    render();
  });
  linkageMode.addEventListener('change', () => {
    updateStaticCorrectionLinkageOptions(dom);
    state.error = '';
    state.message = 'Endpoint linkage mode updated.';
    render();
  });
  modelKind.addEventListener('change', () => {
    updateModelPresetControls(dom);
    state.error = '';
    state.message = 'Refraction static model preset updated.';
    render();
  });
  updateBedrockVelocityControls(dom);
  bedrockVelocityMode.addEventListener('change', () => {
    updateBedrockVelocityControls(dom);
    state.error = '';
    state.message = bedrockVelocityMode.disabled
      ? 'Cell V2/T1 presets solve V2 by refractor cell.'
      : bedrockVelocityMode.value === 'fixed_global'
      ? 'Fixed global V2 will be submitted for the one-layer model.'
      : 'Global V2 will be solved from the submitted picks.';
    render();
  });
  fieldCorrectionsEnabled.addEventListener('change', () => {
    updateStaticCorrectionFieldCorrectionOptions(dom);
    state.error = '';
    state.message = fieldCorrectionsEnabled.checked
      ? 'Field correction options are enabled.'
      : 'Field corrections are disabled for this run.';
    render();
  });
  fieldSourceDepthMode.addEventListener('change', () => {
    updateStaticCorrectionFieldCorrectionOptions(dom);
    state.error = '';
    state.message = 'Source-depth correction mode updated.';
    render();
  });
  fieldUpholeMode.addEventListener('change', () => {
    updateStaticCorrectionFieldCorrectionOptions(dom);
    state.error = '';
    state.message = 'Uphole correction mode updated.';
    render();
  });
  fieldManualStaticMode.addEventListener('change', () => {
    updateStaticCorrectionFieldCorrectionOptions(dom);
    state.error = '';
    state.message = 'Manual static mode updated.';
    render();
  });

  window.addEventListener('pagehide', () => saveStaticCorrectionDraft());
  window.addEventListener('beforeunload', () => saveStaticCorrectionDraft());

  render();
  restoreStaticCorrectionDraftIfAvailable();
}

const staticCorrectionPublicApi = {
  applyStaticCorrectionGeometryPreset,
  buildOneLayerRefractionModel,
  buildRefractionStaticApplyRequest,
  buildStaticCorrectionGeometry,
  buildStaticCorrectionGeometryRequest,
  buildStaticCorrectionFormData,
  buildStaticCorrectionLinkage,
  buildStaticCorrectionLinkageRequest,
  buildStaticCorrectionPickSource,
  buildStaticCorrectionRequest,
  buildRefractionQcUrl,
  buildStaticLinkageBuildRequest,
  clearRestoredPickNpz,
  clearStaticCorrectionDraft,
  cancelStaticCorrectionJob,
  collectGeometryInputs,
  collectFieldCorrectionInputs,
  collectInputs,
  collectLinkageInputs,
  collectModelInputs,
  collectOutputInputs,
  collectPresetInputs,
  applyPresetValues,
  hasNpzExtension,
  deleteSelectedPreset,
  getStaticCorrectionValidationSnapshot,
  loadStaticArtifacts,
  loadSelectedPreset,
  loadPickNpzFromIndexedDb,
  normalizeStaticJobState,
  pollStaticCorrectionJobUntilTerminal,
  pollStaticCorrectionStatus,
  pollStaticJobUntilReady,
  render,
  runStaticCorrection,
  saveCurrentPreset,
  saveStaticCorrectionDraft,
  stopStaticCorrectionPolling,
  updateModelPresetControls,
  updateBedrockVelocityControls,
  updateStaticCorrectionFieldCorrectionOptions,
  updateStaticCorrectionLinkageOptions,
  validateStaticCorrectionFieldCorrections,
  validateCellRefractionModel,
  validateMultilayerRefractionModel,
  validateOneLayerRefractionModel,
  validateRefractionStaticModel,
  validateStaticCorrectionGeometryRequest,
  validateStaticCorrectionLinkageRequest,
  validateStaticCorrectionInputs,
};

function installStaticCorrectionPublicSurface() {
  window.refractionStaticRunState = state;
  window.refractionStaticRunUI = staticCorrectionPublicApi;
}

function initStaticCorrectionPage() {
  installStaticCorrectionPublicSurface();
  init();
  return staticCorrectionPublicApi;
}

installStaticCorrectionPublicSurface();

export {
  ACTIVE_VIEWER_TARGET_STORAGE_KEY,
  GEOMETRY_DEFAULTS,
  GEOMETRY_HEADER_FIELDS,
  PRESET_STORAGE_KEY,
  STATIC_DRAFT_STORAGE_KEY,
  STATIC_PICK_DB_NAME,
  STATIC_PICK_STORE,
  applyPresetValues,
  applyStaticCorrectionGeometryPreset,
  buildOneLayerRefractionModel,
  buildRefractionQcUrl,
  buildRefractionStaticApplyRequest,
  buildStaticCorrectionFormData,
  buildStaticCorrectionGeometry,
  buildStaticCorrectionGeometryRequest,
  buildStaticCorrectionLinkage,
  buildStaticCorrectionLinkageRequest,
  buildStaticCorrectionPickSource,
  buildStaticCorrectionRequest,
  buildStaticLinkageBuildRequest,
  cancelStaticCorrectionJob,
  clearRestoredPickNpz,
  clearStaticCorrectionDraft,
  collectFieldCorrectionInputs,
  collectGeometryInputs,
  collectInputs,
  collectLinkageInputs,
  collectModelInputs,
  collectOutputInputs,
  collectPresetInputs,
  deletePickNpzFromIndexedDb,
  deleteSelectedPreset,
  getStandaloneStaticCorrectionTargetState,
  getStaticCorrectionTarget,
  getStaticCorrectionTargetState,
  getStaticCorrectionValidationSnapshot,
  hasNpzExtension,
  initStaticCorrectionPage,
  installStaticCorrectionPublicSurface,
  loadPickNpzFromIndexedDb,
  loadSelectedPreset,
  loadStaticArtifacts,
  normalizeStandaloneTargetCandidate,
  normalizeStaticJobState,
  pickRecordMetadata,
  pollStaticCorrectionJobUntilTerminal,
  pollStaticCorrectionStatus,
  pollStaticJobUntilReady,
  readStaticCorrectionDraft,
  readStoredPresets,
  render,
  restoreStaticCorrectionDraftIfAvailable,
  saveCurrentPreset,
  savePickNpzToIndexedDb,
  saveStaticCorrectionDraft,
  selectedPickNpzFile,
  staticCorrectionPublicApi,
  state,
  stopStaticCorrectionPolling,
  updateBedrockVelocityControls,
  updateModelPresetControls,
  updateStaticCorrectionFieldCorrectionOptions,
  updateStaticCorrectionLinkageOptions,
  validateCellRefractionModel,
  validateMultilayerRefractionModel,
  validateOneLayerRefractionModel,
  validateRefractionStaticModel,
  validateStaticCorrectionFieldCorrections,
  validateStaticCorrectionGeometryRequest,
  validateStaticCorrectionInputs,
  validateStaticCorrectionLinkageRequest,
  validateStaticCorrectionTarget,
  writeStoredPresets,
};
