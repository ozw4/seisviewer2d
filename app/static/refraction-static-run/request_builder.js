import { LINKAGE_ARTIFACT_NAME, UPLOADED_PICK_KIND } from './constants.js';
import {
  collectInputs,
  hasNpzExtension,
  selectedPickNpzFile,
} from './form_collectors.js';
import { getStaticCorrectionDom } from './runtime.js';
import { state } from './state.js';
import { validateStaticCorrectionTarget } from './target.js';
import { trimValue } from './utils.js';
import {
  validationError,
  validateOneLayerRefractionModel,
  validateStaticCorrectionGeometryRequest,
  validateStaticCorrectionLinkageRequest,
  validateRefractionStaticModel,
  validateStaticCorrectionFieldCorrections,
  validateStaticCorrectionOutput,
} from './validation.js';

export function buildStaticCorrectionPickSource(targetDom = getStaticCorrectionDom()) {
  const values = collectInputs(targetDom);
  const errors = [];
  if (!values) {
    throw validationError(['Static correction form is not available.']);
  }
  const pickFile = selectedPickNpzFile(targetDom);
  if (!pickFile) {
    errors.push('First-break pick NPZ is required.');
  } else if (!hasNpzExtension(pickFile.name)) {
    errors.push('First-break pick file must use the .npz extension.');
  }
  if (errors.length) {
    throw validationError(errors);
  }
  return { kind: UPLOADED_PICK_KIND };
}
export function buildStaticCorrectionGeometryRequest(targetDom = getStaticCorrectionDom()) {
  const result = validateStaticCorrectionGeometryRequest(targetDom);
  if (result.errors.length) {
    throw validationError(result.errors);
  }
  return result.payload;
}
export function buildStaticCorrectionGeometry(targetDom = getStaticCorrectionDom()) {
  return buildStaticCorrectionGeometryRequest(targetDom).geometry;
}
export function buildStaticCorrectionLinkageRequest(targetDom = getStaticCorrectionDom()) {
  const result = validateStaticCorrectionLinkageRequest(targetDom);
  if (result.errors.length) {
    throw validationError(result.errors);
  }
  return result.payload;
}
export function buildStaticCorrectionLinkage(targetDom = getStaticCorrectionDom(), linkageJobId = '') {
  const jobId = trimValue(linkageJobId);
  if (jobId) {
    return {
      mode: 'required',
      job_id: jobId,
      artifact_name: LINKAGE_ARTIFACT_NAME,
    };
  }
  return { mode: 'none' };
}
export function buildOneLayerRefractionModel(targetDom = getStaticCorrectionDom()) {
  const result = validateOneLayerRefractionModel(targetDom);
  if (result.errors.length) {
    throw validationError(result.errors);
  }
  return result.payload;
}
export function buildRefractionStaticApplyRequest(targetDom = getStaticCorrectionDom(), options = {}) {
  const values = collectInputs(targetDom);
  const errors = [];
  if (!values) {
    throw validationError(['Static correction form is not available.']);
  }
  const targetResult = validateStaticCorrectionTarget();
  errors.push(...targetResult.errors);
  const target = targetResult.target;
  const key1Byte = target ? target.key1_byte : null;
  const key2Byte = target ? target.key2_byte : null;

  let pickSource = null;
  try {
    pickSource = buildStaticCorrectionPickSource(targetDom);
  } catch (error) {
    errors.push(...(error.errors || [error.message || String(error)]));
  }

  const geometryResult = validateStaticCorrectionGeometryRequest(targetDom);
  errors.push(...geometryResult.errors);
  const modelResult = validateRefractionStaticModel(targetDom);
  errors.push(...modelResult.errors);
  const fieldCorrectionResult = validateStaticCorrectionFieldCorrections(targetDom);
  errors.push(...fieldCorrectionResult.errors);
  const outputResult = validateStaticCorrectionOutput(targetDom);
  errors.push(...outputResult.errors);
  const linkage = buildStaticCorrectionLinkage(targetDom, options.linkageJobId || '');

  if (errors.length) {
    throw validationError(errors);
  }

  return {
    file_id: target.file_id,
    key1_byte: key1Byte,
    key2_byte: key2Byte,
    pick_source: pickSource,
    geometry: geometryResult.payload.geometry,
    linkage,
    model: modelResult.payload,
    moveout: {
      ...geometryResult.payload.moveout,
      ...modelResult.moveout,
      model: 'head_wave_linear_offset',
      distance_source: 'geometry',
    },
    conversion: modelResult.conversion,
    ...(fieldCorrectionResult.payload || {}),
    ...outputResult.payload,
  };
}
export function buildStaticCorrectionRequest() {
  const values = collectInputs(getStaticCorrectionDom());
  const errors = [];
  if (!values) {
    return { payload: null, errors: ['Static correction form is not available.'] };
  }

  const targetResult = validateStaticCorrectionTarget();
  errors.push(...targetResult.errors);
  const target = targetResult.target;
  const key1Byte = target ? target.key1_byte : null;
  const key2Byte = target ? target.key2_byte : null;
  let pickSource = null;
  try {
    pickSource = buildStaticCorrectionPickSource(getStaticCorrectionDom());
  } catch (error) {
    errors.push(...(error.errors || [error.message || String(error)]));
  }
  const geometryResult = validateStaticCorrectionGeometryRequest(getStaticCorrectionDom());
  errors.push(...geometryResult.errors);
  const linkageResult = validateStaticCorrectionLinkageRequest(getStaticCorrectionDom());
  errors.push(...linkageResult.errors);
  const modelResult = validateRefractionStaticModel(getStaticCorrectionDom());
  errors.push(...modelResult.errors);
  const fieldCorrectionResult = validateStaticCorrectionFieldCorrections(getStaticCorrectionDom());
  errors.push(...fieldCorrectionResult.errors);
  const outputResult = validateStaticCorrectionOutput(getStaticCorrectionDom());
  errors.push(...outputResult.errors);

  if (errors.length) {
    return { payload: null, errors };
  }
  return {
    payload: {
      file_id: target.file_id,
      key1_byte: key1Byte,
      key2_byte: key2Byte,
      pick_source: pickSource,
      geometry: geometryResult.payload.geometry,
      ...linkageResult.payload,
      model: modelResult.payload,
      moveout: {
        ...geometryResult.payload.moveout,
        ...modelResult.moveout,
        model: 'head_wave_linear_offset',
        distance_source: 'geometry',
      },
      conversion: modelResult.conversion,
      ...(fieldCorrectionResult.payload || {}),
      ...outputResult.payload,
    },
    errors,
  };
}
export function buildStaticCorrectionPreviewRequest(targetDom = getStaticCorrectionDom()) {
  const preview = buildRefractionStaticApplyRequest(targetDom, {
    linkageJobId: state.lastLinkageJobId,
  });
  if (targetDom && targetDom.enableLinkage && targetDom.enableLinkage.checked && !state.lastLinkageJobId) {
    preview.linkage = {
      mode: 'required',
      artifact_name: LINKAGE_ARTIFACT_NAME,
    };
  }
  return preview;
}
export function getStaticCorrectionValidationSnapshot(targetDom = getStaticCorrectionDom()) {
  try {
    return {
      payload: buildStaticCorrectionPreviewRequest(targetDom),
      errors: [],
    };
  } catch (error) {
    return {
      payload: null,
      errors: error && Array.isArray(error.errors)
        ? error.errors
        : [error && error.message ? error.message : String(error)],
    };
  }
}
export function buildStaticLinkageBuildRequest(targetDom = getStaticCorrectionDom()) {
  const values = collectInputs(targetDom);
  const errors = [];
  if (!values) {
    throw validationError(['Static correction form is not available.']);
  }
  const targetResult = validateStaticCorrectionTarget();
  errors.push(...targetResult.errors);
  const target = targetResult.target;
  const key1Byte = target ? target.key1_byte : null;
  const key2Byte = target ? target.key2_byte : null;
  const geometryResult = validateStaticCorrectionGeometryRequest(targetDom);
  errors.push(...geometryResult.errors);
  const linkageResult = validateStaticCorrectionLinkageRequest(targetDom);
  errors.push(...linkageResult.errors);
  if (errors.length) {
    throw validationError(errors);
  }
  return {
    file_id: target.file_id,
    key1_byte: key1Byte,
    key2_byte: key2Byte,
    geometry: { ...geometryResult.payload.geometry },
    linkage: { ...linkageResult.payload.linkage },
  };
}
export function buildStaticCorrectionFormData(request, targetDom = getStaticCorrectionDom()) {
  const pickFile = selectedPickNpzFile(targetDom);
  if (!pickFile) {
    throw validationError(['First-break pick NPZ is required.']);
  }
  const formData = new FormData();
  formData.append('request_json', JSON.stringify(request));
  formData.append('pick_npz', pickFile, pickFile.name || 'first_breaks.npz');
  return formData;
}
