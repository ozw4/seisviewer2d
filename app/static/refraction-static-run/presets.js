import { GEOMETRY_DEFAULTS, PRESET_STORAGE_KEY } from './constants.js';
import { collectPresetInputs } from './form_collectors.js';
import { getStaticCorrectionDom, requestStaticCorrectionRender } from './runtime.js';
import { state } from './state.js';
import {
  applyStaticCorrectionGeometryPreset,
  updateModelPresetControls,
  updateStaticCorrectionFieldCorrectionOptions,
  updateStaticCorrectionLinkageOptions,
} from './ui.js';
import { setElementChecked, setElementValue, trimValue } from './utils.js';

export function readStoredPresets() {
  let parsed = null;
  try {
    parsed = JSON.parse(window.localStorage.getItem(PRESET_STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) {
    return [];
  }
  return parsed
    .filter((entry) => entry && typeof entry.name === 'string' && entry.values)
    .map((entry) => ({
      name: trimValue(entry.name),
      values: entry.values,
    }))
    .filter((entry) => entry.name)
    .sort((left, right) => left.name.localeCompare(right.name));
}
export function writeStoredPresets(presets) {
  window.localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(presets));
}
export function applyPresetValues(values, targetDom = getStaticCorrectionDom()) {
  if (!targetDom || !values) return;

  const geometry = values.geometry || {};
  setElementValue(targetDom.geometryPreset, geometry.preset);
  setElementValue(targetDom.sourceIdByte, geometry.source_id_byte);
  setElementValue(targetDom.receiverIdByte, geometry.receiver_id_byte);
  setElementValue(targetDom.sourceXByte, geometry.source_x_byte);
  setElementValue(targetDom.sourceYByte, geometry.source_y_byte);
  setElementValue(targetDom.receiverXByte, geometry.receiver_x_byte);
  setElementValue(targetDom.receiverYByte, geometry.receiver_y_byte);
  setElementValue(targetDom.sourceElevationByte, geometry.source_elevation_byte);
  setElementValue(targetDom.receiverElevationByte, geometry.receiver_elevation_byte);
  setElementValue(targetDom.coordinateScalarByte, geometry.coordinate_scalar_byte);
  setElementValue(targetDom.elevationScalarByte, geometry.elevation_scalar_byte);
  setElementValue(targetDom.sourceDepthByte, geometry.source_depth_byte);
  setElementValue(targetDom.coordinateUnit, geometry.coordinate_unit);
  setElementValue(targetDom.elevationUnit, geometry.elevation_unit);
  setElementValue(targetDom.offsetByte, geometry.offset_byte);

  const linkage = values.linkage || {};
  setElementChecked(targetDom.enableLinkage, linkage.enabled);
  setElementValue(targetDom.linkageMode, linkage.mode);
  setElementValue(targetDom.linkageThresholdM, linkage.threshold_m);
  setElementValue(targetDom.receiverLocationIntervalM, linkage.receiver_location_interval_m);
  setElementChecked(targetDom.preferReceiverAnchor, linkage.prefer_receiver_anchor);

  const model = values.model || {};
  setElementValue(targetDom.modelKind, model.model_preset);
  setElementValue(targetDom.weatheringVelocityMS, model.weathering_velocity_m_s);
  setElementValue(targetDom.bedrockVelocityMode, model.bedrock_velocity_mode);
  setElementValue(targetDom.initialBedrockVelocityMS, model.initial_bedrock_velocity_m_s);
  setElementValue(targetDom.fixedBedrockVelocityMS, model.bedrock_velocity_m_s);
  setElementValue(targetDom.minOffsetM, model.min_offset_m);
  setElementValue(targetDom.maxOffsetM, model.max_offset_m);
  setElementValue(targetDom.conversionMode, model.conversion_mode);
  setElementValue(targetDom.v3MinOffsetM, model.v3_min_offset_m);
  setElementValue(targetDom.v3MaxOffsetM, model.v3_max_offset_m);
  setElementValue(targetDom.initialV3VelocityMS, model.initial_v3_velocity_m_s);
  setElementValue(targetDom.vsubMinOffsetM, model.vsub_min_offset_m);
  setElementValue(targetDom.initialVsubVelocityMS, model.initial_vsub_velocity_m_s);
  setElementValue(targetDom.cellXOriginM, model.cell_x_origin_m);
  setElementValue(targetDom.cellYOriginM, model.cell_y_origin_m);
  setElementValue(targetDom.cellCountX, model.cell_count_x);
  setElementValue(targetDom.cellCountY, model.cell_count_y);
  setElementValue(targetDom.cellSizeXM, model.cell_size_x_m);
  setElementValue(targetDom.cellSizeYM, model.cell_size_y_m);
  setElementValue(targetDom.cellMinObservations, model.cell_min_observations);
  setElementValue(targetDom.cellSmoothingWeight, model.cell_smoothing_weight);
  setElementValue(targetDom.lineOriginXM, model.line_origin_x_m);
  setElementValue(targetDom.lineOriginYM, model.line_origin_y_m);
  setElementValue(targetDom.lineAzimuthDeg, model.line_azimuth_deg);

  const fieldCorrections = values.field_corrections || {};
  setElementChecked(targetDom.fieldCorrectionsEnabled, fieldCorrections.enabled);
  setElementValue(targetDom.fieldSourceDepthMode, fieldCorrections.source_depth_mode);
  setElementValue(targetDom.fieldSourceDepthByte, fieldCorrections.source_depth_byte);
  setElementValue(targetDom.fieldUpholeMode, fieldCorrections.uphole_mode);
  setElementValue(targetDom.fieldUpholeTimeByte, fieldCorrections.uphole_time_byte);
  setElementValue(targetDom.fieldManualStaticMode, fieldCorrections.manual_static_mode);
  setElementValue(
    targetDom.fieldManualStaticSignConvention,
    fieldCorrections.manual_static_sign_convention
  );
  setElementValue(targetDom.fieldManualSourceJobId, fieldCorrections.manual_source_job_id);
  setElementValue(
    targetDom.fieldManualSourceArtifactName,
    fieldCorrections.manual_source_artifact_name
  );
  setElementValue(targetDom.fieldManualReceiverJobId, fieldCorrections.manual_receiver_job_id);
  setElementValue(
    targetDom.fieldManualReceiverArtifactName,
    fieldCorrections.manual_receiver_artifact_name
  );
  setElementChecked(targetDom.fieldApplyToTraceShift, fieldCorrections.apply_to_trace_shift);

  const output = values.output || {};
  setElementChecked(targetDom.registerCorrectedFile, output.register_corrected_file);
  setElementChecked(targetDom.exportEnabled, output.export_enabled);
  if (Array.isArray(output.export_formats)) {
    for (const checkbox of targetDom.exportFormatInputs || []) {
      checkbox.checked = output.export_formats.includes(checkbox.value);
    }
  }

  applyStaticCorrectionGeometryPreset(
    targetDom,
    trimValue(targetDom.geometryPreset && targetDom.geometryPreset.value) || GEOMETRY_DEFAULTS.preset
  );
  updateModelPresetControls(targetDom);
  updateStaticCorrectionLinkageOptions(targetDom);
  updateStaticCorrectionFieldCorrectionOptions(targetDom);
}
export function saveCurrentPreset() {
  const dom = getStaticCorrectionDom();
  if (!dom) return;
  const name = trimValue(dom.presetName && dom.presetName.value);
  if (!name) {
    state.error = 'Preset name is required.';
    state.message = 'Enter a preset name before saving.';
    requestStaticCorrectionRender();
    return;
  }
  const presets = readStoredPresets().filter((entry) => entry.name !== name);
  presets.push({
    name,
    values: collectPresetInputs(dom),
  });
  presets.sort((left, right) => left.name.localeCompare(right.name));
  writeStoredPresets(presets);
  state.presets = presets;
  if (dom.presetSelect) {
    dom.presetSelect.value = name;
  }
  state.error = '';
  state.message = `Saved preset ${name}.`;
  requestStaticCorrectionRender();
}
export function loadSelectedPreset() {
  const dom = getStaticCorrectionDom();
  if (!dom || !dom.presetSelect) return;
  const name = trimValue(dom.presetSelect.value);
  const preset = state.presets.find((entry) => entry.name === name);
  if (!preset) {
    state.error = 'Choose a saved preset to load.';
    state.message = 'No preset was loaded.';
    requestStaticCorrectionRender();
    return;
  }
  applyPresetValues(preset.values, dom);
  if (dom.presetName) {
    dom.presetName.value = preset.name;
  }
  state.error = '';
  state.showValidationSummary = false;
  state.validationErrors = [];
  state.message = `Loaded preset ${preset.name}. Current viewer target and selected pick file were kept.`;
  requestStaticCorrectionRender();
}
export function deleteSelectedPreset() {
  const dom = getStaticCorrectionDom();
  if (!dom || !dom.presetSelect) return;
  const name = trimValue(dom.presetSelect.value);
  if (!name) {
    state.error = 'Choose a saved preset to delete.';
    state.message = 'No preset was deleted.';
    requestStaticCorrectionRender();
    return;
  }
  const presets = readStoredPresets().filter((entry) => entry.name !== name);
  writeStoredPresets(presets);
  state.presets = presets;
  if (dom.presetName && dom.presetName.value === name) {
    dom.presetName.value = '';
  }
  state.error = '';
  state.message = `Deleted preset ${name}.`;
  requestStaticCorrectionRender();
}
