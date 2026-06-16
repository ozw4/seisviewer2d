import { UPLOADED_PICK_KIND } from './constants.js';
import { getStaticCorrectionDom } from './runtime.js';
import { getStaticCorrectionTarget } from './target.js';
import { trimValue } from './utils.js';

export function hasNpzExtension(name) {
  return trimValue(name).toLowerCase().endsWith('.npz');
}
export function selectedPickNpzFile(targetDom = getStaticCorrectionDom()) {
  if (!targetDom || !targetDom.pickNpzFile || !targetDom.pickNpzFile.files) {
    return null;
  }
  return targetDom.pickNpzFile.files[0] || null;
}
export function formatFileSize(bytes) {
  const size = Number(bytes);
  if (!Number.isFinite(size) || size < 0) {
    return '';
  }
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KiB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}
export function collectInputs(targetDom = getStaticCorrectionDom()) {
  if (!targetDom) return null;
  const target = getStaticCorrectionTarget();
  const pickFile = selectedPickNpzFile(targetDom);
  return {
    file_id: target ? target.file_id : '',
    key1_byte: target ? String(target.key1_byte) : '',
    key2_byte: target ? String(target.key2_byte) : '',
    pick_source: {
      kind: UPLOADED_PICK_KIND,
    },
    pick_npz: {
      name: pickFile ? trimValue(pickFile.name) : '',
      size: pickFile ? pickFile.size : null,
    },
  };
}
export function collectLinkageInputs(targetDom = getStaticCorrectionDom()) {
  if (!targetDom) return null;
  return {
    enabled: Boolean(targetDom.enableLinkage && targetDom.enableLinkage.checked),
    mode: trimValue(targetDom.linkageMode && targetDom.linkageMode.value),
    threshold_m: trimValue(targetDom.linkageThresholdM && targetDom.linkageThresholdM.value),
    receiver_location_interval_m: trimValue(
      targetDom.receiverLocationIntervalM && targetDom.receiverLocationIntervalM.value
    ),
    prefer_receiver_anchor: Boolean(
      targetDom.preferReceiverAnchor && targetDom.preferReceiverAnchor.checked
    ),
  };
}
export function collectGeometryInputs(targetDom = getStaticCorrectionDom()) {
  if (!targetDom) return null;
  return {
    preset: trimValue(targetDom.geometryPreset.value),
    source_id_byte: trimValue(targetDom.sourceIdByte.value),
    receiver_id_byte: trimValue(targetDom.receiverIdByte.value),
    source_x_byte: trimValue(targetDom.sourceXByte.value),
    source_y_byte: trimValue(targetDom.sourceYByte.value),
    receiver_x_byte: trimValue(targetDom.receiverXByte.value),
    receiver_y_byte: trimValue(targetDom.receiverYByte.value),
    source_elevation_byte: trimValue(targetDom.sourceElevationByte.value),
    receiver_elevation_byte: trimValue(targetDom.receiverElevationByte.value),
    coordinate_scalar_byte: trimValue(targetDom.coordinateScalarByte.value),
    elevation_scalar_byte: trimValue(targetDom.elevationScalarByte.value),
    source_depth_byte: trimValue(targetDom.sourceDepthByte.value),
    coordinate_unit: trimValue(targetDom.coordinateUnit.value),
    elevation_unit: trimValue(targetDom.elevationUnit.value),
    offset_byte: trimValue(targetDom.offsetByte.value),
  };
}
export function collectModelInputs(targetDom = getStaticCorrectionDom()) {
  if (!targetDom) return null;
  return {
    model_preset: trimValue(targetDom.modelKind && targetDom.modelKind.value),
    weathering_velocity_m_s: trimValue(
      targetDom.weatheringVelocityMS && targetDom.weatheringVelocityMS.value
    ),
    bedrock_velocity_mode: trimValue(
      targetDom.bedrockVelocityMode && targetDom.bedrockVelocityMode.value
    ),
    initial_bedrock_velocity_m_s: trimValue(
      targetDom.initialBedrockVelocityMS && targetDom.initialBedrockVelocityMS.value
    ),
    bedrock_velocity_m_s: trimValue(
      targetDom.fixedBedrockVelocityMS && targetDom.fixedBedrockVelocityMS.value
    ),
    min_offset_m: trimValue(targetDom.minOffsetM && targetDom.minOffsetM.value),
    max_offset_m: trimValue(targetDom.maxOffsetM && targetDom.maxOffsetM.value),
    conversion_mode: trimValue(targetDom.conversionMode && targetDom.conversionMode.value),
    v3_min_offset_m: trimValue(targetDom.v3MinOffsetM && targetDom.v3MinOffsetM.value),
    v3_max_offset_m: trimValue(targetDom.v3MaxOffsetM && targetDom.v3MaxOffsetM.value),
    initial_v3_velocity_m_s: trimValue(
      targetDom.initialV3VelocityMS && targetDom.initialV3VelocityMS.value
    ),
    vsub_min_offset_m: trimValue(
      targetDom.vsubMinOffsetM && targetDom.vsubMinOffsetM.value
    ),
    initial_vsub_velocity_m_s: trimValue(
      targetDom.initialVsubVelocityMS && targetDom.initialVsubVelocityMS.value
    ),
    cell_x_origin_m: trimValue(targetDom.cellXOriginM && targetDom.cellXOriginM.value),
    cell_y_origin_m: trimValue(targetDom.cellYOriginM && targetDom.cellYOriginM.value),
    cell_count_x: trimValue(targetDom.cellCountX && targetDom.cellCountX.value),
    cell_count_y: trimValue(targetDom.cellCountY && targetDom.cellCountY.value),
    cell_size_x_m: trimValue(targetDom.cellSizeXM && targetDom.cellSizeXM.value),
    cell_size_y_m: trimValue(targetDom.cellSizeYM && targetDom.cellSizeYM.value),
    cell_min_observations: trimValue(
      targetDom.cellMinObservations && targetDom.cellMinObservations.value
    ),
    cell_smoothing_weight: trimValue(
      targetDom.cellSmoothingWeight && targetDom.cellSmoothingWeight.value
    ),
    line_origin_x_m: trimValue(targetDom.lineOriginXM && targetDom.lineOriginXM.value),
    line_origin_y_m: trimValue(targetDom.lineOriginYM && targetDom.lineOriginYM.value),
    line_azimuth_deg: trimValue(targetDom.lineAzimuthDeg && targetDom.lineAzimuthDeg.value),
  };
}
export function collectFieldCorrectionInputs(targetDom = getStaticCorrectionDom()) {
  if (!targetDom) return null;
  return {
    enabled: Boolean(targetDom.fieldCorrectionsEnabled && targetDom.fieldCorrectionsEnabled.checked),
    source_depth_mode: trimValue(
      targetDom.fieldSourceDepthMode && targetDom.fieldSourceDepthMode.value
    ),
    source_depth_byte: trimValue(
      targetDom.fieldSourceDepthByte && targetDom.fieldSourceDepthByte.value
    ),
    uphole_mode: trimValue(targetDom.fieldUpholeMode && targetDom.fieldUpholeMode.value),
    uphole_time_byte: trimValue(
      targetDom.fieldUpholeTimeByte && targetDom.fieldUpholeTimeByte.value
    ),
    manual_static_mode: trimValue(
      targetDom.fieldManualStaticMode && targetDom.fieldManualStaticMode.value
    ),
    manual_static_sign_convention: trimValue(
      targetDom.fieldManualStaticSignConvention
      && targetDom.fieldManualStaticSignConvention.value
    ),
    manual_source_job_id: trimValue(
      targetDom.fieldManualSourceJobId && targetDom.fieldManualSourceJobId.value
    ),
    manual_source_artifact_name: trimValue(
      targetDom.fieldManualSourceArtifactName
      && targetDom.fieldManualSourceArtifactName.value
    ),
    manual_receiver_job_id: trimValue(
      targetDom.fieldManualReceiverJobId && targetDom.fieldManualReceiverJobId.value
    ),
    manual_receiver_artifact_name: trimValue(
      targetDom.fieldManualReceiverArtifactName
      && targetDom.fieldManualReceiverArtifactName.value
    ),
    apply_to_trace_shift: Boolean(
      targetDom.fieldApplyToTraceShift && targetDom.fieldApplyToTraceShift.checked
    ),
  };
}
export function collectOutputInputs(targetDom = getStaticCorrectionDom()) {
  if (!targetDom) return null;
  const formats = [];
  for (const checkbox of targetDom.exportFormatInputs || []) {
    if (checkbox.checked) {
      formats.push(checkbox.value);
    }
  }
  return {
    register_corrected_file: Boolean(
      targetDom.registerCorrectedFile && targetDom.registerCorrectedFile.checked
    ),
    export_enabled: Boolean(targetDom.exportEnabled && targetDom.exportEnabled.checked),
    export_formats: formats,
  };
}
export function collectPresetInputs(targetDom = getStaticCorrectionDom()) {
  return {
    geometry: collectGeometryInputs(targetDom),
    linkage: collectLinkageInputs(targetDom),
    model: collectModelInputs(targetDom),
    field_corrections: collectFieldCorrectionInputs(targetDom),
    output: collectOutputInputs(targetDom),
  };
}
