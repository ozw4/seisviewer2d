import {
  CELL_MODEL_PRESETS,
  DEFAULTS,
  FIELD_MANUAL_SIGN_CONVENTIONS,
  FIELD_MANUAL_STATIC_MODES,
  FIELD_SOURCE_DEPTH_MODES,
  FIELD_UPHOLE_MODES,
  GEOMETRY_HEADER_FIELDS,
  LINKAGE_THRESHOLD_MODES,
  MODEL_PRESETS,
  MULTILAYER_MODEL_PRESETS,
} from './constants.js';
import {
  collectFieldCorrectionInputs,
  collectGeometryInputs,
  collectLinkageInputs,
  collectModelInputs,
  collectOutputInputs,
} from './form_collectors.js';
import { getStaticCorrectionDom } from './runtime.js';
import { trimValue } from './utils.js';

export function parsePositiveInteger(value, label, errors) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    errors.push(`${label} must be a positive integer.`);
    return null;
  }
  return parsed;
}
export function parseOptionalArtifactRef(jobId, artifactName, label, errors) {
  if (!jobId && !artifactName) {
    return null;
  }
  if (!jobId) {
    errors.push(`${label}.job_id is required when ${label}.artifact_name is set.`);
  }
  if (!artifactName) {
    errors.push(`${label}.artifact_name is required when ${label}.job_id is set.`);
  }
  if (artifactName && /[\\/]/.test(artifactName)) {
    errors.push(`${label}.artifact_name must be a plain file name.`);
  }
  if (artifactName && !artifactName.toLowerCase().endsWith('.csv')) {
    errors.push(`${label}.artifact_name must be a .csv artifact.`);
  }
  if (!jobId || !artifactName || /[\\/]/.test(artifactName) || !artifactName.toLowerCase().endsWith('.csv')) {
    return null;
  }
  return {
    job_id: jobId,
    artifact_name: artifactName,
  };
}
export function parsePositiveFloat(value, label, errors, { optional = false } = {}) {
  const trimmed = trimValue(value);
  if (optional && trimmed === '') {
    return null;
  }
  const parsed = Number(trimmed);
  if (!trimmed || !Number.isFinite(parsed) || parsed <= 0) {
    errors.push(`${label} must be a finite number greater than 0.`);
    return null;
  }
  return parsed;
}
export function parseFiniteFloat(value, label, errors, { optional = false } = {}) {
  const trimmed = trimValue(value);
  if (optional && trimmed === '') {
    return null;
  }
  const parsed = Number(trimmed);
  if (!trimmed || !Number.isFinite(parsed)) {
    errors.push(`${label} must be a finite number.`);
    return null;
  }
  return parsed;
}
export function parseNonNegativeFloat(value, label, errors, { optional = false } = {}) {
  const trimmed = trimValue(value);
  if (optional && trimmed === '') {
    return null;
  }
  const parsed = Number(trimmed);
  if (!trimmed || !Number.isFinite(parsed) || parsed < 0) {
    errors.push(`${label} must be a finite number greater than or equal to 0.`);
    return null;
  }
  return parsed;
}
export function parseHeaderByte(value, label, errors, { optional = false } = {}) {
  const trimmed = trimValue(value);
  if (optional && trimmed === '') {
    return null;
  }
  if (!/^\d+$/.test(trimmed)) {
    errors.push(`${label} must be an integer SEG-Y trace header byte from 1 to 240.`);
    return null;
  }
  const parsed = Number(trimmed);
  if (parsed < 1 || parsed > 240) {
    errors.push(`${label} must be an integer SEG-Y trace header byte from 1 to 240.`);
    return null;
  }
  return parsed;
}
export function parseUnit(value, label, errors) {
  const unit = trimValue(value);
  if (!['m', 'ft'].includes(unit)) {
    errors.push(`${label} must be m or ft.`);
    return null;
  }
  return unit;
}
export function validationError(errors) {
  const error = new Error(errors.join(' '));
  error.errors = errors;
  return error;
}
export function validateStaticCorrectionGeometryRequest(targetDom = getStaticCorrectionDom()) {
  const values = collectGeometryInputs(targetDom);
  const errors = [];
  if (!values) {
    return { payload: null, errors: ['Static correction geometry form is not available.'] };
  }

  const geometry = {};
  for (const field of GEOMETRY_HEADER_FIELDS) {
    const parsed = parseHeaderByte(
      values[field.requestKey],
      field.label,
      errors,
      { optional: Boolean(field.optional) }
    );
    geometry[field.requestKey] = parsed;
  }

  const coordinateUnit = parseUnit(values.coordinate_unit, 'geometry.coordinate_unit', errors);
  const elevationUnit = parseUnit(values.elevation_unit, 'geometry.elevation_unit', errors);
  const offsetByte = parseHeaderByte(values.offset_byte, 'moveout.offset_byte', errors);
  geometry.coordinate_unit = coordinateUnit;
  geometry.elevation_unit = elevationUnit;
  if (
    geometry.source_id_byte !== null
    && geometry.receiver_id_byte !== null
    && geometry.source_id_byte === geometry.receiver_id_byte
  ) {
    errors.push('geometry.source_id_byte and receiver_id_byte must differ.');
  }

  if (errors.length) {
    return { payload: null, errors };
  }

  return {
    payload: {
      // UI names follow the backend schema here. The issue's coordinate_units
      // term maps to geometry.coordinate_unit, and offset_byte belongs to
      // moveout.offset_byte rather than the geometry object.
      geometry,
      moveout: {
        offset_byte: offsetByte,
      },
    },
    errors,
  };
}
export function validateStaticCorrectionLinkageRequest(targetDom = getStaticCorrectionDom()) {
  const values = collectLinkageInputs(targetDom);
  const errors = [];
  if (!values) {
    return { payload: null, errors: ['Static correction linkage form is not available.'] };
  }

  if (!values.enabled) {
    return {
      payload: {
        linkage: {
          mode: 'none',
        },
      },
      errors,
    };
  }

  const linkage = {
    mode: values.mode,
    prefer_receiver_anchor: values.prefer_receiver_anchor,
  };
  if (!values.mode) {
    errors.push('linkage.mode is required when endpoint linkage is enabled.');
  }

  if (LINKAGE_THRESHOLD_MODES.has(values.mode)) {
    const thresholdM = parsePositiveFloat(values.threshold_m, 'linkage.threshold_m', errors);
    if (thresholdM !== null) {
      linkage.threshold_m = thresholdM;
    }
  }

  const receiverLocationIntervalM = parsePositiveFloat(
    values.receiver_location_interval_m,
    'linkage.receiver_location_interval_m',
    errors,
    { optional: true }
  );
  if (receiverLocationIntervalM !== null) {
    linkage.receiver_location_interval_m = receiverLocationIntervalM;
  }

  if (errors.length) {
    return { payload: null, errors };
  }
  return { payload: { linkage }, errors };
}
export function validateOneLayerRefractionModel(targetDom = getStaticCorrectionDom()) {
  const values = collectModelInputs(targetDom);
  const errors = [];
  if (!values) {
    return { payload: null, moveout: null, conversion: null, errors: ['Static correction model form is not available.'] };
  }

  const weatheringVelocity = parsePositiveFloat(
    values.weathering_velocity_m_s,
    'model.first_layer.weathering_velocity_m_s',
    errors
  );
  const bedrockVelocityMode = values.bedrock_velocity_mode;
  if (!['solve_global', 'fixed_global'].includes(bedrockVelocityMode)) {
    errors.push('model.bedrock_velocity_mode must be solve_global or fixed_global.');
  }

  let initialBedrockVelocity = null;
  let fixedBedrockVelocity = null;
  if (bedrockVelocityMode === 'solve_global') {
    initialBedrockVelocity = parsePositiveFloat(
      values.initial_bedrock_velocity_m_s,
      'model.initial_bedrock_velocity_m_s',
      errors
    );
  } else if (bedrockVelocityMode === 'fixed_global') {
    fixedBedrockVelocity = parsePositiveFloat(
      values.bedrock_velocity_m_s,
      'model.bedrock_velocity_m_s',
      errors
    );
  }
  const minOffsetM = parseNonNegativeFloat(values.min_offset_m, 'moveout.min_offset_m', errors);
  const maxOffsetM = parsePositiveFloat(values.max_offset_m, 'moveout.max_offset_m', errors);
  if (minOffsetM !== null && maxOffsetM !== null && minOffsetM >= maxOffsetM) {
    errors.push('moveout.min_offset_m must be less than moveout.max_offset_m.');
  }
  if (values.conversion_mode !== 't1lsst_1layer') {
    errors.push('conversion.mode must be t1lsst_1layer for the one-layer preset.');
  }
  if (
    weatheringVelocity !== null
    && bedrockVelocityMode === 'solve_global'
    && initialBedrockVelocity !== null
    && initialBedrockVelocity <= weatheringVelocity
  ) {
    errors.push('model.initial_bedrock_velocity_m_s must be greater than model.first_layer.weathering_velocity_m_s.');
  }
  if (
    weatheringVelocity !== null
    && bedrockVelocityMode === 'fixed_global'
    && fixedBedrockVelocity !== null
    && fixedBedrockVelocity <= weatheringVelocity
  ) {
    errors.push('model.bedrock_velocity_m_s must be greater than model.first_layer.weathering_velocity_m_s.');
  }

  if (errors.length) {
    return { payload: null, moveout: null, conversion: null, errors };
  }

  const model = {
    method: 'gli_variable_thickness',
    first_layer: {
      mode: 'constant',
      weathering_velocity_m_s: weatheringVelocity,
    },
    bedrock_velocity_mode: bedrockVelocityMode,
    min_bedrock_velocity_m_s: 1200,
    max_bedrock_velocity_m_s: 6000,
  };
  if (bedrockVelocityMode === 'solve_global') {
    model.initial_bedrock_velocity_m_s = initialBedrockVelocity;
  } else {
    model.bedrock_velocity_m_s = fixedBedrockVelocity;
  }

  return {
    payload: model,
    moveout: {
      min_offset_m: minOffsetM,
      max_offset_m: maxOffsetM,
    },
    conversion: {
      mode: 't1lsst_1layer',
    },
    errors,
  };
}
export function validateCellRefractionModel(targetDom = getStaticCorrectionDom(), preset) {
  const values = collectModelInputs(targetDom);
  const errors = [];
  if (!values) {
    return { payload: null, moveout: null, conversion: null, errors: ['Static correction model form is not available.'] };
  }

  const weatheringVelocity = parsePositiveFloat(
    values.weathering_velocity_m_s,
    'model.first_layer.weathering_velocity_m_s',
    errors
  );
  const initialBedrockVelocity = parsePositiveFloat(
    values.initial_bedrock_velocity_m_s,
    'model.initial_bedrock_velocity_m_s',
    errors
  );
  const minOffsetM = parseNonNegativeFloat(values.min_offset_m, 'moveout.min_offset_m', errors);
  const maxOffsetM = parsePositiveFloat(values.max_offset_m, 'moveout.max_offset_m', errors);
  if (minOffsetM !== null && maxOffsetM !== null && minOffsetM >= maxOffsetM) {
    errors.push('moveout.min_offset_m must be less than moveout.max_offset_m.');
  }
  if (values.conversion_mode !== 't1lsst_1layer') {
    errors.push('conversion.mode must be t1lsst_1layer for the cell V2/T1 presets.');
  }
  if (
    weatheringVelocity !== null
    && initialBedrockVelocity !== null
    && initialBedrockVelocity <= weatheringVelocity
  ) {
    errors.push('model.initial_bedrock_velocity_m_s must be greater than model.first_layer.weathering_velocity_m_s.');
  }

  const cellCountX = parsePositiveInteger(
    values.cell_count_x,
    'model.refractor_cell.number_of_cell_x',
    errors
  );
  const cellSizeXM = parsePositiveFloat(
    values.cell_size_x_m,
    'model.refractor_cell.size_of_cell_x_m',
    errors
  );
  const cellXOriginM = parseFiniteFloat(
    values.cell_x_origin_m,
    'model.refractor_cell.x_coordinate_origin_m',
    errors
  );
  const cellYOriginM = parseFiniteFloat(
    values.cell_y_origin_m,
    'model.refractor_cell.y_coordinate_origin_m',
    errors
  );
  const cellMinObservations = parsePositiveInteger(
    values.cell_min_observations,
    'model.refractor_cell.min_observations_per_cell',
    errors
  );
  const cellSmoothingWeight = parseNonNegativeFloat(
    values.cell_smoothing_weight,
    'model.refractor_cell.velocity_smoothing_weight',
    errors
  );
  const isLine2D = preset === 'cell_v2_t1_line_2d';
  const cellCountY = isLine2D
    ? 1
    : parsePositiveInteger(values.cell_count_y, 'model.refractor_cell.number_of_cell_y', errors);
  const cellSizeYM = isLine2D
    ? null
    : parsePositiveFloat(values.cell_size_y_m, 'model.refractor_cell.size_of_cell_y_m', errors);
  const lineOriginXM = isLine2D
    ? parseFiniteFloat(values.line_origin_x_m, 'model.refractor_cell.line_origin_x_m', errors)
    : null;
  const lineOriginYM = isLine2D
    ? parseFiniteFloat(values.line_origin_y_m, 'model.refractor_cell.line_origin_y_m', errors)
    : null;
  const lineAzimuthDeg = isLine2D
    ? parseFiniteFloat(values.line_azimuth_deg, 'model.refractor_cell.line_azimuth_deg', errors)
    : null;

  if (errors.length) {
    return { payload: null, moveout: null, conversion: null, errors };
  }

  const refractorCell = {
    number_of_cell_x: cellCountX,
    size_of_cell_x_m: cellSizeXM,
    x_coordinate_origin_m: cellXOriginM,
    number_of_cell_y: cellCountY,
    size_of_cell_y_m: cellSizeYM,
    y_coordinate_origin_m: cellYOriginM,
    assignment_mode: 'midpoint',
    outside_grid_policy: 'reject',
    coordinate_mode: isLine2D ? 'line_2d_projected' : 'grid_3d',
    min_observations_per_cell: cellMinObservations,
    velocity_smoothing_weight: cellSmoothingWeight,
  };
  if (isLine2D) {
    refractorCell.line_origin_x_m = lineOriginXM;
    refractorCell.line_origin_y_m = lineOriginYM;
    refractorCell.line_azimuth_deg = lineAzimuthDeg;
  }

  return {
    payload: {
      method: 'gli_variable_thickness',
      first_layer: {
        mode: 'constant',
        weathering_velocity_m_s: weatheringVelocity,
      },
      bedrock_velocity_mode: 'solve_cell',
      initial_bedrock_velocity_m_s: initialBedrockVelocity,
      min_bedrock_velocity_m_s: 1200,
      max_bedrock_velocity_m_s: 6000,
      refractor_cell: refractorCell,
    },
    moveout: {
      min_offset_m: minOffsetM,
      max_offset_m: maxOffsetM,
    },
    conversion: {
      mode: 't1lsst_1layer',
    },
    errors,
  };
}
export function validateMultilayerRefractionModel(targetDom = getStaticCorrectionDom(), preset) {
  const values = collectModelInputs(targetDom);
  const errors = [];
  if (!values) {
    return { payload: null, moveout: null, conversion: null, errors: ['Static correction model form is not available.'] };
  }

  const weatheringVelocity = parsePositiveFloat(
    values.weathering_velocity_m_s,
    'model.first_layer.weathering_velocity_m_s',
    errors
  );
  const bedrockVelocityMode = values.bedrock_velocity_mode;
  if (!['solve_global', 'fixed_global'].includes(bedrockVelocityMode)) {
    errors.push('model.layers v2_t1 velocity_mode must be solve_global or fixed_global.');
  }
  const v2MinOffsetM = parseNonNegativeFloat(
    values.min_offset_m,
    'model.layers v2_t1 min_offset_m',
    errors
  );
  const v2MaxOffsetM = parsePositiveFloat(
    values.max_offset_m,
    'model.layers v2_t1 max_offset_m',
    errors
  );
  let initialV2Velocity = null;
  let fixedV2Velocity = null;
  if (bedrockVelocityMode === 'solve_global') {
    initialV2Velocity = parsePositiveFloat(
      values.initial_bedrock_velocity_m_s,
      'model.layers v2_t1 initial_velocity_m_s',
      errors
    );
  } else if (bedrockVelocityMode === 'fixed_global') {
    fixedV2Velocity = parsePositiveFloat(
      values.bedrock_velocity_m_s,
      'model.layers v2_t1 fixed_velocity_m_s',
      errors
    );
  }

  const v3MinOffsetM = parseNonNegativeFloat(
    values.v3_min_offset_m,
    'model.layers v3_t2 min_offset_m',
    errors
  );
  const v3MaxOffsetM = preset === 'three_layer_global'
    ? parsePositiveFloat(values.v3_max_offset_m, 'model.layers v3_t2 max_offset_m', errors)
    : parsePositiveFloat(values.v3_max_offset_m, 'model.layers v3_t2 max_offset_m', errors, { optional: true });
  const initialV3Velocity = parsePositiveFloat(
    values.initial_v3_velocity_m_s,
    'model.layers v3_t2 initial_velocity_m_s',
    errors
  );
  const isThreeLayer = preset === 'three_layer_global';
  const vsubMinOffsetM = isThreeLayer
    ? parseNonNegativeFloat(values.vsub_min_offset_m, 'model.layers vsub_t3 min_offset_m', errors)
    : null;
  const initialVsubVelocity = isThreeLayer
    ? parsePositiveFloat(
      values.initial_vsub_velocity_m_s,
      'model.layers vsub_t3 initial_velocity_m_s',
      errors
    )
    : null;

  if (values.conversion_mode !== 't1lsst_multilayer') {
    errors.push('conversion.mode must be t1lsst_multilayer for multi-layer presets.');
  }
  if (v2MinOffsetM !== null && v2MaxOffsetM !== null && v2MinOffsetM >= v2MaxOffsetM) {
    errors.push('model.layers v2_t1 min_offset_m must be less than max_offset_m.');
  }
  if (v2MaxOffsetM !== null && v3MinOffsetM !== null && v2MaxOffsetM > v3MinOffsetM) {
    errors.push('model.layers v2_t1 and v3_t2 offset gates must not overlap.');
  }
  if (v3MinOffsetM !== null && v3MaxOffsetM !== null && v3MinOffsetM >= v3MaxOffsetM) {
    errors.push('model.layers v3_t2 min_offset_m must be less than max_offset_m.');
  }
  if (isThreeLayer && v3MaxOffsetM !== null && vsubMinOffsetM !== null && v3MaxOffsetM > vsubMinOffsetM) {
    errors.push('model.layers v3_t2 and vsub_t3 offset gates must not overlap.');
  }
  for (const [label, velocity] of [
    ['model.layers v2_t1 velocity', initialV2Velocity || fixedV2Velocity],
    ['model.layers v3_t2 initial_velocity_m_s', initialV3Velocity],
    ['model.layers vsub_t3 initial_velocity_m_s', initialVsubVelocity],
  ]) {
    if (weatheringVelocity !== null && velocity !== null && velocity <= weatheringVelocity) {
      errors.push(`${label} must be greater than model.first_layer.weathering_velocity_m_s.`);
    }
  }

  if (errors.length) {
    return { payload: null, moveout: null, conversion: null, errors };
  }

  const v2Layer = {
    kind: 'v2_t1',
    enabled: true,
    min_offset_m: v2MinOffsetM,
    max_offset_m: v2MaxOffsetM,
    velocity_mode: bedrockVelocityMode,
    min_velocity_m_s: 1200,
    max_velocity_m_s: 5000,
  };
  if (bedrockVelocityMode === 'solve_global') {
    v2Layer.initial_velocity_m_s = initialV2Velocity;
  } else {
    v2Layer.fixed_velocity_m_s = fixedV2Velocity;
  }

  const layers = [
    v2Layer,
    {
      kind: 'v3_t2',
      enabled: true,
      min_offset_m: v3MinOffsetM,
      max_offset_m: v3MaxOffsetM,
      velocity_mode: 'solve_global',
      initial_velocity_m_s: initialV3Velocity,
      min_velocity_m_s: 2600,
      max_velocity_m_s: 7000,
    },
  ];
  if (isThreeLayer) {
    layers.push({
      kind: 'vsub_t3',
      enabled: true,
      min_offset_m: vsubMinOffsetM,
      max_offset_m: null,
      velocity_mode: 'solve_global',
      initial_velocity_m_s: initialVsubVelocity,
      min_velocity_m_s: 3800,
      max_velocity_m_s: 9000,
    });
  }

  const model = {
    method: 'multilayer_time_term',
    first_layer: {
      mode: 'constant',
      weathering_velocity_m_s: weatheringVelocity,
    },
    layers,
  };
  if (bedrockVelocityMode === 'solve_global') {
    model.initial_bedrock_velocity_m_s = initialV2Velocity;
  } else {
    model.bedrock_velocity_m_s = fixedV2Velocity;
  }

  return {
    payload: model,
    moveout: {},
    conversion: {
      mode: 't1lsst_multilayer',
      layer_count: isThreeLayer ? 3 : 2,
    },
    errors,
  };
}
export function validateRefractionStaticModel(targetDom = getStaticCorrectionDom()) {
  const values = collectModelInputs(targetDom);
  const preset = values && values.model_preset ? values.model_preset : DEFAULTS.modelPreset;
  if (!MODEL_PRESETS.has(preset)) {
    return {
      payload: null,
      moveout: null,
      conversion: null,
      errors: ['model preset must be a supported refraction static preset.'],
    };
  }
  if (preset === 'one_layer_global') {
    return validateOneLayerRefractionModel(targetDom);
  }
  if (CELL_MODEL_PRESETS.has(preset)) {
    return validateCellRefractionModel(targetDom, preset);
  }
  return validateMultilayerRefractionModel(targetDom, preset);
}
export function validateStaticCorrectionFieldCorrections(targetDom = getStaticCorrectionDom()) {
  const values = collectFieldCorrectionInputs(targetDom);
  const geometryValues = collectGeometryInputs(targetDom);
  const errors = [];
  if (!values) {
    return { payload: null, errors: ['Static correction field-correction form is not available.'] };
  }
  if (!values.enabled) {
    return { payload: null, errors };
  }

  const sourceDepthMode = values.source_depth_mode || 'none';
  const upholeMode = values.uphole_mode || 'none';
  const manualMode = values.manual_static_mode || 'none';
  if (!FIELD_SOURCE_DEPTH_MODES.has(sourceDepthMode)) {
    errors.push('field_corrections.source_depth.mode must be none or weathering_velocity_time.');
  }
  if (!FIELD_UPHOLE_MODES.has(upholeMode)) {
    errors.push('field_corrections.uphole.mode must be none or header_time.');
  }
  if (!FIELD_MANUAL_STATIC_MODES.has(manualMode)) {
    errors.push('field_corrections.manual_static.mode must be none or artifact_table.');
  }

  const fieldCorrections = {
    source_depth: { mode: sourceDepthMode },
    uphole: { mode: upholeMode },
    manual_static: { mode: manualMode },
    composition: {
      enabled: true,
      apply_to_trace_shift: values.apply_to_trace_shift,
    },
  };

  if (sourceDepthMode === 'weathering_velocity_time') {
    const sourceDepthByte = parseHeaderByte(
      values.source_depth_byte,
      'field_corrections.source_depth.source_depth_byte',
      errors,
      { optional: true }
    );
    if (sourceDepthByte !== null) {
      fieldCorrections.source_depth.source_depth_byte = sourceDepthByte;
    } else if (!geometryValues || !trimValue(geometryValues.source_depth_byte)) {
      errors.push(
        'field_corrections.source_depth.source_depth_byte or geometry.source_depth_byte is required when source-depth correction is enabled.'
      );
    }
  }

  if (upholeMode === 'header_time') {
    const upholeTimeByte = parseHeaderByte(
      values.uphole_time_byte,
      'field_corrections.uphole.uphole_time_byte',
      errors
    );
    if (upholeTimeByte !== null) {
      fieldCorrections.uphole.uphole_time_byte = upholeTimeByte;
    }
  }

  if (manualMode === 'artifact_table') {
    const signConvention = values.manual_static_sign_convention;
    if (!FIELD_MANUAL_SIGN_CONVENTIONS.has(signConvention)) {
      errors.push('field_corrections.manual_static.sign_convention must be applied_shift_s or delay_positive_ms.');
    } else {
      fieldCorrections.manual_static.sign_convention = signConvention;
    }
    const sourceRef = parseOptionalArtifactRef(
      values.manual_source_job_id,
      values.manual_source_artifact_name,
      'field_corrections.manual_static.source_table_artifact',
      errors
    );
    const receiverRef = parseOptionalArtifactRef(
      values.manual_receiver_job_id,
      values.manual_receiver_artifact_name,
      'field_corrections.manual_static.receiver_table_artifact',
      errors
    );
    if (sourceRef) {
      fieldCorrections.manual_static.source_table_artifact = sourceRef;
    }
    if (receiverRef) {
      fieldCorrections.manual_static.receiver_table_artifact = receiverRef;
    }
    if (!sourceRef && !receiverRef) {
      errors.push(
        'field_corrections.manual_static.source_table_artifact or receiver_table_artifact is required when manual static mode is artifact_table.'
      );
    }
  }

  if (errors.length) {
    return { payload: null, errors };
  }
  return { payload: { field_corrections: fieldCorrections }, errors };
}
export function validateStaticCorrectionOutput(targetDom = getStaticCorrectionDom()) {
  const values = collectOutputInputs(targetDom);
  const errors = [];
  if (!values) {
    return { payload: null, errors: ['Static correction output form is not available.'] };
  }
  if (values.export_enabled && values.export_formats.length === 0) {
    errors.push('export.formats must include at least one format when export.enabled is true.');
  }
  if (errors.length) {
    return { payload: null, errors };
  }
  return {
    payload: {
      export: {
        enabled: values.export_enabled,
        formats: values.export_formats,
      },
      apply: {
        register_corrected_file: values.register_corrected_file,
      },
    },
    errors,
  };
}
