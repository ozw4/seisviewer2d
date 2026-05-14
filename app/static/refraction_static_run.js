(function () {
  const DEFAULTS = {
    key1Byte: '189',
    key2Byte: '193',
    modelPreset: 'one_layer_global',
    pickKind: 'batch_predicted_npz',
    pickArtifactName: 'predicted_picks_time_s.npz',
    linkageMode: 'auto_threshold',
    linkageThresholdM: '25',
    weatheringVelocityMS: '800',
    bedrockVelocityMode: 'solve_global',
    initialBedrockVelocityMS: '2400',
    fixedBedrockVelocityMS: '2400',
    minOffsetM: '300',
    maxOffsetM: '4000',
    conversionMode: 't1lsst_1layer',
    initialV3VelocityMS: '3600',
    v3MinOffsetM: '4000',
    v3MaxOffsetM: '6000',
    initialVsubVelocityMS: '5000',
    vsubMinOffsetM: '6000',
    cellXOriginM: '0',
    cellYOriginM: '0',
    cellCountX: '20',
    cellCountY: '1',
    cellSizeXM: '500',
    cellSizeYM: '500',
    cellMinObservations: '5',
    cellSmoothingWeight: '0',
    lineOriginXM: '0',
    lineOriginYM: '0',
    lineAzimuthDeg: '0',
  };
  const MODEL_PRESETS = new Set([
    'one_layer_global',
    'two_layer_global',
    'three_layer_global',
    'cell_v2_t1_line_2d',
    'cell_v2_t1_grid_3d',
  ]);
  const CELL_MODEL_PRESETS = new Set(['cell_v2_t1_line_2d', 'cell_v2_t1_grid_3d']);
  const MULTILAYER_MODEL_PRESETS = new Set(['two_layer_global', 'three_layer_global']);
  const GEOMETRY_PRESET_SEG_Y_DEFAULT = 'segy_default';
  const GEOMETRY_PRESET_CUSTOM = 'custom';
  const GEOMETRY_DEFAULTS = {
    preset: GEOMETRY_PRESET_SEG_Y_DEFAULT,
    sourceIdByte: '9',
    receiverIdByte: '13',
    sourceXByte: '73',
    sourceYByte: '77',
    receiverXByte: '81',
    receiverYByte: '85',
    sourceElevationByte: '45',
    receiverElevationByte: '41',
    coordinateScalarByte: '71',
    elevationScalarByte: '69',
    sourceDepthByte: '',
    coordinateUnit: 'm',
    elevationUnit: 'm',
    offsetByte: '37',
  };
  const GEOMETRY_HEADER_FIELDS = [
    { domKey: 'sourceIdByte', requestKey: 'source_id_byte', label: 'geometry.source_id_byte' },
    { domKey: 'receiverIdByte', requestKey: 'receiver_id_byte', label: 'geometry.receiver_id_byte' },
    { domKey: 'sourceXByte', requestKey: 'source_x_byte', label: 'geometry.source_x_byte' },
    { domKey: 'sourceYByte', requestKey: 'source_y_byte', label: 'geometry.source_y_byte' },
    { domKey: 'receiverXByte', requestKey: 'receiver_x_byte', label: 'geometry.receiver_x_byte' },
    { domKey: 'receiverYByte', requestKey: 'receiver_y_byte', label: 'geometry.receiver_y_byte' },
    {
      domKey: 'sourceElevationByte',
      requestKey: 'source_elevation_byte',
      label: 'geometry.source_elevation_byte',
    },
    {
      domKey: 'receiverElevationByte',
      requestKey: 'receiver_elevation_byte',
      label: 'geometry.receiver_elevation_byte',
    },
    {
      domKey: 'coordinateScalarByte',
      requestKey: 'coordinate_scalar_byte',
      label: 'geometry.coordinate_scalar_byte',
    },
    {
      domKey: 'elevationScalarByte',
      requestKey: 'elevation_scalar_byte',
      label: 'geometry.elevation_scalar_byte',
    },
    {
      domKey: 'sourceDepthByte',
      requestKey: 'source_depth_byte',
      label: 'geometry.source_depth_byte',
      optional: true,
    },
  ];
  const ARTIFACT_PICK_KINDS = new Set(['batch_predicted_npz', 'manual_npz_artifact']);
  const LINKAGE_THRESHOLD_MODES = new Set(['auto_threshold']);
  const LINKAGE_ARTIFACT_NAME = 'geometry_linkage.npz';
  const LINKAGE_READY_STATES = new Set(['done', 'ready']);
  const LINKAGE_FAILED_STATES = new Set(['error', 'failed', 'cancelled', 'canceled', 'expired']);
  const STATIC_READY_STATES = new Set(['done', 'ready']);
  const STATIC_ACTIVE_STATES = new Set(['queued', 'running', 'cancel_requested']);
  const STATIC_TERMINAL_STATES = new Set(['done', 'ready', 'error', 'cancelled', 'expired']);
  const FIELD_SOURCE_DEPTH_MODES = new Set(['none', 'weathering_velocity_time']);
  const FIELD_UPHOLE_MODES = new Set(['none', 'header_time']);
  const FIELD_MANUAL_STATIC_MODES = new Set(['none', 'artifact_table']);
  const FIELD_MANUAL_SIGN_CONVENTIONS = new Set(['applied_shift_s', 'delay_positive_ms']);
  const PRESET_STORAGE_KEY = 'sv.static_correction.presets';

  const state = {
    ready: false,
    message: 'Enter a SEG-Y/TraceStore file_id and a first-break pick artifact usable by refraction statics.',
    error: '',
    loadingPickArtifacts: false,
    pickArtifacts: [],
    lastRequest: null,
    lastResponse: null,
    lastLinkageBuildRequest: null,
    lastLinkageJobId: '',
    lastStaticCorrectionJobId: '',
    lastStaticCorrectionState: '',
    lastStaticCorrectionMessage: '',
    lastStaticCorrectionProgress: 0,
    staticArtifacts: [],
    loadingStaticArtifacts: false,
    presets: [],
    validationErrors: [],
    showValidationSummary: false,
    phase: 'idle',
    pollIntervalMs: 1000,
  };

  let dom = null;
  let staticPollToken = 0;

  function trimValue(value) {
    return String(value || '').trim();
  }

  function setDefaultValue(element, value) {
    if (element && trimValue(element.value) === '') {
      element.value = value;
    }
  }

  function geometryValueElements(targetDom) {
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

  function setGeometryDefaults(targetDom) {
    if (!targetDom) return;
    for (const [key, value] of Object.entries(GEOMETRY_DEFAULTS)) {
      if (key === 'preset') continue;
      if (targetDom[key]) {
        targetDom[key].value = value;
      }
    }
  }

  function applyStaticCorrectionGeometryPreset(targetDom = dom, presetName = null) {
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

  function isLikelyPickArtifact(name) {
    const normalized = trimValue(name);
    return (
      normalized === DEFAULTS.pickArtifactName
      || /^manual_picks_time.*\.npz$/i.test(normalized)
    );
  }

  function collectInputs(targetDom = dom) {
    if (!targetDom) return null;
    return {
      file_id: trimValue(targetDom.fileId.value),
      key1_byte: trimValue(targetDom.key1Byte.value),
      key2_byte: trimValue(targetDom.key2Byte.value),
      pick_source: {
        kind: trimValue(targetDom.pickKind.value),
        job_id: trimValue(targetDom.pickJobId.value),
        artifact_name: trimValue(targetDom.pickArtifactName.value),
      },
    };
  }

  function collectLinkageInputs(targetDom = dom) {
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

  function collectGeometryInputs(targetDom = dom) {
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

  function collectModelInputs(targetDom = dom) {
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

  function collectFieldCorrectionInputs(targetDom = dom) {
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

  function collectOutputInputs(targetDom = dom) {
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

  function collectPresetInputs(targetDom = dom) {
    const inputValues = collectInputs(targetDom);
    return {
      key1_byte: inputValues ? inputValues.key1_byte : '',
      key2_byte: inputValues ? inputValues.key2_byte : '',
      pick_source: {
        kind: inputValues && inputValues.pick_source ? inputValues.pick_source.kind : '',
        artifact_name: inputValues && inputValues.pick_source
          ? inputValues.pick_source.artifact_name
          : '',
      },
      geometry: collectGeometryInputs(targetDom),
      linkage: collectLinkageInputs(targetDom),
      model: collectModelInputs(targetDom),
      field_corrections: collectFieldCorrectionInputs(targetDom),
      output: collectOutputInputs(targetDom),
    };
  }

  function readStoredPresets() {
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

  function writeStoredPresets(presets) {
    window.localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(presets));
  }

  function setElementValue(element, value) {
    if (element && value !== undefined && value !== null) {
      element.value = String(value);
    }
  }

  function setElementChecked(element, value) {
    if (element && value !== undefined && value !== null) {
      element.checked = Boolean(value);
    }
  }

  function parsePositiveInteger(value, label, errors) {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      errors.push(`${label} must be a positive integer.`);
      return null;
    }
    return parsed;
  }

  function parseOptionalArtifactRef(jobId, artifactName, label, errors) {
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

  function parsePositiveFloat(value, label, errors, { optional = false } = {}) {
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

  function parseFiniteFloat(value, label, errors, { optional = false } = {}) {
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

  function parseNonNegativeFloat(value, label, errors, { optional = false } = {}) {
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

  function parseHeaderByte(value, label, errors, { optional = false } = {}) {
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

  function parseUnit(value, label, errors) {
    const unit = trimValue(value);
    if (!['m', 'ft'].includes(unit)) {
      errors.push(`${label} must be m or ft.`);
      return null;
    }
    return unit;
  }

  function validationError(errors) {
    const error = new Error(errors.join(' '));
    error.errors = errors;
    return error;
  }

  function buildStaticCorrectionPickSource(targetDom = dom) {
    const values = collectInputs(targetDom);
    const errors = [];
    if (!values) {
      throw validationError(['Static correction form is not available.']);
    }
    const pickSource = { ...values.pick_source };
    const pickKind = pickSource.kind;
    if (!pickKind) {
      errors.push('pick_source.kind is required.');
    }
    if (ARTIFACT_PICK_KINDS.has(pickKind)) {
      if (!pickSource.job_id) {
        errors.push('pick_source.job_id is required for artifact-backed pick sources.');
      }
      if (!pickSource.artifact_name) {
        errors.push('pick_source.artifact_name is required for artifact-backed pick sources.');
      }
    }
    if (pickSource.artifact_name && !pickSource.artifact_name.toLowerCase().endsWith('.npz')) {
      errors.push('pick_source.artifact_name must be an .npz artifact.');
    }
    if (errors.length) {
      throw validationError(errors);
    }
    return pickSource;
  }

  function validateStaticCorrectionGeometryRequest(targetDom = dom) {
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

  function buildStaticCorrectionGeometryRequest(targetDom = dom) {
    const result = validateStaticCorrectionGeometryRequest(targetDom);
    if (result.errors.length) {
      throw validationError(result.errors);
    }
    return result.payload;
  }

  function buildStaticCorrectionGeometry(targetDom = dom) {
    return buildStaticCorrectionGeometryRequest(targetDom).geometry;
  }

  function linkageValueElements(targetDom) {
    if (!targetDom) return [];
    return [
      targetDom.linkageMode,
      targetDom.linkageThresholdM,
      targetDom.receiverLocationIntervalM,
      targetDom.preferReceiverAnchor,
    ].filter(Boolean);
  }

  function updateStaticCorrectionLinkageOptions(targetDom = dom) {
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

  function fieldCorrectionValueElements(targetDom) {
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

  function updateStaticCorrectionFieldCorrectionOptions(targetDom = dom) {
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

  function updateBedrockVelocityControls(targetDom = dom) {
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

  function setHidden(element, hidden) {
    if (element) {
      element.hidden = hidden;
    }
  }

  function setDisabled(elements, disabled) {
    for (const element of elements) {
      if (element) {
        element.disabled = disabled;
      }
    }
  }

  function updateModelPresetControls(targetDom = dom) {
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

  function validateStaticCorrectionLinkageRequest(targetDom = dom) {
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

  function buildStaticCorrectionLinkageRequest(targetDom = dom) {
    const result = validateStaticCorrectionLinkageRequest(targetDom);
    if (result.errors.length) {
      throw validationError(result.errors);
    }
    return result.payload;
  }

  function buildStaticCorrectionLinkage(targetDom = dom, linkageJobId = '') {
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

  function validateOneLayerRefractionModel(targetDom = dom) {
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

  function buildOneLayerRefractionModel(targetDom = dom) {
    const result = validateOneLayerRefractionModel(targetDom);
    if (result.errors.length) {
      throw validationError(result.errors);
    }
    return result.payload;
  }

  function validateCellRefractionModel(targetDom = dom, preset) {
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

  function validateMultilayerRefractionModel(targetDom = dom, preset) {
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

  function validateRefractionStaticModel(targetDom = dom) {
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

  function validateStaticCorrectionFieldCorrections(targetDom = dom) {
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

  function validateStaticCorrectionOutput(targetDom = dom) {
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

  function buildRefractionStaticApplyRequest(targetDom = dom, options = {}) {
    const values = collectInputs(targetDom);
    const errors = [];
    if (!values) {
      throw validationError(['Static correction form is not available.']);
    }
    if (!values.file_id) {
      errors.push('file_id is required.');
    }
    const key1Byte = parsePositiveInteger(values.key1_byte, 'key1_byte', errors);
    const key2Byte = parsePositiveInteger(values.key2_byte, 'key2_byte', errors);

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
      file_id: values.file_id,
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

  function buildStaticCorrectionRequest() {
    const values = collectInputs(dom);
    const errors = [];
    if (!values) {
      return { payload: null, errors: ['Static correction form is not available.'] };
    }

    if (!values.file_id) {
      errors.push('file_id is required.');
    }
    const key1Byte = parsePositiveInteger(values.key1_byte, 'key1_byte', errors);
    const key2Byte = parsePositiveInteger(values.key2_byte, 'key2_byte', errors);
    let pickSource = null;
    try {
      pickSource = buildStaticCorrectionPickSource(dom);
    } catch (error) {
      errors.push(...(error.errors || [error.message || String(error)]));
    }
    const geometryResult = validateStaticCorrectionGeometryRequest(dom);
    errors.push(...geometryResult.errors);
    const linkageResult = validateStaticCorrectionLinkageRequest(dom);
    errors.push(...linkageResult.errors);
    const modelResult = validateRefractionStaticModel(dom);
    errors.push(...modelResult.errors);
    const fieldCorrectionResult = validateStaticCorrectionFieldCorrections(dom);
    errors.push(...fieldCorrectionResult.errors);
    const outputResult = validateStaticCorrectionOutput(dom);
    errors.push(...outputResult.errors);

    if (errors.length) {
      return { payload: null, errors };
    }
    return {
      payload: {
        file_id: values.file_id,
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

  function buildStaticCorrectionPreviewRequest(targetDom = dom) {
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

  function getStaticCorrectionValidationSnapshot(targetDom = dom) {
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

  function applyPresetValues(values, targetDom = dom) {
    if (!targetDom || !values) return;

    setElementValue(targetDom.key1Byte, values.key1_byte);
    setElementValue(targetDom.key2Byte, values.key2_byte);
    if (values.pick_source) {
      setElementValue(targetDom.pickKind, values.pick_source.kind);
      setElementValue(targetDom.pickArtifactName, values.pick_source.artifact_name);
    }

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

  function saveCurrentPreset() {
    if (!dom) return;
    const name = trimValue(dom.presetName && dom.presetName.value);
    if (!name) {
      state.error = 'Preset name is required.';
      state.message = 'Enter a preset name before saving.';
      render();
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
    render();
  }

  function loadSelectedPreset() {
    if (!dom || !dom.presetSelect) return;
    const name = trimValue(dom.presetSelect.value);
    const preset = state.presets.find((entry) => entry.name === name);
    if (!preset) {
      state.error = 'Choose a saved preset to load.';
      state.message = 'No preset was loaded.';
      render();
      return;
    }
    applyPresetValues(preset.values, dom);
    if (dom.presetName) {
      dom.presetName.value = preset.name;
    }
    state.error = '';
    state.showValidationSummary = false;
    state.validationErrors = [];
    state.message = `Loaded preset ${preset.name}. Current file_id and pick_source.job_id were kept.`;
    render();
  }

  function deleteSelectedPreset() {
    if (!dom || !dom.presetSelect) return;
    const name = trimValue(dom.presetSelect.value);
    if (!name) {
      state.error = 'Choose a saved preset to delete.';
      state.message = 'No preset was deleted.';
      render();
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
    render();
  }

  function buildStaticLinkageBuildRequest(staticCorrectionPayload) {
    if (!staticCorrectionPayload) {
      throw validationError(['Static correction payload is required before building linkage.']);
    }
    return {
      file_id: staticCorrectionPayload.file_id,
      key1_byte: staticCorrectionPayload.key1_byte,
      key2_byte: staticCorrectionPayload.key2_byte,
      geometry: { ...staticCorrectionPayload.geometry },
      linkage: { ...staticCorrectionPayload.linkage },
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

  function normalizeStaticJobState(value) {
    const normalized = trimValue(value).toLowerCase();
    if (normalized === 'completed') return 'done';
    if (normalized === 'failed') return 'error';
    if (normalized === 'canceled') return 'cancelled';
    return normalized || 'unknown';
  }

  function isStaticJobActive(value = state.lastStaticCorrectionState) {
    return STATIC_ACTIVE_STATES.has(normalizeStaticJobState(value));
  }

  function isStaticJobTerminal(value = state.lastStaticCorrectionState) {
    return STATIC_TERMINAL_STATES.has(normalizeStaticJobState(value));
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

  async function readResponseError(response, operation = 'request') {
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
    return `${operation} ${response.status}${detail ? `: ${detail}` : ''}`;
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

  function renderStaticArtifacts() {
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

  function renderStaticJobPanel() {
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
  }

  function renderValidationSummary() {
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

  function renderPresetSelect() {
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

  function render() {
    if (!dom) return;
    const preview = getStaticCorrectionValidationSnapshot(dom);
    if (state.showValidationSummary) {
      state.validationErrors = preview.errors;
    }
    updateStaticCorrectionLinkageOptions(dom);
    dom.status.textContent = state.message;
    dom.error.hidden = !state.error;
    dom.error.textContent = state.error;
    dom.runButton.disabled = state.phase !== 'idle' || isStaticJobActive();
    if (dom.loadPickArtifactsButton) {
      dom.loadPickArtifactsButton.disabled = state.loadingPickArtifacts;
    }
    if (dom.requestPreview) {
      dom.requestPreview.textContent = preview.payload
        ? JSON.stringify(preview.payload, null, 2)
        : JSON.stringify({ validation_errors: preview.errors }, null, 2);
    }
    renderValidationSummary();
    renderPresetSelect();
    renderPickArtifactList();
    renderStaticJobPanel();
    renderStaticArtifacts();
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
        throw new Error(await readResponseError(response, 'batch job files'));
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

  function delay(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
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

  async function loadStaticArtifacts(jobId = state.lastStaticCorrectionJobId) {
    const safeJobId = trimValue(jobId);
    if (!safeJobId) {
      state.staticArtifacts = [];
      render();
      return [];
    }

    state.loadingStaticArtifacts = true;
    state.staticArtifacts = [];
    render();
    try {
      const response = await fetch(`/statics/job/${encodeURIComponent(safeJobId)}/files`);
      if (!response.ok) {
        throw new Error(await readResponseError(response, 'static job files'));
      }
      const payload = await response.json();
      const files = Array.isArray(payload.files) ? payload.files : [];
      state.staticArtifacts = sortedArtifactFiles(files);
      state.message = files.length
        ? `Loaded ${files.length} static correction artifact${files.length === 1 ? '' : 's'} for ${safeJobId}.`
        : `Static correction job ${safeJobId} finished without generated artifact files.`;
      state.error = '';
      return state.staticArtifacts;
    } catch (error) {
      state.staticArtifacts = [];
      state.error = error instanceof Error ? error.message : String(error);
      state.message = `Static correction job ${safeJobId} finished, but artifacts could not be loaded.`;
      return [];
    } finally {
      state.loadingStaticArtifacts = false;
      render();
    }
  }

  async function autoLoadRefractionQc(jobId) {
    const safeJobId = trimValue(jobId);
    const refractionQc = window.RefractionQc;
    if (!safeJobId || !refractionQc || typeof refractionQc.loadJob !== 'function') {
      return null;
    }
    return refractionQc.loadJob(safeJobId, { activateTab: true });
  }

  async function pollStaticCorrectionStatus(jobId) {
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
    render();
    return snapshot;
  }

  function stopStaticCorrectionPolling() {
    staticPollToken += 1;
  }

  async function pollStaticCorrectionJobUntilTerminal(jobId) {
    const token = staticPollToken + 1;
    staticPollToken = token;
    state.staticArtifacts = [];
    render();

    while (token === staticPollToken) {
      try {
        const snapshot = await pollStaticCorrectionStatus(jobId);
        if (STATIC_READY_STATES.has(snapshot.state)) {
          await loadStaticArtifacts(jobId);
          if (token !== staticPollToken) {
            return null;
          }
          await autoLoadRefractionQc(jobId);
          return snapshot;
        }
        if (isStaticJobTerminal(snapshot.state)) {
          return snapshot;
        }
      } catch (error) {
        if (token !== staticPollToken) {
          return null;
        }
        state.error = error instanceof Error ? error.message : String(error);
        state.message = `Static correction status polling failed: ${state.error}`;
        render();
        return null;
      }
      await delay(Math.max(0, Number(state.pollIntervalMs) || 0));
    }
    return null;
  }

  async function pollStaticJobUntilReady(jobId) {
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
      render();
      await delay(Math.max(0, Number(state.pollIntervalMs) || 0));
    }
  }

  async function submitStaticCorrection(payload) {
    state.phase = 'submitting_static_correction';
    state.message = state.lastLinkageJobId
      ? `Linkage job ${state.lastLinkageJobId} is ready. Submitting static correction...`
      : 'Submitting static correction...';
    render();
    const responsePayload = await postJson(
      '/statics/refraction/apply',
      payload,
      'refraction static apply'
    );
    state.lastStaticCorrectionJobId = trimValue(responsePayload && responsePayload.job_id);
    setStaticJobSnapshot(responsePayload);
    state.lastResponse = responsePayload;
    state.phase = 'idle';
    const initialState = state.lastStaticCorrectionState
      ? ` Initial state: ${state.lastStaticCorrectionState}.`
      : '';
    state.message = state.lastStaticCorrectionJobId
      ? `Static correction job ${state.lastStaticCorrectionJobId} submitted.${initialState}`
      : 'Static correction job submitted.';
    render();
    if (state.lastStaticCorrectionJobId) {
      pollStaticCorrectionJobUntilTerminal(state.lastStaticCorrectionJobId);
    }
    return responsePayload;
  }

  function submitRefractionStaticApply(request) {
    return postJson('/statics/refraction/apply', request, 'refraction static apply');
  }

  async function runStaticCorrection() {
    const { payload, errors } = buildStaticCorrectionRequest();
    state.lastRequest = payload;
    state.lastResponse = null;
    state.lastLinkageBuildRequest = null;
    state.lastLinkageJobId = '';
    state.lastStaticCorrectionJobId = '';
    state.lastStaticCorrectionState = '';
    state.lastStaticCorrectionMessage = '';
    state.lastStaticCorrectionProgress = 0;
    state.staticArtifacts = [];
    stopStaticCorrectionPolling();
    if (errors.length) {
      state.ready = false;
      state.error = errors.join(' ');
      state.validationErrors = errors;
      state.showValidationSummary = true;
      state.message = 'Fix input errors before running refraction statics.';
      state.phase = 'idle';
      render();
      return;
    }

    state.ready = true;
    state.error = '';
    state.validationErrors = [];
    state.showValidationSummary = false;
    state.phase = 'submitting_static_correction';
    render();

    try {
      let applyPayload = payload;
      if (dom.enableLinkage.checked) {
        const linkageBuildPayload = buildStaticLinkageBuildRequest(payload);
        state.lastLinkageBuildRequest = linkageBuildPayload;
        state.phase = 'building_linkage';
        state.message = 'Building endpoint geometry linkage...';
        render();

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
        render();

        await pollStaticJobUntilReady(linkageJobId);
        state.phase = 'linkage_ready';
        applyPayload = {
          ...payload,
          linkage: buildStaticCorrectionLinkage(dom, linkageJobId),
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
      render();
    }
  }

  function handleRun(event) {
    if (event) {
      event.preventDefault();
    }
    runStaticCorrection();
  }

  async function cancelStaticCorrectionJob() {
    const jobId = trimValue(state.lastStaticCorrectionJobId);
    if (!jobId || !isStaticJobActive()) {
      return null;
    }

    try {
      state.message = `Cancelling static correction job ${jobId}...`;
      state.error = '';
      render();
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
      render();
      return snapshot;
    } catch (error) {
      state.error = error instanceof Error ? error.message : String(error);
      state.message = `Failed to cancel static correction job ${jobId}.`;
      render();
      return null;
    }
  }

  function init() {
    const form = document.getElementById('staticCorrectionForm');
    const status = document.getElementById('staticCorrectionStatus');
    const error = document.getElementById('staticCorrectionError');
    const runButton = document.getElementById('staticCorrectionRunButton');
    const fileId = document.getElementById('staticCorrectionFileId');
    const key1Byte = document.getElementById('staticCorrectionKey1Byte');
    const key2Byte = document.getElementById('staticCorrectionKey2Byte');
    const presetSelect = document.getElementById('staticCorrectionPresetSelect');
    const presetName = document.getElementById('staticCorrectionPresetName');
    const savePresetButton = document.getElementById('staticCorrectionSavePresetButton');
    const loadPresetButton = document.getElementById('staticCorrectionLoadPresetButton');
    const deletePresetButton = document.getElementById('staticCorrectionDeletePresetButton');
    const pickKind = document.getElementById('staticCorrectionPickKind');
    const pickJobId = document.getElementById('staticCorrectionPickJobId');
    const pickArtifactName = document.getElementById('staticCorrectionPickArtifactName');
    const loadPickArtifactsButton = document.getElementById('staticCorrectionLoadPickArtifactsButton');
    const pickArtifactList = document.getElementById('staticCorrectionPickArtifactList');
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
    if (
      !form || !status || !error || !runButton || !fileId || !key1Byte || !key2Byte
      || !presetSelect || !presetName || !savePresetButton || !loadPresetButton
      || !deletePresetButton
      || !pickKind || !pickJobId || !pickArtifactName || !loadPickArtifactsButton
      || !pickArtifactList || !geometryPreset || !sourceIdByte || !receiverIdByte
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
      || !registerCorrectedFile || !exportEnabled || !validationSummary || !requestPreview
      || !cancelButton || !staticJobPanel || !staticJobIdValue || !staticJobStateValue
      || !staticJobMessageValue || !staticJobProgress || !staticJobProgressValue
      || !staticArtifactTable || !staticArtifactBody || !staticArtifactEmpty
      || exportFormatInputs.length === 0
    ) {
      return;
    }

    setDefaultValue(key1Byte, DEFAULTS.key1Byte);
    setDefaultValue(key2Byte, DEFAULTS.key2Byte);
    setDefaultValue(pickKind, DEFAULTS.pickKind);
    setDefaultValue(pickArtifactName, DEFAULTS.pickArtifactName);
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
      runButton,
      fileId,
      key1Byte,
      key2Byte,
      presetSelect,
      presetName,
      savePresetButton,
      loadPresetButton,
      deletePresetButton,
      pickKind,
      pickJobId,
      pickArtifactName,
      loadPickArtifactsButton,
      pickArtifactList,
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
    };
    state.presets = readStoredPresets();
    applyStaticCorrectionGeometryPreset(dom, trimValue(geometryPreset.value) || GEOMETRY_DEFAULTS.preset);
    updateModelPresetControls(dom);
    updateStaticCorrectionLinkageOptions(dom);
    updateStaticCorrectionFieldCorrectionOptions(dom);

    form.addEventListener('submit', handleRun);
    form.addEventListener('input', () => {
      if (state.showValidationSummary) {
        state.validationErrors = getStaticCorrectionValidationSnapshot(dom).errors;
      }
      render();
    });
    form.addEventListener('change', () => {
      if (state.showValidationSummary) {
        state.validationErrors = getStaticCorrectionValidationSnapshot(dom).errors;
      }
      render();
    });
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
    loadPickArtifactsButton.addEventListener('click', (event) => {
      event.preventDefault();
      loadPickArtifacts();
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

    render();
  }

  window.refractionStaticRunState = state;
  window.refractionStaticRunUI = {
    applyStaticCorrectionGeometryPreset,
    buildOneLayerRefractionModel,
    buildRefractionStaticApplyRequest,
    buildStaticCorrectionGeometry,
    buildStaticCorrectionGeometryRequest,
    buildStaticCorrectionLinkage,
    buildStaticCorrectionLinkageRequest,
    buildStaticCorrectionPickSource,
    buildStaticCorrectionRequest,
    buildStaticLinkageBuildRequest,
    cancelStaticCorrectionJob,
    collectGeometryInputs,
    collectFieldCorrectionInputs,
    collectInputs,
    collectLinkageInputs,
    collectModelInputs,
    collectOutputInputs,
    collectPresetInputs,
    applyPresetValues,
    isLikelyPickArtifact,
    deleteSelectedPreset,
    getStaticCorrectionValidationSnapshot,
    loadStaticArtifacts,
    loadPickArtifacts,
    loadSelectedPreset,
    normalizeStaticJobState,
    pollStaticCorrectionJobUntilTerminal,
    pollStaticCorrectionStatus,
    pollStaticJobUntilReady,
    render,
    runStaticCorrection,
    saveCurrentPreset,
    stopStaticCorrectionPolling,
    submitRefractionStaticApply,
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
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
