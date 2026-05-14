(function () {
  const DEFAULTS = {
    key1Byte: '189',
    key2Byte: '193',
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
  };
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
    phase: 'idle',
    pollIntervalMs: 1000,
  };

  let dom = null;

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

  function parsePositiveInteger(value, label, errors) {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      errors.push(`${label} must be a positive integer.`);
      return null;
    }
    return parsed;
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

  function updateBedrockVelocityControls(targetDom = dom) {
    if (!targetDom || !targetDom.bedrockVelocityMode) return;
    const fixedMode = targetDom.bedrockVelocityMode.value === 'fixed_global';
    if (targetDom.initialBedrockVelocityMS) {
      targetDom.initialBedrockVelocityMS.disabled = fixedMode;
    }
    if (targetDom.fixedBedrockVelocityMS) {
      targetDom.fixedBedrockVelocityMS.disabled = !fixedMode;
    }
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

    const initialBedrockVelocity = parsePositiveFloat(
      values.initial_bedrock_velocity_m_s,
      'model.initial_bedrock_velocity_m_s',
      errors,
      { optional: bedrockVelocityMode !== 'solve_global' }
    );
    const fixedBedrockVelocity = parsePositiveFloat(
      values.bedrock_velocity_m_s,
      'model.bedrock_velocity_m_s',
      errors,
      { optional: bedrockVelocityMode !== 'fixed_global' }
    );
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
    const modelResult = validateOneLayerRefractionModel(targetDom);
    errors.push(...modelResult.errors);
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
    const modelResult = validateOneLayerRefractionModel(dom);
    errors.push(...modelResult.errors);
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
        ...outputResult.payload,
      },
      errors,
    };
  }

  function buildStaticLinkageBuildRequest(staticCorrectionPayload) {
    if (!staticCorrectionPayload) {
      throw validationError(['Static correction payload is required before building linkage.']);
    }
    return {
      file_id: staticCorrectionPayload.file_id,
      key1_byte: staticCorrectionPayload.key1_byte,
      key2_byte: staticCorrectionPayload.key2_byte,
      geometry: {
        source_x_byte: staticCorrectionPayload.geometry.source_x_byte,
        source_y_byte: staticCorrectionPayload.geometry.source_y_byte,
        receiver_x_byte: staticCorrectionPayload.geometry.receiver_x_byte,
        receiver_y_byte: staticCorrectionPayload.geometry.receiver_y_byte,
        coordinate_scalar_byte: staticCorrectionPayload.geometry.coordinate_scalar_byte,
      },
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

  function render() {
    if (!dom) return;
    updateStaticCorrectionLinkageOptions(dom);
    dom.status.textContent = state.message;
    dom.error.hidden = !state.error;
    dom.error.textContent = state.error;
    dom.runButton.disabled = false;
    if (dom.loadPickArtifactsButton) {
      dom.loadPickArtifactsButton.disabled = state.loadingPickArtifacts;
    }
    if (dom.requestPreview) {
      dom.requestPreview.hidden = !state.lastRequest;
      dom.requestPreview.textContent = state.lastRequest
        ? JSON.stringify(state.lastRequest, null, 2)
        : '';
    }
    renderPickArtifactList();
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

  async function pollStaticJobUntilReady(jobId) {
    const encodedJobId = encodeURIComponent(jobId);
    while (true) {
      const response = await fetch(`/statics/job/${encodedJobId}/status`);
      if (!response.ok) {
        throw new Error(await readResponseError(response, 'static job status'));
      }
      const payload = await response.json();
      const stateValue = trimValue(payload && payload.state);
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
    state.lastStaticCorrectionState = trimValue(responsePayload && responsePayload.state);
    state.lastResponse = responsePayload;
    state.phase = 'idle';
    const initialState = state.lastStaticCorrectionState
      ? ` Initial state: ${state.lastStaticCorrectionState}.`
      : '';
    state.message = state.lastStaticCorrectionJobId
      ? `Static correction job ${state.lastStaticCorrectionJobId} submitted.${initialState}`
      : 'Static correction job submitted.';
    render();
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
    if (errors.length) {
      state.ready = false;
      state.error = errors.join(' ');
      state.message = 'Fix input errors before running refraction statics.';
      state.phase = 'idle';
      render();
      return;
    }

    state.ready = true;
    state.error = '';
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
      state.ready = false;
      state.phase = state.phase === 'building_linkage' ? 'linkage_failed' : state.phase;
      state.error = error instanceof Error ? error.message : String(error);
      state.message = state.phase === 'linkage_failed'
        ? 'Geometry linkage failed. Static correction was not submitted.'
        : 'Static correction submission failed.';
      render();
    }
  }

  function handleRun(event) {
    if (event) {
      event.preventDefault();
    }
    runStaticCorrection();
  }

  function init() {
    const form = document.getElementById('staticCorrectionForm');
    const status = document.getElementById('staticCorrectionStatus');
    const error = document.getElementById('staticCorrectionError');
    const runButton = document.getElementById('staticCorrectionRunButton');
    const fileId = document.getElementById('staticCorrectionFileId');
    const key1Byte = document.getElementById('staticCorrectionKey1Byte');
    const key2Byte = document.getElementById('staticCorrectionKey2Byte');
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
    const weatheringVelocityMS = document.getElementById('staticCorrectionWeatheringVelocityMS');
    const bedrockVelocityMode = document.getElementById('staticCorrectionBedrockVelocityMode');
    const initialBedrockVelocityMS = document.getElementById('staticCorrectionInitialBedrockVelocityMS');
    const fixedBedrockVelocityMS = document.getElementById('staticCorrectionFixedBedrockVelocityMS');
    const minOffsetM = document.getElementById('staticCorrectionMinOffsetM');
    const maxOffsetM = document.getElementById('staticCorrectionMaxOffsetM');
    const conversionMode = document.getElementById('staticCorrectionConversionMode');
    const registerCorrectedFile = document.getElementById('staticCorrectionRegisterCorrectedFile');
    const exportEnabled = document.getElementById('staticCorrectionExportEnabled');
    const exportFormatInputs = Array.from(document.querySelectorAll('[data-static-correction-export-format]'));
    const requestPreview = document.getElementById('staticCorrectionRequestPreview');
    if (
      !form || !status || !error || !runButton || !fileId || !key1Byte || !key2Byte
      || !pickKind || !pickJobId || !pickArtifactName || !loadPickArtifactsButton
      || !pickArtifactList || !geometryPreset || !sourceIdByte || !receiverIdByte
      || !sourceXByte || !sourceYByte || !receiverXByte || !receiverYByte
      || !sourceElevationByte || !receiverElevationByte || !coordinateScalarByte
      || !elevationScalarByte || !sourceDepthByte || !coordinateUnit || !elevationUnit
      || !offsetByte || !enableLinkage || !linkageOptions || !linkageMode
      || !linkageThresholdM || !receiverLocationIntervalM || !preferReceiverAnchor
      || !weatheringVelocityMS || !bedrockVelocityMode || !initialBedrockVelocityMS
      || !fixedBedrockVelocityMS || !minOffsetM || !maxOffsetM || !conversionMode
      || !registerCorrectedFile || !exportEnabled || !requestPreview
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
    setDefaultValue(weatheringVelocityMS, DEFAULTS.weatheringVelocityMS);
    setDefaultValue(bedrockVelocityMode, DEFAULTS.bedrockVelocityMode);
    setDefaultValue(initialBedrockVelocityMS, DEFAULTS.initialBedrockVelocityMS);
    setDefaultValue(fixedBedrockVelocityMS, DEFAULTS.fixedBedrockVelocityMS);
    setDefaultValue(minOffsetM, DEFAULTS.minOffsetM);
    setDefaultValue(maxOffsetM, DEFAULTS.maxOffsetM);
    setDefaultValue(conversionMode, DEFAULTS.conversionMode);

    dom = {
      form,
      status,
      error,
      runButton,
      fileId,
      key1Byte,
      key2Byte,
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
      weatheringVelocityMS,
      bedrockVelocityMode,
      initialBedrockVelocityMS,
      fixedBedrockVelocityMS,
      minOffsetM,
      maxOffsetM,
      conversionMode,
      registerCorrectedFile,
      exportEnabled,
      exportFormatInputs,
      requestPreview,
    };
    applyStaticCorrectionGeometryPreset(dom, trimValue(geometryPreset.value) || GEOMETRY_DEFAULTS.preset);
    updateStaticCorrectionLinkageOptions(dom);

    form.addEventListener('submit', handleRun);
    runButton.addEventListener('click', handleRun);
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
    updateBedrockVelocityControls(dom);
    bedrockVelocityMode.addEventListener('change', () => {
      updateBedrockVelocityControls(dom);
      state.error = '';
      state.message = bedrockVelocityMode.value === 'fixed_global'
        ? 'Fixed global V2 will be submitted for the one-layer model.'
        : 'Global V2 will be solved from the submitted picks.';
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
    collectGeometryInputs,
    collectInputs,
    collectLinkageInputs,
    collectModelInputs,
    collectOutputInputs,
    isLikelyPickArtifact,
    loadPickArtifacts,
    pollStaticJobUntilReady,
    render,
    runStaticCorrection,
    submitRefractionStaticApply,
    updateBedrockVelocityControls,
    updateStaticCorrectionLinkageOptions,
    validateOneLayerRefractionModel,
    validateStaticCorrectionGeometryRequest,
    validateStaticCorrectionLinkageRequest,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
