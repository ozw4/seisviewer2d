(function () {
  const DEFAULTS = {
    key1Byte: '189',
    key2Byte: '193',
    pickKind: 'batch_predicted_npz',
    pickArtifactName: 'predicted_picks_time_s.npz',
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

  const state = {
    ready: false,
    message: [
      'Enter a SEG-Y/TraceStore file_id and a first-break pick artifact usable by refraction statics.',
      'Static correction job submission is not enabled in this milestone.',
    ].join(' '),
    error: '',
    loadingPickArtifacts: false,
    pickArtifacts: [],
    lastRequest: null,
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

  function collectInputs() {
    if (!dom) return null;
    return {
      file_id: trimValue(dom.fileId.value),
      key1_byte: trimValue(dom.key1Byte.value),
      key2_byte: trimValue(dom.key2Byte.value),
      pick_source: {
        kind: trimValue(dom.pickKind.value),
        job_id: trimValue(dom.pickJobId.value),
        artifact_name: trimValue(dom.pickArtifactName.value),
      },
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

  function parsePositiveInteger(value, label, errors) {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      errors.push(`${label} must be a positive integer.`);
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

  function buildStaticCorrectionRequest() {
    const values = collectInputs();
    const errors = [];
    if (!values) {
      return { payload: null, errors: ['Static correction form is not available.'] };
    }

    if (!values.file_id) {
      errors.push('file_id is required.');
    }
    const key1Byte = parsePositiveInteger(values.key1_byte, 'key1_byte', errors);
    const key2Byte = parsePositiveInteger(values.key2_byte, 'key2_byte', errors);
    const pickKind = values.pick_source.kind;
    if (!pickKind) {
      errors.push('pick_source.kind is required.');
    }
    if (ARTIFACT_PICK_KINDS.has(pickKind)) {
      if (!values.pick_source.job_id) {
        errors.push('pick_source.job_id is required for artifact-backed pick sources.');
      }
      if (!values.pick_source.artifact_name) {
        errors.push('pick_source.artifact_name is required for artifact-backed pick sources.');
      }
    }
    if (
      values.pick_source.artifact_name
      && !values.pick_source.artifact_name.toLowerCase().endsWith('.npz')
    ) {
      errors.push('pick_source.artifact_name must be an .npz artifact.');
    }
    const geometryResult = validateStaticCorrectionGeometryRequest(dom);
    errors.push(...geometryResult.errors);

    if (errors.length) {
      return { payload: null, errors };
    }
    return {
      payload: {
        file_id: values.file_id,
        key1_byte: key1Byte,
        key2_byte: key2Byte,
        pick_source: values.pick_source,
        ...geometryResult.payload,
      },
      errors,
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

  async function readResponseError(response) {
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
    return `batch job files ${response.status}${detail ? `: ${detail}` : ''}`;
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
    dom.status.textContent = state.message;
    dom.error.hidden = !state.error;
    dom.error.textContent = state.error;
    dom.runButton.disabled = false;
    if (dom.loadPickArtifactsButton) {
      dom.loadPickArtifactsButton.disabled = state.loadingPickArtifacts;
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
        throw new Error(await readResponseError(response));
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

  function handleRun(event) {
    if (event) {
      event.preventDefault();
    }
    const { payload, errors } = buildStaticCorrectionRequest();
    state.lastRequest = payload;
    if (errors.length) {
      state.ready = false;
      state.error = errors.join(' ');
      state.message = 'Fix input errors before running refraction statics.';
      render();
      return;
    }

    state.ready = true;
    state.error = '';
    state.message = 'Inputs are valid. Static correction job submission is not enabled in this milestone.';
    render();
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
    if (
      !form || !status || !error || !runButton || !fileId || !key1Byte || !key2Byte
      || !pickKind || !pickJobId || !pickArtifactName || !loadPickArtifactsButton
      || !pickArtifactList || !geometryPreset || !sourceIdByte || !receiverIdByte
      || !sourceXByte || !sourceYByte || !receiverXByte || !receiverYByte
      || !sourceElevationByte || !receiverElevationByte || !coordinateScalarByte
      || !elevationScalarByte || !sourceDepthByte || !coordinateUnit || !elevationUnit
      || !offsetByte
    ) {
      return;
    }

    setDefaultValue(key1Byte, DEFAULTS.key1Byte);
    setDefaultValue(key2Byte, DEFAULTS.key2Byte);
    setDefaultValue(pickKind, DEFAULTS.pickKind);
    setDefaultValue(pickArtifactName, DEFAULTS.pickArtifactName);

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
    };
    applyStaticCorrectionGeometryPreset(dom, trimValue(geometryPreset.value) || GEOMETRY_DEFAULTS.preset);

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

    render();
  }

  window.refractionStaticRunState = state;
  window.refractionStaticRunUI = {
    applyStaticCorrectionGeometryPreset,
    buildStaticCorrectionGeometryRequest,
    buildStaticCorrectionRequest,
    collectGeometryInputs,
    collectInputs,
    isLikelyPickArtifact,
    loadPickArtifacts,
    render,
    validateStaticCorrectionGeometryRequest,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
