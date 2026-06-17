function parseFiniteNumberField(value, label, errors) {
  const parsed = Number.parseFloat(String(value ?? '').trim());
  if (!Number.isFinite(parsed)) {
    errors.push(`${label} must be a finite number.`);
    return NaN;
  }
  return parsed;
}

function parsePositiveIntegerField(value, label, errors) {
  const text = String(value ?? '').trim();
  const parsed = Number(text);
  if (!text || !Number.isInteger(parsed) || parsed < 1) {
    errors.push(`${label} must be a positive integer.`);
    return NaN;
  }
  return parsed;
}

function parseIntegerField(value, label, errors) {
  const text = String(value ?? '').trim();
  const parsed = Number(text);
  if (!text || !Number.isInteger(parsed)) {
    errors.push(`${label} must be an integer.`);
    return NaN;
  }
  return parsed;
}

export function buildGatherPreviewRequest(state, jobId, context) {
  const errors = [];
  const fileId = String(context.fileId || '').trim();
  if (!jobId) errors.push('Job ID is required.');
  if (!fileId) errors.push('File ID is required.');

  const key1Byte = parsePositiveIntegerField(context.key1Byte, 'key1 byte', errors);
  const key2Byte = parsePositiveIntegerField(context.key2Byte, 'key2 byte', errors);
  const timeStart = parseFiniteNumberField(state.gatherTimeStartS, 'Time start', errors);
  const timeEnd = parseFiniteNumberField(state.gatherTimeEndS, 'Time end', errors);
  const maxTraces = parsePositiveIntegerField(state.gatherMaxTraces, 'Max traces', errors);
  if (Number.isFinite(timeStart) && timeStart < 0) {
    errors.push('Time start must be greater than or equal to 0.');
  }
  if (Number.isFinite(timeEnd) && Number.isFinite(timeStart) && timeEnd <= timeStart) {
    errors.push('Time end must be greater than time start.');
  }

  const payload = {
    job_id: jobId,
    file_id: fileId,
    key1_byte: key1Byte,
    key2_byte: key2Byte,
    gather_axis: state.gatherAxis,
    time_start_s: timeStart,
    time_end_s: timeEnd,
    max_traces: maxTraces,
    scaling: 'amax',
  };

  if (state.gatherAxis === 'section') {
    const key1 = parseIntegerField(context.key1, 'Section key1', errors);
    const x0 = parseIntegerField(context.x0, 'Trace start', errors);
    const x1 = parseIntegerField(context.x1, 'Trace end', errors);
    if (Number.isInteger(x0) && Number.isInteger(x1) && x1 < x0) {
      errors.push('Trace end must be greater than or equal to trace start.');
    }
    payload.key1 = key1;
    payload.x0 = x0;
    payload.x1 = x1;
  } else {
    const endpointKey = String(state.gatherEndpointKey || '').trim();
    if (!endpointKey) {
      const label = state.gatherAxis === 'receiver' ? 'Receiver station' : 'Source station';
      errors.push(`${label} を選択してください。`);
    }
    payload.endpoint_key = endpointKey;
  }

  if (state.gatherDisplayMode === 'reduced_time') {
    const velocity = parseFiniteNumberField(
      state.gatherReductionVelocity,
      'Reduction velocity',
      errors,
    );
    if (Number.isFinite(velocity) && velocity <= 0) {
      errors.push('Reduction velocity must be greater than 0.');
    }
    payload.reduction_velocity_m_s = velocity;
  }

  return { payload, errors };
}
