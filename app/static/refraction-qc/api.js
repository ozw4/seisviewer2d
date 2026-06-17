export async function readError(response, label = 'QC bundle request') {
  try {
    const text = await response.text();
    if (!text) return `${label} failed with status ${response.status}`;
    try {
      const payload = JSON.parse(text);
      if (payload && typeof payload.detail === 'string') return payload.detail;
      if (payload && payload.detail) return JSON.stringify(payload.detail);
    } catch (_) {
    }
    return text;
  } catch (_) {
  }
  return `${label} failed with status ${response.status}`;
}

async function jsonFromResponse(response, label) {
  if (!response.ok) {
    throw new Error(await readError(response, label));
  }
  return response.json();
}

export async function fetchQcBundle({ jobId, include, maxPoints }) {
  const response = await fetch('/statics/refraction/qc', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      job_id: jobId,
      include,
      max_points: maxPoints,
      coordinate_mode: 'auto',
    }),
  });
  return jsonFromResponse(response, 'QC bundle request');
}

export async function fetchQcDrilldown({ jobId, target, maxObservations, label = 'Cell drilldown request' }) {
  const body = {
    job_id: jobId,
    target,
  };
  if (maxObservations !== undefined) body.max_observations = maxObservations;
  const response = await fetch('/statics/refraction/qc/drilldown', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return jsonFromResponse(response, label);
}

export async function fetchPickMapUpload(payload, file) {
  const formData = new FormData();
  formData.append('request_json', JSON.stringify(payload));
  formData.append('pick_npz', file, file.name || 'first_breaks.npz');
  const response = await fetch('/statics/refraction/qc/pick-map', {
    method: 'POST',
    body: formData,
  });
  return jsonFromResponse(response, 'Pick Map request');
}

export async function fetchCompletedPickMap(jobId) {
  const response = await fetch('/statics/refraction/qc/pick-map', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  });
  return jsonFromResponse(response, 'Pick Map request');
}

export async function fetchStationStructure(payload) {
  const response = await fetch('/statics/refraction/qc/station-structure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonFromResponse(response, 'Station-structure QC request');
}

export async function fetchGatherPreview(payload) {
  const response = await fetch('/statics/refraction/qc/gather-preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonFromResponse(response, 'Gather preview request');
}
