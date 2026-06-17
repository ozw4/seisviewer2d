export function buildToolUrl(path, options = {}) {
  const location = options.location || window.location;
  const storage = options.storage || window.localStorage;
  const params = new URLSearchParams(location.search || '');
  const target = new URL(path, location.origin);
  const fileId = params.get('file_id') || storage.getItem('file_id') || '';
  const key1Byte = params.get('key1_byte') || storage.getItem('key1_byte') || '';
  const key2Byte = params.get('key2_byte') || storage.getItem('key2_byte') || '';
  const refractionJobId = params.get('refraction_job_id') || params.get('refraction_qc_job_id') || '';

  if (fileId) target.searchParams.set('file_id', fileId);
  if (key1Byte) target.searchParams.set('key1_byte', key1Byte);
  if (key2Byte) target.searchParams.set('key2_byte', key2Byte);
  if (refractionJobId && path.includes('refraction-qc')) {
    target.searchParams.set('refraction_job_id', refractionJobId);
  }
  return target.toString();
}

export function hydrateToolLinks(root = document) {
  for (const link of root.querySelectorAll('[data-tool-link]')) {
    link.href = buildToolUrl(link.getAttribute('href'));
  }
}
