export function cleanJobId(jobId) {
  return String(jobId || '').trim();
}

export function isTerminalJobState(state) {
  return ['done', 'error', 'expired', 'cancelled'].includes(String(state || '').trim());
}

export function isActiveJobState(state) {
  const normalized = String(state || '').trim();
  return Boolean(normalized) && !isTerminalJobState(normalized);
}
