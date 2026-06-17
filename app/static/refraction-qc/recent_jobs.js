import { MAX_RECENT_JOBS, RECENT_JOBS_KEY } from './constants.js';

export function readRecentJobs(storage = localStorage) {
  try {
    const raw = storage.getItem(RECENT_JOBS_KEY);
    const parsed = JSON.parse(raw || '[]');
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => typeof item === 'string' && item).slice(0, MAX_RECENT_JOBS);
  } catch (_) {
    return [];
  }
}

export function writeRecentJob(jobId, storage = localStorage) {
  const cleanJobId = String(jobId || '').trim();
  if (!cleanJobId) return;
  const recent = readRecentJobs(storage).filter((item) => item !== cleanJobId);
  recent.unshift(cleanJobId);
  try {
    storage.setItem(RECENT_JOBS_KEY, JSON.stringify(recent.slice(0, MAX_RECENT_JOBS)));
  } catch (_) {
  }
}
