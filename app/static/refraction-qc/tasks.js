import { TASK_DEFS, TASK_VIEW_IDS, VIEW_TASKS } from './constants.js';

export function taskForView(viewId) {
  return VIEW_TASKS[viewId] || 'overview';
}

export function defaultViewForTask(taskId) {
  const task = TASK_DEFS.find((item) => item.id === taskId);
  return task ? task.defaultView : 'summary';
}

export function viewIdsForTask(taskId) {
  return TASK_VIEW_IDS[taskId] || ['summary'];
}
