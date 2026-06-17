export const RECENT_JOBS_KEY = 'sv.refraction_qc.recent_jobs';
export const MAX_RECENT_JOBS = 8;
export const DEFAULT_MAX_POINTS = 20000;

export const VIEW_DEFS = [
  {
    id: 'summary',
    include: 'summary',
    viewKeys: [],
    unavailableKeys: ['summary'],
  },
  {
    id: 'first_break_residuals',
    include: 'first_break',
    viewKeys: ['first_break_fit', 'first_break_residual'],
    unavailableKeys: ['first_break'],
  },
  {
    id: 'reduced_time',
    include: 'reduced_time',
    viewKeys: ['reduced_time'],
    unavailableKeys: ['reduced_time'],
  },
  {
    id: 'profiles_2d',
    include: 'profiles',
    viewKeys: ['line_profiles'],
    unavailableKeys: ['profiles'],
  },
  {
    id: 'cell_maps_3d',
    include: 'cells',
    viewKeys: ['refraction_grid_map_qc'],
    unavailableKeys: ['cells'],
  },
  {
    id: 'static_components',
    include: 'static_components',
    viewKeys: ['static_components'],
    unavailableKeys: ['static_components'],
  },
  {
    id: 'pick_map',
    include: 'summary',
    viewKeys: [],
    unavailableKeys: [],
  },
  {
    id: 'offset_time',
    include: 'summary',
    viewKeys: [],
    unavailableKeys: [],
  },
  {
    id: 'station_structure',
    include: 'summary',
    viewKeys: [],
    unavailableKeys: [],
  },
  {
    id: 'gather_preview',
    include: 'gather_preview',
    viewKeys: [],
    unavailableKeys: ['gather_preview'],
  },
  {
    id: 'artifacts',
    include: 'summary',
    viewKeys: [],
    unavailableKeys: [],
  },
];

export const INCLUDE_ALL = Array.from(new Set(VIEW_DEFS.map((view) => view.include)));

export const TASK_DEFS = [
  { id: 'overview', defaultView: 'summary' },
  { id: 'find_problems', defaultView: 'first_break_residuals' },
  { id: 'inspect_station', defaultView: 'profiles_2d' },
  { id: 'inspect_cell', defaultView: 'cell_maps_3d' },
  { id: 'preview_gather', defaultView: 'gather_preview' },
  { id: 'artifacts', defaultView: 'artifacts' },
];

export const VIEW_TASKS = {
  summary: 'overview',
  first_break_residuals: 'find_problems',
  reduced_time: 'find_problems',
  pick_map: 'find_problems',
  offset_time: 'find_problems',
  profiles_2d: 'inspect_station',
  static_components: 'inspect_station',
  station_structure: 'inspect_station',
  cell_maps_3d: 'inspect_cell',
  gather_preview: 'preview_gather',
  artifacts: 'artifacts',
};

export const TASK_VIEW_IDS = {
  overview: ['summary'],
  find_problems: ['first_break_residuals', 'reduced_time', 'pick_map', 'offset_time'],
  inspect_station: ['profiles_2d', 'static_components', 'station_structure'],
  inspect_cell: ['cell_maps_3d'],
  preview_gather: ['gather_preview'],
  artifacts: ['artifacts'],
};

export const LAYER_CONTROL_OPTIONS = [
  ['all', 'All'],
  ['v1_direct_arrival', 'V1 direct'],
  ['v2_t1', 'V2/T1'],
  ['v3_t2', 'V3/T2'],
  ['vsub_t3', 'Vsub/T3'],
];

export const X_AXIS_CONTROL_OPTIONS = [
  ['offset', 'Offset'],
  ['inline', 'Inline'],
  ['trace', 'Trace index'],
];

export const FIRST_BREAK_SORT_CONTROL_OPTIONS = [
  ['largest_residual', 'Largest residual'],
  ['trace', 'Trace'],
  ['offset', 'Offset'],
];

export const PROFILE_GROUP_CONTROL_OPTIONS = [
  ['time_terms', 'Time terms'],
  ['velocities', 'Velocities'],
  ['thickness_elevation', 'Thickness / elevations'],
  ['statics', 'Static components'],
  ['qc_metrics', 'QC metrics'],
];

export const PROFILE_UNITS_CONTROL_OPTIONS = [
  ['auto', 'Auto'],
  ['ms', 'Milliseconds'],
  ['s', 'Seconds'],
];

export const STATUS_FILTER_CONTROL_OPTIONS = [
  ['all', 'All statuses'],
  ['valid', 'Valid only'],
  ['invalid', 'Invalid only'],
];

export const CELL_MAP_QUANTITY_CONTROL_OPTIONS = [
  ['velocity', 'Velocity'],
  ['velocity_update', 'Velocity update'],
  ['fold', 'Fold / observation count'],
  ['residual_rms', 'Residual RMS'],
  ['residual_mad', 'Residual MAD'],
  ['status', 'Status'],
];

export const ARTIFACT_TYPE_CONTROL_OPTIONS = [
  ['all', 'All artifacts'],
  ['manifest', 'Manifest entries'],
  ['view', 'View artifacts'],
];

export const ENDPOINT_KIND_CONTROL_OPTIONS = [
  ['both', 'both'],
  ['source', 'source'],
  ['receiver', 'receiver'],
];
