export const DEFAULTS = {
  modelPreset: 'one_layer_global',
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

export const MODEL_PRESETS = new Set([
  'one_layer_global',
  'two_layer_global',
  'three_layer_global',
  'cell_v2_t1_line_2d',
  'cell_v2_t1_grid_3d',
]);
export const CELL_MODEL_PRESETS = new Set(['cell_v2_t1_line_2d', 'cell_v2_t1_grid_3d']);
export const MULTILAYER_MODEL_PRESETS = new Set(['two_layer_global', 'three_layer_global']);
export const GEOMETRY_PRESET_SEG_Y_DEFAULT = 'segy_default';
export const GEOMETRY_PRESET_CUSTOM = 'custom';
export const GEOMETRY_DEFAULTS = {
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
export const GEOMETRY_HEADER_FIELDS = [
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
export const UPLOADED_PICK_KIND = 'uploaded_npz';
export const LINKAGE_THRESHOLD_MODES = new Set(['auto_threshold']);
export const LINKAGE_ARTIFACT_NAME = 'geometry_linkage.npz';
export const LINKAGE_READY_STATES = new Set(['done', 'ready']);
export const LINKAGE_FAILED_STATES = new Set(['error', 'failed', 'cancelled', 'canceled', 'expired']);
export const STATIC_READY_STATES = new Set(['done', 'ready']);
export const STATIC_ACTIVE_STATES = new Set(['queued', 'running', 'cancel_requested']);
export const STATIC_TERMINAL_STATES = new Set(['done', 'ready', 'error', 'cancelled', 'expired']);
export const FIELD_SOURCE_DEPTH_MODES = new Set(['none', 'weathering_velocity_time']);
export const FIELD_UPHOLE_MODES = new Set(['none', 'header_time']);
export const FIELD_MANUAL_STATIC_MODES = new Set(['none', 'artifact_table']);
export const FIELD_MANUAL_SIGN_CONVENTIONS = new Set(['applied_shift_s', 'delay_positive_ms']);
export const PRESET_STORAGE_KEY = 'sv.static_correction.presets';
export const ACTIVE_VIEWER_TARGET_STORAGE_KEY = 'sv.active_viewer_target';
export const STATIC_DRAFT_STORAGE_KEY = 'sv.static_correction.form_draft.v1';
export const STATIC_PICK_DB_NAME = 'seisviewer2d-static-correction';
export const STATIC_PICK_DB_VERSION = 1;
export const STATIC_PICK_STORE = 'pick_npz_blobs';
export const STATIC_PICK_MAX_RECORDS = 10;
export const STATIC_PICK_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;
export const NO_ACTIVE_TARGET_ERROR = (
  'No active viewer file. Open an SGY/TraceStore in the viewer before running Static Correction.'
);
