import { DEFAULT_MAX_POINTS } from './constants.js';

function initialState() {
  return {
  selectedJobId: '',
  qcBundle: null,
  activeTask: 'overview',
  selectedView: 'summary',
  selectedObject: { kind: null, key: '', payload: {} },
  selectedFirstBreakPick: null,
  selectedProfileEndpoint: null,
  firstBreakDrilldown: null,
  firstBreakDrilldownLoading: false,
  firstBreakDrilldownError: null,
  selectedLayerKind: 'all',
  firstBreakXAxis: 'offset',
  showRejectedFirstBreaks: true,
  firstBreakResidualThresholdMs: '',
  firstBreakSortBy: 'largest_residual',
  selectedEndpointKind: 'source',
  selectedCell: null,
  selectedCellMapQuantity: 'velocity',
  qcDrilldown: null,
  qcDrilldownLoading: false,
  qcDrilldownError: null,
  qcDrilldownTarget: null,
  selectedEndpoint: '',
  selectedTraceIndex: '',
  selectedProfileGroup: 'time_terms',
  selectedProfileUnits: 'auto',
  profileStatusFilter: 'all',
  maxPoints: DEFAULT_MAX_POINTS,
  gatherAxis: 'source',
  gatherDisplayMode: 'side_by_side',
  gatherEndpointKey: '',
  gatherEndpointSearch: '',
  gatherFileId: '',
  gatherKey1Byte: '',
  gatherKey2Byte: '',
  gatherSectionKey1: '',
  gatherX0: '',
  gatherX1: '',
  gatherTimeStartS: '0',
  gatherTimeEndS: '1',
  gatherMaxTraces: 120,
  gatherReductionVelocity: '1500',
  gatherPreview: null,
  gatherLoading: false,
  gatherError: null,
  artifactTypeFilter: 'all',
  artifactSearch: '',
  pickMap: null,
  pickMapLoading: false,
  pickMapError: null,
  pickMapDisplayMode: 'before',
  pickMapGatherStart: '',
  pickMapGatherEnd: '',
  pickMapCachedFile: null,
  pickMapCachedMeta: null,
  pickMapCacheStatus: '',
  stationStructure: null,
  stationStructureLoading: false,
  stationStructureError: null,
  stationStructureGatherStart: '',
  stationStructureGatherEnd: '',
  stationStructureVelocityField: 'auto',
  stationStructureDepthField: 'auto',
  error: null,
  loading: false,
  };
}

export const state = initialState();

export function resetState() {
  for (const key of Object.keys(state)) {
    delete state[key];
  }
  Object.assign(state, initialState());
}

export function clearSelectedObject() {
  state.selectedObject = { kind: null, key: '', payload: {} };
}
