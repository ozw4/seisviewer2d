import { expect, test, type Page } from '@playwright/test';

function lineProfileRecords() {
	return [
		{
			endpoint_kind: 'source',
			endpoint_key: 'S001',
			node_id: '1',
			inline_m: '0',
			crossline_m: '0',
			x_m: '1000',
			y_m: '2000',
			surface_elevation_m: '120',
			pick_count: '8',
			used_pick_count: '7',
			residual_rms_ms: '3.1',
			residual_mad_ms: '2.2',
			v1_m_s: '800',
			v2_m_s: '2400',
			v3_m_s: '3600',
			vsub_m_s: '',
			t1_ms: '14',
			t2_ms: '28',
			t3_ms: '',
			sh1_weathering_thickness_m: '9.5',
			sh2_weathering_thickness_m: '16.2',
			sh3_weathering_thickness_m: '',
			layer1_base_elevation_m: '110.5',
			layer2_base_elevation_m: '94.3',
			final_refractor_elevation_m: '94.3',
			weathering_correction_ms: '-8',
			elevation_correction_ms: '2',
			source_field_shift_ms: '4.5',
			receiver_field_shift_ms: '',
			field_correction_ms: '4.5',
			manual_static_shift_ms: '0.5',
			manual_static_ms: '0.5',
			total_applied_shift_ms: '-4',
			source_total_with_field_shift_ms: '0.5',
			receiver_total_with_field_shift_ms: '',
			static_status: 'ok',
			solution_status: 'ok',
		},
		{
			endpoint_kind: 'source',
			endpoint_key: 'S002',
			node_id: '2',
			inline_m: '100',
			crossline_m: '0',
			x_m: '1100',
			y_m: '2100',
			surface_elevation_m: '124',
			pick_count: '5',
			used_pick_count: '3',
			residual_rms_ms: '6.2',
			residual_mad_ms: '4',
			v1_m_s: '800',
			v2_m_s: '2500',
			v3_m_s: '3700',
			vsub_m_s: '',
			t1_ms: '18',
			t2_ms: '35',
			t3_ms: '',
			sh1_weathering_thickness_m: '12.0',
			sh2_weathering_thickness_m: '18.0',
			sh3_weathering_thickness_m: '',
			layer1_base_elevation_m: '112.0',
			layer2_base_elevation_m: '94.0',
			final_refractor_elevation_m: '94.0',
			weathering_correction_ms: '-9',
			elevation_correction_ms: '2.5',
			source_field_shift_ms: '6.5',
			receiver_field_shift_ms: '',
			field_correction_ms: '6.5',
			manual_static_shift_ms: '1',
			manual_static_ms: '1',
			total_applied_shift_ms: '-5.5',
			source_total_with_field_shift_ms: '1',
			receiver_total_with_field_shift_ms: '',
			static_status: 'invalid_endpoint',
			solution_status: 'missing_solution',
		},
		{
			endpoint_kind: 'receiver',
			endpoint_key: 'R001',
			node_id: '10',
			inline_m: '20',
			crossline_m: '0',
			x_m: '1020',
			y_m: '2020',
			surface_elevation_m: '118',
			pick_count: '9',
			used_pick_count: '9',
			residual_rms_ms: '2',
			residual_mad_ms: '1.1',
			v1_m_s: '800',
			v2_m_s: '2380',
			v3_m_s: '3550',
			vsub_m_s: '',
			t1_ms: '12',
			t2_ms: '24',
			t3_ms: '',
			sh1_weathering_thickness_m: '8.2',
			sh2_weathering_thickness_m: '15.0',
			sh3_weathering_thickness_m: '',
			layer1_base_elevation_m: '109.8',
			layer2_base_elevation_m: '94.8',
			final_refractor_elevation_m: '94.8',
			weathering_correction_ms: '-7',
			elevation_correction_ms: '1',
			source_field_shift_ms: '',
			receiver_field_shift_ms: '2',
			field_correction_ms: '2',
			manual_static_shift_ms: '0',
			manual_static_ms: '0',
			total_applied_shift_ms: '-4',
			source_total_with_field_shift_ms: '',
			receiver_total_with_field_shift_ms: '-2',
			static_status: 'ok',
			solution_status: 'ok',
		},
		{
			endpoint_kind: 'receiver',
			endpoint_key: 'R002',
			node_id: '11',
			inline_m: '120',
			crossline_m: '0',
			x_m: '1120',
			y_m: '2120',
			surface_elevation_m: '121',
			pick_count: '7',
			used_pick_count: '6',
			residual_rms_ms: '3.4',
			residual_mad_ms: '2',
			v1_m_s: '800',
			v2_m_s: '2450',
			v3_m_s: '3650',
			vsub_m_s: '',
			t1_ms: '16',
			t2_ms: '31',
			t3_ms: '',
			sh1_weathering_thickness_m: '10.1',
			sh2_weathering_thickness_m: '17.3',
			sh3_weathering_thickness_m: '',
			layer1_base_elevation_m: '110.9',
			layer2_base_elevation_m: '93.6',
			final_refractor_elevation_m: '93.6',
			weathering_correction_ms: '-8.5',
			elevation_correction_ms: '1.5',
			source_field_shift_ms: '',
			receiver_field_shift_ms: '2.5',
			field_correction_ms: '2.5',
			manual_static_shift_ms: '0',
			manual_static_ms: '0',
			total_applied_shift_ms: '-4.5',
			source_total_with_field_shift_ms: '',
			receiver_total_with_field_shift_ms: '-2',
			static_status: 'ok',
			solution_status: 'ok',
		},
	];
}

function staticComponentEndpointRecords() {
	return [
		{
			endpoint_kind: 'source',
			endpoint_key: 'S001',
			weathering_correction_ms: '-8',
			elevation_correction_ms: '2',
			source_depth_correction_ms: '5',
			uphole_correction_ms: '-1',
			manual_static_ms: '0.5',
			field_correction_ms: '4.5',
			computed_field_correction_ms: '4.5',
			applied_field_correction_ms: '4.5',
			total_static_ms: '-4',
			total_applied_shift_ms: '-4',
			total_with_field_shift_ms: '0.5',
			apply_to_trace_shift: 'true',
			static_status: 'ok',
			sign_convention: 'corrected(t) = raw(t - shift_s)',
		},
		{
			endpoint_kind: 'source',
			endpoint_key: 'S002',
			weathering_correction_ms: '',
			elevation_correction_ms: '2.5',
			source_depth_correction_ms: '',
			uphole_correction_ms: '',
			manual_static_ms: '1',
			field_correction_ms: '6.5',
			computed_field_correction_ms: '6.5',
			applied_field_correction_ms: '6.5',
			total_static_ms: '-5.5',
			total_applied_shift_ms: '-5.5',
			total_with_field_shift_ms: '1',
			apply_to_trace_shift: 'true',
			static_status: 'invalid_weathering',
			sign_convention: 'corrected(t) = raw(t - shift_s)',
		},
		{
			endpoint_kind: 'receiver',
			endpoint_key: 'R001',
			weathering_correction_ms: '-7',
			elevation_correction_ms: '1',
			source_depth_correction_ms: '',
			uphole_correction_ms: '',
			manual_static_ms: '0',
			field_correction_ms: '2',
			computed_field_correction_ms: '2',
			applied_field_correction_ms: '2',
			total_static_ms: '-4',
			total_applied_shift_ms: '-4',
			total_with_field_shift_ms: '-2',
			apply_to_trace_shift: 'true',
			static_status: 'ok',
			sign_convention: 'corrected(t) = raw(t - shift_s)',
		},
	];
}

function staticComponentTraceRecords() {
	return [
		{
			trace_index_sorted: '0',
			source_endpoint_key: 'S001',
			receiver_endpoint_key: 'R001',
			refraction_shift_ms: '-8',
			weathering_shift_ms: '-15',
			datum_shift_ms: '3',
			field_shift_ms: '6.5',
			computed_field_shift_ms: '6.5',
			applied_field_shift_ms: '6.5',
			trace_field_static_status: 'ok',
			manual_static_shift_ms: '0.5',
			source_depth_shift_ms: '5',
			uphole_shift_ms: '-1',
			final_trace_shift_ms: '-1.5',
			applied_trace_shift_ms: '-1.5',
			apply_to_trace_shift: 'true',
			static_status: 'ok',
			sign_convention: 'corrected(t) = raw(t - shift_s)',
		},
		{
			trace_index_sorted: '1',
			source_endpoint_key: 'S002',
			receiver_endpoint_key: 'R001',
			refraction_shift_ms: '-5.5',
			weathering_shift_ms: '-7',
			datum_shift_ms: '1.5',
			field_shift_ms: '6.5',
			computed_field_shift_ms: '6.5',
			applied_field_shift_ms: '6.5',
			trace_field_static_status: 'ok',
			manual_static_shift_ms: '1',
			source_depth_shift_ms: '',
			uphole_shift_ms: '',
			final_trace_shift_ms: '1.0',
			applied_trace_shift_ms: '1.0',
			apply_to_trace_shift: 'true',
			static_status: 'invalid_weathering',
			sign_convention: 'corrected(t) = raw(t - shift_s)',
		},
	];
}

function oneLayerLineProfileRecords() {
	return lineProfileRecords().map((record) => ({
		...record,
		t2_ms: '',
		t3_ms: '',
		v3_m_s: '',
		vsub_m_s: '',
		sh2_weathering_thickness_m: '',
		sh3_weathering_thickness_m: '',
		layer2_base_elevation_m: '',
	}));
}

function gridMapRecords() {
	return [
		{
			layer_kind: 'v2_t1',
			cell_ix: '0',
			cell_iy: '0',
			cell_center_x_m: '25',
			cell_center_y_m: '25',
			velocity_m_s: '2400',
			initial_velocity_m_s: '2300',
			velocity_update_from_initial_m_s: '100',
			slowness_s_per_m: '0.0004167',
			n_observations: '8',
			n_sources: '3',
			n_receivers: '4',
			residual_rms_ms: '4.5',
			residual_mad_ms: '3.0',
			status: 'solved',
			status_reason: 'ok',
		},
		{
			layer_kind: 'v2_t1',
			cell_ix: '1',
			cell_iy: '0',
			cell_center_x_m: '75',
			cell_center_y_m: '25',
			velocity_m_s: '2500',
			initial_velocity_m_s: '2300',
			velocity_update_from_initial_m_s: '200',
			slowness_s_per_m: '0.0004',
			n_observations: '2',
			n_sources: '1',
			n_receivers: '2',
			residual_rms_ms: '9.0',
			residual_mad_ms: '6.0',
			status: 'low_fold',
			status_reason: 'below_min_observations_per_cell',
		},
		{
			layer_kind: 'v2_t1',
			cell_ix: '0',
			cell_iy: '1',
			cell_center_x_m: '25',
			cell_center_y_m: '75',
			velocity_m_s: '',
			initial_velocity_m_s: '2300',
			velocity_update_from_initial_m_s: '',
			slowness_s_per_m: '',
			n_observations: '0',
			n_sources: '0',
			n_receivers: '0',
			residual_rms_ms: '',
			residual_mad_ms: '',
			status: 'inactive',
			status_reason: 'no_observations',
		},
		{
			layer_kind: 'v2_t1',
			cell_ix: '1',
			cell_iy: '1',
			cell_center_x_m: '75',
			cell_center_y_m: '75',
			velocity_m_s: '2600',
			initial_velocity_m_s: '2300',
			velocity_update_from_initial_m_s: '300',
			slowness_s_per_m: '0.0003846',
			n_observations: '11',
			n_sources: '5',
			n_receivers: '5',
			residual_rms_ms: '2.5',
			residual_mad_ms: '1.5',
			status: 'solved',
			status_reason: 'ok',
		},
		{
			layer_kind: 'v3_t2',
			cell_ix: '0',
			cell_iy: '0',
			cell_center_x_m: '25',
			cell_center_y_m: '25',
			velocity_m_s: '3600',
			initial_velocity_m_s: '3500',
			velocity_update_from_initial_m_s: '100',
			slowness_s_per_m: '0.0002778',
			n_observations: '7',
			n_sources: '3',
			n_receivers: '4',
			residual_rms_ms: '5.5',
			residual_mad_ms: '4.0',
			status: 'solved',
			status_reason: 'ok',
		},
		{
			layer_kind: 'v3_t2',
			cell_ix: '1',
			cell_iy: '0',
			cell_center_x_m: '75',
			cell_center_y_m: '25',
			velocity_m_s: '3700',
			initial_velocity_m_s: '3500',
			velocity_update_from_initial_m_s: '200',
			slowness_s_per_m: '0.0002703',
			n_observations: '9',
			n_sources: '4',
			n_receivers: '4',
			residual_rms_ms: '4.0',
			residual_mad_ms: '2.5',
			status: 'solved',
			status_reason: 'ok',
		},
	];
}

function qcBundlePayload(jobId: string) {
	const profileRecords = lineProfileRecords();
	const staticEndpointRecords = staticComponentEndpointRecords();
	const staticTraceRecords = staticComponentTraceRecords();
	return {
		job_id: jobId,
		statics_kind: 'refraction',
		sign_convention: 'corrected(t) = raw(t - shift_s)',
		coordinate_mode: 'line_2d_projected',
		summary: {
			status: 'ok',
			job_state: 'done',
			workflow: 'refraction_statics',
			method: 'multilayer_time_term',
			conversion_mode: 't1lsst_multilayer',
			layer_count: 2,
			observation_gates: [
				{
					layer_kind: 'v1_direct_arrival',
					min_direct_offset_m: 20,
					max_direct_offset_m: 140,
				},
				{
					layer_kind: 'v2_t1',
					min_offset_m: 0,
					max_offset_m: 1800,
				},
				{
					layer_kind: 'v3_t2',
					min_offset_m: 1800,
					max_offset_m: 3200,
				},
				{
					layer_kind: 'vsub_t3',
					min_offset_m: 3200,
					max_offset_m: null,
				},
			],
		},
		artifacts: {
			first_break_residuals: 'first_break_residuals.csv',
			refraction_first_break_fit_qc_csv: 'refraction_first_break_fit_qc.csv',
			refraction_reduced_time_qc: 'refraction_reduced_time_qc.csv',
			near_surface_model: 'near_surface_model.csv',
			refraction_line_profile_qc_combined: 'refraction_line_profile_qc_combined.csv',
			refraction_refractor_velocity_cells: 'refraction_refractor_velocity_cells.csv',
			refraction_static_components: 'refraction_static_components.csv',
			refraction_static_component_qc_endpoint: 'refraction_static_component_qc_endpoint.csv',
			refraction_static_component_qc_trace: 'refraction_static_component_qc_trace.csv',
		},
		available_views: [
			'summary',
			'first_break_fit',
			'first_break_residual',
			'reduced_time',
			'line_profiles',
			'refractor_cells',
			'static_component_qc_endpoint',
			'static_component_qc_trace',
			'static_components',
		],
		unavailable_views: ['gather_preview'],
		unavailable_view_reasons: {},
		views: {
			first_break_fit: {
				artifact: 'refraction_first_break_fit_qc.csv',
				columns: [
					'observation_index',
					'trace_index_sorted',
					'offset_m',
					'inline_m',
					'observed_first_break_time_s',
					'modeled_first_break_time_s',
					'residual_time_ms',
					'layer_kind',
					'used_in_solve',
					'reject_reason',
					'status',
				],
				total_points: 5,
				returned_points: 3,
				downsampled: true,
				downsampling_method: 'even_index_floor_first_last',
				records: [
					{
						observation_index: '0',
						trace_index_sorted: '0',
						offset_m: '100',
						inline_m: '10',
						observed_first_break_time_s: '0.100',
						modeled_first_break_time_s: '0.095',
						residual_time_ms: '5.0',
						layer_kind: 'v2_t1',
						used_in_solve: 'true',
						reject_reason: '',
						status: 'ok',
					},
					{
						observation_index: '1',
						trace_index_sorted: '1',
						offset_m: '200',
						inline_m: '20',
						observed_first_break_time_s: '0.140',
						modeled_first_break_time_s: '0.142',
						residual_time_ms: '-2.0',
						layer_kind: 'v3_t2',
						used_in_solve: 'true',
						reject_reason: '',
						status: 'ok',
					},
					{
						observation_index: '2',
						trace_index_sorted: '2',
						offset_m: '300',
						inline_m: '30',
						observed_first_break_time_s: '0.180',
						modeled_first_break_time_s: '0.170',
						residual_time_ms: '10.0',
						layer_kind: 'v2_t1',
						used_in_solve: 'false',
						reject_reason: 'outlier',
						status: 'rejected',
					},
				],
			},
			first_break_residual: {
				artifact: 'first_break_residuals.csv',
				columns: ['trace', 'first_break_residual_ms'],
				total_points: 2,
				returned_points: 2,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: [
					{ trace: '0', first_break_residual_ms: '1.25' },
					{ trace: '1', first_break_residual_ms: '-0.50' },
				],
			},
			reduced_time: {
				artifact: 'refraction_reduced_time_qc.csv',
				columns: [
					'trace_index_sorted',
					'source_endpoint_key',
					'receiver_endpoint_key',
					'offset_m',
					'inline_m',
					'layer_gate_kind',
					'reduction_velocity_m_s',
					'observed_first_break_time_s',
					'reduced_time_s',
					'reduced_time_ms',
					'within_v1_gate',
					'within_v2_t1_gate',
					'within_v3_t2_gate',
					'within_vsub_t3_gate',
					'used_for_inversion',
					'status',
				],
				total_points: 5,
				returned_points: 4,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: [
					{
						trace_index_sorted: '0',
						source_endpoint_key: 'S001',
						receiver_endpoint_key: 'R001',
						offset_m: '100',
						inline_m: '10',
						layer_gate_kind: 'v2_t1',
						reduction_velocity_m_s: '2000',
						observed_first_break_time_s: '0.150',
						reduced_time_s: '0.100',
						reduced_time_ms: '100',
						within_v1_gate: 'false',
						within_v2_t1_gate: 'false',
						within_v3_t2_gate: 'false',
						within_vsub_t3_gate: 'false',
						used_for_inversion: 'true',
						status: 'ok',
					},
					{
						trace_index_sorted: '1',
						source_endpoint_key: 'S002',
						receiver_endpoint_key: 'R002',
						offset_m: '2200',
						inline_m: '220',
						layer_gate_kind: 'v3_t2',
						reduction_velocity_m_s: '4000',
						observed_first_break_time_s: '0.800',
						reduced_time_s: '0.250',
						reduced_time_ms: '250',
						within_v1_gate: 'false',
						within_v2_t1_gate: 'false',
						within_v3_t2_gate: 'false',
						within_vsub_t3_gate: 'false',
						used_for_inversion: 'true',
						status: 'ok',
					},
					{
						trace_index_sorted: '2',
						source_endpoint_key: 'S003',
						receiver_endpoint_key: 'R003',
						offset_m: '300',
						inline_m: '30',
						layer_gate_kind: 'v2_t1',
						reduction_velocity_m_s: '',
						observed_first_break_time_s: '0.180',
						reduced_time_s: '',
						reduced_time_ms: '',
						within_v1_gate: 'false',
						within_v2_t1_gate: 'false',
						within_v3_t2_gate: 'false',
						within_vsub_t3_gate: 'false',
						used_for_inversion: 'true',
						status: 'missing_reduction_velocity',
					},
					{
						trace_index_sorted: '3',
						source_endpoint_key: 'S004',
						receiver_endpoint_key: 'R004',
						offset_m: '3600',
						inline_m: '360',
						layer_gate_kind: 'vsub_t3',
						reduction_velocity_m_s: '5000',
						observed_first_break_time_s: '1.100',
						reduced_time_s: '0.380',
						reduced_time_ms: '380',
						within_v1_gate: 'false',
						within_v2_t1_gate: 'false',
						within_v3_t2_gate: 'false',
						within_vsub_t3_gate: 'false',
						used_for_inversion: 'false',
						status: 'ok',
					},
				],
			},
			line_profiles: {
				artifact: 'refraction_line_profile_qc_combined.csv',
				columns: Object.keys(profileRecords[0]),
				total_points: profileRecords.length,
				returned_points: profileRecords.length,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: profileRecords,
			},
			refractor_cells: {
				artifact: 'refraction_refractor_velocity_cells.csv',
				columns: ['cell_ix', 'cell_iy', 'velocity_m_s'],
				total_points: 1,
				returned_points: 1,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: [{ cell_ix: '0', cell_iy: '0', velocity_m_s: '2500.0' }],
			},
			static_components: {
				artifact: 'refraction_static_components.csv',
				columns: [
					'kind',
					'endpoint_key',
					'weathering_status',
					'datum_status',
					'source_depth_status',
					'uphole_status',
					'manual_static_status',
					'field_status',
					'static_status',
				],
				total_points: 3,
				returned_points: 3,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: [
					{
						kind: 'source',
						endpoint_key: 'S001',
						weathering_status: 'ok',
						datum_status: 'ok',
						source_depth_status: 'ok',
						uphole_status: 'ok',
						manual_static_status: 'ok',
						field_status: 'ok',
						static_status: 'ok',
					},
					{
						kind: 'source',
						endpoint_key: 'S002',
						weathering_status: 'invalid_weathering',
						datum_status: 'ok',
						source_depth_status: 'missing',
						uphole_status: 'not_enabled',
						manual_static_status: 'ok',
						field_status: 'ok',
						static_status: 'invalid_weathering',
					},
					{
						kind: 'receiver',
						endpoint_key: 'R001',
						weathering_status: 'ok',
						datum_status: 'ok',
						source_depth_status: 'not_applicable',
						uphole_status: 'not_applicable',
						manual_static_status: 'ok',
						field_status: 'ok',
						static_status: 'ok',
					},
				],
			},
			static_component_qc_endpoint: {
				artifact: 'refraction_static_component_qc_endpoint.csv',
				columns: Object.keys(staticEndpointRecords[0]),
				total_points: staticEndpointRecords.length,
				returned_points: staticEndpointRecords.length,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: staticEndpointRecords,
			},
			static_component_qc_trace: {
				artifact: 'refraction_static_component_qc_trace.csv',
				columns: Object.keys(staticTraceRecords[0]),
				total_points: staticTraceRecords.length,
				returned_points: staticTraceRecords.length,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: staticTraceRecords,
			},
		},
		downsampling: {
			first_break_fit: {
				total_points: 5,
				returned_points: 3,
				downsampled: true,
				method: 'even_index_floor_first_last',
			},
		},
	};
}

function grid3dQcBundlePayload(jobId: string) {
	const payload = qcBundlePayload(jobId) as any;
	const records = gridMapRecords();
	payload.coordinate_mode = 'grid_3d';
	payload.artifacts.refraction_grid_map_qc = 'refraction_grid_map_qc.csv';
	payload.available_views = [
		...payload.available_views.filter((view: string) => view !== 'refraction_grid_map_qc'),
		'refraction_grid_map_qc',
	];
	payload.unavailable_views = payload.unavailable_views.filter((view: string) => view !== 'cells');
	payload.views.refraction_grid_map_qc = {
		artifact: 'refraction_grid_map_qc.csv',
		columns: Object.keys(records[0]),
		total_points: records.length,
		returned_points: records.length,
		downsampled: false,
		downsampling_method: 'even_index_floor_first_last',
		records,
	};
	payload.downsampling.refraction_grid_map_qc = {
		total_points: records.length,
		returned_points: records.length,
		downsampled: false,
		method: 'even_index_floor_first_last',
	};
	return payload;
}

function globalVelocityQcBundlePayload(jobId: string) {
	const payload = qcBundlePayload(jobId) as any;
	payload.coordinate_mode = 'auto';
	payload.available_views = payload.available_views.filter((view: string) => (
		view !== 'refraction_grid_map_qc'
		&& view !== 'refractor_cells'
		&& view !== 'v3_refractor_cells'
		&& view !== 'vsub_refractor_cells'
	));
	payload.unavailable_views = Array.from(new Set([...payload.unavailable_views, 'cells']));
	delete payload.views.refraction_grid_map_qc;
	delete payload.views.refractor_cells;
	delete payload.views.v3_refractor_cells;
	delete payload.views.vsub_refractor_cells;
	delete payload.artifacts.refraction_grid_map_qc;
	delete payload.artifacts.refraction_refractor_velocity_cells;
	return payload;
}

function gatherPreviewPayload(jobId: string, options: {
	correctedStatus?: 'ok' | 'not_registered' | 'unavailable';
	correctedSource?: string;
} = {}) {
	const correctedStatus = options.correctedStatus ?? 'ok';
	return {
		job_id: jobId,
		statics_kind: 'refraction',
		sign_convention: 'corrected(t) = raw(t - shift_s)',
		raw_window_ref: {
			status: 'ok',
			endpoint: '/get_section_window_bin',
			query: { file_id: 'raw-preview-file', key1: 100 },
		},
		corrected_window_ref: {
			status: correctedStatus,
			source: 'corrected_tracestore',
			endpoint: correctedStatus === 'ok' ? '/get_section_window_bin' : undefined,
			query: correctedStatus === 'ok' ? { file_id: 'corrected-preview-file', key1: 100 } : undefined,
			message: correctedStatus === 'ok'
				? undefined
				: 'Registered corrected TraceStore is not available for this job.',
		},
		raw_samples: [
			[0, 0],
			[1, 0.25],
			[0.2, 1],
			[0, 0.1],
		],
		corrected_samples: [
			[0, 0.1],
			[0.4, 0],
			[1, 0.2],
			[0.2, 1],
		],
		corrected_samples_source: options.correctedSource ?? (
			correctedStatus === 'ok'
				? 'corrected_tracestore'
				: 'raw_tracestore_shifted_on_the_fly'
		),
		dt_s: 0.1,
		shape: [4, 2],
		window: {
			key1: 100,
			y0: 0,
			y1: 3,
			sample_start: 0,
			sample_stop: 3,
			requested_trace_count: 2,
			returned_trace_count: 2,
			requested_sample_count: 4,
			returned_sample_count: 4,
			effective_step_y: 1,
			trace_capped: false,
			sample_capped: false,
		},
		gather: {
			axis: 'source',
			endpoint_key: 'S001',
			overlay_status: 'available',
		},
		x_indices: [0, 1],
		trace_indices: [10, 11],
		offset_m: [100, 200],
		source_endpoint_key: ['S001', 'S001'],
		receiver_endpoint_key: ['R001', 'R002'],
		observed_pick_time_s: [0.1, 0.2],
		modeled_pick_time_s: [0.11, 0.19],
		residual_s: [-0.01, 0.01],
		final_trace_shift_s: [0.02, -0.01],
		corrected_observed_pick_time_s: [0.12, 0.19],
		corrected_modeled_pick_time_s: [0.13, 0.18],
		reduced_observed_time_s: [0.05, 0.1],
		reduced_modeled_time_s: [0.06, 0.09],
		overlay_status: {
			first_break_fit: 'available',
			shift_field: 'final_trace_shift_s',
			reduction_velocity_m_s: null,
		},
		artifacts: {
			qc: 'refraction_static_qc.json',
			first_break_fit_qc_npz: 'refraction_first_break_fit_qc.npz',
			refraction_static_solution_npz: 'refraction_static_solution.npz',
		},
	};
}

async function loadRefractionQcBundle(page: Page, jobId: string) {
	await openRefractionQcTab(page);
	await page.getByTestId('refraction-qc-job-id').fill(jobId);
	await page.getByTestId('refraction-qc-load').click();
	await expect(page.getByTestId('refraction-qc-status')).toContainText(`Loaded ${jobId}`);
}

async function openGatherPreview(page: Page, jobId: string) {
	await loadRefractionQcBundle(page, jobId);
	await page.getByTestId('refraction-qc-view-gather-button').click();
	await page.getByTestId('refraction-qc-gather-file-id').fill('raw-preview-file');
	await page.getByTestId('refraction-qc-endpoint').fill('S001');
}

async function gatherPlotSummary(page: Page, testId: string) {
	return page.getByTestId(testId).evaluate((node) => {
		const plot = node as HTMLElement & {
			data?: Array<{
				name?: string;
				type?: string;
				x?: number[];
				y?: number[];
				z?: number[][];
			}>;
			layout?: {
				xaxis?: { title?: { text?: string } };
				yaxis?: { title?: { text?: string } };
			};
		};
		return {
			xAxisTitle: plot.layout?.xaxis?.title?.text ?? '',
			yAxisTitle: plot.layout?.yaxis?.title?.text ?? '',
			traces: (plot.data ?? []).map((trace) => ({
				name: trace.name ?? '',
				type: trace.type ?? '',
				x: Array.isArray(trace.x) ? trace.x : [],
				y: Array.isArray(trace.y) ? trace.y : [],
				z: Array.isArray(trace.z) ? trace.z : [],
			})),
		};
	});
}

async function residualPlotPointCount(page: Page) {
	return page.getByTestId('refraction-qc-first-break-residual-plot').evaluate((node) => {
		const plot = node as HTMLElement & { data?: Array<{ x?: unknown[] }> };
		return plot.data?.reduce((total, trace) => total + (Array.isArray(trace.x) ? trace.x.length : 0), 0) ?? 0;
	});
}

async function reducedTimePlotPointCount(page: Page) {
	return page.getByTestId('refraction-qc-reduced-time-plot').evaluate((node) => {
		const plot = node as HTMLElement & { data?: Array<{ x?: unknown[] }> };
		return plot.data?.reduce((total, trace) => total + (Array.isArray(trace.x) ? trace.x.length : 0), 0) ?? 0;
	});
}

async function profilePlotSummary(page: Page) {
	return page.getByTestId('refraction-qc-profile-plot').evaluate((node) => {
		const plot = node as HTMLElement & {
			data?: Array<{
				name?: string;
				x?: unknown[];
				y?: number[];
				line?: { dash?: string };
				marker?: { symbol?: string[] };
			}>;
			layout?: { yaxis?: { title?: { text?: string } } };
		};
		return {
			axisTitle: plot.layout?.yaxis?.title?.text ?? '',
			traces: (plot.data ?? []).map((trace) => ({
				name: trace.name ?? '',
				pointCount: Array.isArray(trace.x) ? trace.x.length : 0,
				values: Array.isArray(trace.y) ? trace.y.map((value) => Math.round(value * 1000) / 1000) : [],
				dash: trace.line?.dash ?? '',
				symbols: Array.isArray(trace.marker?.symbol) ? trace.marker?.symbol : [],
			})),
		};
	});
}

async function staticComponentPlotSummary(page: Page, testId = 'refraction-qc-static-waterfall') {
	return page.getByTestId(testId).evaluate((node) => {
		const plot = node as HTMLElement & {
			data?: Array<{
				name?: string;
				x?: number[];
				y?: string[];
				text?: string[];
			}>;
			layout?: { xaxis?: { title?: { text?: string } } };
		};
		const trace = plot.data?.[0];
		return {
			name: trace?.name ?? '',
			axisTitle: plot.layout?.xaxis?.title?.text ?? '',
			components: (trace?.y ?? []).map((label, index) => ({
				label,
				value: Math.round(((trace?.x ?? [])[index] ?? 0) * 1000) / 1000,
				text: (trace?.text ?? [])[index] ?? '',
			})),
		};
	});
}

async function cellMapPlotSummary(page: Page) {
	return page.getByTestId('refraction-qc-cell-map-plot').evaluate((node) => {
		const plot = node as HTMLElement & {
			data?: Array<{
				name?: string;
				z?: Array<Array<number | null>>;
				text?: string[][];
				customdata?: Array<Array<{ cell_ix?: number; cell_iy?: number; layer_kind?: string } | null>>;
				colorbar?: { title?: { text?: string } };
			}>;
			layout?: {
				title?: { text?: string };
				xaxis?: { title?: { text?: string } };
				yaxis?: { title?: { text?: string } };
			};
		};
		const heatmap = plot.data?.[0];
		return {
			title: plot.layout?.title?.text ?? '',
			xAxisTitle: plot.layout?.xaxis?.title?.text ?? '',
			yAxisTitle: plot.layout?.yaxis?.title?.text ?? '',
			colorbarTitle: heatmap?.colorbar?.title?.text ?? '',
			z: heatmap?.z ?? [],
			text: heatmap?.text ?? [],
			customdata: heatmap?.customdata ?? [],
			traceNames: (plot.data ?? []).map((trace) => trace.name ?? ''),
		};
	});
}

async function emitCellMapClick(page: Page, cellIx: number, cellIy: number) {
	await page.getByTestId('refraction-qc-cell-map-plot').evaluate(
		(node, target) => {
			const plot = node as HTMLElement & {
				data?: Array<{
					customdata?: Array<Array<{ cell_ix?: number; cell_iy?: number; layer_kind?: string } | null>>;
				}>;
				emit?: (name: string, event: unknown) => void;
			};
			const heatmap = plot.data?.[0];
			const cells = heatmap?.customdata?.flat() ?? [];
			const cell = cells.find((item) => (
				item?.cell_ix === target.cellIx
				&& item?.cell_iy === target.cellIy
			));
			if (!cell || typeof plot.emit !== 'function') {
				throw new Error('Cell map plot click target is unavailable');
			}
			plot.emit('plotly_click', { points: [{ customdata: cell }] });
		},
		{ cellIx, cellIy },
	);
}

async function openRefractionQcTab(page: Page) {
	await page.goto('/');
	await page.getByTestId('refraction-qc-tab').click();
	await expect(page.getByTestId('refraction-qc-panel')).toBeVisible();
}

test('refraction QC tab loads', async ({ page }) => {
	await openRefractionQcTab(page);

	await expect(page.getByTestId('refraction-qc-job-id')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-status')).toContainText('No QC bundle loaded.');
	await expect(page.getByTestId('refraction-qc-layer-kind')).toHaveValue('all');
	await expect(page.getByTestId('refraction-qc-x-axis')).toHaveValue('offset');
	await expect(page.getByTestId('refraction-qc-profile-group')).toHaveValue('time_terms');
	await expect(page.getByTestId('refraction-qc-profile-units')).toHaveValue('auto');
	await expect(page.getByTestId('refraction-qc-status-filter')).toHaveValue('all');
	await expect(page.getByTestId('refraction-qc-map-quantity')).toHaveValue('velocity');
	await expect(page.getByTestId('refraction-qc-show-rejected')).toBeChecked();
	await expect(page.getByTestId('refraction-qc-endpoint-kind')).toHaveValue('source');
	await expect(page.getByTestId('refraction-qc-view-summary-button')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-first-break-button')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-reduced-time-button')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-profiles-button')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-cells-button')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-statics-button')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-gather-button')).toBeVisible();
});

test('static correction tab scaffold switches side panels', async ({ page }) => {
	const staticsRequests: string[] = [];
	await page.route('**/statics/refraction/**', async (route) => {
		staticsRequests.push(route.request().url());
		await route.abort();
	});

	await page.goto('/');

	await expect(page.getByTestId('static-correction-tab')).toBeVisible();
	await expect(page.getByTestId('static-correction-tab')).toHaveAttribute('aria-selected', 'false');
	await expect(page.locator('#staticCorrectionTabPanel')).toHaveAttribute('role', 'tabpanel');
	await expect(page.locator('#staticCorrectionTabPanel')).toHaveAttribute(
		'aria-labelledby',
		'staticCorrectionSidebarTab',
	);

	await page.getByTestId('static-correction-tab').click();

	const panel = page.getByTestId('static-correction-panel');
	await expect(panel).toBeVisible();
	await expect(panel.getByRole('heading', { name: 'Static Correction' })).toBeVisible();
	for (const heading of ['Input', 'First-break picks', 'Geometry', 'Linkage', 'Model', 'Output', 'Run']) {
		await expect(panel.getByRole('heading', { name: heading })).toBeVisible();
	}
	await expect(page.getByTestId('static-correction-form')).toBeVisible();
	await expect(page.getByTestId('static-correction-status')).toContainText(
		'Enter a SEG-Y/TraceStore file_id and a first-break pick artifact usable by refraction statics.',
	);
	await expect(page.getByTestId('static-correction-file-id')).toBeVisible();
	await expect(page.getByTestId('static-correction-key1-byte')).toHaveValue('189');
	await expect(page.getByTestId('static-correction-key2-byte')).toHaveValue('193');
	await expect(page.getByTestId('static-correction-pick-kind')).toHaveValue('batch_predicted_npz');
	await expect(page.getByTestId('static-correction-pick-job-id')).toBeVisible();
	await expect(page.getByTestId('static-correction-pick-artifact-name')).toHaveValue('predicted_picks_time_s.npz');
	await expect(panel).toContainText('/statics/refraction/apply');
	await expect(panel).toContainText('viewer first-break probability cache is not a valid statics pick artifact');
	await expect(page.getByTestId('static-correction-error')).toBeHidden();
	await expect(page.getByTestId('static-correction-run')).toBeVisible();
	await expect(page.getByTestId('static-correction-run')).toBeEnabled();
	await expect(page.getByTestId('pipeline-sidebar-tab')).toHaveAttribute('aria-selected', 'false');
	await expect(page.getByTestId('refraction-qc-tab')).toHaveAttribute('aria-selected', 'false');
	await expect(page.getByTestId('static-correction-tab')).toHaveAttribute('aria-selected', 'true');
	await expect(page.locator('#pipelineTabPanel')).toBeHidden();
	await expect(page.getByTestId('refraction-qc-panel')).toBeHidden();

	await page.getByTestId('refraction-qc-tab').click();
	await expect(page.getByTestId('refraction-qc-panel')).toBeVisible();
	await expect(panel).toBeHidden();
	await expect(page.getByTestId('refraction-qc-tab')).toHaveAttribute('aria-selected', 'true');
	await expect(page.getByTestId('static-correction-tab')).toHaveAttribute('aria-selected', 'false');

	expect(staticsRequests).toEqual([]);
});

test('static correction tab validates required file and pick inputs without submitting', async ({ page }) => {
	const staticsRequests: string[] = [];
	await page.route('**/statics/refraction/**', async (route) => {
		staticsRequests.push(route.request().url());
		await route.abort();
	});

	await page.goto('/');
	await page.getByTestId('static-correction-tab').click();
	await page.getByTestId('static-correction-run').click();

	await expect(page.getByTestId('static-correction-error')).toBeVisible();
	await expect(page.getByTestId('static-correction-error')).toContainText('file_id is required');
	await expect(page.getByTestId('static-correction-error')).toContainText('pick_source.job_id is required');
	await expect(page.getByTestId('static-correction-status')).toContainText(
		'Fix input errors before running refraction statics.',
	);
	expect(staticsRequests).toEqual([]);
});

test('static correction one-layer builder defaults to no linkage and solved global V2', async ({ page }) => {
	await page.goto('/');
	await page.getByTestId('static-correction-tab').click();
	await page.getByTestId('static-correction-file-id').fill('line-a-store');
	await page.getByTestId('static-correction-pick-job-id').fill('pick-job');

	const request = await page.evaluate(() => (
		(window as any).refractionStaticRunUI.buildRefractionStaticApplyRequest()
	));

	expect(request).toMatchObject({
		file_id: 'line-a-store',
		pick_source: {
			kind: 'batch_predicted_npz',
			job_id: 'pick-job',
			artifact_name: 'predicted_picks_time_s.npz',
		},
		linkage: {
			mode: 'none',
		},
		model: {
			method: 'gli_variable_thickness',
			first_layer: {
				mode: 'constant',
				weathering_velocity_m_s: 800,
			},
			bedrock_velocity_mode: 'solve_global',
			initial_bedrock_velocity_m_s: 2400,
		},
		moveout: {
			model: 'head_wave_linear_offset',
			distance_source: 'geometry',
			offset_byte: 37,
			min_offset_m: 300,
			max_offset_m: 4000,
		},
		conversion: {
			mode: 't1lsst_1layer',
		},
		export: {
			enabled: true,
			formats: ['canonical_static_table', 'time_term_spreadsheet'],
		},
		apply: {
			register_corrected_file: false,
		},
	});
});

test('static correction run submits one-layer refraction apply request', async ({ page }) => {
	let applyRequest: Record<string, unknown> | null = null;
	await page.route('**/statics/refraction/apply', async (route) => {
		applyRequest = JSON.parse(route.request().postData() || '{}');
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({
				job_id: 'refraction-job-559',
				state: 'queued',
				requested_formats: ['canonical_static_table', 'time_term_spreadsheet'],
			}),
		});
	});

	await page.goto('/');
	await page.getByTestId('static-correction-tab').click();
	await page.getByTestId('static-correction-file-id').fill('line-a-store');
	await page.getByTestId('static-correction-pick-job-id').fill('pick-job');
	await page.getByTestId('static-correction-run').click();

	await expect(page.getByTestId('static-correction-status')).toContainText(
		'Static correction job refraction-job-559 submitted. Initial state: queued.',
	);
	await expect(page.getByTestId('static-correction-request-preview')).toBeVisible();
	expect(applyRequest).toMatchObject({
		file_id: 'line-a-store',
		linkage: { mode: 'none' },
		model: {
			first_layer: {
				mode: 'constant',
				weathering_velocity_m_s: 800,
			},
			bedrock_velocity_mode: 'solve_global',
		},
		conversion: { mode: 't1lsst_1layer' },
		export: {
			enabled: true,
			formats: ['canonical_static_table', 'time_term_spreadsheet'],
		},
		apply: { register_corrected_file: false },
	});
});

test('static correction submit errors keep user input visible', async ({ page }) => {
	await page.route('**/statics/refraction/apply', async (route) => {
		await route.fulfill({
			status: 422,
			contentType: 'application/json',
			body: JSON.stringify({ detail: 'model.initial_bedrock_velocity_m_s is outside available picks' }),
		});
	});

	await page.goto('/');
	await page.getByTestId('static-correction-tab').click();
	await page.getByTestId('static-correction-file-id').fill('line-a-store');
	await page.getByTestId('static-correction-pick-job-id').fill('pick-job');
	await page.getByTestId('static-correction-run').click();

	await expect(page.getByTestId('static-correction-error')).toBeVisible();
	await expect(page.getByTestId('static-correction-error')).toContainText(
		'model.initial_bedrock_velocity_m_s is outside available picks',
	);
	await expect(page.getByTestId('static-correction-status')).toContainText(
		'Static correction submission failed.',
	);
	await expect(page.getByTestId('static-correction-file-id')).toHaveValue('line-a-store');
	await expect(page.getByTestId('static-correction-pick-job-id')).toHaveValue('pick-job');
});

test('static correction tab loads likely first-break pick artifacts', async ({ page }) => {
	await page.route('**/batch/job/pick-job/files', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({
				files: [
					{ name: 'job_meta.json', size_bytes: 12 },
					{ name: 'predicted_picks_time_s.npz', size_bytes: 128 },
					{ name: 'manual_picks_time_lineA.npz', size_bytes: 96 },
				],
			}),
		});
	});

	await page.goto('/');
	await page.getByTestId('static-correction-tab').click();
	await page.getByTestId('static-correction-pick-job-id').fill('pick-job');
	await page.getByTestId('static-correction-load-pick-artifacts').click();

	const list = page.getByTestId('static-correction-pick-artifact-list');
	await expect(list).toBeVisible();
	await expect(list).toContainText('predicted_picks_time_s.npz');
	await expect(list).toContainText('manual_picks_time_lineA.npz');
	await expect(list).toContainText('first-break candidate');

	await list.getByRole('button', { name: 'manual_picks_time_lineA.npz' }).click();
	await expect(page.getByTestId('static-correction-pick-artifact-name')).toHaveValue(
		'manual_picks_time_lineA.npz',
	);
});

test('static correction tab displays pick artifact load errors', async ({ page }) => {
	await page.route('**/batch/job/missing-pick-job/files', async (route) => {
		await route.fulfill({
			status: 404,
			contentType: 'application/json',
			body: JSON.stringify({ detail: 'Job ID not found' }),
		});
	});

	await page.goto('/');
	await page.getByTestId('static-correction-tab').click();
	await page.getByTestId('static-correction-pick-job-id').fill('missing-pick-job');
	await page.getByTestId('static-correction-load-pick-artifacts').click();

	await expect(page.getByTestId('static-correction-error')).toBeVisible();
	await expect(page.getByTestId('static-correction-error')).toContainText('batch job files 404: Job ID not found');
	await expect(page.getByTestId('static-correction-status')).toContainText('Unable to load pick artifacts.');
	await expect(page.getByTestId('static-correction-pick-artifact-list')).toBeHidden();
});

test('refraction QC tab fetches bundle for job', async ({ page }) => {
	let requestPayload: Record<string, unknown> | null = null;
	await page.route('**/statics/refraction/qc', async (route) => {
		requestPayload = JSON.parse(route.request().postData() || '{}');
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-1')),
		});
	});

	await openRefractionQcTab(page);
	await page.getByTestId('refraction-qc-job-id').fill('refraction-job-1');
	await page.getByTestId('refraction-qc-max-points').fill('3');
	await page.getByTestId('refraction-qc-load').click();

	await expect(page.getByTestId('refraction-qc-status')).toContainText('Loaded refraction-job-1');
	await expect(page.getByTestId('refraction-qc-sign')).toContainText('corrected(t) = raw(t - shift_s)');
	await expect(page.getByTestId('refraction-qc-view-summary')).toContainText('multilayer_time_term');
	expect(requestPayload).toMatchObject({
		job_id: 'refraction-job-1',
		max_points: 3,
		coordinate_mode: 'auto',
	});
	expect(requestPayload?.include).toEqual([
		'summary',
		'first_break',
		'reduced_time',
		'profiles',
		'cells',
		'static_components',
		'gather_preview',
	]);
	expect(await page.evaluate(() => (window as any).refractionQcState.qcBundle.job_id)).toBe('refraction-job-1');
});

test('refraction QC tab shows error for missing job', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 404,
			contentType: 'application/json',
			body: JSON.stringify({ detail: 'Job ID not found' }),
		});
	});

	await openRefractionQcTab(page);
	await page.getByTestId('refraction-qc-job-id').fill('missing-job');
	await page.getByTestId('refraction-qc-load').click();

	await expect(page.getByTestId('refraction-qc-error')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-error')).toContainText('Job ID not found');
	await expect(page.getByTestId('refraction-qc-status')).toContainText('No QC bundle loaded.');
});

test('refraction QC error state for missing artifact', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		const payload = qcBundlePayload('refraction-missing-artifact') as any;
		delete payload.views.first_break_fit;
		delete payload.views.first_break_residual;
		payload.available_views = payload.available_views.filter((view: string) => (
			view !== 'first_break_fit' && view !== 'first_break_residual'
		));
		payload.unavailable_views = Array.from(new Set([
			...payload.unavailable_views,
			'first_break',
		]));
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(payload),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-missing-artifact');
	await page.getByTestId('refraction-qc-view-first-break-button').click();

	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText(
		'This view is unavailable from the loaded QC bundle artifacts.',
	);
	await expect(page.getByTestId('refraction-qc-first-break-time-plot')).toHaveCount(0);
	await expect(page.getByTestId('refraction-qc-first-break-residual-plot')).toHaveCount(0);
});

test('refraction QC tab view switching', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-2')),
		});
	});

	await openRefractionQcTab(page);
	await page.getByTestId('refraction-qc-job-id').fill('refraction-job-2');
	await page.getByTestId('refraction-qc-load').click();
	await expect(page.getByTestId('refraction-qc-status')).toContainText('Loaded refraction-job-2');

	await page.getByTestId('refraction-qc-view-first-break-button').click();
	await expect(page.getByTestId('refraction-qc-view-first-break')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText('refraction_first_break_fit_qc.csv');
	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText('Downsampling: 3 of 5; downsampled');
	await expect(page.getByTestId('refraction-qc-first-break-time-plot')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-first-break-residual-plot')).toBeVisible();

	await page.getByTestId('refraction-qc-view-reduced-time-button').click();
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('refraction_reduced_time_qc.csv');

	await page.getByTestId('refraction-qc-view-gather-button').click();
	await expect(page.getByTestId('refraction-qc-view-gather')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-gather-controls')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-gather')).toContainText('Load a bounded gather preview from the M6 API');
	expect(await page.evaluate(() => (window as any).refractionQcState.selectedView)).toBe('gather_preview');
});

test('refraction gather preview UI raw only fetches bounded API data', async ({ page }) => {
	let requestPayload: Record<string, unknown> | null = null;
	await page.route('**/statics/refraction/qc/gather-preview', async (route) => {
		requestPayload = JSON.parse(route.request().postData() || '{}');
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(gatherPreviewPayload('refraction-gather-raw')),
		});
	});
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-gather-raw')),
		});
	});

	await openGatherPreview(page, 'refraction-gather-raw');
	await page.getByTestId('refraction-qc-gather-display').selectOption('raw');
	await page.getByTestId('refraction-qc-gather-time-start').fill('0');
	await page.getByTestId('refraction-qc-gather-time-end').fill('0.3');
	await page.getByTestId('refraction-qc-gather-max-traces').fill('2');
	await page.getByTestId('refraction-qc-gather-load').click();

	await expect(page.getByTestId('refraction-qc-gather-raw-plot')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-gather-corrected-plot')).toHaveCount(0);
	expect(requestPayload).toMatchObject({
		job_id: 'refraction-gather-raw',
		file_id: 'raw-preview-file',
		gather_axis: 'source',
		endpoint_key: 'S001',
		time_start_s: 0,
		time_end_s: 0.3,
		max_traces: 2,
	});
});

test('refraction gather preview UI requests midpoint CMP window through bounded API data', async ({ page }) => {
	let requestPayload: Record<string, unknown> | null = null;
	await page.route('**/statics/refraction/qc/gather-preview', async (route) => {
		requestPayload = JSON.parse(route.request().postData() || '{}');
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(gatherPreviewPayload('refraction-gather-cmp-window')),
		});
	});
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-gather-cmp-window')),
		});
	});

	await openGatherPreview(page, 'refraction-gather-cmp-window');
	await page.getByTestId('refraction-qc-gather-axis').selectOption({ label: 'Midpoint/CMP window' });
	await page.getByTestId('refraction-qc-gather-key1').fill('100');
	await page.getByTestId('refraction-qc-gather-x0').fill('4');
	await page.getByTestId('refraction-qc-gather-x1').fill('12');
	await page.getByTestId('refraction-qc-gather-time-start').fill('0');
	await page.getByTestId('refraction-qc-gather-time-end').fill('0.3');
	await page.getByTestId('refraction-qc-gather-load').click();

	expect(requestPayload).toMatchObject({
		job_id: 'refraction-gather-cmp-window',
		file_id: 'raw-preview-file',
		gather_axis: 'section',
		key1: 100,
		x0: 4,
		x1: 12,
		time_start_s: 0,
		time_end_s: 0.3,
	});
});

test('refraction gather preview UI raw corrected side by side renders overlays', async ({ page }) => {
	await page.route('**/statics/refraction/qc/gather-preview', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(gatherPreviewPayload('refraction-gather-side-by-side')),
		});
	});
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-gather-side-by-side')),
		});
	});

	await openGatherPreview(page, 'refraction-gather-side-by-side');
	await page.getByTestId('refraction-qc-gather-display').selectOption('side_by_side');
	await page.getByTestId('refraction-qc-gather-load').click();

	await expect(page.getByTestId('refraction-qc-gather-raw-plot')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-gather-corrected-plot')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-gather-corrected-status')).toContainText('corrected_tracestore');
	await expect.poll(async () => gatherPlotSummary(page, 'refraction-qc-gather-raw-plot')).toMatchObject({
		xAxisTitle: 'Offset (m)',
		yAxisTitle: 'Time (s)',
		traces: [
			expect.objectContaining({ name: 'Raw gather', type: 'heatmap' }),
			expect.objectContaining({ name: 'Observed first break', x: [100, 200], y: [0.1, 0.2] }),
			expect.objectContaining({ name: 'Modeled first break', x: [100, 200], y: [0.11, 0.19] }),
		],
	});
	await expect.poll(async () => gatherPlotSummary(page, 'refraction-qc-gather-corrected-plot')).toMatchObject({
		traces: [
			expect.objectContaining({ name: 'Corrected gather', type: 'heatmap' }),
			expect.objectContaining({ name: 'Corrected observed first break', x: [100, 200], y: [0.12, 0.19] }),
			expect.objectContaining({ name: 'Corrected modeled first break', x: [100, 200], y: [0.13, 0.18] }),
		],
	});
});

test('refraction gather preview UI missing corrected status is clear', async ({ page }) => {
	await page.route('**/statics/refraction/qc/gather-preview', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(gatherPreviewPayload('refraction-gather-missing-corrected', {
				correctedStatus: 'not_registered',
			})),
		});
	});
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-gather-missing-corrected')),
		});
	});

	await openGatherPreview(page, 'refraction-gather-missing-corrected');
	await page.getByTestId('refraction-qc-gather-load').click();

	await expect(page.getByTestId('refraction-qc-gather-corrected-status')).toContainText('Missing corrected data');
	await expect(page.getByTestId('refraction-qc-gather-corrected-status')).toContainText('not_registered');
	await expect(page.getByTestId('refraction-qc-gather-corrected-status')).toContainText('Registered corrected TraceStore is not available');
});

test('refraction gather preview UI validates time range and max traces', async ({ page }) => {
	let gatherRequests = 0;
	await page.route('**/statics/refraction/qc/gather-preview', async (route) => {
		gatherRequests += 1;
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(gatherPreviewPayload('refraction-gather-validation')),
		});
	});
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-gather-validation')),
		});
	});

	await openGatherPreview(page, 'refraction-gather-validation');
	await page.getByTestId('refraction-qc-gather-time-start').fill('0.4');
	await page.getByTestId('refraction-qc-gather-time-end').fill('0.2');
	await page.getByTestId('refraction-qc-gather-max-traces').fill('0');
	await page.getByTestId('refraction-qc-gather-load').click();

	await expect(page.getByTestId('refraction-qc-gather-error')).toContainText('Time end must be greater than time start');
	await expect(page.getByTestId('refraction-qc-gather-error')).toContainText('Max traces must be a positive integer');
	expect(gatherRequests).toBe(0);

	await page.getByTestId('refraction-qc-gather-axis').selectOption({ label: 'Midpoint/CMP window' });
	await page.getByTestId('refraction-qc-gather-key1-byte').fill('189.5');
	await page.getByTestId('refraction-qc-gather-key1').fill('100');
	await page.getByTestId('refraction-qc-gather-x0').fill('0.5');
	await page.getByTestId('refraction-qc-gather-x1').fill('4');
	await page.getByTestId('refraction-qc-gather-time-start').fill('0');
	await page.getByTestId('refraction-qc-gather-time-end').fill('0.2');
	await page.getByTestId('refraction-qc-gather-max-traces').fill('1.5');
	await page.getByTestId('refraction-qc-gather-load').click();

	await expect(page.getByTestId('refraction-qc-gather-error')).toContainText('key1 byte must be a positive integer');
	await expect(page.getByTestId('refraction-qc-gather-error')).toContainText('Trace start must be an integer');
	await expect(page.getByTestId('refraction-qc-gather-error')).toContainText('Max traces must be a positive integer');
	expect(gatherRequests).toBe(0);
});

test('3D cell map renders velocity quantity', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(grid3dQcBundlePayload('refraction-job-cell-1')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-cell-1');
	await page.getByTestId('refraction-qc-view-cells-button').click();
	await expect(page.getByTestId('refraction-qc-cell-map-plot')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-cells')).toContainText('refraction_grid_map_qc.csv');
	await expect(page.getByTestId('refraction-qc-view-cells')).toContainText('low_fold: 1');
	await expect(page.getByTestId('refraction-qc-view-cells')).toContainText('inactive: 1');

	await expect.poll(async () => cellMapPlotSummary(page)).toMatchObject({
		title: 'V2/T1 Velocity map',
		xAxisTitle: 'Cell center X (m)',
		yAxisTitle: 'Cell center Y (m)',
		colorbarTitle: 'Velocity (m/s)',
		z: [
			[2400, 2500],
			[null, 2600],
		],
		traceNames: ['Velocity', 'Flagged cells'],
	});
	const summary = await cellMapPlotSummary(page);
	expect(summary.text.flat().some((text) => text.includes('Status: low_fold'))).toBe(true);
	expect(summary.text.flat().some((text) => text.includes('Status reason: no_observations'))).toBe(true);
});

test('3D cell map quantity selector', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(grid3dQcBundlePayload('refraction-job-cell-2')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-cell-2');
	await page.getByTestId('refraction-qc-view-cells-button').click();
	await page.getByTestId('refraction-qc-map-quantity').selectOption('residual_rms');

	await expect(page.getByTestId('refraction-qc-view-cells')).toContainText('Residual RMS');
	await expect.poll(async () => cellMapPlotSummary(page)).toMatchObject({
		title: 'V2/T1 Residual RMS map',
		colorbarTitle: 'Residual RMS (ms)',
		z: [
			[4.5, 9],
			[null, 2.5],
		],
	});
});

test('3D cell map layer selector', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(grid3dQcBundlePayload('refraction-job-cell-3')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-cell-3');
	await page.getByTestId('refraction-qc-view-cells-button').click();
	await page.getByTestId('refraction-qc-layer-kind').selectOption('v3_t2');

	await expect(page.getByTestId('refraction-qc-view-cells')).toContainText('Plotted layer');
	await expect(page.getByTestId('refraction-qc-view-cells')).toContainText('V3/T2');
	await expect.poll(async () => cellMapPlotSummary(page)).toMatchObject({
		title: 'V3/T2 Velocity map',
		z: [[3600, 3700]],
		traceNames: ['Velocity'],
	});
});

test('3D cell map click selects cell', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(grid3dQcBundlePayload('refraction-job-cell-4')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-cell-4');
	await page.getByTestId('refraction-qc-view-cells-button').click();
	await expect.poll(async () => cellMapPlotSummary(page)).toMatchObject({
		z: [
			[2400, 2500],
			[null, 2600],
		],
	});

	await emitCellMapClick(page, 1, 0);

	await expect(page.getByTestId('refraction-qc-cell')).toHaveValue('1,0');
	await expect.poll(async () => page.evaluate(() => (window as any).refractionQcState.selectedCell)).toEqual({
		cell_ix: 1,
		cell_iy: 0,
		layer_kind: 'v2_t1',
	});
});

test('3D cell map unavailable for global velocity', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(globalVelocityQcBundlePayload('refraction-job-global')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-global');
	await page.getByTestId('refraction-qc-view-cells-button').click();

	await expect(page.getByTestId('refraction-qc-view-cells')).toContainText(
		'3D cell maps are unavailable for global-velocity jobs',
	);
	await expect(page.getByTestId('refraction-qc-cell-map-plot')).toHaveCount(0);
});

test('first-break QC plot renders observed and modeled series', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-3')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-3');
	await page.getByTestId('refraction-qc-view-first-break-button').click();
	await expect(page.getByTestId('refraction-qc-first-break-time-plot')).toBeVisible();

	await expect.poll(async () => page.getByTestId('refraction-qc-first-break-time-plot').evaluate((node) => {
		const plot = node as HTMLElement & { data?: Array<{ name?: string }> };
		return plot.data?.map((trace) => trace.name) ?? [];
	})).toEqual(['Observed', 'Modeled']);
});

test('first-break QC residual plot labels ms and uses observed minus modeled', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-4')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-4');
	await page.getByTestId('refraction-qc-view-first-break-button').click();
	await expect(page.getByTestId('refraction-qc-first-break-residual-note')).toContainText('Residual = observed - modeled, shown in ms.');

	await expect.poll(async () => page.getByTestId('refraction-qc-first-break-residual-plot').evaluate((node) => {
		const plot = node as HTMLElement & {
			data?: Array<{ y?: number[] }>;
			layout?: { yaxis?: { title?: { text?: string } } };
		};
		const values = (plot.data ?? []).flatMap((trace) => Array.isArray(trace.y) ? trace.y : []);
		return {
			axisTitle: plot.layout?.yaxis?.title?.text ?? '',
			values: values.map((value) => Math.round(value * 1000) / 1000).sort((a, b) => a - b),
		};
	})).toEqual({
		axisTitle: 'Residual (ms)',
		values: [-2, 5, 10],
	});
});

test('first-break QC layer filter limits plotted layer_kind records', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-5')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-5');
	await page.getByTestId('refraction-qc-view-first-break-button').click();
	await page.getByTestId('refraction-qc-layer-kind').selectOption('v3_t2');

	await expect.poll(async () => residualPlotPointCount(page)).toBe(1);
	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText('Layer filter');
	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText('V3/T2');
});

test('first-break QC rejected picks can be hidden and shown', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-6')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-6');
	await page.getByTestId('refraction-qc-view-first-break-button').click();
	await expect.poll(async () => residualPlotPointCount(page)).toBe(3);

	await page.getByTestId('refraction-qc-show-rejected').uncheck();
	await expect.poll(async () => residualPlotPointCount(page)).toBe(2);
	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText('Rejected picks');
	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText('hidden');

	await page.getByTestId('refraction-qc-show-rejected').check();
	await expect.poll(async () => residualPlotPointCount(page)).toBe(3);
});

test('reduced-time plot renders values with ms axis and gate overlays', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-7')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-7');
	await page.getByTestId('refraction-qc-view-reduced-time-button').click();
	await expect(page.getByTestId('refraction-qc-reduced-time-formula-note')).toContainText(
		'Reduced time = observed first-break time - offset / reduction velocity, shown in ms.',
	);
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('V2/T1');
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('V1 direct');
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('20.0-140.0 m');
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('0.0-1800.0 m');
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('>= 3200.0 m');

	await expect.poll(async () => page.getByTestId('refraction-qc-reduced-time-plot').evaluate((node) => {
		const plot = node as HTMLElement & {
			data?: Array<{ y?: number[] }>;
			layout?: {
				yaxis?: { title?: { text?: string } };
				shapes?: unknown[];
			};
		};
		const values = (plot.data ?? []).flatMap((trace) => Array.isArray(trace.y) ? trace.y : []);
		return {
			axisTitle: plot.layout?.yaxis?.title?.text ?? '',
			values: values.map((value) => Math.round(value * 1000) / 1000).sort((a, b) => a - b),
			shapeCount: plot.layout?.shapes?.length ?? 0,
		};
	})).toEqual({
		axisTitle: 'Reduced time (ms)',
		values: [100, 250, 380],
		shapeCount: 4,
	});
});

test('reduced-time plot displays reduction velocity', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-8')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-8');
	await page.getByTestId('refraction-qc-view-reduced-time-button').click();
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('Reduction velocity');
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('2000.00-5000.00 m/s');

	await expect.poll(async () => page.getByTestId('refraction-qc-reduced-time-plot').evaluate((node) => {
		const plot = node as HTMLElement & { data?: Array<{ text?: string[] }> };
		return (plot.data ?? [])
			.flatMap((trace) => Array.isArray(trace.text) ? trace.text : [])
			.some((text) => text.includes('Reduction velocity: 2000.00 m/s'));
	})).toBe(true);
});

test('reduced-time plot layer gate filter limits plotted layer records', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-9')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-9');
	await page.getByTestId('refraction-qc-view-reduced-time-button').click();
	await page.getByTestId('refraction-qc-layer-kind').selectOption('v3_t2');

	await expect.poll(async () => reducedTimePlotPointCount(page)).toBe(1);
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('Layer filter');
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('V3/T2');
});

test('reduced-time plot handles missing velocity status', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-10')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-10');
	await page.getByTestId('refraction-qc-view-reduced-time-button').click();

	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('Unavailable rows');
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('missing_reduction_velocity: 1');
	await expect.poll(async () => reducedTimePlotPointCount(page)).toBe(3);
});

test('reduced-time plot hides rows not used for inversion', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-10b')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-10b');
	await page.getByTestId('refraction-qc-view-reduced-time-button').click();
	await expect.poll(async () => reducedTimePlotPointCount(page)).toBe(3);

	await page.getByTestId('refraction-qc-show-rejected').uncheck();
	await expect.poll(async () => reducedTimePlotPointCount(page)).toBe(2);
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('Unused picks');
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('hidden');
});

test('reduced-time gate overlays read documented layer arrays', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		const payload = qcBundlePayload('refraction-job-10c') as any;
		payload.summary.layers = [
			{ kind: 'v2_t1', enabled: true, min_offset_m: 0, max_offset_m: 1800 },
			{ kind: 'v3_t2', enabled: true, min_offset_m: 1800, max_offset_m: 3200 },
			{ kind: 'vsub_t3', enabled: true, min_offset_m: 3200, max_offset_m: null },
		];
		delete payload.summary.observation_gates;
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(payload),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-10c');
	await page.getByTestId('refraction-qc-view-reduced-time-button').click();

	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('V2/T1');
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('V3/T2');
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('Vsub/T3');
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('0.0-1800.0 m');
	await expect(page.getByTestId('refraction-qc-reduced-time-gates')).toContainText('>= 3200.0 m');
});

test('2D profile plot renders time terms', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-11')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-11');
	await page.getByTestId('refraction-qc-view-profiles-button').click();
	await expect(page.getByTestId('refraction-qc-profile-plot')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-profiles')).toContainText('refraction_line_profile_qc_combined.csv');
	await expect(page.getByTestId('refraction-qc-view-profiles')).toContainText('Unavailable fields');
	await expect(page.getByTestId('refraction-qc-view-profiles')).toContainText('T3');

	await expect.poll(async () => profilePlotSummary(page)).toMatchObject({
		axisTitle: 'Time term (ms)',
		traces: [
			{ name: 'T1 (ms) source', pointCount: 2, values: [14, 18] },
			{ name: 'T2 (ms) source', pointCount: 2, values: [28, 35] },
		],
	});
});

test('2D profiles tab shows unavailable line-profile reason', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		const payload = qcBundlePayload('refraction-job-unavailable-profiles') as any;
		payload.available_views = payload.available_views.filter((view: string) => view !== 'line_profiles');
		payload.unavailable_views = Array.from(new Set([...payload.unavailable_views, 'profiles']));
		payload.unavailable_view_reasons = {
			profiles: 'no_projected_inline_coordinate_model',
		};
		payload.artifacts.refraction_line_profile_qc_json = 'refraction_line_profile_qc.json';
		delete payload.views.line_profiles;
		delete payload.downsampling.line_profiles;
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(payload),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-unavailable-profiles');
	await page.getByTestId('refraction-qc-view-profiles-button').click();

	await expect(page.getByTestId('refraction-qc-view-profiles')).toContainText(
		'This view is unavailable from refraction_line_profile_qc_* artifacts',
	);
	await expect(page.getByTestId('refraction-qc-view-profiles')).toContainText(
		'no_projected_inline_coordinate_model',
	);
	await expect(page.getByTestId('refraction-qc-view-profiles')).not.toContainText(
		'No sampled line-profile records are present',
	);
	await expect(page.getByTestId('refraction-qc-profile-plot')).toHaveCount(0);
});

test('2D profile plot renders static components with units', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-12')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-12');
	await page.getByTestId('refraction-qc-view-profiles-button').click();
	await page.getByTestId('refraction-qc-profile-group').selectOption('statics');
	await page.getByTestId('refraction-qc-profile-units').selectOption('s');
	await expect(page.getByTestId('refraction-qc-profile-sign-note')).toContainText('corrected(t) = raw(t - shift_s)');
	await expect(page.getByTestId('refraction-qc-view-profiles')).toContainText('positive shift_s delays displayed events');

	await expect.poll(async () => profilePlotSummary(page)).toMatchObject({
		axisTitle: 'Static shift (s)',
		traces: expect.arrayContaining([
			expect.objectContaining({
				name: 'Weathering correction (s) source',
				values: [-0.008, -0.009],
			}),
			expect.objectContaining({
				name: 'Final applied static (s) source',
				values: [-0.004, -0.005],
			}),
		]),
	});
});

test('static component view renders waterfall values', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-15')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-15');
	await page.getByTestId('refraction-qc-view-statics-button').click();

	await expect(page.getByTestId('refraction-qc-view-statics')).toContainText('Endpoint static components');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('Weathering correction');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('-8.000 ms');
	await expect(page.getByTestId('refraction-qc-static-trace-component-list')).toContainText('Final applied trace shift');
	await expect(page.getByTestId('refraction-qc-static-trace-component-list')).toContainText('-1.500 ms');

	await expect.poll(async () => staticComponentPlotSummary(page)).toMatchObject({
		name: 'Endpoint static components',
		axisTitle: 'Shift (ms)',
		components: expect.arrayContaining([
			expect.objectContaining({ label: 'Weathering correction', value: -8 }),
			expect.objectContaining({ label: 'Computed field correction', value: 4.5 }),
			expect.objectContaining({ label: 'Applied field correction', value: 4.5 }),
			expect.objectContaining({ label: 'Final endpoint shift', value: -4 }),
		]),
	});
});

test('static component view displays sign convention', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-16')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-16');
	await page.getByTestId('refraction-qc-view-statics-button').click();

	await expect(page.getByTestId('refraction-qc-static-sign-note')).toContainText('corrected(t) = raw(t - shift_s)');
	await expect(page.getByTestId('refraction-qc-static-sign-note')).toContainText('positive shift_s delays displayed events');
	await expect(page.getByTestId('refraction-qc-static-sign-note')).toContainText('negative shift_s advances displayed events');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('advances displayed events');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('delays displayed events');
});

test('static component view shows field and final shifts when apply is false', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		const payload = qcBundlePayload('refraction-job-17') as any;
		for (const record of payload.views.static_component_qc_endpoint.records) {
			record.apply_to_trace_shift = 'false';
			record.applied_field_correction_ms = '0';
		}
		for (const record of payload.views.static_component_qc_trace.records) {
			record.apply_to_trace_shift = 'false';
			record.applied_field_shift_ms = '0';
			record.final_trace_shift_ms = record.refraction_shift_ms;
			record.applied_trace_shift_ms = record.refraction_shift_ms;
		}
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(payload),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-17');
	await page.getByTestId('refraction-qc-view-statics-button').click();

	await expect(page.getByTestId('refraction-qc-view-statics')).toContainText('Apply field shift');
	await expect(page.getByTestId('refraction-qc-view-statics')).toContainText('false');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('Computed field correction');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('4.500 ms');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('Applied field correction');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('0.000 ms');
	await expect(page.getByTestId('refraction-qc-static-trace-component-list')).toContainText('Final applied trace shift');
	await expect(page.getByTestId('refraction-qc-static-trace-component-list')).toContainText('-8.000 ms');
});

test('static component view endpoint selection updates details and statuses', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-18')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-18');
	await page.getByTestId('refraction-qc-view-statics-button').click();
	await page.getByTestId('refraction-qc-endpoint').fill('S002');

	await expect(page.getByTestId('refraction-qc-view-statics')).toContainText('S002');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('invalid_weathering');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('missing');
	await expect.poll(async () => staticComponentPlotSummary(page)).toMatchObject({
		components: expect.arrayContaining([
			expect.objectContaining({
				label: 'Computed field correction',
				value: 6.5,
			}),
		]),
	});
	await expect.poll(async () => staticComponentPlotSummary(page, 'refraction-qc-static-trace-waterfall')).toMatchObject({
		components: expect.arrayContaining([
			expect.objectContaining({
				label: 'Computed field shift',
				value: 6.5,
				text: expect.stringContaining('ok'),
			}),
		]),
	});

	await page.getByTestId('refraction-qc-endpoint-kind').selectOption('receiver');
	await page.getByTestId('refraction-qc-endpoint').fill('R001');
	await expect(page.getByTestId('refraction-qc-view-statics')).toContainText('R001');
	await expect(page.getByTestId('refraction-qc-static-component-list')).toContainText('not_applicable');
	await expect.poll(async () => staticComponentPlotSummary(page)).toMatchObject({
		components: expect.arrayContaining([
			expect.objectContaining({ label: 'Weathering correction', value: -7 }),
			expect.objectContaining({ label: 'Computed field correction', value: 2 }),
		]),
	});
});

test('static component view shows no match for unknown endpoint filter', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-18b')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-18b');
	await page.getByTestId('refraction-qc-view-statics-button').click();
	await page.getByTestId('refraction-qc-endpoint').fill('S999');

	await expect(page.getByTestId('refraction-qc-view-statics')).toContainText(
		'No source/receiver endpoint component rows match the current endpoint selector.',
	);
	await expect(page.getByTestId('refraction-qc-view-statics')).toContainText(
		'No trace component row matches the current trace or endpoint selector.',
	);
	await expect(page.getByTestId('refraction-qc-view-statics')).not.toContainText('Selected endpoint S001');
});

test('2D profile plot endpoint kind filter', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(qcBundlePayload('refraction-job-13')),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-13');
	await page.getByTestId('refraction-qc-view-profiles-button').click();
	await page.getByTestId('refraction-qc-endpoint-kind').selectOption('both');

	await expect.poll(async () => profilePlotSummary(page)).toMatchObject({
		traces: expect.arrayContaining([
			expect.objectContaining({
				name: 'T1 (ms) source',
				dash: 'solid',
				symbols: ['circle', 'x'],
			}),
			expect.objectContaining({
				name: 'T1 (ms) receiver',
				dash: 'dot',
				symbols: ['diamond', 'diamond'],
			}),
		]),
	});

	await page.getByTestId('refraction-qc-status-filter').selectOption('invalid');
	await expect.poll(async () => profilePlotSummary(page)).toMatchObject({
		traces: [
			{ name: 'T1 (ms) source', pointCount: 1, symbols: ['x'] },
			{ name: 'T2 (ms) source', pointCount: 1, symbols: ['x'] },
		],
	});
});

test('2D profile plot handles one-layer missing T2 T3', async ({ page }) => {
	await page.route('**/statics/refraction/qc', async (route) => {
		const payload = qcBundlePayload('refraction-job-14');
		const records = oneLayerLineProfileRecords();
		payload.summary.conversion_mode = 't1lsst_1layer';
		payload.summary.layer_count = 1;
		payload.views.line_profiles.records = records;
		payload.views.line_profiles.columns = Object.keys(records[0]);
		payload.views.line_profiles.total_points = records.length;
		payload.views.line_profiles.returned_points = records.length;
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify(payload),
		});
	});

	await loadRefractionQcBundle(page, 'refraction-job-14');
	await page.getByTestId('refraction-qc-view-profiles-button').click();
	await expect(page.getByTestId('refraction-qc-view-profiles')).toContainText('Unavailable fields');
	await expect(page.getByTestId('refraction-qc-view-profiles')).toContainText('T2, T3');

	await expect.poll(async () => profilePlotSummary(page)).toMatchObject({
		axisTitle: 'Time term (ms)',
		traces: [{ name: 'T1 (ms) source', pointCount: 2, values: [14, 18] }],
	});
});
