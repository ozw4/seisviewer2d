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
			sh1_m: '9.5',
			sh2_m: '16.2',
			sh3_m: '',
			layer1_base_elevation_m: '110.5',
			layer2_base_elevation_m: '94.3',
			final_refractor_elevation_m: '94.3',
			weathering_correction_ms: '-8',
			elevation_correction_ms: '2',
			field_correction_ms: '1.5',
			manual_static_ms: '0.5',
			total_static_ms: '-4',
			total_applied_shift_ms: '-4',
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
			sh1_m: '12.0',
			sh2_m: '18.0',
			sh3_m: '',
			layer1_base_elevation_m: '112.0',
			layer2_base_elevation_m: '94.0',
			final_refractor_elevation_m: '94.0',
			weathering_correction_ms: '-9',
			elevation_correction_ms: '2.5',
			field_correction_ms: '0',
			manual_static_ms: '1',
			total_static_ms: '-5.5',
			total_applied_shift_ms: '-5.5',
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
			sh1_m: '8.2',
			sh2_m: '15.0',
			sh3_m: '',
			layer1_base_elevation_m: '109.8',
			layer2_base_elevation_m: '94.8',
			final_refractor_elevation_m: '94.8',
			weathering_correction_ms: '-7',
			elevation_correction_ms: '1',
			field_correction_ms: '2',
			manual_static_ms: '0',
			total_static_ms: '-4',
			total_applied_shift_ms: '-4',
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
			sh1_m: '10.1',
			sh2_m: '17.3',
			sh3_m: '',
			layer1_base_elevation_m: '110.9',
			layer2_base_elevation_m: '93.6',
			final_refractor_elevation_m: '93.6',
			weathering_correction_ms: '-8.5',
			elevation_correction_ms: '1.5',
			field_correction_ms: '2.5',
			manual_static_ms: '0',
			total_static_ms: '-4.5',
			total_applied_shift_ms: '-4.5',
			static_status: 'ok',
			solution_status: 'ok',
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
		sh2_m: '',
		sh3_m: '',
		layer2_base_elevation_m: '',
	}));
}

function qcBundlePayload(jobId: string) {
	const profileRecords = lineProfileRecords();
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
			observation_gates: {
				v1_direct_arrival: {
					enabled: true,
					min_direct_offset_m: 20,
					max_direct_offset_m: 140,
				},
				v2_t1: {
					enabled: true,
					min_offset_m: 0,
					max_offset_m: 1800,
				},
				v3_t2: {
					enabled: true,
					min_offset_m: 1800,
					max_offset_m: 3200,
				},
				vsub_t3: {
					enabled: true,
					min_offset_m: 3200,
					max_offset_m: null,
				},
			},
		},
		artifacts: {
			first_break_residuals: 'first_break_residuals.csv',
			refraction_first_break_fit_qc_csv: 'refraction_first_break_fit_qc.csv',
			refraction_reduced_time_qc: 'refraction_reduced_time_qc.csv',
			near_surface_model: 'near_surface_model.csv',
			refraction_line_profile_qc_combined: 'refraction_line_profile_qc_combined.csv',
			refraction_refractor_velocity_cells: 'refraction_refractor_velocity_cells.csv',
			refraction_static_components: 'refraction_static_components.csv',
		},
		available_views: [
			'summary',
			'first_break_fit',
			'first_break_residual',
			'reduced_time',
			'line_profiles',
			'refractor_cells',
			'static_components',
		],
		unavailable_views: ['gather_preview'],
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
				columns: ['endpoint_key', 'total_applied_shift_ms'],
				total_points: 1,
				returned_points: 1,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: [{ endpoint_key: 'S001', total_applied_shift_ms: '-8.0' }],
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

async function loadRefractionQcBundle(page: Page, jobId: string) {
	await openRefractionQcTab(page);
	await page.getByTestId('refraction-qc-job-id').fill(jobId);
	await page.getByTestId('refraction-qc-load').click();
	await expect(page.getByTestId('refraction-qc-status')).toContainText(`Loaded ${jobId}`);
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
	await expect(page.getByTestId('refraction-qc-view-gather')).toContainText('Gather preview is not included');
	expect(await page.evaluate(() => (window as any).refractionQcState.selectedView)).toBe('gather_preview');
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
