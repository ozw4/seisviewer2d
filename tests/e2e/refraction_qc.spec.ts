import { expect, test, type Page } from '@playwright/test';

function qcBundlePayload(jobId: string) {
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
		},
		artifacts: {
			first_break_residuals: 'first_break_residuals.csv',
			refraction_first_break_fit_qc_csv: 'refraction_first_break_fit_qc.csv',
			refraction_reduced_time_qc: 'refraction_reduced_time_qc.csv',
			near_surface_model: 'near_surface_model.csv',
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
				columns: ['trace', 'reduced_time_ms'],
				total_points: 1,
				returned_points: 1,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: [{ trace: '0', reduced_time_ms: '12.5' }],
			},
			line_profiles: {
				artifact: 'near_surface_model.csv',
				columns: ['endpoint_key', 'station_x_m'],
				total_points: 1,
				returned_points: 1,
				downsampled: false,
				downsampling_method: 'even_index_floor_first_last',
				records: [{ endpoint_key: 'S001', station_x_m: '1000.0' }],
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
	await expect(page.getByTestId('refraction-qc-show-rejected')).toBeChecked();
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
