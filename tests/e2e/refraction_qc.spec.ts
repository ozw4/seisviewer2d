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
			refraction_reduced_time_qc: 'refraction_reduced_time_qc.csv',
			near_surface_model: 'near_surface_model.csv',
			refraction_refractor_velocity_cells: 'refraction_refractor_velocity_cells.csv',
			refraction_static_components: 'refraction_static_components.csv',
		},
		available_views: [
			'summary',
			'first_break_residual',
			'reduced_time',
			'line_profiles',
			'refractor_cells',
			'static_components',
		],
		unavailable_views: ['gather_preview'],
		views: {
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
		downsampling: {},
	};
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
	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText('first_break_residuals.csv');
	await expect(page.getByTestId('refraction-qc-view-first-break')).toContainText('1.25');

	await page.getByTestId('refraction-qc-view-reduced-time-button').click();
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-reduced-time')).toContainText('refraction_reduced_time_qc.csv');

	await page.getByTestId('refraction-qc-view-gather-button').click();
	await expect(page.getByTestId('refraction-qc-view-gather')).toBeVisible();
	await expect(page.getByTestId('refraction-qc-view-gather')).toContainText('Gather preview is not included');
	expect(await page.evaluate(() => (window as any).refractionQcState.selectedView)).toBe('gather_preview');
});
