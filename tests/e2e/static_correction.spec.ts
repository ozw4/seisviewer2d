import { expect, test, type Route } from '@playwright/test';

declare global {
	interface Window {
		SeisViewerState?: {
			syncActiveFileTarget?: (state: {
				fileId: string;
				displayName: string;
				key1Byte: number;
				key2Byte: number;
				isFileLoaded: boolean;
			}) => unknown;
		};
	}
}

async function fulfillJson(route: Route, payload: unknown, status = 200) {
	await route.fulfill({
		status,
		contentType: 'application/json',
		body: JSON.stringify(payload),
	});
}

test('direct NPZ Static Correction posts multipart request and auto-loads Refraction QC', async ({
	page,
}) => {
	const calls: string[] = [];
	let multipartBody = '';
	let qcRequestBody = '';

	await page.route('**/statics/**', async (route) => {
		const request = route.request();
		const url = new URL(request.url());
		calls.push(`${request.method()} ${url.pathname}`);

		if (request.method() === 'POST' && url.pathname === '/statics/refraction/apply-with-picks') {
			multipartBody = request.postData() || '';
			await fulfillJson(route, { job_id: 'static-job-e2e', state: 'queued' });
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/static-job-e2e/status') {
			await fulfillJson(route, {
				state: 'ready',
				message: 'finished',
				progress: 1,
			});
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/static-job-e2e/files') {
			await fulfillJson(route, {
				files: [
					{ name: 'refraction_static_qc.json', size_bytes: 512 },
					{ name: 'refraction_static_artifacts.json', size_bytes: 256 },
				],
			});
			return;
		}
		if (request.method() === 'POST' && url.pathname === '/statics/refraction/qc') {
			qcRequestBody = request.postData() || '';
			await fulfillJson(route, {
				job_id: 'static-job-e2e',
				summary: { status: 'ready', workflow: 'refraction' },
				available_views: ['summary'],
				unavailable_views: [],
				coordinate_mode: 'auto',
			});
			return;
		}

		throw new Error(`Unexpected Static Correction request: ${request.method()} ${url.pathname}`);
	});

	await page.goto('/');
	await page.waitForFunction(() => Boolean(window.SeisViewerState?.syncActiveFileTarget));
	await page.evaluate(() => {
		window.SeisViewerState?.syncActiveFileTarget?.({
			fileId: 'viewer-line',
			displayName: 'viewer-line.sgy',
			key1Byte: 9,
			key2Byte: 13,
			isFileLoaded: true,
		});
	});

	await page.getByTestId('static-correction-tab').click();
	await expect(page.locator('#staticCorrectionTargetFile')).toContainText('viewer-line');
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'uploaded-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.locator('#staticCorrectionRunButton').click();

	await expect(page.getByTestId('refraction-qc-tab')).toHaveAttribute('aria-selected', 'true');
	await expect(page.locator('#refractionQcJobId')).toHaveValue('static-job-e2e');
	await expect(page.locator('#refractionQcStatus')).toContainText('Loaded static-job-e2e');

	expect(calls).toEqual([
		'POST /statics/refraction/apply-with-picks',
		'GET /statics/job/static-job-e2e/status',
		'GET /statics/job/static-job-e2e/files',
		'POST /statics/refraction/qc',
	]);
	expect(multipartBody).toContain('name="request_json"');
	expect(multipartBody).toContain('"file_id":"viewer-line"');
	expect(multipartBody).toContain('"key1_byte":9');
	expect(multipartBody).toContain('"key2_byte":13');
	expect(multipartBody).toContain('"pick_source":{"kind":"uploaded_npz"}');
	expect(multipartBody).toContain('"linkage":{"mode":"none"}');
	expect(multipartBody).toContain('name="pick_npz"; filename="uploaded-picks.npz"');
	expect(qcRequestBody).toContain('"job_id":"static-job-e2e"');
});

test('direct NPZ Static Correction builds checked linkage before multipart apply', async ({
	page,
}) => {
	const calls: string[] = [];
	let linkageBuildBody = '';
	let multipartBody = '';
	let qcRequestBody = '';

	await page.route('**/statics/**', async (route) => {
		const request = route.request();
		const url = new URL(request.url());
		calls.push(`${request.method()} ${url.pathname}`);

		if (request.method() === 'POST' && url.pathname === '/statics/linkage/build') {
			linkageBuildBody = request.postData() || '';
			await fulfillJson(route, { job_id: 'linkage-job-e2e', state: 'queued' });
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/linkage-job-e2e/status') {
			await fulfillJson(route, {
				job_id: 'linkage-job-e2e',
				state: 'done',
				message: 'linked',
				progress: 1,
			});
			return;
		}
		if (request.method() === 'POST' && url.pathname === '/statics/refraction/apply-with-picks') {
			multipartBody = request.postData() || '';
			await fulfillJson(route, { job_id: 'static-job-linked-e2e', state: 'queued' });
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/static-job-linked-e2e/status') {
			await fulfillJson(route, {
				state: 'ready',
				message: 'finished',
				progress: 1,
			});
			return;
		}
		if (request.method() === 'GET' && url.pathname === '/statics/job/static-job-linked-e2e/files') {
			await fulfillJson(route, {
				files: [
					{ name: 'refraction_static_qc.json', size_bytes: 512 },
					{ name: 'refraction_static_artifacts.json', size_bytes: 256 },
				],
			});
			return;
		}
		if (request.method() === 'POST' && url.pathname === '/statics/refraction/qc') {
			qcRequestBody = request.postData() || '';
			await fulfillJson(route, {
				job_id: 'static-job-linked-e2e',
				summary: { status: 'ready', workflow: 'refraction' },
				available_views: ['summary'],
				unavailable_views: [],
				coordinate_mode: 'auto',
			});
			return;
		}

		throw new Error(`Unexpected Static Correction request: ${request.method()} ${url.pathname}`);
	});

	await page.goto('/');
	await page.waitForFunction(() => Boolean(window.SeisViewerState?.syncActiveFileTarget));
	await page.evaluate(() => {
		window.SeisViewerState?.syncActiveFileTarget?.({
			fileId: 'viewer-linked-line',
			displayName: 'viewer-linked-line.sgy',
			key1Byte: 17,
			key2Byte: 21,
			isFileLoaded: true,
		});
	});

	await page.getByTestId('static-correction-tab').click();
	await page.locator('#staticCorrectionPickNpz').setInputFiles({
		name: 'uploaded-linked-picks.npz',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('npz-bytes'),
	});
	await page.getByTestId('static-correction-enable-linkage').check();
	await page.getByTestId('static-correction-linkage-threshold-m').fill('12.5');
	await page.locator('#staticCorrectionRunButton').click();

	await expect(page.getByTestId('refraction-qc-tab')).toHaveAttribute('aria-selected', 'true');
	await expect(page.locator('#refractionQcJobId')).toHaveValue('static-job-linked-e2e');
	await expect(page.locator('#refractionQcStatus')).toContainText('Loaded static-job-linked-e2e');

	expect(calls).toEqual([
		'POST /statics/linkage/build',
		'GET /statics/job/linkage-job-e2e/status',
		'POST /statics/refraction/apply-with-picks',
		'GET /statics/job/static-job-linked-e2e/status',
		'GET /statics/job/static-job-linked-e2e/files',
		'POST /statics/refraction/qc',
	]);
	expect(JSON.parse(linkageBuildBody)).toMatchObject({
		file_id: 'viewer-linked-line',
		key1_byte: 17,
		key2_byte: 21,
		linkage: { mode: 'auto_threshold', threshold_m: 12.5 },
	});
	expect(multipartBody).toContain('"file_id":"viewer-linked-line"');
	expect(multipartBody).toContain('"key1_byte":17');
	expect(multipartBody).toContain('"key2_byte":21');
	expect(multipartBody).toContain('"pick_source":{"kind":"uploaded_npz"}');
	expect(multipartBody).toContain(
		'"linkage":{"mode":"required","job_id":"linkage-job-e2e","artifact_name":"geometry_linkage.npz"}',
	);
	expect(multipartBody).toContain('name="pick_npz"; filename="uploaded-linked-picks.npz"');
	expect(qcRequestBody).toContain('"job_id":"static-job-linked-e2e"');
});
