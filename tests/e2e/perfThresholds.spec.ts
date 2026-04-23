import { expect, test } from '@playwright/test';
import type { SvPerfArtifactMetadata, SvPerfRow } from './helpers/perfArtifacts';
import {
	buildPerfThresholdFailureMessage,
	evaluatePerfThresholds,
	type PerfThresholdArtifact,
} from './helpers/perfThresholds';
import type {
	PerfCaseLabel,
	PerfScenarioResult,
	WindowResponseMetadata,
} from './helpers/perfScenarios';

const PERF_ENV_NAMES = [
	'SV_PERF_CI',
	'SV_PERF_MAX_COLD_TOTAL_MS',
	'SV_PERF_MAX_WARM_TOTAL_MS',
	'SV_PERF_MAX_ZOOM_TOTAL_MS',
	'SV_PERF_MAX_PAN_TOTAL_MS',
	'SV_PERF_MAX_COLD_PLOTLY_MS',
	'SV_PERF_MAX_WARM_PLOTLY_MS',
	'SV_PERF_MAX_ZOOM_PLOTLY_MS',
	'SV_PERF_MAX_PAN_PLOTLY_MS',
	'SV_PERF_MAX_ANY_FETCH_MS',
	'SV_PERF_MAX_ANY_DECODE_MS',
	'SV_PERF_MAX_ANY_PLOTLY_MS',
	'SV_PERF_MAX_SERVER_MS',
	'SV_PERF_MAX_BUILD_MS',
	'SV_PERF_MAX_PACK_MS',
] as const;

const BASE_METADATA: SvPerfArtifactMetadata = {
	label: 'viewer-perf-thresholds',
	url: 'http://127.0.0.1:8000/?perf=1',
	browserName: 'chromium',
	viewport: { width: 1280, height: 720 },
	createdAt: '2026-04-23T00:00:00.000Z',
};

function setPerfEnv(name: (typeof PERF_ENV_NAMES)[number], value: string | null): void {
	if (value === null) {
		delete process.env[name];
		return;
	}
	process.env[name] = value;
}

function buildResponseHeaders(overrides: Partial<WindowResponseMetadata['headers']> = {}) {
	return {
		serverTiming: null,
		xSvCache: 'miss',
		xSvServerMs: null,
		xSvBuildMs: null,
		xSvPackMs: null,
		xSvBytes: '4096',
		...overrides,
	};
}

function buildResponse(
	overrides: Partial<WindowResponseMetadata> = {},
): WindowResponseMetadata {
	return {
		url: 'http://127.0.0.1:8000/get_section_window_bin',
		status: 200,
		contentLength: 4096,
		requestTiming: {
			startTime: 0,
			domainLookupStart: 0,
			domainLookupEnd: 0,
			connectStart: 0,
			connectEnd: 0,
			secureConnectionStart: 0,
			requestStart: 0,
			responseStart: 0,
			responseEnd: 0,
		},
		headers: buildResponseHeaders(),
		...overrides,
	};
}

function buildMeasuredScenario(
	label: PerfCaseLabel,
	windowOverrides: Partial<SvPerfRow> = {},
	responseHeaders: Partial<WindowResponseMetadata['headers']> = {},
): PerfScenarioResult {
	const latestWindowRow: SvPerfRow = {
		kind: 'window',
		fetch_ms: 1_200,
		decode_ms: 250,
		plotly_ms: 900,
		total_ms: 2_000,
		...windowOverrides,
	};

	return {
		label,
		status: 'measured',
		windowRowsBefore: 0,
		windowRowsAfter: 1,
		deltaRows: [latestWindowRow],
		deltaWindowRows: [latestWindowRow],
		latestWindowRow,
		responses: [buildResponse({ headers: buildResponseHeaders(responseHeaders) })],
	};
}

function buildSkippedScenario(label: PerfCaseLabel, skipReason = 'range unavailable'): PerfScenarioResult {
	return {
		label,
		status: 'skipped',
		skipReason,
		windowRowsBefore: 0,
		windowRowsAfter: 0,
		deltaRows: [],
		deltaWindowRows: [],
		latestWindowRow: null,
		responses: [],
	};
}

test.afterEach(() => {
	for (const name of PERF_ENV_NAMES) {
		delete process.env[name];
	}
});

test('evaluatePerfThresholds uses CI defaults and reports exceeded metrics', async () => {
	setPerfEnv('SV_PERF_CI', '1');

	const artifact = evaluatePerfThresholds(
		[
			buildMeasuredScenario('cold-initial', { total_ms: 2_200 }),
			buildMeasuredScenario('warm-initial', { total_ms: 3_100 }),
			buildMeasuredScenario('zoom-in', { plotly_ms: 5_200, total_ms: null, fetch_ms: null, decode_ms: null }),
			buildSkippedScenario('pan-after-zoom'),
		],
		BASE_METADATA,
	);

	expect(artifact.metadata.thresholdProfile).toBe('ci-default');
	expect(artifact.thresholds).toEqual(
		expect.arrayContaining([
			expect.objectContaining({
				caseLabel: 'cold-initial',
				metric: 'total_ms',
				max: 8_000,
				required: true,
			}),
			expect.objectContaining({
				caseLabel: 'any',
				metric: 'fetch_ms',
				max: 6_000,
				required: false,
			}),
		]),
	);
	expect(artifact.failures).toEqual([
		expect.objectContaining({
			caseLabel: 'zoom-in',
			metric: 'plotly_ms',
			actual: 5_200,
			max: 5_000,
			status: 'fail',
		}),
	]);
	expect(buildPerfThresholdFailureMessage(artifact.failures)).toBe(
		'Performance threshold failed: zoom-in plotly_ms actual=5200 max=5000',
	);
	expect(artifact.results).toEqual(
		expect.arrayContaining([
			expect.objectContaining({
				caseLabel: 'zoom-in',
				metric: 'fetch_ms',
				status: 'missing-allowed',
			}),
			expect.objectContaining({
				caseLabel: 'pan-after-zoom',
				metric: 'plotly_ms',
				status: 'skipped',
			}),
		]),
	);
});

test('evaluatePerfThresholds lets explicit env overrides replace CI defaults and require server headers', async () => {
	setPerfEnv('SV_PERF_CI', '1');
	setPerfEnv('SV_PERF_MAX_COLD_TOTAL_MS', '1200');
	setPerfEnv('SV_PERF_MAX_SERVER_MS', '50');

	const artifact = evaluatePerfThresholds(
		[
			buildMeasuredScenario('cold-initial', { total_ms: 1_500 }),
			buildMeasuredScenario('warm-initial', { total_ms: 2_500 }, { xSvServerMs: '45.5' }),
		],
		BASE_METADATA,
	);

	expect(artifact.metadata.thresholdProfile).toBe('ci-default+env-overrides');
	expect(artifact.thresholds).toEqual(
		expect.arrayContaining([
			expect.objectContaining({
				caseLabel: 'cold-initial',
				metric: 'total_ms',
				max: 1_200,
			}),
			expect.objectContaining({
				caseLabel: 'any',
				metric: 'server_ms',
				max: 50,
				required: true,
			}),
		]),
	);
	expect(artifact.failures).toEqual(
		expect.arrayContaining([
			expect.objectContaining({
				caseLabel: 'cold-initial',
				metric: 'total_ms',
				status: 'fail',
			}),
			expect.objectContaining({
				caseLabel: 'cold-initial',
				metric: 'server_ms',
				status: 'missing-required',
			}),
		]),
	);
});

test('evaluatePerfThresholds keeps explicit zoom total checks optional for cached paths', async () => {
	setPerfEnv('SV_PERF_MAX_ZOOM_TOTAL_MS', '1000');
	setPerfEnv('SV_PERF_MAX_ZOOM_PLOTLY_MS', '1000');

	const artifact = evaluatePerfThresholds(
		[
			buildMeasuredScenario('zoom-in', {
				total_ms: null,
				fetch_ms: null,
				decode_ms: null,
				plotly_ms: 400,
			}),
		],
		BASE_METADATA,
	);

	expect(artifact.metadata.thresholdProfile).toBe('env-only');
	expect(artifact.failures).toEqual([]);
	expect(artifact.results).toEqual(
		expect.arrayContaining([
			expect.objectContaining({
				caseLabel: 'zoom-in',
				metric: 'total_ms',
				status: 'missing-allowed',
			}),
			expect.objectContaining({
				caseLabel: 'zoom-in',
				metric: 'plotly_ms',
				status: 'pass',
			}),
		]),
	);
});

test('evaluatePerfThresholds uses the latest finite metric row inside a measured scenario', async () => {
	setPerfEnv('SV_PERF_CI', '1');

	const firstRow: SvPerfRow = {
		kind: 'window',
		total_ms: 1_800,
		plotly_ms: 700,
		fetch_ms: 1_000,
		decode_ms: 180,
	};
	const cachedRow: SvPerfRow = {
		kind: 'window',
		total_ms: null,
		plotly_ms: 350,
		fetch_ms: null,
		decode_ms: null,
	};

	const artifact = evaluatePerfThresholds(
		[
			{
				label: 'cold-initial',
				status: 'measured',
				windowRowsBefore: 0,
				windowRowsAfter: 2,
				deltaRows: [firstRow, cachedRow],
				deltaWindowRows: [firstRow, cachedRow],
				latestWindowRow: cachedRow,
				responses: [buildResponse()],
			},
		],
		BASE_METADATA,
	);

	expect(artifact.failures).toEqual([]);
	expect(artifact.results).toEqual(
		expect.arrayContaining([
			expect.objectContaining({
				caseLabel: 'cold-initial',
				metric: 'total_ms',
				actual: 1_800,
				status: 'pass',
			}),
		]),
	);
});

test('evaluatePerfThresholds still emits an artifact when thresholds are disabled', async () => {
	const artifact = evaluatePerfThresholds(
		[
			buildMeasuredScenario('cold-initial', { total_ms: 1_500 }),
			buildSkippedScenario('zoom-in', 'no relayout'),
		],
		BASE_METADATA,
	);

	expect(artifact).toEqual<PerfThresholdArtifact>({
		metadata: {
			...BASE_METADATA,
			ci: false,
			thresholdProfile: 'none',
		},
		thresholds: [],
		scenarios: [
			{ label: 'cold-initial', status: 'measured', skipReason: null },
			{ label: 'zoom-in', status: 'skipped', skipReason: 'no relayout' },
		],
		results: [],
		failures: [],
	});
	expect(buildPerfThresholdFailureMessage(artifact.failures)).toBeNull();
});
