import type { SvPerfArtifactMetadata } from './perfArtifacts';
import {
	findLatestWindowRowWithMetric,
	toFiniteNumber,
	type PerfCaseLabel,
	type PerfScenarioResult,
} from './perfScenarios';

export type NumericMetricName =
	| 'fetch_ms'
	| 'decode_ms'
	| 'lut_ms'
	| 'prep_ms'
	| 'plotly_ms'
	| 'total_ms'
	| 'server_ms'
	| 'build_ms'
	| 'pack_ms';

export type PerfThreshold = {
	caseLabel: PerfCaseLabel | 'any';
	metric: NumericMetricName;
	max: number;
	required: boolean;
};

export type PerfThresholdResult = {
	caseLabel: PerfCaseLabel;
	metric: NumericMetricName;
	actual: number | null;
	max: number;
	required: boolean;
	status: 'pass' | 'fail' | 'missing-allowed' | 'missing-required' | 'skipped';
};

export type PerfThresholdArtifact = {
	metadata: SvPerfArtifactMetadata & {
		ci: boolean;
		thresholdProfile: 'none' | 'env-only' | 'ci-default' | 'ci-default+env-overrides';
	};
	thresholds: PerfThreshold[];
	scenarios: Array<{
		label: PerfCaseLabel;
		status: PerfScenarioResult['status'];
		skipReason: string | null;
	}>;
	results: PerfThresholdResult[];
	failures: PerfThresholdResult[];
};

type ThresholdDefinition = {
	caseLabel: PerfCaseLabel | 'any';
	metric: NumericMetricName;
	envName: string;
	requiredWhenPresent: boolean;
	ciDefaultMax?: number;
};

const THRESHOLD_DEFINITIONS: ThresholdDefinition[] = [
	{
		caseLabel: 'cold-initial',
		metric: 'total_ms',
		envName: 'SV_PERF_MAX_COLD_TOTAL_MS',
		requiredWhenPresent: true,
		ciDefaultMax: 8_000,
	},
	{
		caseLabel: 'warm-initial',
		metric: 'total_ms',
		envName: 'SV_PERF_MAX_WARM_TOTAL_MS',
		requiredWhenPresent: true,
		ciDefaultMax: 6_000,
	},
	{
		caseLabel: 'zoom-in',
		metric: 'total_ms',
		envName: 'SV_PERF_MAX_ZOOM_TOTAL_MS',
		requiredWhenPresent: false,
	},
	{
		caseLabel: 'pan-after-zoom',
		metric: 'total_ms',
		envName: 'SV_PERF_MAX_PAN_TOTAL_MS',
		requiredWhenPresent: false,
	},
	{
		caseLabel: 'cold-initial',
		metric: 'plotly_ms',
		envName: 'SV_PERF_MAX_COLD_PLOTLY_MS',
		requiredWhenPresent: true,
	},
	{
		caseLabel: 'warm-initial',
		metric: 'plotly_ms',
		envName: 'SV_PERF_MAX_WARM_PLOTLY_MS',
		requiredWhenPresent: true,
	},
	{
		caseLabel: 'zoom-in',
		metric: 'plotly_ms',
		envName: 'SV_PERF_MAX_ZOOM_PLOTLY_MS',
		requiredWhenPresent: true,
		ciDefaultMax: 5_000,
	},
	{
		caseLabel: 'pan-after-zoom',
		metric: 'plotly_ms',
		envName: 'SV_PERF_MAX_PAN_PLOTLY_MS',
		requiredWhenPresent: true,
		ciDefaultMax: 5_000,
	},
	{
		caseLabel: 'any',
		metric: 'fetch_ms',
		envName: 'SV_PERF_MAX_ANY_FETCH_MS',
		requiredWhenPresent: false,
		ciDefaultMax: 6_000,
	},
	{
		caseLabel: 'any',
		metric: 'decode_ms',
		envName: 'SV_PERF_MAX_ANY_DECODE_MS',
		requiredWhenPresent: false,
		ciDefaultMax: 3_000,
	},
	{
		caseLabel: 'any',
		metric: 'plotly_ms',
		envName: 'SV_PERF_MAX_ANY_PLOTLY_MS',
		requiredWhenPresent: true,
	},
	{
		caseLabel: 'any',
		metric: 'server_ms',
		envName: 'SV_PERF_MAX_SERVER_MS',
		requiredWhenPresent: false,
	},
	{
		caseLabel: 'any',
		metric: 'build_ms',
		envName: 'SV_PERF_MAX_BUILD_MS',
		requiredWhenPresent: false,
	},
	{
		caseLabel: 'any',
		metric: 'pack_ms',
		envName: 'SV_PERF_MAX_PACK_MS',
		requiredWhenPresent: false,
	},
];

function buildThresholdKey(threshold: Pick<PerfThreshold, 'caseLabel' | 'metric'>): string {
	return `${threshold.caseLabel}:${threshold.metric}`;
}

function parseOptionalThresholdEnv(name: string): number | null {
	const raw = process.env[name];
	if (!raw) {
		return null;
	}
	const value = Number(raw);
	if (!Number.isFinite(value)) {
		throw new Error(`${name} must be a number`);
	}
	return value;
}

export function buildPerfThresholds(): {
	ci: boolean;
	thresholdProfile: PerfThresholdArtifact['metadata']['thresholdProfile'];
	thresholds: PerfThreshold[];
} {
	const ci = process.env.SV_PERF_CI === '1';
	const thresholdsByKey = new Map<string, PerfThreshold>();
	let explicitThresholdCount = 0;

	for (const definition of THRESHOLD_DEFINITIONS) {
		if (ci && definition.ciDefaultMax !== undefined) {
			const threshold: PerfThreshold = {
				caseLabel: definition.caseLabel,
				metric: definition.metric,
				max: definition.ciDefaultMax,
				required: definition.requiredWhenPresent,
			};
			thresholdsByKey.set(buildThresholdKey(threshold), threshold);
		}
	}

	for (const definition of THRESHOLD_DEFINITIONS) {
		const explicitMax = parseOptionalThresholdEnv(definition.envName);
		if (explicitMax === null) {
			continue;
		}
		explicitThresholdCount += 1;
		const threshold: PerfThreshold = {
			caseLabel: definition.caseLabel,
			metric: definition.metric,
			max: explicitMax,
			required: definition.requiredWhenPresent,
		};
		thresholdsByKey.set(buildThresholdKey(threshold), threshold);
	}

	let thresholdProfile: PerfThresholdArtifact['metadata']['thresholdProfile'] = 'none';
	if (ci && explicitThresholdCount > 0) {
		thresholdProfile = 'ci-default+env-overrides';
	} else if (ci) {
		thresholdProfile = 'ci-default';
	} else if (explicitThresholdCount > 0) {
		thresholdProfile = 'env-only';
	}

	const thresholds = THRESHOLD_DEFINITIONS
		.map((definition) =>
			thresholdsByKey.get(
				buildThresholdKey({
					caseLabel: definition.caseLabel,
					metric: definition.metric,
				}),
			),
		)
		.filter((threshold): threshold is PerfThreshold => threshold !== undefined);

	return { ci, thresholdProfile, thresholds };
}

function readLatestResponseMetric(
	scenario: Extract<PerfScenarioResult, { status: 'measured' }>,
	metric: Extract<NumericMetricName, 'server_ms' | 'build_ms' | 'pack_ms'>,
): number | null {
	for (let index = scenario.responses.length - 1; index >= 0; index -= 1) {
		const response = scenario.responses[index];
		const rawValue =
			metric === 'server_ms'
				? response.headers.xSvServerMs
				: metric === 'build_ms'
					? response.headers.xSvBuildMs
					: response.headers.xSvPackMs;
		if (!rawValue) {
			continue;
		}
		const parsedValue = toFiniteNumber(Number(rawValue));
		if (parsedValue !== null) {
			return parsedValue;
		}
	}
	return null;
}

function readScenarioMetric(
	scenario: PerfScenarioResult,
	metric: NumericMetricName,
): number | null {
	if (scenario.status !== 'measured') {
		return null;
	}
	if (metric === 'server_ms' || metric === 'build_ms' || metric === 'pack_ms') {
		return readLatestResponseMetric(scenario, metric);
	}
	const latestMetricRow = findLatestWindowRowWithMetric(scenario.deltaWindowRows, metric);
	return toFiniteNumber(latestMetricRow?.[metric]);
}

function thresholdMatchesScenario(
	threshold: PerfThreshold,
	scenario: PerfScenarioResult,
): boolean {
	return threshold.caseLabel === 'any' || threshold.caseLabel === scenario.label;
}

export function evaluatePerfThresholds(
	cases: PerfScenarioResult[],
	metadata: SvPerfArtifactMetadata,
): PerfThresholdArtifact {
	const thresholdConfig = buildPerfThresholds();
	const results: PerfThresholdResult[] = [];

	for (const scenario of cases) {
		for (const threshold of thresholdConfig.thresholds) {
			if (!thresholdMatchesScenario(threshold, scenario)) {
				continue;
			}

			if (scenario.status === 'skipped') {
				results.push({
					caseLabel: scenario.label,
					metric: threshold.metric,
					actual: null,
					max: threshold.max,
					required: threshold.required,
					status: 'skipped',
				});
				continue;
			}

			const actual = readScenarioMetric(scenario, threshold.metric);
			if (actual === null) {
				results.push({
					caseLabel: scenario.label,
					metric: threshold.metric,
					actual,
					max: threshold.max,
					required: threshold.required,
					status: threshold.required ? 'missing-required' : 'missing-allowed',
				});
				continue;
			}

			results.push({
				caseLabel: scenario.label,
				metric: threshold.metric,
				actual,
				max: threshold.max,
				required: threshold.required,
				status: actual <= threshold.max ? 'pass' : 'fail',
			});
		}
	}

	const failures = results.filter(
		(result) => result.status === 'fail' || result.status === 'missing-required',
	);

	return {
		metadata: {
			...metadata,
			ci: thresholdConfig.ci,
			thresholdProfile: thresholdConfig.thresholdProfile,
		},
		thresholds: thresholdConfig.thresholds,
		scenarios: cases.map((scenario) => ({
			label: scenario.label,
			status: scenario.status,
			skipReason: scenario.status === 'skipped' ? scenario.skipReason : null,
		})),
		results,
		failures,
	};
}

function formatActualValue(actual: number | null): string {
	return actual === null ? 'null' : `${actual}`;
}

export function buildPerfThresholdFailureMessage(
	failures: PerfThresholdResult[],
): string | null {
	if (failures.length === 0) {
		return null;
	}

	const lines = failures.map(
		(failure) =>
			`- ${failure.caseLabel} ${failure.metric} actual=${formatActualValue(failure.actual)} max=${failure.max}`,
	);
	if (lines.length === 1) {
		return `Performance threshold failed: ${lines[0].slice(2)}`;
	}
	return `Performance thresholds failed:\n${lines.join('\n')}`;
}
