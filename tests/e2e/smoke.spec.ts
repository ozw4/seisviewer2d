import { test, expect } from '@playwright/test';

const BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:8000';

type LinearMoveoutState = {
	enabled: boolean;
	velocityMps: number;
	offsetByte: number;
	offsetScale: number;
	offsetMode: string;
	refMode: string;
	refTrace: number;
	polarity: number;
};

declare global {
	interface Window {
		__lmoEvents: unknown[];
		getCurrentLinearMoveout(): LinearMoveoutState;
		setCurrentLinearMoveout(patch: Partial<LinearMoveoutState>): LinearMoveoutState;
		currentLmoKey(): string;
	}
}

test('upload page loads', async ({ page }) => {
	await page.goto(`${BASE_URL}/upload`);
	await expect(page).toHaveTitle(/Open SEG-Y/i);
});

test('main viewer page loads', async ({ page }) => {
	await page.goto(`${BASE_URL}/`);
	await expect(page.locator('body')).toBeVisible();
});

test('viewer linear moveout controls persist normalized state', async ({ page }) => {
	await page.goto(`${BASE_URL}/`);

	await expect(page.getByText('Linear Moveout')).toBeVisible();
	await expect(page.locator('#lmoEnabled')).not.toBeChecked();
	await expect(page.locator('#lmoVelocityMps')).toHaveValue('1500');
	await expect(page.locator('#lmoOffsetByte')).toHaveValue('37');
	await expect(page.locator('#lmoOffsetScale')).toHaveValue('1');
	await expect(page.locator('#lmoOffsetMode')).toHaveValue('absolute');
	await expect(page.locator('#lmoRefMode')).toHaveValue('min');
	await expect(page.locator('#lmoRefTrace')).toHaveValue('0');
	await expect(page.locator('#lmoRefTrace')).toBeDisabled();
	await expect(page.locator('#lmoPolarity')).toHaveValue('1');

	await page.evaluate(() => {
		window.__lmoEvents = [];
		window.addEventListener('lmo:change', (event) => {
			window.__lmoEvents.push((event as CustomEvent).detail);
		});
	});

	await page.locator('#lmoEnabled').check();
	await page.locator('#lmoVelocityMps').fill('2000');
	await page.locator('#lmoOffsetByte').fill('41');
	await page.locator('#lmoOffsetScale').fill('2.5');
	await page.locator('#lmoOffsetMode').selectOption('signed');
	await page.locator('#lmoRefMode').selectOption('trace');
	await expect(page.locator('#lmoRefTrace')).toBeEnabled();
	await page.locator('#lmoRefTrace').fill('12');
	await page.locator('#lmoPolarity').selectOption('-1');

	await expect.poll(() => page.evaluate(() => window.getCurrentLinearMoveout())).toEqual({
		enabled: true,
		velocityMps: 2000,
		offsetByte: 41,
		offsetScale: 2.5,
		offsetMode: 'signed',
		refMode: 'trace',
		refTrace: 12,
		polarity: -1,
	});
	await expect(page.locator('#lmoRefTrace')).toBeEnabled();
	expect(await page.evaluate(() => window.currentLmoKey())).toBe(
		'lmo:on|v=2000|ob=41|os=2.5|om=signed|rm=trace|rt=12|p=-1',
	);
	expect(await page.evaluate(() => localStorage.getItem('lmo_velocity_mps'))).toBe('2000');

	const cloneMutationResult = await page.evaluate(() => {
		const clone = window.getCurrentLinearMoveout();
		clone.velocityMps = 999;
		return window.getCurrentLinearMoveout().velocityMps;
	});
	expect(cloneMutationResult).toBe(2000);

	const repeatedSetEventCount = await page.evaluate(() => {
		window.__lmoEvents = [];
		window.setCurrentLinearMoveout({ velocityMps: 2000 });
		return window.__lmoEvents.length;
	});
	expect(repeatedSetEventCount).toBe(0);

	const changedSetEvent = await page.evaluate(() => {
		window.__lmoEvents = [];
		window.setCurrentLinearMoveout({ enabled: false, velocityMps: 3500 });
		return {
			count: window.__lmoEvents.length,
			key: window.currentLmoKey(),
			storedEnabled: localStorage.getItem('lmo_enabled'),
		};
	});
	expect(changedSetEvent).toEqual({
		count: 1,
		key: 'lmo:off',
		storedEnabled: 'false',
	});

	await page.reload();
	await expect(page.locator('#lmoEnabled')).not.toBeChecked();
	await expect(page.locator('#lmoVelocityMps')).toHaveValue('3500');
	await expect(page.locator('#lmoOffsetMode')).toHaveValue('signed');
	await expect(page.locator('#lmoRefMode')).toHaveValue('trace');
	await expect(page.locator('#lmoRefTrace')).toBeEnabled();
	expect(await page.evaluate(() => window.currentLmoKey())).toBe('lmo:off');

	await page.evaluate(() => {
		localStorage.setItem('lmo_enabled', 'yes');
		localStorage.setItem('lmo_velocity_mps', '-1');
		localStorage.setItem('lmo_offset_byte', '241');
		localStorage.setItem('lmo_offset_scale', '0');
		localStorage.setItem('lmo_offset_mode', 'bad');
		localStorage.setItem('lmo_ref_mode', 'bad');
		localStorage.setItem('lmo_ref_trace', '-2');
		localStorage.setItem('lmo_polarity', '0');
	});
	await page.reload();
	expect(await page.evaluate(() => window.getCurrentLinearMoveout())).toEqual({
		enabled: false,
		velocityMps: 1500,
		offsetByte: 37,
		offsetScale: 1,
		offsetMode: 'absolute',
		refMode: 'min',
		refTrace: 0,
		polarity: 1,
	});
	await expect(page.locator('#lmoRefTrace')).toBeDisabled();
	expect(await page.evaluate(() => localStorage.getItem('lmo_velocity_mps'))).toBe('1500');
});
