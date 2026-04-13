import { test, expect } from '@playwright/test';

test('upload page loads', async ({ page }) => {
	await page.goto('/upload');
	await expect(page).toHaveTitle(/Seis/i);
});

test('main viewer page loads', async ({ page }) => {
	await page.goto('/');
	await expect(page.locator('body')).toBeVisible();
});
