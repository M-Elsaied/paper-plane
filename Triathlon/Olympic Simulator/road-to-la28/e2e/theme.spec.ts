import { test, expect } from "@playwright/test";

test("theme toggle flips and persists across reloads", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 }); // desktop → sidebar toggle
  await page.goto("/rankings");
  // default is dark
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

  // the sidebar theme toggle flips to light
  await page.getByRole("button", { name: /Switch to light theme/i }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");

  // persists after reload (FOUC script reads localStorage)
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});

test("desktop shows the sidebar; mobile shows the bottom nav", async ({ page }) => {
  // desktop viewport → sidebar visible
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/");
  await expect(page.getByRole("link", { name: /Race Week/i }).first()).toBeVisible();

  // mobile viewport → bottom nav (fixed) present
  await page.setViewportSize({ width: 400, height: 800 });
  await page.reload();
  await expect(page.getByRole("link", { name: /Rankings/i }).first()).toBeVisible();
});
